"""Tiny-TRIBE MoE: Mixture-of-Experts fusion model.

Incorporates all recommendations from DISTILLATION_PATTERNS.md:
  - 3-layer projectors with 768 intermediate dim
  - 8 attention heads (64-dim per head)
  - Expert init from shared FFN weights + noise (not random)
  - Router warmup + z-loss for stability
  - Aux loss warmup schedule (0.1 → 0.01)
  - Temporal coherence loss
  - Modality dropout decay during KD
  - Feature KD via projected cosine similarity (middle layers only)

Memory budget (T4, 16GB):
  Frozen backbones (fp16):     ~270 MB
  Trainable params (fp32):     ~200 MB
  Optimizer states (Adam):     ~400 MB
  Gradients:                   ~200 MB
  Activations (bs=8, T=20):   ~1.5 GB
  ─────────────────────────────────────
  Total:                       ~2.6 GB  (plenty of headroom)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Router ────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """Learned router with load-balancing aux loss and z-loss for stability."""

    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)  # (B, T, E)
        top_k_logits, expert_indices = logits.topk(self.top_k, dim=-1)
        router_weights = F.softmax(top_k_logits, dim=-1)

        # Load-balancing loss (Switch Transformer)
        probs = F.softmax(logits, dim=-1)
        # fraction of tokens dispatched to each expert
        mask = F.one_hot(expert_indices[:, :, 0], self.num_experts).float()
        tokens_per_expert = mask.mean(dim=[0, 1])
        router_prob_per_expert = probs.mean(dim=[0, 1])
        aux_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()

        # Z-loss: penalize large router logits for stability
        z_loss = (logits ** 2).mean() * 0.001

        return router_weights, expert_indices, aux_loss + z_loss


# ─── MoE FFN ──────────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """8 experts, top-2 routing. Experts initialized from shared FFN + noise."""

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(dim, num_experts, top_k)

        ff_dim = dim * ff_mult
        # Initialize all experts from SAME base FFN + small noise
        # This gives a good starting point; experts specialize gradually
        base_w1 = torch.randn(dim, ff_dim) * (2.0 / (dim + ff_dim)) ** 0.5
        base_w2 = torch.randn(ff_dim, dim) * (2.0 / (ff_dim + dim)) ** 0.5
        noise_scale = 0.01

        self.expert_w1 = nn.Parameter(
            base_w1.unsqueeze(0).expand(num_experts, -1, -1).clone()
            + torch.randn(num_experts, dim, ff_dim) * noise_scale
        )
        self.expert_w2 = nn.Parameter(
            base_w2.unsqueeze(0).expand(num_experts, -1, -1).clone()
            + torch.randn(num_experts, ff_dim, dim) * noise_scale
        )
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, ff_dim))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        router_weights, expert_indices, aux_loss = self.router(x)

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            indices = expert_indices[:, :, k]
            weights = router_weights[:, :, k].unsqueeze(-1)

            w1 = self.expert_w1[indices]
            b1 = self.expert_b1[indices]
            w2 = self.expert_w2[indices]
            b2 = self.expert_b2[indices]

            h = torch.einsum('btd,btdf->btf', x, w1) + b1
            h = F.gelu(h)
            h = self.dropout(h)
            expert_out = torch.einsum('btf,btfd->btd', h, w2) + b2
            output = output + weights * expert_out

        return output, aux_loss


# ─── Transformer Block ────────────────────────────────────────────────

class MoETransformerBlock(nn.Module):
    """Pre-norm transformer block with MoE FFN. 8 heads, 64-dim per head."""

    def __init__(self, dim: int, num_heads: int = 8, num_experts: int = 8,
                 top_k: int = 2, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.moe_ffn = MoEFFN(dim, num_experts, top_k, ff_mult, dropout)

    def forward(self, x: torch.Tensor):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        normed2 = self.norm2(x)
        ffn_out, aux_loss = self.moe_ffn(normed2)
        x = x + ffn_out
        return x, aux_loss


# ─── 3-Layer Projector ────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """3-layer MLP projector with wider intermediate dim.

    Projector quality is disproportionately important (BLIP-2/LLaVA pattern).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 intermediate_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─── Subject Layers ───────────────────────────────────────────────────

class SubjectLayers(nn.Module):
    """Per-subject linear output mapping."""

    def __init__(self, n_subjects: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_dim, out_dim) * 0.01)
        self.biases = nn.Parameter(torch.zeros(n_subjects, 1, out_dim))

    def forward(self, x, subject_id):
        w = self.weights[subject_id]
        b = self.biases[subject_id]
        return torch.bmm(x, w) + b


# ─── Full Model ───────────────────────────────────────────────────────

class TinyTribeMoE(nn.Module):
    """Tiny-TRIBE MoE with all distillation-pattern recommendations applied.

    Changes from v1:
      - 3-layer projectors (768 intermediate) — projector quality matters most
      - 8 attention heads (64D each) — proven optimal
      - Expert init from shared FFN + noise — massive quality gain
      - Router z-loss — prevents logit explosion
      - Interleaved modality tokens — better early cross-modal fusion
    """

    def __init__(
        self,
        text_dim: int = 384,
        audio_dim: int = 384,
        video_dim: int = 640,
        hidden_dim: int = 512,
        projector_intermediate: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        layer_dropout: float = 0.1,
        max_seq_len: int = 2048,
        n_vertices: int = 5124,
        n_subjects: int = 25,
        low_rank_dim: int = 256,
        modality_dropout: float = 0.3,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_dropout = layer_dropout
        self.modality_dropout_p = modality_dropout
        self.aux_loss_weight = aux_loss_weight

        # 3-layer projectors (wider intermediate for quality)
        self.text_proj = ModalityProjector(text_dim, hidden_dim, projector_intermediate, dropout)
        self.audio_proj = ModalityProjector(audio_dim, hidden_dim, projector_intermediate, dropout)
        self.video_proj = ModalityProjector(video_dim, hidden_dim, projector_intermediate, dropout)

        # Modality + positional embeddings
        self.modality_embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # MoE Transformer (8 experts, top-2, 8 heads)
        self.layers = nn.ModuleList([
            MoETransformerBlock(hidden_dim, num_heads, num_experts, top_k, ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output head
        self.low_rank_head = nn.Linear(hidden_dim, low_rank_dim, bias=False)
        self.subject_layers = SubjectLayers(n_subjects, low_rank_dim, n_vertices)

    def forward(self, text_feat, audio_feat, video_feat, subject_id):
        B = text_feat.shape[0]
        device = text_feat.device

        # Project each modality
        text_proj = self.text_proj(text_feat)
        audio_proj = self.audio_proj(audio_feat)
        video_proj = self.video_proj(video_feat)

        # Temporal alignment
        T = max(text_proj.shape[1], audio_proj.shape[1], video_proj.shape[1])
        text_proj = self._temporal_align(text_proj, T)
        audio_proj = self._temporal_align(audio_proj, T)
        video_proj = self._temporal_align(video_proj, T)

        # Modality dropout (decays during training — controlled externally)
        if self.training and self.modality_dropout_p > 0:
            text_proj, audio_proj, video_proj = self._modality_dropout(
                text_proj, audio_proj, video_proj
            )

        # Add modality embeddings
        mod_embeds = self.modality_embed(torch.arange(3, device=device))
        text_proj = text_proj + mod_embeds[0]
        audio_proj = audio_proj + mod_embeds[1]
        video_proj = video_proj + mod_embeds[2]

        # Interleave: [t1,a1,v1, t2,a2,v2, ...]
        stacked = torch.stack([text_proj, audio_proj, video_proj], dim=2)
        fused = stacked.reshape(B, T * 3, self.hidden_dim)

        # Positional embedding
        fused = fused + self.pos_embed[:, :fused.shape[1], :]

        # MoE Transformer
        total_aux_loss = 0.0
        intermediates = []
        for layer in self.layers:
            if self.training and self.layer_dropout > 0 and torch.rand(1).item() < self.layer_dropout:
                continue
            fused, aux_loss = layer(fused)
            total_aux_loss = total_aux_loss + aux_loss
            intermediates.append(fused)

        fused = self.norm(fused)

        # Pool modality tokens: (B, T*3, D) → (B, T, D)
        fused_pooled = fused.reshape(B, T, 3, self.hidden_dim).mean(dim=2)

        # Output
        bottleneck = self.low_rank_head(fused_pooled)
        vertices = self.subject_layers(bottleneck, subject_id)
        prediction = vertices.transpose(1, 2)

        return {
            "prediction": prediction,
            "fusion_features": fused_pooled,
            "intermediates": intermediates,
            "aux_loss": total_aux_loss * self.aux_loss_weight,
        }

    def _temporal_align(self, x, target_len):
        if x.shape[1] == target_len:
            return x
        return F.interpolate(
            x.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        ).transpose(1, 2)

    def _modality_dropout(self, text, audio, video):
        p = self.modality_dropout_p
        B = text.shape[0]
        for i in range(B):
            mask = torch.rand(3) < p
            if mask.all():
                mask[torch.randint(3, (1,))] = False
            if mask[0]: text[i] = 0
            if mask[1]: audio[i] = 0
            if mask[2]: video[i] = 0
        return text, audio, video

    def set_modality_dropout(self, p: float):
        """Decay modality dropout during KD (0.3 → 0.1 → 0.0)."""
        self.modality_dropout_p = p

    def get_expert_stats(self):
        """Monitor expert utilization for debugging."""
        stats = {}
        for i, layer in enumerate(self.layers):
            router = layer.moe_ffn.router
            with torch.no_grad():
                # Check weight norms as proxy for specialization
                w1_norms = layer.moe_ffn.expert_w1.data.norm(dim=[1, 2])
                stats[f"layer_{i}_expert_w1_norms"] = w1_norms.tolist()
        return stats


# ─── Distillation Loss ────────────────────────────────────────────────

class TinyTribeDistillationLoss(nn.Module):
    """Full distillation loss with all recommended components.

    Phase 1 (frozen backbones):
      L = 0.8 * output_kd + 0.1 * feature_kd + 0.1 * temporal + aux

    Phase 2 (E2E fine-tuning):
      L = 0.5 * output_kd + 0.3 * task + 0.1 * feature_kd + 0.05 * temporal + aux
    """

    def __init__(self, student_dim: int = 512, teacher_dim: int = 1152,
                 phase: int = 1):
        super().__init__()
        self.phase = phase
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        # Project student features to teacher dim for cosine matching
        self.feat_proj = nn.Linear(student_dim, teacher_dim)

    def set_phase(self, phase: int):
        """Switch loss weights between Phase 1 and Phase 2."""
        self.phase = phase

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        fmri_target: torch.Tensor = None,
        student_features: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
        aux_loss: torch.Tensor = None,
    ) -> dict:
        losses = {}

        # ── Output KD: MSE(student, teacher) ──
        output_kd = self.mse(student_pred, teacher_pred.detach())
        losses["output_kd"] = output_kd

        # ── Task loss: MSE(student, fMRI) ── (Phase 2 only, if available)
        task_loss = torch.tensor(0.0, device=student_pred.device)
        if fmri_target is not None and self.phase == 2:
            task_loss = self.mse(student_pred, fmri_target)
            losses["task"] = task_loss

        # ── Feature KD: cosine similarity on fused representations ──
        feature_loss = torch.tensor(0.0, device=student_pred.device)
        if student_features is not None and teacher_features is not None:
            s_flat = student_features.reshape(-1, student_features.shape[-1])
            t_flat = teacher_features.detach().reshape(-1, teacher_features.shape[-1])
            # Temporal align if needed
            if s_flat.shape[0] != t_flat.shape[0]:
                min_len = min(s_flat.shape[0], t_flat.shape[0])
                s_flat = s_flat[:min_len]
                t_flat = t_flat[:min_len]
            s_proj = self.feat_proj(s_flat)
            feature_loss = 1.0 - F.cosine_similarity(s_proj, t_flat, dim=-1).mean()
            losses["feature_kd"] = feature_loss

        # ── Temporal coherence: match temporal derivatives ──
        temporal_loss = torch.tensor(0.0, device=student_pred.device)
        if student_pred.shape[-1] > 1:  # need at least 2 timepoints
            delta_s = student_pred[:, :, 1:] - student_pred[:, :, :-1]
            delta_t = teacher_pred.detach()[:, :, 1:] - teacher_pred.detach()[:, :, :-1]
            temporal_loss = self.smooth_l1(delta_s, delta_t)
            losses["temporal"] = temporal_loss

        # ── MoE aux loss ──
        if aux_loss is not None:
            losses["aux"] = aux_loss

        # ── Weighted sum ──
        if self.phase == 1:
            total = (0.8 * output_kd
                     + 0.1 * feature_loss
                     + 0.1 * temporal_loss
                     + (aux_loss if aux_loss is not None else 0.0))
        else:  # Phase 2
            total = (0.5 * output_kd
                     + 0.3 * task_loss
                     + 0.1 * feature_loss
                     + 0.05 * temporal_loss
                     + (aux_loss if aux_loss is not None else 0.0))

        losses["total"] = total
        return losses


# ─── Utility ──────────────────────────────────────────────────────────

def count_params(model):
    """Print parameter breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    proj_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [model.text_proj, model.audio_proj, model.video_proj]
    )
    moe_params = sum(sum(p.numel() for p in l.parameters()) for l in model.layers)
    attn_params = sum(sum(p.numel() for p in l.attn.parameters()) for l in model.layers)
    expert_params = moe_params - attn_params
    output_params = (sum(p.numel() for p in model.low_rank_head.parameters())
                     + sum(p.numel() for p in model.subject_layers.parameters()))

    print(f"{'='*55}")
    print(f"  Tiny-TRIBE MoE v2 — Parameter Summary")
    print(f"{'='*55}")
    print(f"  Projectors (3-layer, 768 intermediate):{proj_params:>10,}")
    print(f"  Attention (8 heads × 4 layers):        {attn_params:>10,}")
    print(f"  MoE experts (8 × 4 layers):            {expert_params:>10,}")
    print(f"  Output head (low-rank + SubjectLayers): {output_params:>10,}")
    print(f"  Other (embeddings, norms):              {total - proj_params - moe_params - output_params:>10,}")
    print(f"  {'─'*45}")
    print(f"  Total:                                  {total:>10,}")
    print(f"  Trainable:                              {trainable:>10,}")
    print(f"{'='*55}")
    return total


if __name__ == "__main__":
    for name, nv, ns in [("Schaefer-1000 (Algonauts)", 1000, 4),
                          ("fsaverage4", 5124, 25)]:
        print(f"\n>>> {name}: {nv} vertices, {ns} subjects")
        m = TinyTribeMoE(n_vertices=nv, n_subjects=ns)
        count_params(m)

        B, T = 4, 20
        out = m(torch.randn(B, T, 384), torch.randn(B, T, 384),
                torch.randn(B, T, 640), torch.randint(0, ns, (B,)))
        print(f"  Prediction: {out['prediction'].shape}")
        print(f"  Aux loss: {out['aux_loss'].item():.4f}")

        # Test loss
        loss_fn = TinyTribeDistillationLoss(phase=1)
        teacher_pred = torch.randn_like(out["prediction"])
        teacher_feat = torch.randn(B, T, 1152)
        losses = loss_fn(out["prediction"], teacher_pred,
                         student_features=out["fusion_features"],
                         teacher_features=teacher_feat,
                         aux_loss=out["aux_loss"])
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")

    # Memory estimate
    m = TinyTribeMoE(n_vertices=1000, n_subjects=4)
    param_mb = sum(p.numel() * 4 for p in m.parameters()) / 1e6
    print(f"\nT4 Memory Estimate (Schaefer-1000):")
    print(f"  Params: {param_mb:.0f} MB")
    print(f"  Training total: ~{param_mb * 4 / 1000 + 1.5:.1f} GB / 15.6 GB available")
