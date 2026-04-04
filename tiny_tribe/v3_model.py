"""Tiny-TRIBE v3 — distilled model with all architecture improvements.

Changes from v2 MoE:
  - Per-modality temporal embeddings (text/audio/video each have own time embed)
  - Temporal Conv1D on video features (captures frame-to-frame motion)
  - HRF decay attention bias in layers 1-2 (neurophysical prior)
  - Full attention in layers 3-4 (global narrative context)
  - Gated modality pooling (learned per-modality weights, not mean)
  - HRF convolution layer (initialized to double-Gamma HRF kernel)
  - FiLM subject conditioning (replaces per-subject weight matrices)
  - Aux loss always accumulated even when layer is stochastically dropped (bug fix)
  - Linear stochastic depth schedule (deeper layers drop more)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gamma as scipy_gamma
import numpy as np


# ── Projector ─────────────────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """3-layer MLP projector. Wider intermediate dim is disproportionately important."""

    def __init__(self, in_dim: int, out_dim: int, intermediate: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Temporal motion module (video only) ───────────────────────────────────────

class TemporalMotionModule(nn.Module):
    """Depthwise Conv1D to capture frame-to-frame motion in video features.

    MobileViT-S is per-frame — this adds temporal dynamics cheaply.
    Groups=dim means each feature has its own 3-tap filter: ~1.5K params.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,  # depthwise — one filter per feature dim
            bias=False,
        )
        # Init: approximate finite difference [−1, 0, +1] to detect change
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[:, 0, 0] = -0.5
        self.conv.weight.data[:, 0, 2] =  0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — transpose for Conv1d which expects (B, D, T)
        motion = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + motion  # residual: original + motion signal


# ── MoE router ────────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """Router with load-balancing aux loss and z-loss for stability."""

    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        logits = self.gate(x)                                   # (B, T, E)
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)  # (B, T, k)
        weights = F.softmax(top_logits, dim=-1)                 # (B, T, k)

        # Load-balancing loss (Switch Transformer style)
        probs = F.softmax(logits, dim=-1)
        mask = F.one_hot(top_idx[:, :, 0], self.num_experts).float()
        frac_tokens = mask.mean(dim=[0, 1])
        frac_probs  = probs.mean(dim=[0, 1])
        aux_loss = self.num_experts * (frac_tokens * frac_probs).sum()

        # Z-loss: penalise large logits (prevents router collapse)
        z_loss = (logits ** 2).mean() * 1e-3

        return weights, top_idx, aux_loss + z_loss


# ── MoE FFN ───────────────────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """8 experts, top-2 routing. Experts init from shared FFN + small noise."""

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = TopKRouter(dim, num_experts, top_k)
        ff = dim * ff_mult

        # All experts start from same base FFN + small noise (prevents collapse)
        scale = math.sqrt(2.0 / (dim + ff))
        base_w1 = torch.randn(dim, ff) * scale
        base_w2 = torch.randn(ff, dim) * scale

        self.w1 = nn.Parameter(
            base_w1.unsqueeze(0).expand(num_experts, -1, -1).clone()
            + torch.randn(num_experts, dim, ff) * 0.01
        )
        self.w2 = nn.Parameter(
            base_w2.unsqueeze(0).expand(num_experts, -1, -1).clone()
            + torch.randn(num_experts, ff, dim) * 0.01
        )
        self.b1 = nn.Parameter(torch.zeros(num_experts, ff))
        self.b2 = nn.Parameter(torch.zeros(num_experts, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        weights, indices, aux_loss = self.router(x)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = indices[:, :, k]       # (B, T)
            w   = weights[:, :, k].unsqueeze(-1)  # (B, T, 1)

            w1 = self.w1[idx]            # (B, T, D, ff)
            b1 = self.b1[idx]            # (B, T, ff)
            w2 = self.w2[idx]            # (B, T, ff, D)
            b2 = self.b2[idx]            # (B, T, D)

            h = torch.einsum("btd,btdf->btf", x, w1) + b1
            h = self.drop(F.gelu(h))
            out = out + w * (torch.einsum("btf,btfd->btd", h, w2) + b2)

        return out, aux_loss


# ── Transformer block (with optional HRF decay attention bias) ────────────────

class MoEBlock(nn.Module):
    """Pre-norm transformer block with MoE FFN.

    hrf_decay: if True, adds a learned temporal decay bias to attention logits.
               Use for layers 1-2 (local context). False for layers 3-4 (global).
    """

    def __init__(self, dim: int, num_heads: int = 8, num_experts: int = 8,
                 top_k: int = 2, ff_mult: int = 2, dropout: float = 0.1,
                 hrf_decay: bool = False):
        super().__init__()
        self.hrf_decay = hrf_decay
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
        self.moe   = MoEFFN(dim, num_experts, top_k, ff_mult, dropout)

        if hrf_decay:
            # Learned decay rate — init so that 6TR (~9s) weight ≈ 0.5
            # bias[i,j] = -alpha * |t_i - t_j| where t = position // 3 (tokens per TR)
            self.log_alpha = nn.Parameter(torch.tensor(math.log(math.log(2) / 6)))

    def _hrf_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build (seq_len, seq_len) attention bias with HRF temporal decay."""
        pos = torch.arange(seq_len, device=device)
        # Each TR has 3 tokens (text, audio, video) — get TR index
        tr_idx = pos // 3
        dist   = (tr_idx.unsqueeze(0) - tr_idx.unsqueeze(1)).abs().float()
        alpha  = self.log_alpha.exp()
        return -alpha * dist   # (seq_len, seq_len)

    def forward(self, x: torch.Tensor):
        B, T3, D = x.shape
        normed = self.norm1(x)

        attn_bias = None
        if self.hrf_decay:
            attn_bias = self._hrf_bias(T3, x.device)  # (T3, T3)

        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=attn_bias,
                                need_weights=False)
        x = x + attn_out

        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss


# ── Gated modality pooling ────────────────────────────────────────────────────

class GatedModalityPool(nn.Module):
    """Learned per-timestep modality gates instead of mean pooling.

    Lets the model learn: visual cortex weights video, Broca's area weights text.
    1,536 params.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_net = nn.Linear(dim, 3, bias=True)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T*3, D)
        B, T3, D = x.shape
        T = T3 // 3
        tokens = x.reshape(B, T, 3, D)        # (B, T, 3, D)

        # Gate input: rough average across modalities
        pool = tokens.mean(dim=2)              # (B, T, D)
        gates = F.softmax(self.gate_net(pool), dim=-1)  # (B, T, 3)

        # Weighted sum across modality dim
        fused = (gates.unsqueeze(-1) * tokens).sum(dim=2)  # (B, T, D)
        return fused


# ── HRF convolution layer ─────────────────────────────────────────────────────

class HRFConvolution(nn.Module):
    """Depthwise Conv1D initialized to canonical double-Gamma HRF.

    Causal: only looks at past TRs (stimulus always precedes BOLD response).
    Fine-tuned during training so region-specific HRF shapes can be learned.
    TR: repetition time in seconds (default 1.5s for nilearn dataset).
    """

    def __init__(self, dim: int, tr: float = 1.5, kernel_trs: int = 8):
        super().__init__()
        self.kernel_trs = kernel_trs

        # Build canonical double-Gamma HRF at the given TR
        t = np.arange(kernel_trs) * tr           # time in seconds
        # SPM double-Gamma: peak at ~6s, undershoot at ~16s
        h = (scipy_gamma.pdf(t, 6, scale=1)
             - scipy_gamma.pdf(t, 16, scale=1) / 6)
        h = h / (np.abs(h).sum() + 1e-8)         # normalise
        h = h[::-1].copy()                        # flip for conv (causal)

        # Depthwise conv: each feature dim has its own HRF kernel
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_trs,
            padding=kernel_trs - 1,              # causal padding
            groups=dim,
            bias=False,
        )
        # Init all channels to canonical HRF
        hrf_tensor = torch.tensor(h, dtype=torch.float32)
        self.conv.weight.data.copy_(
            hrf_tensor.unsqueeze(0).unsqueeze(0).expand(dim, 1, -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        # Conv1d expects (B, D, T)
        out = self.conv(x.transpose(1, 2))
        # Remove non-causal future padding
        out = out[:, :, :T]
        return x + out.transpose(1, 2)  # residual


# ── FiLM subject conditioning ─────────────────────────────────────────────────

class FiLMConditioner(nn.Module):
    """Per-subject scale (gamma) and shift (beta) vectors.

    Replaces per-subject weight matrices (33M params in v2) with
    per-subject FiLM vectors (2 × dim × n_subjects params).
    For 25 subjects at dim=512: 25,600 params vs 32,800,000.

    New subject: freeze everything, learn only 1024 floats.
    """

    def __init__(self, dim: int, n_subjects: int):
        super().__init__()
        self.gamma = nn.Embedding(n_subjects, dim)
        self.beta  = nn.Embedding(n_subjects, dim)
        # Init: identity transform (gamma=1, beta=0)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), subject_id: (B,)
        g = self.gamma(subject_id).unsqueeze(1)  # (B, 1, D)
        b = self.beta(subject_id).unsqueeze(1)   # (B, 1, D)
        return g * x + b


# ── Full Model ─────────────────────────────────────────────────────────────────

class TinyTribeV3(nn.Module):
    """Tiny-TRIBE v3: full distilled brain encoder with all v3 improvements.

    Input (pre-extracted backbone features, NOT raw video/audio/text):
      text_feat:  (B, T, text_dim)   from all-MiniLM-L6-v2
      audio_feat: (B, T, audio_dim)  from Whisper-Tiny encoder
      video_feat: (B, T, video_dim)  from MobileViT-S

    Output:
      prediction: (B, n_vertices, T_out)  brain vertex predictions
    """

    def __init__(
        self,
        text_dim:    int = 384,
        audio_dim:   int = 384,
        video_dim:   int = 640,
        hidden_dim:  int = 512,
        proj_inter:  int = 768,
        num_layers:  int = 4,
        num_heads:   int = 8,
        num_experts: int = 8,
        top_k:       int = 2,
        ff_mult:     int = 2,
        dropout:     float = 0.1,
        max_seq_len: int = 2048,
        n_vertices:  int = 400,        # Schaefer-400 for smoke test; 5124 for full
        n_subjects:  int = 3,          # 3 for nilearn; 4 for CNeuroMod; 25 for all
        low_rank_dim: int = 256,
        tr:          float = 1.5,      # TR in seconds
        modality_dropout: float = 0.3,
        aux_loss_weight:  float = 0.01,
        stoch_depth_max:  float = 0.2, # max drop prob (applied to last layer)
    ):
        super().__init__()
        self.hidden_dim       = hidden_dim
        self.num_layers       = num_layers
        self.modality_dropout = modality_dropout
        self.aux_loss_weight  = aux_loss_weight
        self.stoch_depth_max  = stoch_depth_max

        # ── Projectors (3-layer MLP per modality) ──────────────────────────
        self.text_proj  = ModalityProjector(text_dim,  hidden_dim, proj_inter, dropout)
        self.audio_proj = ModalityProjector(audio_dim, hidden_dim, proj_inter, dropout)
        self.video_proj = ModalityProjector(video_dim, hidden_dim, proj_inter, dropout)

        # ── Temporal motion (video only) ────────────────────────────────────
        self.video_motion = TemporalMotionModule(hidden_dim)

        # ── Per-modality type embeddings ────────────────────────────────────
        self.modality_embed = nn.Embedding(3, hidden_dim)

        # ── Per-modality temporal embeddings (separate for each modality) ───
        self.text_time_embed  = nn.Embedding(max_seq_len, hidden_dim)
        self.audio_time_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.video_time_embed = nn.Embedding(max_seq_len, hidden_dim)

        # ── Shared positional embedding (after interleaving) ─────────────────
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len * 3, hidden_dim) * 0.02
        )

        # ── MoE Transformer (4 layers) ───────────────────────────────────────
        # Layers 0-1: local attention with HRF decay bias
        # Layers 2-3: full global attention
        self.layers = nn.ModuleList([
            MoEBlock(hidden_dim, num_heads, num_experts, top_k, ff_mult, dropout,
                     hrf_decay=(i < 2))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # ── Gated modality pooling ───────────────────────────────────────────
        self.gate_pool = GatedModalityPool(hidden_dim)

        # ── HRF convolution (Gamma-initialized) ─────────────────────────────
        self.hrf_conv = HRFConvolution(hidden_dim, tr=tr, kernel_trs=8)

        # ── Output: shared MLP + FiLM + vertex projection ────────────────────
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp  = nn.Linear(hidden_dim, hidden_dim)
        self.film        = FiLMConditioner(hidden_dim, n_subjects)
        self.vertex_proj = nn.Linear(hidden_dim, n_vertices, bias=False)

        # ── Feature projection for KD loss (student → teacher dim) ───────────
        # Used externally by the loss function
        self.feat_proj = nn.Linear(hidden_dim, 1152, bias=False)

    def forward(
        self,
        text_feat:   torch.Tensor,   # (B, T, 384)
        audio_feat:  torch.Tensor,   # (B, T, 384)
        video_feat:  torch.Tensor,   # (B, T, 640)
        subject_id:  torch.Tensor,   # (B,)
        n_out_trs:   int = None,
    ) -> dict:

        B, T, _ = text_feat.shape
        device   = text_feat.device

        # ── Project each modality ────────────────────────────────────────────
        tp = self.text_proj(text_feat)       # (B, T, D)
        ap = self.audio_proj(audio_feat)     # (B, T, D)
        vp = self.video_proj(video_feat)     # (B, T, D)
        vp = self.video_motion(vp)           # (B, T, D) + motion signal

        # ── Per-modality temporal embeddings ─────────────────────────────────
        t_idx = torch.arange(T, device=device)
        tp = tp + self.text_time_embed(t_idx)
        ap = ap + self.audio_time_embed(t_idx)
        vp = vp + self.video_time_embed(t_idx)

        # ── Per-modality type embeddings ─────────────────────────────────────
        mod = self.modality_embed(torch.arange(3, device=device))
        tp = tp + mod[0]
        ap = ap + mod[1]
        vp = vp + mod[2]

        # ── Modality dropout (training only) ─────────────────────────────────
        if self.training and self.modality_dropout > 0:
            tp, ap, vp = self._modality_dropout(tp, ap, vp, B)

        # ── Temporal alignment to same T ─────────────────────────────────────
        T_max = max(tp.shape[1], ap.shape[1], vp.shape[1])
        tp = self._align(tp, T_max)
        ap = self._align(ap, T_max)
        vp = self._align(vp, T_max)

        # ── Interleave: [t1, a1, v1, t2, a2, v2, ...] ───────────────────────
        x = torch.stack([tp, ap, vp], dim=2)     # (B, T, 3, D)
        x = x.reshape(B, T_max * 3, self.hidden_dim)  # (B, T*3, D)
        x = x + self.pos_embed[:, :x.shape[1]]

        # ── MoE Transformer ───────────────────────────────────────────────────
        total_aux = torch.tensor(0.0, device=device)
        intermediates = []

        for i, layer in enumerate(self.layers):
            # Always compute aux_loss (bug fix: don't skip inside the if)
            x_out, aux = layer(x)
            total_aux = total_aux + aux

            # Stochastic depth: linear schedule (deeper = higher drop prob)
            drop_prob = (i / (self.num_layers - 1)) * self.stoch_depth_max
            if self.training and drop_prob > 0 and torch.rand(1).item() < drop_prob:
                pass   # skip update, keep x — but aux was already accumulated
            else:
                x = x_out

            intermediates.append(x)

        x = self.norm(x)

        # ── Gated modality pooling → (B, T, D) ───────────────────────────────
        fused = self.gate_pool(x)

        # ── HRF convolution ───────────────────────────────────────────────────
        fused = self.hrf_conv(fused)

        # ── Output MLP + FiLM ─────────────────────────────────────────────────
        out = F.gelu(self.output_mlp(self.output_norm(fused)))  # (B, T, D)
        out = self.film(out, subject_id)                         # (B, T, D)
        vertices = self.vertex_proj(out)                         # (B, T, n_vertices)

        # ── Pool to output TRs ────────────────────────────────────────────────
        vertices = vertices.transpose(1, 2)   # (B, n_vertices, T)
        if n_out_trs is not None and n_out_trs != T_max:
            vertices = F.adaptive_avg_pool1d(vertices, n_out_trs)

        return {
            "prediction":    vertices,                                   # (B, n_v, T)
            "fusion_feat":   fused,                                      # (B, T, D)
            "intermediates": intermediates,
            "aux_loss":      total_aux * self.aux_loss_weight,
        }

    def _align(self, x: torch.Tensor, target: int) -> torch.Tensor:
        if x.shape[1] == target:
            return x
        return F.interpolate(x.transpose(1, 2), size=target,
                             mode="linear", align_corners=False).transpose(1, 2)

    def _modality_dropout(self, t, a, v, B):
        p = self.modality_dropout
        for i in range(B):
            mask = torch.rand(3) < p
            if mask.all():
                mask[torch.randint(3, (1,))] = False
            if mask[0]: t = t.clone(); t[i] = 0
            if mask[1]: a = a.clone(); a[i] = 0
            if mask[2]: v = v.clone(); v[i] = 0
        return t, a, v

    def set_modality_dropout(self, p: float):
        self.modality_dropout = p

    def count_params(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "projectors":  n(self.text_proj) + n(self.audio_proj) + n(self.video_proj),
            "video_motion": n(self.video_motion),
            "embeddings":  n(self.modality_embed) + n(self.text_time_embed)
                           + n(self.audio_time_embed) + n(self.video_time_embed),
            "transformer": sum(n(l) for l in self.layers),
            "gate_pool":   n(self.gate_pool),
            "hrf_conv":    n(self.hrf_conv),
            "output":      n(self.output_mlp) + n(self.film) + n(self.vertex_proj),
            "total":       n(self),
        }
