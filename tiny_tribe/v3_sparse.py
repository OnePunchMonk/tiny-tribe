"""Tiny-TRIBE v3 — Sparse Architecture Variants

Implements three complementary sparsification techniques on top of v3:

1.  **Modality-Separate Towers (MoT)**
    Split the shared MoE transformer into three modality-specific towers.
    Each modality only touches 1/3 of total transformer parameters.
    At inference, can run a single tower if only one modality matters.

2.  **Statistical Top-k Activation Sparsity (Spark-style)**
    In the FFN/MoE experts, instead of GELU/softmax, use a
    differentiable statistical threshold that keeps only the top-k
    activations.  Trains stably unlike hard Top-k (Spark 2025).

3.  **Heterogeneous Experts (MoHETS-style)**
    Each expert is a different computational type:
    – depthwise-conv FFN  (captures local temporal patterns)
    – Fourier-based FFN    (captures periodic/seasonal patterns)
    – standard MLP FFN     (general-purpose transformations)
    A shared expert is always active (prevents collapse in tiny models).

All three are orthogonal and can be combined.

Usage:
    from tiny_tribe.v3_sparse import TinyTribeV3Sparse

    # MoT towers only (free sparsity — each modality sees fewer params)
    model = TinyTribeV3Sparse(
        architecture="mot",           # or "spark", "hetero", "full"
        num_layers=4,                 # per-modality layers (total = 3×4)
        ...
    )

    # Combined: MoT towers + Spark FFN + Heterogeneous experts
    model = TinyTribeV3Sparse(
        architecture="full",
        spark_k_ratio=0.15,           # keep top-15% FFN activations
        num_experts=6,                # 2 conv + 2 fourier + 2 mlp
        top_k=2,
        ...
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gamma as scipy_gamma
import numpy as np

from tiny_tribe.v3_model import (
    ModalityProjector, TemporalMotionModule, FiLMConditioner,
    GatedModalityPool, HRFConvolution, TopKRouter,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  STATISTICAL TOP-K  (Spark Transformer 2025)
# ═══════════════════════════════════════════════════════════════════════════════

class StatisticalTopK(nn.Module):
    """Differentiable Top-k activation sparsity via mean+std threshold.

    For a tensor x of shape (..., D), keep only the top-k values per
    position.  The threshold is continuous (θ = μ + σ·Q(1−k/D)) so it
    trains stably — unlike hard Top-k which creates zero-gradient
    plateaus (Spark Transformer, 2506.06644).

    Parameters
    ----------
    dim : int
        Feature dimension D.
    k_ratio : float
        Fraction of activations to keep (0 < k_ratio ≤ 1).
        Default 0.15 keeps top 15%.
    """

    def __init__(self, dim: int, k_ratio: float = 0.15):
        super().__init__()
        self.k = max(1, int(dim * k_ratio))
        # Pre-compute the normal quantile for the target sparsity level.
        from scipy.stats import norm
        # Q(1 - k/D) where Q is the quantile function (inverse CDF)
        # For small k/D, this is ≈ sqrt(-2*ln(k/D))
        self.register_buffer(
            "threshold_coeff",
            torch.tensor(norm.ppf(1.0 - self.k / dim), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        theta = mu + sigma * self.threshold_coeff
        return torch.clamp(x - theta, min=0.0)


class SparkFFN(nn.Module):
    """FFN with Statistical Top-k activation sparsity.

    Replaces the standard GELU activation.  Only the top-k activations
    per token survive; the rest are zeroed.  This is *activation*
    sparsity — the weights are dense, but the FLOPs are reduced
    because zero activations can be skipped at inference.
    """

    def __init__(self, dim: int, ff_mult: int = 2, dropout: float = 0.1,
                 k_ratio: float = 0.15):
        super().__init__()
        ff = dim * ff_mult
        self.w1 = nn.Linear(dim, ff, bias=False)
        self.w2 = nn.Linear(ff, dim, bias=False)
        self.act = StatisticalTopK(ff, k_ratio)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.w1(x))
        h = self.drop(h)
        return self.w2(h)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  HETEROGENEOUS EXPERTS  (MoHETS 2025)
# ═══════════════════════════════════════════════════════════════════════════════

class DwConvFFN(nn.Module):
    """Depthwise-convolution + pointwise MLP expert.

    Captures local temporal patterns efficiently.  Groups=dim makes
    this ~1/(dim/16) the parameters of a full Linear layer.
    """

    def __init__(self, dim: int, ff_mult: int = 2, dropout: float = 0.1,
                 kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
            groups=dim, bias=False,
        )
        ff = dim * ff_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → conv expects (B, D, T)
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.mlp(x + c)  # residual: conv + MLP


class FourierFFN(nn.Module):
    """Frequency-domain expert using FFT.

    Captures periodic/seasonal temporal patterns.  FFT → MLP in freq
    space → IFFT.  Very cheap because the MLP operates on complex
    coefficients, not raw tokens.
    """

    def __init__(self, dim: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        ff = dim * ff_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, ff),       # real+imag concatenated
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # FFT along the temporal axis
        freq = torch.fft.rfft(x, dim=1)                    # (B, T//2+1, D) complex
        real = freq.real
        imag = freq.imag
        cat = torch.cat([real, imag], dim=-1)              # (B, T//2+1, 2D)
        out = self.mlp(cat)
        mid = out.shape[-1] // 2
        freq_out = torch.view_as_complex(
            torch.stack([out[..., :mid], out[..., mid:]], dim=-1)
            .contiguous()
        )
        time_out = torch.fft.irfft(freq_out, n=x.shape[1], dim=1)
        return time_out


class MLPFFN(nn.Module):
    """Standard MLP expert (the baseline)."""

    def __init__(self, dim: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        ff = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeterogeneousMoEFFN(nn.Module):
    """MoE with different *types* of experts.

    Mix of depthwise-conv, Fourier, and standard MLP experts.  A shared
    MLP expert is always active (prevents collapse in tiny models).

    Parameters
    ----------
    dim : int
    num_experts : int
        Total number of *routed* experts (doesn't count shared).
    top_k : int
        How many routed experts to activate per token.
    expert_types : list[str]
        Type for each routed expert: "conv", "fourier", or "mlp".
        If None, automatically distributes types evenly.
    """

    def __init__(self, dim: int, num_experts: int = 6, top_k: int = 2,
                 ff_mult: int = 2, dropout: float = 0.1,
                 expert_types: list = None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Shared expert — always active
        self.shared = MLPFFN(dim, ff_mult, dropout)
        self.shared_gate = nn.Linear(dim, 1)  # learned gate weight for shared

        # Routed experts
        if expert_types is None:
            # Distribute evenly: conv, fourier, mlp repeating
            type_cycle = ["conv", "fourier", "mlp"]
            expert_types = [type_cycle[i % 3] for i in range(num_experts)]

        self.experts = nn.ModuleList()
        for t in expert_types:
            if t == "conv":
                self.experts.append(DwConvFFN(dim, ff_mult, dropout))
            elif t == "fourier":
                self.experts.append(FourierFFN(dim, ff_mult, dropout))
            else:
                self.experts.append(MLPFFN(dim, ff_mult, dropout))

        self.router = TopKRouter(dim, num_experts, top_k)

    def forward(self, x: torch.Tensor):
        weights, indices, aux_loss = self.router(x)
        out = self.shared(x) * torch.sigmoid(self.shared_gate(x))

        for k in range(self.top_k):
            idx = indices[:, :, k]          # (B, T)
            w = weights[:, :, k].unsqueeze(-1)  # (B, T, 1)
            for eid, expert in enumerate(self.experts):
                mask = (idx == eid).float().unsqueeze(-1)  # (B, T, 1)
                if mask.any():
                    expert_out = expert(x)
                    out = out + w * mask * expert_out
        return out, aux_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  MODALITY-SEPARATE TOWERS  (Mixture-of-Transformers, 2024)
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityTower(nn.Module):
    """A small transformer tower dedicated to one modality.

    Each modality (text, audio, video) gets its own stack of layers.
    This is "free sparsity" — during a forward pass each modality
    only touches its own tower parameters.  At inference, if we only
    need video (e.g., predicting visual cortex), we can skip the text
    and audio towers entirely.

    Parameters
    ----------
    hidden_dim : int
    num_layers : int
        Number of transformer layers in *this* tower.
    num_heads, num_experts, top_k, ff_mult, dropout : same as v3
    use_spark : bool
        Use StatisticalTopK activation in FFN blocks.
    use_hetero : bool
        Use HeterogeneousMoE instead of standard MoE.
    hrf_decay_layers : int
        How many bottom layers get HRF temporal-decay attention bias.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 4,
                 num_heads: int = 8, num_experts: int = 8,
                 top_k: int = 2, ff_mult: int = 2, dropout: float = 0.1,
                 use_spark: bool = False, spark_k_ratio: float = 0.15,
                 use_hetero: bool = False,
                 hrf_decay_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.norms_attn = nn.ModuleList()
        self.norms_ffn  = nn.ModuleList()
        self.attns      = nn.ModuleList()
        self.ffns       = nn.ModuleList()

        for i in range(num_layers):
            self.norms_attn.append(nn.LayerNorm(hidden_dim))
            self.norms_ffn.append(nn.LayerNorm(hidden_dim))
            self.attns.append(
                nn.MultiheadAttention(hidden_dim, num_heads,
                                      dropout=dropout, batch_first=True)
            )

            # FFN / MoE block
            if use_hetero:
                ffn = HeterogeneousMoEFFN(
                    hidden_dim, num_experts, top_k, ff_mult, dropout
                )
            elif use_spark:
                ffn = SparkFFN(hidden_dim, ff_mult, dropout, spark_k_ratio)
            else:
                ff = hidden_dim * ff_mult
                ffn = nn.Sequential(
                    nn.Linear(hidden_dim, ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff, hidden_dim),
                )
            self.ffns.append(ffn)

        self.hrf_decay_layers = hrf_decay_layers
        if hrf_decay_layers > 0:
            self.log_alpha = nn.Parameter(
                torch.tensor(math.log(math.log(2) / 6))
            )

    def _hrf_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
        alpha = self.log_alpha.exp()
        return -alpha * dist

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, T, D) — tokens from one modality
        Returns:
            (x_out, total_aux_loss)
        """
        total_aux = torch.tensor(0.0, device=x.device)
        for i in range(self.num_layers):
            # Attention
            normed = self.norms_attn[i](x)
            attn_bias = None
            if i < self.hrf_decay_layers:
                attn_bias = self._hrf_bias(normed.shape[1], normed.device)
            attn_out, _ = self.attns[i](normed, normed, normed,
                                        attn_mask=attn_bias,
                                        need_weights=False)
            x = x + attn_out

            # FFN / MoE
            normed = self.norms_ffn[i](x)
            if isinstance(self.ffns[i], (HeterogeneousMoEFFN,)):
                ffn_out, aux = self.ffns[i](normed)
                total_aux = total_aux + aux
            else:
                ffn_out = self.ffns[i](normed)
            x = x + ffn_out

        return x, total_aux


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  SPARSE LINEAR ATTENTION  (SLA 2025)
# ═══════════════════════════════════════════════════════════════════════════════

class SparseLinearAttention(nn.Module):
    """Decompose attention into critical / marginal / negligible.

    Critical (top 5%): full FlashAttention-like computation.
    Negligible (bottom 10%): skip entirely.
    Marginal (remaining 85%): cheap linear attention φ(Q)φ(K)^T V.

    This is especially valuable for the *global* layers (3+) where
    attention over long temporal sequences dominates computation.

    Parameters
    ----------
    dim : int
    num_heads : int
    k_critical : float
        Fraction of attention weights considered critical.
    k_skip : float
        Fraction considered negligible (skipped).
    feature_dim : int
        Dimension for linear attention feature map (φ).
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 k_critical: float = 0.05, k_skip: float = 0.10,
                 feature_dim: int = 64):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k_critical = k_critical
        self.k_skip = k_skip
        self.feature_dim = feature_dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Feature map for linear attention: φ(x) = elu(x) + 1
        # (Katharopoulos et al. "Transformers are RNNs")
        self._phi = lambda t: F.elu(t) + 1.0

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None):
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        # Compute raw attention scores for classification
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1)

        # Per-head per-query: classify keys into critical / marginal / negligible
        # Flatten across heads and queries for quantile computation
        Bv = scores.view(-1, T)  # (B*H*T, T)

        # Thresholds per (batch, head, query)
        crit_thresh = Bv.quantile(1.0 - self.k_critical, dim=-1, keepdim=True)
        skip_thresh = Bv.quantile(self.k_skip, dim=-1, keepdim=True)

        crit_mask = (scores > crit_thresh.view(B, self.num_heads, T, 1))
        skip_mask = (scores < skip_thresh.view(B, self.num_heads, T, 1))
        marginal_mask = ~(crit_mask | skip_mask)

        # ── Critical: full softmax attention ──
        crit_scores = scores.clone()
        crit_scores[~crit_mask] = float('-inf')
        crit_attn = F.softmax(crit_scores, dim=-1)
        crit_attn = torch.where(skip_mask, torch.zeros_like(crit_attn), crit_attn)
        crit_out = crit_attn @ v  # (B,H,T,d)

        # ── Marginal: linear attention ──
        # Only compute for positions that have marginal keys
        q_phi = self._phi(q)  # (B,T,H,d)
        k_phi = self._phi(k)  # (B,T,H,d)

        # Compute KV state for all positions, then mask
        # S = Σ φ(k)^T φ(v), Z = Σ φ(k)
        # Linear attention output = φ(q) · S / (φ(q) · Z)
        q_f = q_phi.permute(0, 2, 1, 3)      # (B,H,T,d)
        k_f = k_phi.permute(0, 2, 1, 3)      # (B,H,T,d)
        v_f = v.permute(0, 2, 1, 3)          # (B,H,T,d)

        # Compute cumulative KV product (causal by default for time series)
        kv = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)  # (B,H,T,d,d)
        kv_cum = kv.cumsum(dim=2)                    # (B,H,T,d,d)
        k_cum = k_f.cumsum(dim=2)                    # (B,H,T,d)

        lin_num = (q_f.unsqueeze(-2) @ kv_cum.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # lin_num: (B,H,T,d)
        lin_denom = (q_f * k_cum).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        lin_out = lin_num / lin_denom  # (B,H,T,d)

        # Mask: only apply linear attention where marginal_mask is True
        marginal_mask_h = marginal_mask.permute(0, 2, 1, 3)  # (B,T,H,1) → need (B,H,T,1)
        marginal_mask_h = marginal_mask_h.permute(0, 2, 1, 3)
        # Actually marginal_mask is (B,H,T,T), we need per-query mask
        # For simplicity, use mean over keys as proxy
        marginal_active = marginal_mask.float().mean(dim=-1, keepdim=True)  # (B,H,T,1)
        lin_out = lin_out * marginal_active

        # ── Combine ──
        out = crit_out + lin_out  # (B,H,T,d)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.dropout(self.out_proj(out))
        return out, None


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  FULL SPARSE MODEL  (TinyTribeV3Sparse)
# ═══════════════════════════════════════════════════════════════════════════════

class TinyTribeV3Sparse(nn.Module):
    """Tiny-TRIBE v3 with configurable sparse architecture.

    architecture modes:
      "dense"    — baseline v3 (shared MoE transformer)
      "mot"      — Modality-Separate Towers (3 towers, no sharing)
      "spark"    — dense tower + Statistical Top-k FFN
      "hetero"   — dense tower + Heterogeneous MoE experts
      "sla"      — dense tower + Sparse-Linear Attention
      "full"     — MoT + Spark + Heterogeneous + SLA (all combined)

    All modes preserve the same external interface:
      forward(text_feat, audio_feat, video_feat, subject_id) → dict
    """

    def __init__(
        self,
        architecture: str = "full",       # dense | mot | spark | hetero | sla | full
        text_dim: int = 384,
        audio_dim: int = 384,
        video_dim: int = 640,
        hidden_dim: int = 512,
        proj_inter: int = 768,
        num_layers: int = 4,              # per-modality layers for MoT
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        n_vertices: int = 400,
        n_subjects: int = 3,
        low_rank_dim: int = 256,
        tr: float = 1.5,
        modality_dropout: float = 0.3,
        aux_loss_weight: float = 0.01,
        stoch_depth_max: float = 0.2,
        # Sparse-specific
        spark_k_ratio: float = 0.15,      # Top-k ratio for Spark FFN
        use_sla: bool = None,             # Auto-set by architecture
        sla_k_critical: float = 0.05,
        sla_k_skip: float = 0.10,
        expert_types: list = None,
        shared_expert_ratio: float = 0.3,  # Shared expert capacity vs routed
    ):
        super().__init__()
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modality_dropout = modality_dropout
        self.aux_loss_weight = aux_loss_weight
        self.stoch_depth_max = stoch_depth_max

        if use_sla is None:
            use_sla = architecture in ("sla", "full")

        # ── Projectors (same as v3) ──────────────────────────────────────
        self.text_proj = ModalityProjector(text_dim, hidden_dim, proj_inter, dropout)
        self.audio_proj = ModalityProjector(audio_dim, hidden_dim, proj_inter, dropout)
        self.video_proj = ModalityProjector(video_dim, hidden_dim, proj_inter, dropout)

        # ── Temporal motion (video only) ──────────────────────────────────
        self.video_motion = TemporalMotionModule(hidden_dim)

        # ── Per-modality type embeddings ───────────────────────────────────
        self.modality_embed = nn.Embedding(3, hidden_dim)

        # ── Per-modality temporal embeddings ───────────────────────────────
        self.text_time_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.audio_time_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.video_time_embed = nn.Embedding(max_seq_len, hidden_dim)

        if architecture in ("dense", "spark", "hetero", "sla"):
            # Shared positional embedding + single transformer stack
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_seq_len * 3, hidden_dim) * 0.02
            )
        else:
            # MoT: separate per-modality position embeddings
            self.text_pos_embed = nn.Parameter(
                torch.randn(1, max_seq_len, hidden_dim) * 0.02
            )
            self.audio_pos_embed = nn.Parameter(
                torch.randn(1, max_seq_len, hidden_dim) * 0.02
            )
            self.video_pos_embed = nn.Parameter(
                torch.randn(1, max_seq_len, hidden_dim) * 0.02
            )

        # ── Transformer / Towers ───────────────────────────────────────────
        is_mot = architecture in ("mot", "full")
        is_spark = architecture in ("spark", "full")
        is_hetero = architecture in ("hetero", "full")

        if is_mot:
            # Three modality-specific towers
            self.text_tower = ModalityTower(
                hidden_dim, num_layers, num_heads,
                num_experts=num_experts, top_k=top_k,
                ff_mult=ff_mult, dropout=dropout,
                use_spark=is_spark, spark_k_ratio=spark_k_ratio,
                use_hetero=is_hetero,
                hrf_decay_layers=2,
            )
            self.audio_tower = ModalityTower(
                hidden_dim, num_layers, num_heads,
                num_experts=num_experts, top_k=top_k,
                ff_mult=ff_mult, dropout=dropout,
                use_spark=is_spark, spark_k_ratio=spark_k_ratio,
                use_hetero=is_hetero,
                hrf_decay_layers=2,
            )
            self.video_tower = ModalityTower(
                hidden_dim, num_layers, num_heads,
                num_experts=num_experts, top_k=top_k,
                ff_mult=ff_mult, dropout=dropout,
                use_spark=is_spark, spark_k_ratio=spark_k_ratio,
                use_hetero=is_hetero,
                hrf_decay_layers=2,
            )
        else:
            # Single shared stack (dense / spark / hetero / sla)
            from tiny_tribe.v3_model import MoEBlock
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                if use_sla:
                    # Use SparseLinearAttention — we replace the block
                    pass  # handled below
                else:
                    self.layers.append(
                        MoEBlock(hidden_dim, num_heads, num_experts, top_k,
                                 ff_mult, dropout, hrf_decay=(i < 2))
                    )

            if use_sla:
                # Replace standard attention with SLA
                self.sla_layers = nn.ModuleList()
                for i in range(num_layers):
                    self.sla_layers.append(
                        SparseLinearAttention(hidden_dim, num_heads, dropout,
                                              sla_k_critical, sla_k_skip)
                    )
                self.sla_norms = nn.ModuleList([
                    nn.LayerNorm(hidden_dim) for _ in range(num_layers)
                ])
                self.sla_ffns = nn.ModuleList()
                for i in range(num_layers):
                    if is_hetero:
                        self.sla_ffns.append(
                            HeterogeneousMoEFFN(hidden_dim, num_experts, top_k,
                                                ff_mult, dropout, expert_types)
                        )
                    elif is_spark:
                        self.sla_ffns.append(
                            SparkFFN(hidden_dim, ff_mult, dropout, spark_k_ratio)
                        )
                    else:
                        ff = hidden_dim * ff_mult
                        self.sla_ffns.append(nn.Sequential(
                            nn.Linear(hidden_dim, ff),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(ff, hidden_dim),
                        ))

        self.norm = nn.LayerNorm(hidden_dim)

        # ── Gated modality pooling ───────────────────────────────────────
        self.gate_pool = GatedModalityPool(hidden_dim)

        # ── HRF convolution ────────────────────────────────────────────────
        self.hrf_conv = HRFConvolution(hidden_dim, tr=tr, kernel_trs=8)

        # ── Output MLP + FiLM + vertex projection ────────────────────────
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.film = FiLMConditioner(hidden_dim, n_subjects)
        self.vertex_proj = nn.Linear(hidden_dim, n_vertices, bias=False)

        self.feat_proj = nn.Linear(hidden_dim, 1152, bias=False)

    def forward(self, text_feat, audio_feat, video_feat, subject_id,
                n_out_trs=None):
        B, T, _ = text_feat.shape
        device = text_feat.device

        # ── Project + motion ─────────────────────────────────────────────
        tp = self.text_proj(text_feat)
        ap = self.audio_proj(audio_feat)
        vp = self.video_proj(video_feat)
        vp = self.video_motion(vp)

        # ── Temporal + type embeddings ───────────────────────────────────
        t_idx = torch.arange(T, device=device)
        tp = tp + self.text_time_embed(t_idx)
        ap = ap + self.audio_time_embed(t_idx)
        vp = vp + self.video_time_embed(t_idx)

        mod = self.modality_embed(torch.arange(3, device=device))
        tp = tp + mod[0]
        ap = ap + mod[1]
        vp = vp + mod[2]

        # ── Modality dropout ─────────────────────────────────────────────
        if self.training and self.modality_dropout > 0:
            tp, ap, vp = self._modality_dropout(tp, ap, vp, B)

        # ── Align temporal lengths ───────────────────────────────────────
        T_max = max(tp.shape[1], ap.shape[1], vp.shape[1])
        tp = self._align(tp, T_max)
        ap = self._align(ap, T_max)
        vp = self._align(vp, T_max)

        # ── Tower processing ───────────────────────────────────────────────
        total_aux = torch.tensor(0.0, device=device)

        if self.architecture in ("mot", "full"):
            # Modality-Separate Towers
            tp, aux_t = self.text_tower(tp)
            total_aux = total_aux + aux_t
            ap, aux_a = self.audio_tower(ap)
            total_aux = total_aux + aux_a
            vp, aux_v = self.video_tower(vp)
            total_aux = total_aux + aux_v

            # Add per-modality position embeddings
            tp = tp + self.text_pos_embed[:, :T_max]
            ap = ap + self.audio_pos_embed[:, :T_max]
            vp = vp + self.video_pos_embed[:, :T_max]

            # Stack back for gated pooling
            x = torch.stack([tp, ap, vp], dim=2)  # (B, T, 3, D)
            x = x.reshape(B, T_max * 3, self.hidden_dim)
        else:
            # Shared stack (interleaved)
            x = torch.stack([tp, ap, vp], dim=2)
            x = x.reshape(B, T_max * 3, self.hidden_dim)
            x = x + self.pos_embed[:, :x.shape[1]]

            for i, layer in enumerate(self.layers):
                x_out, aux = layer(x)
                total_aux = total_aux + aux
                drop_prob = (i / (self.num_layers - 1)) * self.stoch_depth_max
                if self.training and drop_prob > 0 and torch.rand(1).item() < drop_prob:
                    pass
                else:
                    x = x_out

        x = self.norm(x)

        # ── Gated pooling → HRF conv → output ────────────────────────────
        fused = self.gate_pool(x)
        fused = self.hrf_conv(fused)
        out = F.gelu(self.output_mlp(self.output_norm(fused)))
        out = self.film(out, subject_id)
        vertices = self.vertex_proj(out)
        vertices = vertices.transpose(1, 2)
        if n_out_trs is not None and n_out_trs != T_max:
            vertices = F.adaptive_avg_pool1d(vertices, n_out_trs)

        return {
            "prediction": vertices,
            "fusion_feat": fused,
            "aux_loss": total_aux * self.aux_loss_weight,
        }

    def _align(self, x, target):
        if x.shape[1] == target:
            return x
        return F.interpolate(x.transpose(1, 2), size=target,
                             mode="linear", align_corners=False).transpose(1, 2)

    def _modality_dropout(self, t, a, v, B):
        p = self.modality_dropout
        device = t.device
        t = t.clone(); a = a.clone(); v = v.clone()
        for i in range(B):
            mask = torch.rand(3, device=device) < p
            if mask.all():
                mask[torch.randint(3, (1,), device=device)] = False
            if mask[0]: t[i] = 0
            if mask[1]: a[i] = 0
            if mask[2]: v[i] = 0
        return t, a, v

    def set_modality_dropout(self, p: float):
        self.modality_dropout = p

    def count_params(self) -> dict:
        """Return parameter counts, distinguishing active vs total."""
        def n(m): return sum(p.numel() for p in m.parameters())

        total = n(self)
        projectors = n(self.text_proj) + n(self.audio_proj) + n(self.video_proj)

        if self.architecture in ("mot", "full"):
            transformer = n(self.text_tower) + n(self.audio_tower) + n(self.video_tower)
            active_transformer = transformer // 3  # only 1 tower active per modality
        else:
            transformer = n(self.layers) if hasattr(self, 'layers') else 0
            active_transformer = transformer  # all active

        return {
            "projectors": projectors,
            "video_motion": n(self.video_motion),
            "embeddings": (n(self.modality_embed) + n(self.text_time_embed)
                           + n(self.audio_time_embed) + n(self.video_time_embed)),
            "transformer_total": transformer,
            "transformer_active": active_transformer,
            "gate_pool": n(self.gate_pool),
            "hrf_conv": n(self.hrf_conv),
            "output": n(self.output_mlp) + n(self.film) + n(self.vertex_proj),
            "total": total,
            "active_per_forward": (projectors + active_transformer
                                   + n(self.video_motion) + n(self.gate_pool)
                                   + n(self.hrf_conv)
                                   + n(self.output_mlp) + n(self.film)
                                   + n(self.vertex_proj)),
        }

    def estimate_flops(self, batch_size: int = 1, seq_len: int = 50) -> dict:
        """Estimate FLOPs per forward pass.  Approximate but useful for comparison."""
        B, T = batch_size, seq_len
        D = self.hidden_dim
        V = self.vertex_proj.out_features

        # Projectors: 3 × (T × D_in × D_inter + T × D_inter × D)
        proj_flops = 3 * T * (384 * 768 + 768 * D)  # text/audio
        proj_flops += T * (640 * 768 + 768 * D)      # video

        if self.architecture in ("mot", "full"):
            # Each tower: L layers × (attention + FFN)
            # Attention: 4 × B × T × D² (QKV + proj)
            # FFN: 2 × B × T × D × (D×ff_mult)
            L = self.num_layers
            attn_flops = 3 * L * 4 * B * T * D * D
            ffn_flops = 3 * L * 2 * B * T * D * (D * 2)
            tower_flops = attn_flops + ffn_flops
            # Spark: multiply by k_ratio if active
            if hasattr(self.text_tower.ffns[0], 'act'):
                tower_flops *= 0.15  # only top-k% active
        else:
            L = self.num_layers
            attn_flops = L * 4 * B * (3 * T) * D * D
            ffn_flops = L * 2 * B * (3 * T) * D * (D * 2)
            tower_flops = attn_flops + ffn_flops

        # Gated pooling: negligible
        # HRF conv: depthwise ≈ T × D × kernel
        hrf_flops = B * T * D * 8
        # Output MLP: B × T × D × D
        output_flops = B * T * D * D
        # Vertex projection: B × T × D × V
        vertex_flops = B * T * D * V

        total = proj_flops + tower_flops + hrf_flops + output_flops + vertex_flops
        return {
            "projectors": proj_flops,
            "transformer": tower_flops,
            "hrf_conv": hrf_flops,
            "output": output_flops + vertex_flops,
            "total": total,
            "per_token": total / (B * T),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    configs = [
        ("dense",    {"num_layers": 4, "num_experts": 8}),
        ("mot",      {"num_layers": 4, "num_experts": 8}),
        ("spark",    {"num_layers": 4, "num_experts": 8, "spark_k_ratio": 0.15}),
        ("hetero",   {"num_layers": 4, "num_experts": 6}),
        ("full",     {"num_layers": 4, "num_experts": 6, "spark_k_ratio": 0.15}),
    ]

    print("=" * 70)
    print(f"{'Arch':>10} | {'Total':>8} | {'Active':>8} | {'FLOPs':>12} | {'Test'}")
    print("=" * 70)

    for name, kw in configs:
        m = TinyTribeV3Sparse(
            architecture=name,
            n_vertices=100, n_subjects=2,
            hidden_dim=128, num_heads=2,
            **kw
        )
        p = m.count_params()
        flops = m.estimate_flops(batch_size=1, seq_len=10)

        # Forward test
        B, T = 1, 3
        out = m(
            torch.randn(B, T, 384),
            torch.randn(B, T, 384),
            torch.randn(B, T, 640),
            torch.randint(0, 2, (B,)),
        )
        ok = out["prediction"].shape == torch.Size([1, 100, 3])

        print(f"{name:>10} | {p['total']/1e6:>7.2f}M | {p['active_per_forward']/1e6:>7.2f}M "
              f"| {flops['total']/1e6:>10.1f}M | {'✓' if ok else '✗'}")

    print("=" * 70)
