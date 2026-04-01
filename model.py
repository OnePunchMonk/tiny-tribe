"""Tiny-TRIBE fusion model: projectors + transformer + output head.

Architecture (from tiny_tribe_research.md):
    Per-modality projectors (3x MLP) → each to hidden//3
    Concatenate → (B, T, hidden)
    + Positional embeddings (max_seq_len)
    4-layer Transformer (512 hidden, 4 heads)
    Low-rank head: 512 → 256
    SubjectLayers: 256 → 5,124 vertices (fsaverage4)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_tribe.config import TinyTribeConfig
from tiny_tribe.backbones import TinyBackboneStack


class ModalityProjector(nn.Module):
    """MLP projector: input_dim → output_dim with LayerNorm + GELU."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class FusionTransformer(nn.Module):
    """Multi-layer transformer for fusing projected modality features."""

    def __init__(self, config: TinyTribeConfig):
        super().__init__()
        fc = config.fusion
        self.layers = nn.ModuleList([
            TransformerBlock(fc.hidden_dim, fc.num_heads, fc.ff_mult, fc.dropout)
            for _ in range(fc.num_layers)
        ])
        self.norm = nn.LayerNorm(fc.hidden_dim)
        self.layer_dropout = config.training.layer_dropout

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns final output and list of intermediate activations (for KD)."""
        intermediates = []
        for layer in self.layers:
            if self.training and self.layer_dropout > 0 and torch.rand(1).item() < self.layer_dropout:
                continue
            x = layer(x)
            intermediates.append(x)
        x = self.norm(x)
        return x, intermediates


class SubjectLayers(nn.Module):
    """Per-subject linear output mapping.

    Each subject gets their own linear layer from bottleneck → n_vertices,
    because brain anatomy varies across subjects.
    """

    def __init__(self, n_subjects: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_dim, out_dim) * 0.01)
        self.biases = nn.Parameter(torch.zeros(n_subjects, 1, out_dim))

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)
            subject_id: (B,) integer subject indices
        Returns:
            (B, T, out_dim)
        """
        # Gather per-subject weights
        w = self.weights[subject_id]  # (B, in_dim, out_dim)
        b = self.biases[subject_id]   # (B, 1, out_dim)
        return torch.bmm(x, w) + b


class TinyTribe(nn.Module):
    """Complete Tiny-TRIBE model: backbones + fusion + output."""

    def __init__(self, config: TinyTribeConfig):
        super().__init__()
        self.config = config
        bc = config.backbone
        fc = config.fusion
        oc = config.output

        # --- Backbones (frozen) ---
        self.backbones = TinyBackboneStack(bc)

        # --- Per-modality projectors → hidden//3 each ---
        proj_dim = fc.hidden_dim // 3
        self.text_proj = ModalityProjector(bc.text_dim, proj_dim, fc.dropout)
        self.audio_proj = ModalityProjector(bc.audio_dim, proj_dim, fc.dropout)
        self.video_proj = ModalityProjector(bc.video_dim, proj_dim, fc.dropout)

        # Combiner: concat of 3 projections → hidden_dim
        # (proj_dim * 3 might not equal hidden_dim exactly due to integer division)
        concat_dim = proj_dim * 3
        self.combiner = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, fc.hidden_dim),
            nn.GELU(),
        )

        # --- Positional embedding ---
        self.pos_embed = nn.Parameter(torch.randn(1, fc.max_seq_len, fc.hidden_dim) * 0.02)

        # --- Fusion transformer ---
        self.fusion = FusionTransformer(config)

        # --- Output head ---
        self.low_rank_head = nn.Linear(fc.hidden_dim, oc.low_rank_dim, bias=False)
        self.subject_layers = SubjectLayers(oc.n_subjects, oc.low_rank_dim, oc.n_vertices)

        # --- Temporal pooling ---
        self.pooler = nn.AdaptiveAvgPool1d(1)  # pool to single TR prediction

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        audio_features: torch.Tensor,
        video_frames: torch.Tensor,
        subject_id: torch.Tensor,
        n_output_trs: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            text_ids: (B, T_text) token ids
            text_mask: (B, T_text) attention mask
            audio_features: (B, 80, T_mel) mel spectrogram
            video_frames: (B, T_vid, 3, H, W)
            subject_id: (B,) integer subject indices
            n_output_trs: number of fMRI TRs to predict

        Returns:
            dict with 'prediction' (B, n_vertices, n_output_trs) and
            'fusion_features' (for KD)
        """
        # --- Extract backbone features ---
        text_feat = self.backbones.text(text_ids, text_mask)     # (B, T_text, 384)
        audio_feat = self.backbones.audio(audio_features)        # (B, T_audio, 384)
        video_feat = self.backbones.video(video_frames)          # (B, T_vid, 640)

        # --- Project each modality ---
        text_proj = self.text_proj(text_feat)     # (B, T_text, proj_dim)
        audio_proj = self.audio_proj(audio_feat)  # (B, T_audio, proj_dim)
        video_proj = self.video_proj(video_feat)  # (B, T_vid, proj_dim)

        # --- Align temporal dimensions via interpolation ---
        target_len = max(text_proj.shape[1], audio_proj.shape[1], video_proj.shape[1])
        text_proj = self._temporal_align(text_proj, target_len)
        audio_proj = self._temporal_align(audio_proj, target_len)
        video_proj = self._temporal_align(video_proj, target_len)

        # --- Modality dropout (training augmentation) ---
        if self.training and self.config.training.modality_dropout > 0:
            text_proj, audio_proj, video_proj = self._apply_modality_dropout(
                text_proj, audio_proj, video_proj
            )

        # --- Feature noise augmentation ---
        if self.training and self.config.training.feature_noise_std > 0:
            noise_std = self.config.training.feature_noise_std
            text_proj = text_proj + torch.randn_like(text_proj) * noise_std
            audio_proj = audio_proj + torch.randn_like(audio_proj) * noise_std
            video_proj = video_proj + torch.randn_like(video_proj) * noise_std

        # --- Concatenate and combine ---
        fused = torch.cat([text_proj, audio_proj, video_proj], dim=-1)  # (B, T, proj_dim*3)
        fused = self.combiner(fused)  # (B, T, hidden_dim)

        # --- Add positional embeddings ---
        T = fused.shape[1]
        fused = fused + self.pos_embed[:, :T, :]

        # --- Fusion transformer ---
        fused, intermediates = self.fusion(fused)  # (B, T, hidden_dim)

        # --- Low-rank projection ---
        bottleneck = self.low_rank_head(fused)  # (B, T, low_rank_dim)

        # --- Subject-specific output ---
        vertices = self.subject_layers(bottleneck, subject_id)  # (B, T, n_vertices)

        # --- Pool to output TRs ---
        vertices = vertices.transpose(1, 2)  # (B, n_vertices, T)
        if n_output_trs == 1:
            out = self.pooler(vertices)  # (B, n_vertices, 1)
        else:
            out = F.adaptive_avg_pool1d(vertices, n_output_trs)

        return {
            "prediction": out,
            "fusion_features": fused,
            "intermediates": intermediates,
        }

    def _temporal_align(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate temporal dimension to target length."""
        if x.shape[1] == target_len:
            return x
        # (B, T, D) → (B, D, T) for interpolation → (B, D, T') → (B, T', D)
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def _apply_modality_dropout(self, text, audio, video):
        """Randomly zero out entire modalities during training."""
        p = self.config.training.modality_dropout
        B = text.shape[0]
        for i in range(B):
            mask = torch.rand(3) < p
            # Ensure at least one modality survives
            if mask.all():
                mask[torch.randint(3, (1,))] = False
            if mask[0]:
                text[i] = 0
            if mask[1]:
                audio[i] = 0
            if mask[2]:
                video[i] = 0
        return text, audio, video


class TinyTribeFusionOnly(nn.Module):
    """Fusion-only model for ONNX export (backbones exported separately).

    Takes pre-extracted features as input instead of raw data.
    """

    def __init__(self, config: TinyTribeConfig):
        super().__init__()
        self.config = config
        bc = config.backbone
        fc = config.fusion
        oc = config.output

        proj_dim = fc.hidden_dim // 3
        self.text_proj = ModalityProjector(bc.text_dim, proj_dim, fc.dropout)
        self.audio_proj = ModalityProjector(bc.audio_dim, proj_dim, fc.dropout)
        self.video_proj = ModalityProjector(bc.video_dim, proj_dim, fc.dropout)

        concat_dim = proj_dim * 3
        self.combiner = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, fc.hidden_dim),
            nn.GELU(),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, fc.max_seq_len, fc.hidden_dim) * 0.02)
        self.fusion = FusionTransformer(config)
        self.low_rank_head = nn.Linear(fc.hidden_dim, oc.low_rank_dim, bias=False)
        self.subject_layers = SubjectLayers(oc.n_subjects, oc.low_rank_dim, oc.n_vertices)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_feat: (B, T, 384) pre-extracted text features
            audio_feat: (B, T, 384) pre-extracted audio features
            video_feat: (B, T, 640) pre-extracted video features
            subject_id: (B,)
        Returns:
            (B, n_vertices, T) vertex predictions
        """
        text_proj = self.text_proj(text_feat)
        audio_proj = self.audio_proj(audio_feat)
        video_proj = self.video_proj(video_feat)

        fused = torch.cat([text_proj, audio_proj, video_proj], dim=-1)
        fused = self.combiner(fused)

        T = fused.shape[1]
        fused = fused + self.pos_embed[:, :T, :]

        fused, _ = self.fusion(fused)
        bottleneck = self.low_rank_head(fused)
        vertices = self.subject_layers(bottleneck, subject_id)

        return vertices.transpose(1, 2)  # (B, n_vertices, T)
