"""ONNX export pipeline for Tiny-TRIBE browser deployment.

Exports each component separately for browser use:
    1. Text backbone → text_encoder.onnx
    2. Audio backbone → audio_encoder.onnx
    3. Video backbone → video_encoder.onnx
    4. Fusion + output → fusion.onnx

Then quantizes all to INT8 for ~4x size reduction.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from tiny_tribe.config import TinyTribeConfig
from tiny_tribe.model import TinyTribe, TinyTribeFusionOnly

logger = logging.getLogger(__name__)


class TextEncoderWrapper(nn.Module):
    """Wrapper for ONNX export of text backbone."""

    def __init__(self, backbone):
        super().__init__()
        self.model = backbone.model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class AudioEncoderWrapper(nn.Module):
    """Wrapper for ONNX export of audio backbone (encoder only)."""

    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone.encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


class VideoEncoderWrapper(nn.Module):
    """Wrapper for ONNX export of video backbone (per-frame)."""

    def __init__(self, backbone):
        super().__init__()
        self.model = backbone.model

    def forward(self, pixel_values):
        """pixel_values: (B, 3, H, W) single frame."""
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)  # (B, D)


class FusionWrapper(nn.Module):
    """Wrapper for ONNX export of fusion model (no layer dropout)."""

    def __init__(self, full_model: TinyTribe):
        super().__init__()
        self.text_proj = full_model.text_proj
        self.audio_proj = full_model.audio_proj
        self.video_proj = full_model.video_proj
        self.combiner = full_model.combiner
        self.pos_embed = full_model.pos_embed
        # Use fusion layers directly without layer dropout
        self.fusion_layers = full_model.fusion.layers
        self.fusion_norm = full_model.fusion.norm
        self.low_rank_head = full_model.low_rank_head
        self.subject_layers_weights = full_model.subject_layers.weights
        self.subject_layers_biases = full_model.subject_layers.biases

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        subject_idx: torch.Tensor,
    ) -> torch.Tensor:
        text_proj = self.text_proj(text_feat)
        audio_proj = self.audio_proj(audio_feat)
        video_proj = self.video_proj(video_feat)

        fused = torch.cat([text_proj, audio_proj, video_proj], dim=-1)
        fused = self.combiner(fused)

        T = fused.shape[1]
        fused = fused + self.pos_embed[:, :T, :]

        for layer in self.fusion_layers:
            fused = layer(fused)
        fused = self.fusion_norm(fused)

        bottleneck = self.low_rank_head(fused)

        # Subject-specific output (single subject for browser)
        w = self.subject_layers_weights[subject_idx]
        b = self.subject_layers_biases[subject_idx]
        out = torch.bmm(bottleneck, w) + b

        return out.transpose(1, 2)  # (B, n_vertices, T)


def export_text_encoder(model: TinyTribe, output_dir: Path):
    logger.info("Exporting text encoder...")
    wrapper = TextEncoderWrapper(model.backbones.text)
    wrapper.eval()

    dummy_ids = torch.randint(0, 1000, (1, 32))
    dummy_mask = torch.ones(1, 32, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(output_dir / "text_encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "embeddings": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )
    logger.info(f"  Saved text_encoder.onnx")


def export_audio_encoder(model: TinyTribe, output_dir: Path):
    logger.info("Exporting audio encoder...")
    wrapper = AudioEncoderWrapper(model.backbones.audio)
    wrapper.eval()

    # Whisper expects (B, 80, 3000) mel spectrogram
    dummy_mel = torch.randn(1, 80, 3000)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(output_dir / "audio_encoder.onnx"),
        input_names=["input_features"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_features": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        opset_version=14,
    )
    logger.info(f"  Saved audio_encoder.onnx")


def export_video_encoder(model: TinyTribe, output_dir: Path):
    logger.info("Exporting video encoder...")
    wrapper = VideoEncoderWrapper(model.backbones.video)
    wrapper.eval()

    dummy_frame = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        wrapper,
        (dummy_frame,),
        str(output_dir / "video_encoder.onnx"),
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        opset_version=14,
    )
    logger.info(f"  Saved video_encoder.onnx")


def export_fusion(model: TinyTribe, output_dir: Path, config: TinyTribeConfig):
    logger.info("Exporting fusion model...")
    wrapper = FusionWrapper(model)
    wrapper.eval()

    T = 20  # example sequence length
    dummy_text = torch.randn(1, T, config.backbone.text_dim)
    dummy_audio = torch.randn(1, T, config.backbone.audio_dim)
    dummy_video = torch.randn(1, T, config.backbone.video_dim)
    dummy_subject = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_text, dummy_audio, dummy_video, dummy_subject),
        str(output_dir / "fusion.onnx"),
        input_names=["text_features", "audio_features", "video_features", "subject_id"],
        output_names=["vertex_predictions"],
        dynamic_axes={
            "text_features": {0: "batch", 1: "time"},
            "audio_features": {0: "batch", 1: "time"},
            "video_features": {0: "batch", 1: "time"},
            "vertex_predictions": {0: "batch", 2: "time"},
        },
        opset_version=14,
    )
    logger.info(f"  Saved fusion.onnx")


def quantize_models(output_dir: Path):
    """Apply INT8 static quantization to all exported ONNX models."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.warning("onnxruntime not installed, skipping quantization. "
                       "Install with: pip install onnxruntime")
        return

    for onnx_file in output_dir.glob("*.onnx"):
        if "_int8" in onnx_file.stem:
            continue
        out_path = onnx_file.with_stem(onnx_file.stem + "_int8")
        logger.info(f"Quantizing {onnx_file.name} → {out_path.name}")
        quantize_dynamic(
            str(onnx_file),
            str(out_path),
            weight_type=QuantType.QInt8,
        )

    # Report sizes
    logger.info("\nExported model sizes:")
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")


def export_all(checkpoint_path: str, output_dir: str):
    config = TinyTribeConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TinyTribe(config)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    export_text_encoder(model, output_dir)
    export_audio_encoder(model, output_dir)
    export_video_encoder(model, output_dir)
    export_fusion(model, output_dir, config)
    quantize_models(output_dir)

    logger.info(f"\nAll models exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export Tiny-TRIBE to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="onnx_export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_all(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
