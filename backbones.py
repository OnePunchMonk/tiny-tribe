"""Tiny backbone wrappers for text, audio, and video.

All backbones are frozen pre-trained models that produce per-timestep
feature embeddings. They are designed to be lightweight and ONNX-exportable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    WhisperModel,
    WhisperFeatureExtractor,
    MobileViTModel,
    MobileViTImageProcessor,
)

from tiny_tribe.config import BackboneConfig


class TextBackbone(nn.Module):
    """all-MiniLM-L6-v2 sentence-transformer encoder.

    Input: list of strings (or pre-tokenized input_ids)
    Output: (B, T, 384) where T = number of tokens
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        self.output_dim = config.text_dim
        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns token-level embeddings (B, T, D)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, T, 384)

    def tokenize(self, texts: list[str], max_length: int = 128) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )


class AudioBackbone(nn.Module):
    """Whisper-Tiny encoder-only backbone.

    Input: raw audio waveform (B, num_samples) at 16kHz
    Output: (B, T, 384) where T = num_samples // 320 (whisper stride)
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        whisper = WhisperModel.from_pretrained(config.audio_model)
        self.encoder = whisper.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.audio_model)
        self.output_dim = config.audio_dim
        self._freeze()

    def _freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: mel spectrogram (B, 80, T_mel) from feature_extractor
        Returns:
            (B, T, 384)
        """
        outputs = self.encoder(input_features)
        return outputs.last_hidden_state  # (B, T, 384)

    def extract_features(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Convert raw waveform to mel spectrogram features."""
        feats = self.feature_extractor(
            audio.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        return feats.input_features.to(audio.device)


class VideoBackbone(nn.Module):
    """MobileViT-S frame-level encoder.

    Processes individual frames and returns per-frame features.
    Input: (B, num_frames, 3, H, W)
    Output: (B, num_frames, 640)
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.model = MobileViTModel.from_pretrained(config.video_model)
        self.feature_extractor = MobileViTImageProcessor.from_pretrained(config.video_model)
        self.output_dim = config.video_dim
        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, num_frames, 3, H, W)
        Returns:
            (B, num_frames, 640) pooled per-frame features
        """
        B, T, C, H, W = pixel_values.shape
        # Process all frames as a batch
        frames_flat = pixel_values.reshape(B * T, C, H, W)
        outputs = self.model(pixel_values=frames_flat)
        # Global average pool spatial dims → (B*T, D)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.reshape(B, T, -1)  # (B, T, 640)


class TinyBackboneStack(nn.Module):
    """Manages all three backbones together."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.text = TextBackbone(config)
        self.audio = AudioBackbone(config)
        self.video = VideoBackbone(config)

    def unfreeze_all(self):
        """Unfreeze for Stage 2 end-to-end fine-tuning."""
        self.audio.unfreeze()
        self.video.unfreeze()
        # Text stays frozen (sentence-transformers are well-pretrained)

    @property
    def output_dims(self) -> dict[str, int]:
        return {
            "text": self.text.output_dim,
            "audio": self.audio.output_dim,
            "video": self.video.output_dim,
        }
