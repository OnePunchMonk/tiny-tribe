"""Configuration for Tiny-TRIBE models and training."""

from dataclasses import dataclass, field


@dataclass
class BackboneConfig:
    # Text: all-MiniLM-L6-v2
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384

    # Audio: Whisper-Tiny encoder-only
    audio_model: str = "openai/whisper-tiny"
    audio_dim: int = 384

    # Video: MobileViT-S
    video_model: str = "apple/mobilevit-small"
    video_dim: int = 640  # MobileViT-S last hidden state


@dataclass
class FusionConfig:
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 1024
    ff_mult: int = 4


@dataclass
class OutputConfig:
    # fsaverage4 = 2562 vertices per hemisphere * 2 = 5124 total
    n_vertices: int = 5124
    low_rank_dim: int = 256
    n_subjects: int = 25


@dataclass
class TrainingConfig:
    # Stage 1: frozen backbones
    stage1_lr: float = 1e-3
    stage1_epochs: int = 5

    # Stage 2: end-to-end with KD
    stage2_lr: float = 1e-4
    stage2_epochs: int = 10

    batch_size: int = 8
    num_workers: int = 4

    # Augmentation
    modality_dropout: float = 0.5
    temporal_dropout: float = 0.1
    feature_noise_std: float = 0.05
    layer_dropout: float = 0.15

    # KD weights
    kd_task_weight: float = 0.7
    kd_output_weight: float = 0.2
    kd_feature_weight: float = 0.1

    # Audio
    audio_sample_rate: int = 16000

    # Video
    video_fps: int = 2
    video_frame_size: int = 256


@dataclass
class TinyTribeConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
