# TRIBE v2 Architecture

## Overview

TRIBE v2 (Transformer-based Representation of Intermodal Brain Encoding, v2) is a multimodal brain encoding model that predicts fMRI responses from naturalistic stimuli (video, audio, text). It uses frozen foundation models for feature extraction and a trainable transformer-based fusion architecture to map multimodal representations to cortical activity.

---

## Full Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          TRIBE v2 ARCHITECTURE                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

                      NATURALISTIC STIMULI INPUT (2 Hz)
              ┌──────────────┬──────────────┬──────────────┐
              │    VIDEO     │    AUDIO     │     TEXT     │
              │  (4s clips)  │  (60s chunks)│ (≤1024 words)│
              └──────┬───────┴──────┬───────┴──────┬───────┘
                     │              │              │
    ╔════════════════╩══════════════╩══════════════╩═══════════════════╗
    ║              FROZEN FOUNDATION MODEL FEATURE EXTRACTORS           ║
    ║                                                                   ║
    ║   ┌──────────────┐  ┌───────────────────┐  ┌─────────────────┐  ║
    ║   │  V-JEPA2     │  │  Wav2Vec-BERT 2.0 │  │  LLaMA 3.2-3B  │  ║
    ║   │  ViT-G       │  │  ~600M params     │  │  3B params      │  ║
    ║   │  1.1B params │  │  1,024 dim        │  │  3,072 dim      │  ║
    ║   │  1,536 dim   │  │  2 layers @ 75%,  │  │  6 layers @     │  ║
    ║   │  2 layers @  │  │  100% depth       │  │  0,20,40,60,    │  ║
    ║   │  75%, 100%   │  │                   │  │  80,100% depth  │  ║
    ║   │  depth       │  │  Bidirectional    │  │  Contextualized │  ║
    ║   │  Spatial avg │  │  encoding         │  │  (k=1024 words) │  ║
    ║   │  over patches│  │                   │  │  via WhisperX   │  ║
    ║   └──────┬───────┘  └────────┬──────────┘  └────────┬────────┘  ║
    ║          │                   │                       │           ║
    ║   (B,2,1536,T)         (B,2,1024,T)           (B,6,3072,T)      ║
    ╚══════════╩═══════════════════╩═══════════════════════╩═══════════╝
               │                   │                       │
               ▼                   ▼                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                  LAYER CONCATENATION (axis=1)                    │
    │                                                                  │
    │   Video: 2 layers × 1,536 dim  →  (B, 3,072, T)                │
    │   Audio: 2 layers × 1,024 dim  →  (B, 2,048, T)                │
    │   Text:  6 layers × 3,072 dim  →  (B, 18,432, T)               │
    └──────────────────────────────────────────────────────────────────┘
               │                   │                       │
               ▼                   ▼                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │               PER-MODALITY MLP PROJECTORS                        │
    │                  (LayerNorm + GELU)                              │
    │                                                                  │
    │   Video: 3,072  →  384   (= 1152 hidden / 3 modalities)        │
    │   Audio: 2,048  →  384                                          │
    │   Text: 18,432  →  384                                          │
    └──────────────────────────────────────────────────────────────────┘
               │                   │                       │
               └───────────────────┼───────────────────────┘
                                   │
                        [MODALITY DROPOUT p=0.3]
                     (zero entire modality, train only)
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                   CONCATENATE MODALITIES                         │
    │                                                                  │
    │        (B, T, 384 + 384 + 384)  →  (B, T, 1152)                │
    └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │              + LEARNABLE POSITIONAL EMBEDDINGS                   │
    │                   (B, T, 1152), max_len=1024                     │
    └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║             BIDIRECTIONAL TRANSFORMER ENCODER (8 layers)         ║
    ║                                                                  ║
    ║   Hidden dim:      1,152                                         ║
    ║   Attention heads: 8  (head_dim = 1152 / 8 = 144)              ║
    ║   Depth:           8 layers                                      ║
    ║   FF expansion:    4×  (inner dim = 1152 × 4 = 4,608)          ║
    ║   Attention dropout: 0.1                                         ║
    ║   Position bias:   Rotary (RoPE)                                 ║
    ║   Norm:            ScaleNorm                                     ║
    ║   Causal:          False (bidirectional)                         ║
    ║                                                                  ║
    ║   ┌──────────────────────────────────────────────────────────┐   ║
    ║   │  Layer 1: Multi-Head Self-Attention (RoPE) + FFN         │   ║
    ║   ├──────────────────────────────────────────────────────────┤   ║
    ║   │  Layer 2: Multi-Head Self-Attention (RoPE) + FFN         │   ║
    ║   ├──────────────────────────────────────────────────────────┤   ║
    ║   │  ...                                                      │   ║
    ║   ├──────────────────────────────────────────────────────────┤   ║
    ║   │  Layer 8: Multi-Head Self-Attention (RoPE) + FFN         │   ║
    ║   └──────────────────────────────────────────────────────────┘   ║
    ║                                                                  ║
    ║   Output: (B, T, 1152)                                           ║
    ╚══════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                    LOW-RANK BOTTLENECK                           │
    │                                                                  │
    │              Linear: 1,152 → 2,048                              │
    │              (B, T, 2048)                                        │
    └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║              SUBJECT-SPECIFIC LINEAR LAYERS                      ║
    ║                                                                  ║
    ║   Weights: (n_subjects + 1, 2048, 20484)                        ║
    ║   Subject dropout: p=0.1 → replaced with avg subject (train)    ║
    ║   Inference mode: uses average subject weights                   ║
    ║                                                                  ║
    ║   Input:  (B, T, 2048)                                          ║
    ║   Output: (B, T, 20,484 vertices)                               ║
    ╚══════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                  ADAPTIVE TEMPORAL POOLING                       │
    │                                                                  │
    │         (B, 20484, T)  →  (B, 20484, T_output)                 │
    └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    fMRI PREDICTIONS (OUTPUT)                     ║
    ║                                                                  ║
    ║   Shape:      (Batch, 20,484 vertices, Time)                    ║
    ║   Space:      fsaverage5 cortical surface                       ║
    ║   Resolution: ~5mm                                               ║
    ║   Coverage:   Full cortex, both hemispheres                     ║
    ║                  LH: 10,242 vertices                            ║
    ║                  RH: 10,242 vertices                            ║
    ╚══════════════════════════════════════════════════════════════════╝
```

---

## Tensor Shapes Through the Network

```
Stage                          Shape
─────────────────────────────────────────────────────
Video features (extracted)     (B, 2, 1536, T)
Audio features (extracted)     (B, 2, 1024, T)
Text features (extracted)      (B, 6, 3072, T)

After layer concat:
  Video                        (B, 3072, T)
  Audio                        (B, 2048, T)
  Text                         (B, 18432, T)

After MLP projection:
  Video                        (B, 384, T)
  Audio                        (B, 384, T)
  Text                         (B, 384, T)

After modality concat          (B, T, 1152)
After pos embedding            (B, T, 1152)
After transformer              (B, T, 1152)
After bottleneck               (B, T, 2048)
After subject layers           (B, T, 20484)
After temporal pooling         (B, 20484, T_out)
```

---

## Key Hyperparameters

| Parameter              | Value                    |
|------------------------|--------------------------|
| Hidden dimension       | 1,152                    |
| Transformer depth      | 8 layers                 |
| Attention heads        | 8 (head_dim = 144)       |
| FF expansion           | 4× (inner = 4,608)       |
| Low-rank bottleneck    | 2,048                    |
| Output vertices        | 20,484 (fsaverage5)      |
| Layer aggregation      | Concatenate              |
| Modality aggregation   | Concatenate              |
| Max sequence length    | 1,024 timesteps          |
| Modality dropout       | p = 0.3                  |
| Subject dropout        | p = 0.1                  |
| Attention dropout      | 0.1                      |
| Position embedding     | Rotary (RoPE)            |
| Normalisation          | ScaleNorm                |
| Causal masking         | No (bidirectional)       |

---

## Feature Extractors (Frozen)

| Modality | Model              | Params  | Output dim | Layers used                    |
|----------|--------------------|---------|------------|--------------------------------|
| Video    | V-JEPA2 ViT-G      | 1.1B    | 1,536      | 75%, 100% depth (2 layers)     |
| Audio    | Wav2Vec-BERT 2.0   | ~600M   | 1,024      | 75%, 100% depth (2 layers)     |
| Text     | LLaMA 3.2-3B       | 3B      | 3,072      | 0, 20, 40, 60, 80, 100% depth (6 layers) |

---

## Training Configuration

| Parameter         | Value                                        |
|-------------------|----------------------------------------------|
| Segment length    | 100 TRs                                      |
| Batch size        | 8                                            |
| Epochs            | 15                                           |
| Optimizer         | Adam, lr = 1e-4, no weight decay             |
| Scheduler         | OneCycleLR (10% warmup + cosine annealing)   |
| Loss              | MSE per vertex                               |
| Evaluation metric | Pearson correlation                          |
| Feature frequency | 2 Hz                                         |
| fMRI frequency    | 1 Hz                                         |
| fMRI offset       | 5 s (hemodynamic delay)                      |

---

## v1 → v2 Key Changes

| Aspect              | v1                          | v2                              |
|---------------------|-----------------------------|---------------------------------|
| Output space        | 1,000 Schaefer parcels      | 20,484 fsaverage5 vertices      |
| Hidden dimension    | 1,024                       | 1,152                           |
| Combiner MLP        | Yes                         | No (transformer handles fusion) |
| Low-rank bottleneck | No                          | Yes (2,048)                     |
| Modality dropout    | 0.2                         | 0.3                             |
| Subject adaptation  | Embedding                   | SubjectLayers + subject_dropout |
| Ensemble            | 1,000 models                | Single model                    |

---

## Source Files

| Component              | File                                  | Lines     |
|------------------------|---------------------------------------|-----------|
| Model config dataclass | `tribev2/tribev2/model.py`            | 49–86     |
| Forward pass           | `tribev2/tribev2/model.py`            | 89–235    |
| Temporal smoothing     | `tribev2/tribev2/model.py`            | 21–46     |
| Default hyperparams    | `tribev2/tribev2/grids/defaults.py`   | 201–214   |
| Training loop          | `tribev2/tribev2/pl_module.py`        | 54–104    |
| Experiment orchestration | `tribev2/tribev2/main.py`           | 82–275    |
| Inference API          | `tribev2/tribev2/demo_utils.py`       | 133–392   |
