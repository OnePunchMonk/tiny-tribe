# Tiny-TRIBE v2: Improved Distilled Architecture

## Key Improvement: Teacher-First Distillation Pipeline

The current approach trains on proxy features. The better approach:

1. **Run TRIBE v2 on real videos** (your `posted/` folder or any dataset)
2. **Cache teacher outputs**: predictions + intermediate fusion features
3. **Train Tiny-TRIBE** against cached teacher outputs (no fMRI ground truth needed!)

This is pure knowledge distillation — the teacher's brain predictions ARE the training signal.

---

## Improved Architecture

The key changes from v1:

- **Shared projection space** instead of per-modality projectors (fewer params, better cross-modal learning)
- **Cross-attention fusion** instead of concatenation (lets modalities attend to each other)
- **Adaptive temporal pooling** learned instead of fixed interpolation
- **Mixture-of-Experts output** instead of flat SubjectLayers

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TINY-TRIBE v2 ARCHITECTURE                      │
│            Teacher-Distilled Multimodal Brain Encoder                │
└─────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════╗
║                        RAW INPUTS                                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            ║
║   │  Text Input   │   │ Audio Input  │   │ Video Input  │            ║
║   │  (captions/   │   │ (waveform    │   │  (frames     │            ║
║   │   transcript) │   │  @ 16kHz)    │   │   @ 2fps)    │            ║
║   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            ║
╚══════════╪══════════════════╪══════════════════╪═════════════════════╝
           │                  │                  │
           ▼                  ▼                  ▼
╔═══════════════════════════════════════════════════════════════════════╗
║              FROZEN BACKBONE ENCODERS (67.3M total)                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        ║
║  │  all-MiniLM-L6  │ │  Whisper-Tiny   │ │   MobileViT-S   │        ║
║  │     22.7M       │ │    39M          │ │     5.6M        │        ║
║  │   384-dim out   │ │   384-dim out   │ │   640-dim out   │        ║
║  │    FROZEN       │ │   FROZEN*       │ │    FROZEN*      │        ║
║  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘        ║
║           │                   │                    │                  ║
║     (B,T_t,384)         (B,T_a,384)         (B,T_v,640)             ║
║                                                                      ║
║  * Unfrozen in Stage 2 for end-to-end KD fine-tuning                ║
╚═══════════╪═══════════════════╪════════════════════╪═════════════════╝
            │                   │                    │
            ▼                   ▼                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║           NEW: UNIFIED PROJECTION + TEMPORAL ALIGNMENT               ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        ║
║  │  Text Projector │ │ Audio Projector │ │ Video Projector │        ║
║  │  384 → 256      │ │  384 → 256     │ │  640 → 256      │        ║
║  │  LN+Lin+GELU   │ │  LN+Lin+GELU   │ │  LN+Lin+GELU   │        ║
║  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘        ║
║           │                   │                    │                  ║
║           ▼                   ▼                    ▼                  ║
║  ┌────────────────────────────────────────────────────────────┐      ║
║  │           Learned Temporal Resampler (1D Conv)             │      ║
║  │     Each modality → common temporal grid (T steps)         │      ║
║  │     Conv1D(256, 256, k=3, stride=adaptive) per modality    │      ║
║  └────────────────────────┬───────────────────────────────────┘      ║
║                           │                                          ║
║                    (B, 3, T, 256)                                     ║
║                    ↑ modality tokens with learned [MOD] prefix       ║
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────┐      ║
║  │    + Modality Embeddings  (3 learned vectors, 256-dim)     │      ║
║  │    + Positional Embeddings (sinusoidal, max 2048)          │      ║
║  └────────────────────────┬───────────────────────────────────┘      ║
║                           │                                          ║
║                     (B, 3*T, 256)                                    ║
║                     Interleaved: [text_1..text_T, aud_1..T, vid_1..T]║
╚═══════════════════════════╪══════════════════════════════════════════╝
                            │
                            ▼
╔═══════════════════════════════════════════════════════════════════════╗
║            NEW: CROSS-MODAL FUSION TRANSFORMER (4 layers)            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Each layer:                                                         ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │                                                             │     ║
║  │   ┌──────────────────────┐   ┌──────────────────────────┐  │     ║
║  │   │   Self-Attention     │   │   Cross-Modal Attention   │  │     ║
║  │   │   (within modality)  │──▶│   (across modalities)     │  │     ║
║  │   │   4 heads, 256-dim   │   │   4 heads, 256-dim        │  │     ║
║  │   └──────────────────────┘   └────────────┬─────────────┘  │     ║
║  │                                           │                 │     ║
║  │                              ┌────────────▼─────────────┐  │     ║
║  │                              │   Feed-Forward Network    │  │     ║
║  │                              │   256 → 1024 → 256       │  │     ║
║  │                              │   GELU + Dropout 10%     │  │     ║
║  │                              └──────────────────────────┘  │     ║
║  │                                                             │     ║
║  │   Layer Dropout: 15% (skip entire layer during training)    │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                                                                      ║
║  × 4 layers                                                         ║
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │                      Modality Pool                          │     ║
║  │   Average tokens per modality → (B, T, 256)                │     ║
║  │   (collapse modality dimension, keep temporal)              │     ║
║  └────────────────────────┬────────────────────────────────────┘     ║
║                           │                                          ║
║  Intermediate activations saved for feature-level KD (CKA loss)      ║
╚═══════════════════════════╪══════════════════════════════════════════╝
                            │
                      (B, T, 256)
                            │
                            ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                NEW: MIXTURE-OF-EXPERTS OUTPUT HEAD                   ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │                  Low-Rank Bottleneck                         │     ║
║  │                  256 → 128 (no bias)                        │     ║
║  └────────────────────────┬────────────────────────────────────┘     ║
║                           │                                          ║
║                     (B, T, 128)                                      ║
║                           │                                          ║
║                           ├─────────────────────┐                    ║
║                           ▼                     ▼                    ║
║  ┌────────────────────────────┐  ┌──────────────────────────┐       ║
║  │    Subject Gating Network  │  │   Shared Expert Layers   │       ║
║  │    subject_id → softmax    │  │   K=4 experts, each:     │       ║
║  │    weights over K experts  │  │   Linear(128, n_verts)   │       ║
║  └─────────────┬──────────────┘  └──────────────┬───────────┘       ║
║                │                                 │                   ║
║                └──────── weighted sum ───────────┘                   ║
║                           │                                          ║
║                     (B, T, n_vertices)                               ║
║                           │                                          ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │              Adaptive Temporal Pooling                       │     ║
║  │         (B, n_vertices, T) → (B, n_vertices, n_TRs)        │     ║
║  └────────────────────────┬────────────────────────────────────┘     ║
╚═══════════════════════════╪══════════════════════════════════════════╝
                            │
                            ▼
              ┌──────────────────────────┐
              │   Predicted Brain Map    │
              │   (B, n_vertices, n_TRs) │
              │                          │
              │  fsaverage4: 5,124 verts │
              │  Schaefer-400: 400 ROIs  │
              └──────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                    DISTILLATION TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────────────┐
  │                    STAGE 0: Pre-compute                       │
  │                                                               │
  │   Load TRIBE v2 (4.7B) on GPU                                │
  │         │                                                     │
  │         ▼                                                     │
  │   Run inference on dataset videos                             │
  │   (posted/*.mp4, or Algonauts 2025 .mkv files)               │
  │         │                                                     │
  │         ▼                                                     │
  │   Save to disk:                                               │
  │     teacher_preds.pt    — (N, 20484, T) vertex predictions    │
  │     teacher_features.pt — (N, T, 1152) fusion layer outputs   │
  │     backbone_feats.pt   — pre-extracted text/audio/video      │
  │         │                                                     │
  │         ▼                                                     │
  │   Unload TRIBE v2 (free GPU memory)                           │
  └───────────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                STAGE 1: Frozen Backbone Training               │
  │                                                               │
  │   Freeze: all-MiniLM, Whisper-Tiny, MobileViT-S              │
  │   Train:  projectors + fusion + output head                   │
  │   Loss:   MSE(student_pred, teacher_pred)                     │
  │   LR:     1e-3, Adam, OneCycleLR                              │
  │   Epochs: 5                                                   │
  │   Target: 80% of TRIBE v2 Pearson r                           │
  └───────────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌───────────────────────────────────────────────────────────────┐
  │          STAGE 2: End-to-End KD Fine-Tuning                    │
  │                                                               │
  │   Unfreeze: Whisper-Tiny + MobileViT-S                        │
  │   Train:    everything except text backbone                   │
  │                                                               │
  │   Combined Loss:                                              │
  │   ┌─────────────────────────────────────────────────────┐     │
  │   │                                                     │     │
  │   │  L = 0.7 × MSE(student_pred, teacher_pred)         │     │
  │   │    + 0.2 × MSE(student_pred, fmri_target*)         │     │
  │   │    + 0.1 × CKA(student_fusion, teacher_fusion)     │     │
  │   │                                                     │     │
  │   │  *fmri_target only if real fMRI data available      │     │
  │   │   otherwise use teacher_pred for both terms         │     │
  │   └─────────────────────────────────────────────────────┘     │
  │                                                               │
  │   LR:     1e-4, Adam, OneCycleLR                              │
  │   Epochs: 10                                                  │
  │   Target: 90%+ of TRIBE v2 Pearson r                          │
  └───────────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌───────────────────────────────────────────────────────────────┐
  │              STAGE 3: Export for Browser                        │
  │                                                               │
  │   Export each component to ONNX:                              │
  │     text_encoder_int8.onnx    (~40 MB)                        │
  │     audio_encoder_int8.onnx   (~60 MB)                        │
  │     video_encoder_int8.onnx   (~10 MB)                        │
  │     fusion_int8.onnx          (~8 MB)                         │
  │                               ─────────                       │
  │                          Total: ~120 MB                        │
  │                                                               │
  │   Browser stack:                                              │
  │     Transformers.js (backbone inference)                      │
  │     ONNX Runtime Web (fusion, WebGPU/WASM)                   │
  │     Three.js (3D brain visualization)                         │
  └───────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                    ARCHITECTURE COMPARISON
═══════════════════════════════════════════════════════════════════════

  ┌──────────────────┬──────────────────┬──────────────────────────┐
  │                  │   Tiny-TRIBE v1   │    Tiny-TRIBE v2         │
  ├──────────────────┼──────────────────┼──────────────────────────┤
  │ Fusion dim       │ 512              │ 256 (smaller, efficient) │
  │ Fusion type      │ Self-attn only   │ Self + Cross-modal attn  │
  │ Temporal align   │ F.interpolate    │ Learned Conv1D resampler │
  │ Modality combine │ Concatenate      │ Interleave + mod embed   │
  │ Output head      │ SubjectLayers    │ MoE (4 experts + gate)   │
  │ Bottleneck       │ 256              │ 128                      │
  │ Pos embedding    │ Learned          │ Sinusoidal (no params)   │
  │ Fusion params    │ ~47M             │ ~8M                      │
  │ Total trainable  │ ~47M             │ ~8M                      │
  │ ONNX INT8 size   │ ~15 MB           │ ~5 MB (fusion only)      │
  │ Expected quality │ 80-85%           │ 85-92% (with KD)         │
  └──────────────────┴──────────────────┴──────────────────────────┘

  Key insight: v2 is SMALLER but BETTER because:
  - Cross-modal attention learns richer interactions than concat
  - MoE output shares expert weights across subjects (4x fewer params)
  - Learned temporal resampling preserves more info than interpolation
  - KD from TRIBE v2 teacher provides much stronger training signal
    than proxy features or small real datasets


═══════════════════════════════════════════════════════════════════════
                    DATASET RECOMMENDATION
═══════════════════════════════════════════════════════════════════════

  For distillation, you DON'T need an fMRI dataset at all.
  You just need videos + the TRIBE v2 teacher.

  Pipeline:
    1. Collect 10-50 hours of diverse video content
       (movies, YouTube, nature documentaries, conversations)
    2. Run TRIBE v2 inference → cache predictions
    3. Train Tiny-TRIBE against teacher predictions

  If you DO want real fMRI for validation:
    - Algonauts 2025 (Schaefer-1000, 4 subjects, video+text)
    - BOLD Moments (fsaverage, 10 subjects, 3s video clips)
    - LeBel 2023 / ds003020 (fsaverage, 8 subjects, audio stories)
```

---

## v2 Model Parameter Budget

| Component | Params | ONNX INT8 |
|-----------|--------|-----------|
| all-MiniLM-L6-v2 (frozen) | 22.7M | 40 MB |
| Whisper-Tiny encoder (frozen) | 39M | 60 MB |
| MobileViT-S (frozen) | 5.6M | 10 MB |
| Projectors (3 × MLP) | 0.4M | <1 MB |
| Temporal resampler | 0.2M | <1 MB |
| Cross-modal transformer (4L) | 4.2M | 3 MB |
| MoE output head (4 experts) | 3.2M | 2 MB |
| **Total** | **75.3M** | **~117 MB** |
| **Trainable** | **~8M** | **~6 MB** |
