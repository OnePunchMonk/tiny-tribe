# Tiny-TRIBE MoE: Architecture Deep Dive

## 1. Context: What TRIBE v2 Does

TRIBE v2 takes multimodal stimuli (video + audio + text) and predicts
whole-brain fMRI responses at every TR (repetition time, ~1.5s).

**Teacher model (TRIBE v2):**
```
LLaMA 3.2-3B (text, 3072D)  ─┐
Wav2Vec-BERT 2.0 (audio, 1024D) ─┤── concat → 8-layer Transformer (1152D) → 20,484 vertices
V-JEPA2-ViT-G (video, 1536D) ─┘    ~4.7B params total, ~10GB
```

Our job: compress this into something that trains on a T4 and runs in a browser.

---

## 2. Design Philosophy

**Why NOT just shrink everything uniformly?**

Uniform shrinking (smaller backbones + smaller fusion) loses quality fast.
The research shows brain alignment saturates early in backbone scale, but
fusion quality depends heavily on **capacity** — the ability to learn
complex cross-modal interactions.

**MoE solves this perfectly:**
- Total params (capacity) ≫ active params (compute/memory)
- 8 experts in each FFN, but only 2 fire per token
- You get 4x the capacity of a dense model at the same compute cost
- Training memory scales with active params, not total params

**The right tradeoff for T4:**
```
                 Dense 512D        MoE 512D (8 experts, top-2)
                 ──────────        ─────────────────────────────
FFN params/layer:  2.1M            8.4M (4x capacity)
Active params:     2.1M            2.1M (same compute!)
Memory (train):    ~8 MB/layer     ~34 MB/layer (weights stored, not all active)
Quality:           baseline        +10-15% on complex cross-modal patterns
```

---

## 3. Full Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              RAW INPUTS                                     ║
║                                                                             ║
║    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐        ║
║    │    Text Input     │  │   Audio Input    │  │   Video Input    │        ║
║    │ transcript/capts  │  │  waveform 16kHz  │  │  frames @ 2fps  │        ║
║    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘        ║
╚═════════════╪══════════════════════╪══════════════════════╪══════════════════╝
              │                      │                      │
              ▼                      ▼                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FROZEN BACKBONE ENCODERS                                ║
║                     (67.3M params total, frozen)                            ║
║                                                                             ║
║  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐      ║
║  │  all-MiniLM-L6-v2  │ │   Whisper-Tiny     │ │    MobileViT-S    │      ║
║  │      22.7M         │ │     39M            │ │      5.6M         │      ║
║  │  sentence-transf.  │ │  encoder-only      │ │   per-frame CNN   │      ║
║  │                    │ │                    │ │  + global avg pool │      ║
║  │  Always frozen     │ │  Unfreeze Stage 2  │ │  Unfreeze Stage 2 │      ║
║  └────────┬───────────┘ └────────┬───────────┘ └────────┬───────────┘      ║
║           │                      │                      │                   ║
║     (B, T_text, 384)       (B, 1500, 384)        (B, T_vid, 640)          ║
║                                                                             ║
║  These are well-pretrained models. Their features are already               ║
║  high-quality. The fusion model's job is to COMBINE them.                   ║
╚═══════════╪══════════════════════╪══════════════════════╪═══════════════════╝
            │                      │                      │
            ▼                      ▼                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PER-MODALITY PROJECTORS                                 ║
║                     (1.5M params, trainable)                                ║
║                                                                             ║
║  Each projector: LayerNorm → Linear → GELU → Dropout → Linear → LayerNorm ║
║                                                                             ║
║  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐      ║
║  │   Text Projector   │ │  Audio Projector   │ │  Video Projector   │      ║
║  │   384 → 512        │ │   384 → 512        │ │   640 → 512        │      ║
║  │   ~400K params     │ │   ~400K params     │ │   ~700K params     │      ║
║  └────────┬───────────┘ └────────┬───────────┘ └────────┬───────────┘      ║
║           │                      │                      │                   ║
║     (B, T_text, 512)       (B, T_audio, 512)     (B, T_vid, 512)          ║
║                                                                             ║
║  All three modalities now live in the same 512-dim space.                   ║
╚═══════════╪══════════════════════╪══════════════════════╪═══════════════════╝
            │                      │                      │
            ▼                      ▼                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TEMPORAL ALIGNMENT + MODALITY FUSION                      ║
║                                                                             ║
║  1. Temporal interpolation: align all modalities to T = max(T_t, T_a, T_v) ║
║     Uses F.interpolate(mode='linear') on the temporal axis                  ║
║                                                                             ║
║  2. Modality dropout (training only, p=0.3):                                ║
║     Randomly zero out entire modalities per sample                          ║
║     At least one modality always survives                                   ║
║     Forces the model to be robust to missing inputs                         ║
║                                                                             ║
║  3. Add learned modality embeddings:                                        ║
║     3 learned vectors (512-dim), one per modality                           ║
║     text_proj += modality_embed[0]                                          ║
║     audio_proj += modality_embed[1]                                         ║
║     video_proj += modality_embed[2]                                         ║
║                                                                             ║
║  4. Interleave into single sequence:                                        ║
║                                                                             ║
║     ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬─────┐                           ║
║     │t_1│a_1│v_1│t_2│a_2│v_2│t_3│a_3│v_3│ ... │                           ║
║     └───┴───┴───┴───┴───┴───┴───┴───┴───┴─────┘                           ║
║     Length: T × 3 tokens                                                    ║
║                                                                             ║
║     WHY interleave instead of concatenate?                                  ║
║     - Adjacent tokens from different modalities can attend to each other    ║
║     - Text at time t sees audio at time t in the next position              ║
║     - Natural cross-modal attention without extra architecture              ║
║                                                                             ║
║  5. Add positional embeddings (learned, max 2048 positions)                 ║
║                                                                             ║
║  Output: (B, T×3, 512)                                                     ║
╚═══════════════════════════════════╪══════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║              MoE FUSION TRANSFORMER (4 layers, ~38M params)                 ║
║                                                                             ║
║  This is where the magic happens. Each layer has:                           ║
║                                                                             ║
║  ┌────────────────────────────────────────────────────────────────────┐     ║
║  │                     TRANSFORMER BLOCK (×4)                         │     ║
║  │                                                                    │     ║
║  │  ┌──────────────────────────────────────────────────────────┐     │     ║
║  │  │                 SELF-ATTENTION                            │     │     ║
║  │  │                                                          │     │     ║
║  │  │  Pre-LayerNorm → MultiHead Self-Attention → Residual     │     │     ║
║  │  │                                                          │     │     ║
║  │  │  • 8 heads, 64-dim per head (512 total)                  │     │     ║
║  │  │  • All modality tokens attend to all others              │     │     ║
║  │  │  • Text token at t=3 can attend to video token at t=1    │     │     ║
║  │  │  • This IS the cross-modal fusion mechanism              │     │     ║
║  │  │  • ~1.05M params per layer                               │     │     ║
║  │  │                                                          │     │     ║
║  │  └──────────────────────────┬───────────────────────────────┘     │     ║
║  │                             │                                      │     ║
║  │                             ▼                                      │     ║
║  │  ┌──────────────────────────────────────────────────────────┐     │     ║
║  │  │              MIXTURE-OF-EXPERTS FFN                       │     │     ║
║  │  │                                                          │     │     ║
║  │  │  Pre-LayerNorm → Router → Top-2 Expert Selection         │     │     ║
║  │  │                                                          │     │     ║
║  │  │  ┌──────────────────────────────────────────────┐       │     │     ║
║  │  │  │              ROUTER                           │       │     │     ║
║  │  │  │                                               │       │     │     ║
║  │  │  │  Linear(512 → 8) → TopK(k=2) → Softmax       │       │     │     ║
║  │  │  │                                               │       │     │     ║
║  │  │  │  For each token, picks 2 of 8 experts         │       │     │     ║
║  │  │  │  Outputs routing weights (sums to 1)          │       │     │     ║
║  │  │  │                                               │       │     │     ║
║  │  │  │  + Load-balancing auxiliary loss               │       │     │     ║
║  │  │  │    (prevents all tokens going to 1 expert)    │       │     │     ║
║  │  │  └──────────────────────┬────────────────────────┘       │     ║
║  │  │                         │                                │     │     ║
║  │  │              ┌──────────┴──────────┐                     │     │     ║
║  │  │              ▼                     ▼                     │     │     ║
║  │  │  ┌─────────────────┐  ┌─────────────────┐               │     │     ║
║  │  │  │   Expert i      │  │   Expert j      │  (+ 6 idle)   │     │     ║
║  │  │  │  512→1024→512   │  │  512→1024→512   │               │     │     ║
║  │  │  │  GELU+Dropout   │  │  GELU+Dropout   │               │     │     ║
║  │  │  │  ~1.05M params  │  │  ~1.05M params  │               │     │     ║
║  │  │  └────────┬────────┘  └────────┬────────┘               │     │     ║
║  │  │           │    weighted sum     │                        │     │     ║
║  │  │           └─────────┬──────────┘                        │     │     ║
║  │  │                     │                                    │     │     ║
║  │  │                     ▼                                    │     │     ║
║  │  │              + Residual connection                       │     │     ║
║  │  │                                                          │     │     ║
║  │  │  8 experts × 1.05M = 8.4M params per layer              │     │     ║
║  │  │  But only 2 experts active = 2.1M compute per token      │     │     ║
║  │  │                                                          │     │     ║
║  │  └──────────────────────────────────────────────────────────┘     │     ║
║  │                                                                    │     ║
║  │  Layer dropout: 10% chance to skip entire block during training    │     ║
║  │                                                                    │     ║
║  └────────────────────────────────────────────────────────────────────┘     ║
║                                                                             ║
║  × 4 layers                                                                ║
║                                                                             ║
║  Per-layer params:  1.05M (attn) + 8.4M (MoE FFN) = 9.45M                ║
║  Total 4 layers:    37.8M params (but only ~12.6M active per token)        ║
║                                                                             ║
║  Final LayerNorm → (B, T×3, 512)                                          ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────┐      ║
║  │  MODALITY POOLING                                                │      ║
║  │                                                                  │      ║
║  │  Reshape: (B, T×3, 512) → (B, T, 3, 512)                      │      ║
║  │  Mean over modality dim: (B, T, 3, 512) → (B, T, 512)          │      ║
║  │                                                                  │      ║
║  │  This collapses the 3 modality tokens per timestep into one     │      ║
║  │  fused representation. Information has already been mixed        │      ║
║  │  by self-attention across modalities.                            │      ║
║  └──────────────────────────────────────────────────────────────────┘      ║
║                                                                             ║
║  Intermediate activations from each layer are saved for feature-level       ║
║  knowledge distillation (CKA/cosine loss against TRIBE v2 fusion layers).  ║
║                                                                             ║
╚═══════════════════════════════════╪══════════════════════════════════════════╝
                                    │
                              (B, T, 512)
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                            OUTPUT HEAD                                      ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────┐      ║
║  │  LOW-RANK BOTTLENECK                                             │      ║
║  │                                                                  │      ║
║  │  Linear(512 → 256, no bias)                                     │      ║
║  │                                                                  │      ║
║  │  Reduces dimensionality before the expensive per-subject layer.  │      ║
║  │  256 is enough to capture the major modes of brain activity.     │      ║
║  │  Saves 2× params in SubjectLayers compared to 512→vertices.     │      ║
║  └──────────────────────────┬───────────────────────────────────────┘      ║
║                             │                                               ║
║                       (B, T, 256)                                          ║
║                             │                                               ║
║                             ▼                                               ║
║  ┌──────────────────────────────────────────────────────────────────┐      ║
║  │  SUBJECT LAYERS                                                  │      ║
║  │                                                    ┌───────────┐ │      ║
║  │  Per-subject linear mapping:                       │Subject ID │ │      ║
║  │  W[s]: (256, n_vertices) + bias                    │  (B,)     │ │      ║
║  │                                                    └─────┬─────┘ │      ║
║  │  output = x @ W[subject_id] + b[subject_id]             │       │      ║
║  │                                                          │       │      ║
║  │  Each subject has unique brain anatomy, so each         │       │      ║
║  │  needs its own mapping from latent → vertices.    ◄─────┘       │      ║
║  │                                                                  │      ║
║  │  Param count depends on target:                                  │      ║
║  │    Schaefer-400:   256 × 400 × n_subj  ≈  0.4M (4 subjects)    │      ║
║  │    Schaefer-1000:  256 × 1000 × n_subj ≈  1.0M (4 subjects)    │      ║
║  │    fsaverage4:     256 × 5124 × n_subj ≈ 33.0M (25 subjects)   │      ║
║  └──────────────────────────┬───────────────────────────────────────┘      ║
║                             │                                               ║
║                       (B, T, n_vertices)                                   ║
║                             │                                               ║
║                             ▼                                               ║
║  ┌──────────────────────────────────────────────────────────────────┐      ║
║  │  TRANSPOSE + TEMPORAL POOLING                                    │      ║
║  │                                                                  │      ║
║  │  (B, T, n_vertices) → (B, n_vertices, T)                       │      ║
║  │  → AdaptiveAvgPool1d(n_output_TRs) → (B, n_vertices, n_TRs)   │      ║
║  └──────────────────────────┬───────────────────────────────────────┘      ║
╚═════════════════════════════╪════════════════════════════════════════════════╝
                              │
                              ▼
                 ┌────────────────────────┐
                 │  PREDICTED BRAIN MAP   │
                 │  (B, n_vertices, n_TRs)│
                 └────────────────────────┘
```

---

## 4. Why Each Design Choice

### Interleaved tokens vs concatenation

**Concatenation** (Tiny-TRIBE v1): `[text_all | audio_all | video_all]`
- Text tokens only attend to nearby text tokens initially
- Cross-modal interaction requires multiple attention layers to propagate
- Same-timestep cross-modal info requires long-range attention

**Interleaving** (Tiny-TRIBE MoE): `[t1,a1,v1, t2,a2,v2, ...]`
- Text at t=5 is adjacent to audio at t=5 and video at t=5
- Cross-modal interaction happens in the FIRST attention layer
- Same compute, much richer early fusion

### 8 experts, top-2 routing

- **8 experts**: standard in MoE literature (Switch, Mixtral). Enough diversity.
- **top-2**: each token gets 2 expert opinions, blended by the router.
  Top-1 is too sparse (quality drop), top-4 is too dense (no savings).
- **ff_mult=2** (not 4): each expert is 512→1024→512. With 8 experts
  this gives the same total capacity as a dense 512→4096→512 FFN,
  but only uses 25% of the compute.

### Load-balancing loss

Without it, the router learns to send all tokens to 1-2 experts (collapse).
The aux loss penalizes uneven expert utilization:
```
L_aux = num_experts × sum(fraction_tokens_i × fraction_prob_i)
```
Weighted at 0.01× main loss — just enough to prevent collapse.

### 512-dim hidden (not smaller)

- 256-dim: too small, attention heads have only 32 dim each → poor attention patterns
- 512-dim with 8 heads: 64 dim per head, well-proven in transformer literature
- The MoE handles the capacity expansion, so we don't need 1024+ hidden dim

### Modality dropout at 30%

- At 50% (our v1): too aggressive, model doesn't learn cross-modal patterns well
- At 30%: forces robustness while still allowing multi-modal learning
- At 0%: model becomes brittle — fails if one modality is noisy

---

## 5. Parameter Budget

### Target: Algonauts 2025 (Schaefer-1000, 4 subjects)

| Component              | Params    | Active/token | Memory (fp32) |
|------------------------|-----------|-------------|---------------|
| Text projector         | 400K      | 400K        | 1.6 MB        |
| Audio projector        | 400K      | 400K        | 1.6 MB        |
| Video projector        | 700K      | 700K        | 2.8 MB        |
| Modality embeddings    | 1.5K      | 1.5K        | 0.006 MB      |
| Positional embeddings  | 1.0M      | 1.0M        | 4.0 MB        |
| Attention (4 layers)   | 4.2M      | 4.2M        | 16.8 MB       |
| MoE experts (4 layers) | 33.6M     | 8.4M        | 134.4 MB      |
| Router (4 layers)      | 16K       | 16K         | 0.06 MB       |
| LayerNorms             | 20K       | 20K         | 0.08 MB       |
| Low-rank head          | 131K      | 131K        | 0.5 MB        |
| SubjectLayers (4 subj) | 1.0M      | 1.0M        | 4.0 MB        |
| **Total**              | **41.5M** | **16.3M**   | **166 MB**    |

### T4 Training Memory (batch_size=8, T=20)

| Component                    | Memory   |
|------------------------------|----------|
| Model params (fp32)          | 166 MB   |
| Optimizer states (Adam, 2×)  | 332 MB   |
| Gradients                    | 166 MB   |
| Activations (estimated)      | 1.5 GB   |
| Frozen backbones (fp16)      | 135 MB   |
| **Total**                    | **~2.3 GB** |
| **T4 available**             | **15.6 GB** |
| **Headroom**                 | **13.3 GB** |

Plenty of room. Could increase batch_size to 32+ or use fsaverage4 with 25 subjects.

---

## 6. Distillation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PHASE 0: Teacher Inference (run once, save to Google Drive)           │
│                                                                         │
│   ┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐  │
│   │  Videos      │────▶│  TRIBE v2 (4.7B) │────▶│  Save to Drive:    │  │
│   │  posted/*.mp4│     │  from_pretrained  │     │                    │  │
│   │              │     │  ('facebook/      │     │  teacher_preds.pt  │  │
│   │              │     │    tribev2')      │     │  (N, 20484)       │  │
│   │              │     │                  │     │                    │  │
│   │              │     │  model.predict()  │     │  Per-video files   │  │
│   │              │     │                  │     │  on Google Drive    │  │
│   └─────────────┘     └──────────────────┘     └────────────────────┘  │
│                                                                         │
│   This takes ~500ms per second of video on T4.                          │
│   10 minutes of video ≈ 5 minutes inference.                            │
│   Store results to Drive so you don't lose them.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PHASE 1: Frozen Backbone Training (5 epochs)                          │
│                                                                         │
│   Load cached teacher predictions from Drive                            │
│   Freeze: all-MiniLM, Whisper-Tiny, MobileViT-S                       │
│   Train: projectors + MoE transformer + output head                     │
│                                                                         │
│   Loss = MSE(student_pred, teacher_pred)                                │
│        + 0.01 × aux_loss (MoE load balancing)                           │
│                                                                         │
│   LR: 1e-3, Adam, OneCycleLR                                           │
│   ~2-3 hours on T4                                                      │
│   Target: 80% of TRIBE v2 correlation                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PHASE 2: End-to-End KD Fine-Tuning (10 epochs)                       │
│                                                                         │
│   Unfreeze: Whisper-Tiny + MobileViT-S (text stays frozen)             │
│   Load teacher fusion features from Drive (if saved)                    │
│                                                                         │
│   Loss = 0.7 × MSE(student_pred, teacher_pred)                         │
│        + 0.2 × MSE(student_pred, fmri_target)  ← if available          │
│        + 0.1 × cosine_sim(student_feat, teacher_feat)                   │
│        + 0.01 × aux_loss                                                │
│                                                                         │
│   LR: 1e-4, Adam, OneCycleLR                                           │
│   ~5-10 hours on T4                                                     │
│   Target: 90%+ of TRIBE v2 correlation                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. What Makes This Different from Standard Model Distillation

| Aspect | Standard KD | Tiny-TRIBE KD |
|--------|------------|---------------|
| Teacher output | Class logits | 20,484 continuous vertex values |
| Temperature | Softmax temperature τ | Not applicable (regression) |
| Feature matching | Optional | Critical (CKA/cosine on fusion layers) |
| Input processing | Same inputs | Different backbones process same raw data |
| Output space | Same labels | May change (fsaverage5 → fsaverage4) |
| MoE routing | N/A | Aux loss prevents expert collapse |

The key insight: in brain encoding, the **output is high-dimensional and continuous**.
Standard KD tricks (temperature, soft labels) don't apply. Instead, we rely on:
1. Direct MSE matching of predictions
2. Feature-level alignment of intermediate representations
3. The MoE's extra capacity to capture patterns the teacher learned

---

## 8. Expected Performance

| Model             | Params  | Active | Mean Pearson r | Inference | Browser? |
|-------------------|---------|--------|----------------|-----------|----------|
| TRIBE v2 (teacher)| 4.7B    | 4.7B   | 0.31           | ~500ms    | No       |
| Tiny-TRIBE v1     | 47M     | 47M    | 0.24-0.26      | ~350ms    | Yes      |
| **Tiny-TRIBE MoE**| **42M** | **16M**| **0.27-0.29**  | **~300ms**| **Yes**  |

The MoE model should outperform v1 despite fewer active params because:
- 4× expert capacity learns richer cross-modal patterns
- Interleaved tokens enable better early fusion
- Better distillation pipeline (teacher outputs, not proxy features)
