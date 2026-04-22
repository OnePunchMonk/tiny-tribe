# Strategy C: Deep Dive
## Self-Supervised Pre-training → Teacher KD → fMRI Fine-tuning

---

## Part 1: Full Architecture Diagram — Tiny-TRIBE v3

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                           TINY-TRIBE v3 COMPLETE FORWARD PASS                        ║
║                    ~14M trainable params, ~45M total, ~280ms/s on T4                 ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 0: INPUTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────┐    ┌───────────────────────┐    ┌──────────────────────────┐
  │   TEXT INPUT          │    │    AUDIO INPUT         │    │    VIDEO INPUT            │
  │                       │    │                        │    │                           │
  │  Word-level events    │    │  Raw waveform 16kHz    │    │  Frames at 2fps           │
  │  from WhisperX ASR    │    │  from video audio      │    │  RGB, 224×224             │
  │  with timestamps      │    │  track                 │    │                           │
  │                       │    │                        │    │  e.g. 60s → 120 frames    │
  │  "the dog [1.2s]      │    │  16000 samples/sec     │    │                           │
  │   ran [1.5s]          │    │  → mel spectrogram     │    │                           │
  │   quickly [1.8s]"     │    │    (80 bins)           │    │                           │
  └──────────┬───────────┘    └───────────┬────────────┘    └───────────┬──────────────┘
             │                            │                              │
             │                            │                              │


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 1: BACKBONE ENCODERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          (67.3M params, ALL frozen in Phase 1+2, unfreeze selectively Phase 3)

             │                            │                              │
             ▼                            ▼                              ▼
  ╔═══════════════════╗        ╔═════════════════════╗        ╔══════════════════════╗
  ║  all-MiniLM-L6-v2 ║        ║   Whisper-Tiny       ║        ║    MobileViT-S        ║
  ║    22.7M params   ║        ║   Encoder-only        ║        ║    5.6M params        ║
  ║                   ║        ║   39M params          ║        ║                       ║
  ║  Sentence-level   ║        ║                       ║        ║  Per-frame spatial    ║
  ║  transformer      ║        ║  1500 audio frames    ║        ║  CNN+attention        ║
  ║                   ║        ║  → 384D each          ║        ║  → 640D per frame     ║
  ║  Computes text    ║        ║                       ║        ║                       ║
  ║  embedding at     ║        ║  Temporal downsampled ║        ║  Each frame is        ║
  ║  each word event  ║        ║  to match 2Hz rate    ║        ║  encoded              ║
  ║                   ║        ║  via mean pooling     ║        ║  independently        ║
  ║  → (B, T, 384)   ║        ║  → (B, T, 384)        ║        ║  → (B, T, 640)        ║
  ║                   ║        ║                       ║        ║                       ║
  ║  ALWAYS FROZEN    ║        ║  Frozen Phase 1,2     ║        ║  Frozen Phase 1,2     ║
  ║  (already great   ║        ║  Unfreeze Phase 3     ║        ║  Unfreeze Phase 3     ║
  ║   for semantics)  ║        ║  LR: 1e-5 (low)       ║        ║  LR: 1e-5 (low)       ║
  ╚════════┬══════════╝        ╚══════════┬════════════╝        ╚═════════┬═════════════╝
           │                             │                                │
      (B, T, 384)                  (B, T, 384)                      (B, T, 640)
           │                             │                                │


━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 2: PER-MODALITY PROJECTORS ━━━━━━━━━━━━━━━━━━━━━━━━━
                                   (2.5M params, always trainable)

           │                             │                                │
           ▼                             ▼                                ▼
  ╔════════════════════╗       ╔═════════════════════╗         ╔═══════════════════════╗
  ║  TEXT PROJECTOR    ║       ║  AUDIO PROJECTOR     ║         ║  VIDEO PROJECTOR      ║
  ║                    ║       ║                      ║         ║                       ║
  ║  LayerNorm(384)    ║       ║  LayerNorm(384)      ║         ║  LayerNorm(640)       ║
  ║  Linear(384→768)   ║       ║  Linear(384→768)     ║         ║  Linear(640→768)      ║
  ║  GELU              ║       ║  GELU                ║         ║  GELU                 ║
  ║  Dropout(0.1)      ║       ║  Dropout(0.1)        ║         ║  Dropout(0.1)         ║
  ║  Linear(768→768)   ║       ║  Linear(768→768)     ║         ║  Linear(768→768)      ║
  ║  GELU              ║       ║  GELU                ║         ║  GELU                 ║
  ║  Dropout(0.1)      ║       ║  Dropout(0.1)        ║         ║  Dropout(0.1)         ║
  ║  Linear(768→512)   ║       ║  Linear(768→512)     ║         ║  Linear(768→512)      ║
  ║  LayerNorm(512)    ║       ║  LayerNorm(512)      ║         ║  LayerNorm(512)       ║
  ║                    ║       ║                      ║         ║                       ║
  ║  ~800K params      ║       ║  ~800K params        ║         ║  ~900K params         ║
  ╚════════┬═══════════╝       ╚══════════┬═══════════╝         ╚══════════┬════════════╝
           │                             │                                 │
      (B, T, 512)                  (B, T, 512)                       (B, T, 512)
           │                             │                                 │
           │                             │                    ┌────────────▼────────────┐
           │                             │                    │ TEMPORAL MOTION MODULE  │
           │                             │                    │                         │
           │                             │                    │  Depthwise Conv1D       │
           │                             │                    │  in_ch=512, out_ch=512  │
           │                             │                    │  kernel_size=3          │
           │                             │                    │  padding=1, groups=512  │
           │                             │                    │  → captures Δframe      │
           │                             │                    │                         │
           │                             │                    │  + Residual connection  │
           │                             │                    │  (original + motion)    │
           │                             │                    │                         │
           │                             │                    │  ~1.5K params           │
           │                             │                    │  Free — depthwise only  │
           │                             │                    └────────────┬────────────┘
           │                             │                                 │
      (B, T, 512)                  (B, T, 512)                       (B, T, 512)


━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 3: MODALITY EMBEDDINGS + ALIGNMENT ━━━━━━━━━━━━━━━━━━━━

           │                             │                                 │
           ▼                             ▼                                 ▼

  Per-modality temporal embeddings (NEW — learned, separate per modality):
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                │
  │  text_time_embed:   Embedding(max_T, 512)  — text temporal position           │
  │  audio_time_embed:  Embedding(max_T, 512)  — audio temporal position          │
  │  video_time_embed:  Embedding(max_T, 512)  — video temporal position          │
  │                                                                                │
  │  text_proj  += text_time_embed[0:T]                                           │
  │  audio_proj += audio_time_embed[0:T]                                          │
  │  video_proj += video_time_embed[0:T]                                          │
  │                                                                                │
  │  WHY: text at t=5 and video at t=5 live in very different temporal contexts   │
  │  Text: discrete word events. Audio: continuous. Video: 2fps frames.           │
  │  Shared positional encoding confuses the transformer. Separate ones do not.   │
  │                                                                                │
  │  + Static modality type embeddings:                                           │
  │  text_proj  += modality_embed[0]    (one learned 512D vector per modality)    │
  │  audio_proj += modality_embed[1]                                               │
  │  video_proj += modality_embed[2]                                               │
  │                                                                                │
  │  Total: 3 × max_T × 512 + 3 × 512 ≈ 3M params (max_T=2048)                 │
  └────────────────────────────────────────────────────────────────────────────────┘

  Temporal alignment:
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  T = max(T_text, T_audio, T_video)                                             │
  │  Each modality: F.interpolate(mode='linear') → (B, T, 512)                   │
  │                                                                                │
  │  Modality dropout (training only):                                            │
  │    Phase 1 self-sup:  p=0.5 (force single-modality robustness)               │
  │    Phase 2 KD:        p=0.3 (teacher signal best with all modalities)         │
  │    Phase 3 fMRI:      p=0.1 → 0.0 (decay, maximize fMRI signal)              │
  └────────────────────────────────────────────────────────────────────────────────┘

           │                             │                                 │
           └─────────────────────────────┼─────────────────────────────────┘
                                         │
                                  INTERLEAVE
                    ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬─────┐
                    │ t₁ │ a₁ │ v₁ │ t₂ │ a₂ │ v₂ │ t₃ │ a₃ │ v₃ │ ... │
                    └────┴────┴────┴────┴────┴────┴────┴────┴────┴─────┘
                                  (B, T×3, 512)
                                         │


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 4: MoE TRANSFORMER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                              (4 layers, ~37.8M params, ~12.6M active)

                                         │
                                         ▼
  ╔═══════════════════════════════════════════════════════════════════════════════════╗
  ║  LAYERS 1 + 2: LOCAL-AWARE FUSION (temporal locality + HRF bias)               ║
  ║                                                                                  ║
  ║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
  ║  │  PRE-LAYERNORM                                                            │   ║
  ║  │  LayerNorm(512)                                                           │   ║
  ║  └────────────────────────────────────┬──────────────────────────────────────┘   ║
  ║                                       │                                          ║
  ║  ┌────────────────────────────────────▼──────────────────────────────────────┐   ║
  ║  │  MULTI-HEAD SELF-ATTENTION (8 heads × 64D = 512D)                        │   ║
  ║  │                                                                            │   ║
  ║  │  Standard QKV attention + TEMPORAL DECAY BIAS:                            │   ║
  ║  │                                                                            │   ║
  ║  │  attn_bias[i,j] = -α × |timestep(i) - timestep(j)|                       │   ║
  ║  │                                                                            │   ║
  ║  │  where α is a LEARNED per-layer scalar (init: log(1/6) for 6TR≈9s decay) │   ║
  ║  │  and timestep(i) = floor(i/3)  (since 3 tokens per TR: t,a,v)            │   ║
  ║  │                                                                            │   ║
  ║  │  Effect: same-TR tokens attend freely, distant-TR tokens are suppressed   │   ║
  ║  │  This matches the HRF shape — recent stimulus dominates fMRI response     │   ║
  ║  │  α is learned so the model can adjust the temporal window per layer       │   ║
  ║  │                                                                            │   ║
  ║  │  Params: 4 × 512 × 512 = 1.05M (Q,K,V,out projections)                  │   ║
  ║  └────────────────────────────────────┬──────────────────────────────────────┘   ║
  ║                                       │                                          ║
  ║  + Residual                           │                                          ║
  ║                                       ▼                                          ║
  ║  ┌────────────────────────────────────────────────────────────────────────────┐   ║
  ║  │  PRE-LAYERNORM + MoE FFN                                                  │   ║
  ║  │                                                                            │   ║
  ║  │  Router: Linear(512 → 8) + Z-loss + TopK(k=2)                            │   ║
  ║  │                                                                            │   ║
  ║  │  8 Experts (only 2 active per token):                                     │   ║
  ║  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│   ║
  ║  │  │ E1   │ │ E2   │ │ E3   │ │ E4   │ │ E5   │ │ E6   │ │ E7   │ │ E8   ││   ║
  ║  │  │512→  │ │512→  │ │512→  │ │512→  │ │512→  │ │512→  │ │512→  │ │512→  ││   ║
  ║  │  │1024→ │ │1024→ │ │1024→ │ │1024→ │ │1024→ │ │1024→ │ │1024→ │ │1024→ ││   ║
  ║  │  │512   │ │512   │ │512   │ │512   │ │512   │ │512   │ │512   │ │512   ││   ║
  ║  │  │GELU  │ │GELU  │ │GELU  │ │GELU  │ │GELU  │ │GELU  │ │GELU  │ │GELU  ││   ║
  ║  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘│   ║
  ║  │                                                                            │   ║
  ║  │  Experts init: all from SAME random FFN + N(0, 0.01) noise each           │   ║
  ║  │  → breaks symmetry while giving shared starting point                     │   ║
  ║  │                                                                            │   ║
  ║  │  8 × 1.05M = 8.4M params/layer, but 2 × 1.05M = 2.1M active per token   │   ║
  ║  │                                                                            │   ║
  ║  │  Aux load-balance loss: *** always accumulated even if layer is dropped *** │   ║
  ║  └────────────────────────────────────┬──────────────────────────────────────┘   ║
  ║                                       │                                          ║
  ║  + Residual                           │                                          ║
  ║                                       │                                          ║
  ║  Stochastic depth: drop prob = (l/L) × 0.2  (layer 1: 5%, layer 2: 10%)        ║
  ║  *** aux_loss accumulated BEFORE drop decision — BUG FIX ***                    ║
  ╚═══════════════════════════════════════╪═══════════════════════════════════════════╝
                                          │

  ╔═══════════════════════════════════════╪═══════════════════════════════════════════╗
  ║  LAYERS 3 + 4: GLOBAL SEMANTIC FUSION (full attention, no locality bias)        ║
  ║                                                                                  ║
  ║  Same MoE block structure but:                                                   ║
  ║    - NO temporal decay bias (full attention over all T×3 tokens)                ║
  ║    - Higher stochastic depth: layer 3: 15%, layer 4: 20%                       ║
  ║    - These layers integrate narrative, semantic context, long-range structure   ║
  ║    - e.g.: "the bomb exploded" at t=200 affects t=150 (anticipation) via       ║
  ║      global attention across the entire 100s segment                           ║
  ║                                                                                  ║
  ╚═══════════════════════════════════════╪═══════════════════════════════════════════╝
                                          │
                                    Final LayerNorm
                                   (B, T×3, 512)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 5: GATED MODALITY POOLING ━━━━━━━━━━━━━━━━━━━━━━━━━

                                          │
                                          ▼
  ╔═══════════════════════════════════════════════════════════════════════════════════╗
  ║  LEARNED MODALITY GATING (replaces mean pool — 1.5K params)                    ║
  ║                                                                                  ║
  ║  Reshape: (B, T×3, 512) → (B, T, 3, 512)                                      ║
  ║                                                                                  ║
  ║  gates = sigmoid(Linear(512, 3))  applied to pooled token:                     ║
  ║  pool_input = mean(modality_tokens)  # rough average as context               ║
  ║  gates = sigmoid(gate_net(pool_input))  # (B, T, 3) — one gate per modality   ║
  ║  gates = gates / gates.sum(dim=-1, keepdim=True)  # normalize to sum=1        ║
  ║                                                                                  ║
  ║  out = Σ_m  gates[:,:,m].unsqueeze(-1) × tokens[:,:,m,:]                      ║
  ║      = (B, T, 512)                                                             ║
  ║                                                                                  ║
  ║  WHY: Visual cortex (V1-V4) should upweight video tokens at every timestep.    ║
  ║       Broca's area should upweight text tokens.                                ║
  ║       The gate learns this from data — no hardcoding needed.                   ║
  ║       During training: if video is masked (modality dropout), gate automatically║
  ║       redistributes weight to text+audio. More robust than mean pool.          ║
  ╚═══════════════════════════════════════╪═══════════════════════════════════════════╝
                                          │
                                    (B, T, 512)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 6: HRF CONVOLUTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                                          │
                                          ▼
  ╔═══════════════════════════════════════════════════════════════════════════════════╗
  ║  HEMODYNAMIC RESPONSE FUNCTION LAYER (learnable, ~0.1M params)                 ║
  ║                                                                                  ║
  ║  Depthwise Conv1D:                                                              ║
  ║    in_channels  = 512                                                           ║
  ║    out_channels = 512                                                           ║
  ║    kernel_size  = 8  (covering ~12s at TR=1.49s, capturing full HRF peak)      ║
  ║    padding      = 7  (causal: only looks at past, not future)                  ║
  ║    groups       = 512  (depthwise — each feature dim has own HRF kernel)       ║
  ║                                                                                  ║
  ║  Initialization (canonical double-Gamma HRF):                                   ║
  ║    t = [0, 1.49, 2.98, 4.47, 5.96, 7.45, 8.94, 10.43]  seconds              ║
  ║    hrf = gamma_pdf(t, a1=6, b1=1) - gamma_pdf(t, a2=16, b2=1)/6               ║
  ║    hrf /= hrf.sum()  # normalize                                               ║
  ║    kernel initialized to hrf for all 512 channels                              ║
  ║                                                                                  ║
  ║    HRF shape (canonical):                                                       ║
  ║    1.0 ┤                                                                        ║
  ║        │         ╭─╮                                                            ║
  ║    0.5 ┤      ╭──╯  ╰──╮                                                       ║
  ║        │    ╭─╯         ╰─╮                                                    ║
  ║    0.0 ┼────╯              ╰────────────                                       ║
  ║   -0.2 ┤                          ╰──╮  ← undershoot                           ║
  ║        └─────────────────────────────────▶ time (s)                            ║
  ║          0  2  4  6  8  10  12  14  16                                         ║
  ║                                                                                  ║
  ║  Fine-tuned during training — different brain regions may have slightly         ║
  ║  different HRF shapes (e.g., primary sensory areas peak earlier)               ║
  ║                                                                                  ║
  ║  + Residual connection: out = conv(x) + x                                      ║
  ║    (if HRF is already implicitly learned, residual preserves original signal)  ║
  ╚═══════════════════════════════════════╪═══════════════════════════════════════════╝
                                          │
                                    (B, T, 512)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STAGE 7: OUTPUT HEAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                                          │
                                          ▼
  ╔═══════════════════════════════════════════════════════════════════════════════════╗
  ║  SHARED OUTPUT MLP + FILM SUBJECT CONDITIONING                                  ║
  ║                                                                                  ║
  ║  Step 1: Shared MLP backbone                                                    ║
  ║  ┌────────────────────────────────────┐                                         ║
  ║  │  LayerNorm(512)                    │                                         ║
  ║  │  Linear(512 → 512)  + GELU        │                                         ║
  ║  └──────────────────┬─────────────────┘                                         ║
  ║                     │                                                            ║
  ║                (B, T, 512)                                                      ║
  ║                     │                                                            ║
  ║  Step 2: FiLM conditioning (Feature-wise Linear Modulation)                    ║
  ║  ┌────────────────────────────────────────────────────────┐                     ║
  ║  │                                                         │                     ║
  ║  │  Per-subject learned vectors:                           │                     ║
  ║  │    γ[subject_id] ∈ R^512  (scale)                      │                     ║
  ║  │    β[subject_id] ∈ R^512  (shift)                      │                     ║
  ║  │                                                         │                     ║
  ║  │  out = γ[s] * x + β[s]                                 │                     ║
  ║  │                                                         │                     ║
  ║  │  Params: 2 × 512 × n_subjects                          │                     ║
  ║  │    = 2 × 512 × 25 = 25,600 params  (vs 33M in v2!)   │                     ║
  ║  │                                                         │                     ║
  ║  │  WHY better than SubjectLayers:                         │                     ║
  ║  │    - 1,300× fewer params                               │                     ║
  ║  │    - Shared MLP learns universal brain→vertex mapping  │                     ║
  ║  │    - Subject-specific shift/scale adapts anatomy        │                     ║
  ║  │    - New subjects: learn only 1024 floats (γ + β)      │                     ║
  ║  │      vs retraining 5.2M params (256×20484) in v2       │                     ║
  ║  └──────────────────────┬─────────────────────────────────┘                     ║
  ║                         │                                                        ║
  ║                    (B, T, 512)                                                  ║
  ║                         │                                                        ║
  ║  Step 3: Vertex projection                                                      ║
  ║  ┌────────────────────────────────────┐                                         ║
  ║  │  Linear(512 → n_vertices, bias=False)                  │                     ║
  ║  │    fsaverage4:  512 × 5124  =  2.6M params            │                     ║
  ║  │    Schaefer-1000: 512 × 1000 = 0.5M params            │                     ║
  ║  └──────────────────────┬─────────────────────────────────┘                     ║
  ║                         │                                                        ║
  ║                  (B, T, n_vertices)                                             ║
  ╚═════════════════════════╪═════════════════════════════════════════════════════════╝
                            │
                     transpose: (B, n_vertices, T)
                            │
                   AdaptiveAvgPool1d(n_output_TRs)
                            │
                    (B, n_vertices, n_TRs)
                            │
                  ╔═════════════════╗
                  ║  BRAIN MAP OUT  ║
                  ╚═════════════════╝
```

---

## Part 2: v2 → v3 Change-by-Change Critique

### Change 1: Temporal motion module (video)

**v2 problem:**
MobileViT-S encodes each frame independently. Visual motion cortex
(area MT+/V5) is one of the most fMRI-predictable regions. It responds to
optic flow — frame-to-frame pixel change — not to absolute frame content.
The v2 model is **completely blind to motion** despite motion being one of
the strongest brain predictors.

**v3 fix:**
Depthwise Conv1D (kernel=3) after the video projector. Groups=512 means
each feature dimension has its own 3-tap temporal filter. The filter learns
to compute something like `frame[t] - frame[t-1]`. Total cost: 1,536 params.
Quality gain: significant for motion-sensitive regions (MT+, V1, parietal cortex).

---

### Change 2: Per-modality temporal embeddings

**v2 problem:**
Text, audio, and video tokens at the same sequence position get the same
positional embedding. But `pos_embed[15]` means very different things
depending on whether it's a text word (word #5 at 2.5s), an audio frame
(at 11.25s), or a video frame (at 7.5s). The model must learn to
disambiguate these from context alone.

**v3 fix:**
Three separate temporal embedding tables — one per modality. Each learns
its own temporal structure. Text embeddings can learn word-boundary patterns.
Audio embeddings can learn phoneme-rate dynamics. Video embeddings can learn
scene-cut patterns. Cost: 3 × 2048 × 512 = 3.1M params (modest, well-spent).

---

### Change 3: HRF temporal decay bias (layers 1-2)

**v2 problem:**
Full self-attention treats all past timesteps equally. In reality, the fMRI
BOLD signal at time t reflects stimulus from roughly t-4s to t-10s (HRF peak
at ~6s). Stimuli more than ~15s ago contribute almost nothing. The model
wastes attention capacity on distant past tokens.

**v3 fix:**
Add a learned bias to the attention logits: `bias[i,j] = -α × |t_i - t_j|`
where α is a scalar per layer, initialized to give ~6s decay timescale.
`α = log(1/6) ÷ TR`. This is like ALiBi (Press et al., 2022) but with
neurobiological motivation. Only in layers 1-2 (local context). Layers 3-4
use full attention for long-range narrative integration.

---

### Change 4: Gated modality pooling

**v2 problem:**
`fused.reshape(B,T,3,512).mean(dim=2)` gives equal weight to text, audio,
and video at every timestep for every brain region. But:
- Primary visual cortex (V1) almost entirely ignores text tokens
- Broca's area almost entirely ignores video tokens
- Superior temporal sulcus integrates all three
Mean pooling ignores this structure entirely.

**v3 fix:**
A small gating network: `gates = softmax(Linear(512, 3))` applied to the
mean-pooled token. Produces per-timestep weights for each modality.
During distillation, the teacher's predictions implicitly teach the gate which
modality matters where. Cost: 512 × 3 = 1,536 params. Quality gain: non-trivial
for modality-selective regions (which are a large fraction of cortex).

---

### Change 5: HRF convolution layer

**v2 problem:**
`AdaptiveAvgPool1d(n_TRs)` just averages features over time — no HRF modelling.
The hemodynamic response function is known physics: a stimulus causes a BOLD
signal that peaks at ~6s with a specific shape (double-Gamma). The model must
learn this entirely from data, which wastes capacity and requires more examples.

**v3 fix:**
Depthwise Conv1D initialized to the canonical double-Gamma HRF kernel. Causal
padding (only looks at past). Fine-tuned during training so different feature
dimensions can learn region-specific HRF shapes (which do vary across cortex —
primary sensory areas peak ~4-5s, higher-order areas ~6-8s). The residual
connection means if HRF is already implicit in the representation, this layer
can become identity.

---

### Change 6: FiLM subject conditioning

**v2 problem:**
SubjectLayers: `n_subjects × low_rank_dim × n_vertices` = 25 × 256 × 5124 = 32.8M params.
These are purely linear, per-subject maps. Problems:
1. Most params (33M/42M = 79%) are in these linear maps
2. Can't generalize to new subjects without retraining everything
3. Linear mapping from 256D is very constrained for 5124 vertices

**v3 fix:**
Shared MLP backbone (Linear 512→512→n_vertices) + per-subject FiLM vectors
(γ, β ∈ R^512 per subject). The MLP learns the universal brain topology — which
latent dimensions predict which cortical regions. FiLM conditioned scale/shift
adapts the representation for individual anatomy. For a new subject: freeze
everything, learn only 1024 floats. This is the core of adapter-style transfer.

---

### Change 7: Stochastic depth linear schedule

**v2 problem:**
Uniform 10% layer dropout. All 4 layers have the same regularization.
But earlier layers do local feature computation (should be more reliable),
later layers do semantic integration (more complex, benefit from more
regularization).

**v3 fix:**
Linear schedule: layer l gets drop probability `(l/L) × 0.2`.
Layer 1: 5%, Layer 2: 10%, Layer 3: 15%, Layer 4: 20%.
Matches the stochastic depth schedule from DeiT and ViT-22B.

---

### Change 8: Aux loss bug fix

**v2 problem (moe_model.py line 280-283):**
```python
for layer in self.layers:
    if self.training and torch.rand(1) < self.layer_dropout:
        continue  # ← also skips aux_loss!
    fused, aux_loss = layer(fused)
    total_aux_loss += aux_loss
```
When a layer is stochastically dropped, its aux_loss is also not computed.
Load-balancing is inconsistent — the router gets contradictory gradients
about whether to balance or not.

**v3 fix:**
```python
for layer in self.layers:
    fused_out, aux_loss = layer(fused)
    total_aux_loss += aux_loss  # always accumulate
    if not (self.training and torch.rand(1) < drop_prob[l]):
        fused = fused_out  # only conditionally update activations
```

---

## Part 3: Dataset — Best fMRI-Video Pairs

### The Winner: Courtois NeuroMod (Algonauts 2025)

This is the primary training dataset. Nothing else comes close for raw data volume.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  COURTOIS NEUROIMAGING OF NATURAL SCENES (CNeuroMod)                     │
│  aka Algonauts 2025 Challenge Dataset                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  Subjects:        4 (sub-01, sub-02, sub-03, sub-04)                    │
│  Hours per subj:  ~66h of scanning (264h total)                          │
│  TR:              1.49s  (fast multiband acquisition)                    │
│  Resolution:      2mm isotropic, whole brain                             │
│  Surface:         fsaverage5 (20,484 cortical vertices)                  │
├──────────────────────────────────────────────────────────────────────────┤
│  STIMULI (the video content):                                            │
│                                                                           │
│  Friends TV Show     S01-S07  (all 7 seasons)                            │
│    - 156 episodes × ~22 min = ~57h                                       │
│    - Dialogue-heavy, social cognition, consistent characters             │
│    - Audio: speech + background music + ambient                          │
│    - Text: word-level subtitles with timing                              │
│                                                                           │
│  DoCu (documentary clips)   ~2h                                          │
│  Raiders of the Lost Ark    ~1.5h (full movie)                          │
│  Forrest Gump               ~2h   (full movie)                          │
│                                                                           │
├──────────────────────────────────────────────────────────────────────────┤
│  WHY IT'S THE BEST:                                                      │
│                                                                           │
│  1. Volume: 264h of paired fMRI is the largest open naturalistic         │
│     neuroimaging dataset in existence by an order of magnitude           │
│                                                                           │
│  2. Naturalistic: continuous TV show watching, not brief flashed clips   │
│     → narrative, temporal structure, social dynamics all represented     │
│                                                                           │
│  3. Multiple repetitions: some stimuli shown multiple times              │
│     → allows noise ceiling estimation (how predictable is the signal?)   │
│                                                                           │
│  4. Quality: 3T scanner, 72-channel head coil, fMRIprep preprocessing   │
│     → clean signal, well-validated preprocessing pipeline                │
│                                                                           │
│  5. Proven: TRIBE v2 trained on this and achieved SOTA (Algonauts 2025) │
│                                                                           │
│  6. Access: openly available at https://www.cneuromod.ca/                │
│     Requires data sharing agreement (academic, free)                     │
├──────────────────────────────────────────────────────────────────────────┤
│  HOW TO USE IN STRATEGY C:                                               │
│                                                                           │
│  Phase 0 (Teacher cache): Run TRIBE v2 on all 264h of Friends + movies  │
│    Cache: predictions (T, 20484) + fusion layers 4,6 (T, 1152)          │
│    Cost: ~132 GPU-hours T4 (~$130)                                       │
│    But: if you already have TRIBE v2 predictions from competition, reuse │
│                                                                           │
│  Phase 3 (fMRI fine-tuning): Train student on real fMRI signal          │
│    Train split: Friends S01-S06 (~50h per subject)                      │
│    Val split:   Friends S07 + movies (~16h per subject)                  │
│    Metric: Pearson r per vertex, averaged across test set                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Secondary: BOLD Moments (Lahner et al., 2024)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BOLD MOMENTS DATASET                                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  Subjects:   10                                                           │
│  Clips:      1,000 unique 3-second video clips                           │
│  Reps:       10 repetitions per clip per subject                         │
│  Total:      ~6.2h per subject, 62h total                                │
│  TR:         1.75s                                                        │
│  Content:    Diverse (animals, sports, nature, people, objects)          │
│  Source:     MIT Moments in Time dataset                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  WHY USEFUL:                                                             │
│  - 10 repetitions → very accurate noise ceiling per clip                │
│  - Short clips = good for evaluating fast visual responses               │
│  - Diverse content → tests generalization across domains                 │
│  - 10 subjects (more than CNeuroMod) → better average-subject model     │
│                                                                           │
│  USE FOR: Validation in Phase 3. Not primary training (too little data) │
└──────────────────────────────────────────────────────────────────────────┘
```

### Tertiary: Lebel et al. 2023

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LEBEL 2023 — Semantic language dataset                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Subjects:  8                                                             │
│  Stimuli:   82 spoken narrative stories (no video)                       │
│  Hours:     6-18h per subject                                            │
│  TR:        2.0s                                                          │
│  Audio:     yes. Video: NO.                                              │
├──────────────────────────────────────────────────────────────────────────┤
│  USE FOR: Audio+text pathway training only (Phase 3, modality dropout)  │
│           Strengthens language and auditory cortex predictions            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Full Dataset Stack for Strategy C

```
PHASE 0 — Teacher cache (run once):
  CNeuroMod (264h) — primary, has paired fMRI
  Nature docs + TED + movies (50h) — diversity
  BOLD Moments clips (6h) — short diverse clips
  ─────────────────────────────────────────────
  Total teacher GPU cost: ~160h T4 (~$160)

PHASE 1 — Self-supervised (no teacher):
  HowTo100M subset    500h   — diverse instructional video+audio+text
  LibriSpeech         960h   — audio+text (no video, use modality dropout)
  VGGSound            200h   — audio+video event pairs
  TED-LIUM            100h   — lecture audio+text
  posted/ videos       10h   — domain-specific content
  ─────────────────────────────────────────────
  Total: 1770h, zero teacher inferences
  GPU cost for feature extraction: ~200h T4 (~$200)

PHASE 2 — KD fine-tuning (cached teacher):
  CNeuroMod predictions (264h cached)
  Additional diverse video (50h cached)
  ─────────────────────────────────────
  No new teacher inferences needed.

PHASE 3 — fMRI fine-tuning:
  CNeuroMod fMRI train split: Friends S01-S06
  Lebel 2023 stories: audio+text pathway
  BOLD Moments: validation only
```

---

## Part 3b: Inference Costing & Dataset Creation Time

This is the distillation bottleneck. Everything below is the real cost
breakdown so you can plan GPU budgets and wall-clock time before starting.

---

### The Distillation Data Pipeline: What Has to Happen

```
Raw video files
      │
      ▼  [Step A] TRIBE v2 teacher inference  ← the expensive part
      │   Input:  raw video
      │   Output: teacher_pred (T, 20484) + fusion features (T, 1152)
      │   Cost:   ~500ms per second of video on T4
      │
      ▼  [Step B] Tiny backbone feature extraction  ← cheap
      │   Input:  raw video
      │   Output: text_feat (T,384), audio_feat (T,384), video_feat (T,640)
      │   Cost:   ~30-50ms per second of video on T4 (10-15x faster than Step A)
      │
      ▼  [Step C] Student training  ← fast, backbone features already cached
          Input:  cached backbone features + cached teacher predictions
          Output: trained student model
          Cost:   ~5-20h T4 per phase depending on dataset size
```

---

### Step A: TRIBE v2 Teacher Inference — Full Cost Breakdown

The teacher model (4.7B params, ~10GB VRAM) is the bottleneck.
Every second of video costs ~500ms on a T4 GPU.

#### Per-video inference cost

```
TRIBE v2 inference on T4 (16GB):
─────────────────────────────────────────────────────────────────────────
  Model load time:            ~45s (cold start, first video only)
  Feature extraction rate:    ~0.5s GPU time per 1s of video (2x realtime)
  Memory (model weights):     ~10GB (fp16)
  Memory (activations):       ~2-3GB per 100-TR segment
  Throughput with bs=1:       ~2 seconds of video per GPU-second
  Throughput with bs=4:       ~3 seconds of video per GPU-second (batch overlap)

  Per segment (100 TRs = 150s of video):
    Inference time:  ~75s
    Output size:     100 × 20484 × 4 bytes = 8.2 MB (fp32)
              +      100 × 1152 × 4 bytes  = 0.5 MB  (fusion l4)
              +      100 × 1152 × 4 bytes  = 0.5 MB  (fusion l6)
    Total per segment: ~9.2 MB
─────────────────────────────────────────────────────────────────────────
```

#### Dataset-by-dataset teacher inference cost

```
┌─────────────────────────────┬────────┬──────────────┬───────────┬──────────────┐
│ Dataset                     │ Hours  │ GPU-hrs (T4) │ Cost ($1) │ Storage      │
├─────────────────────────────┼────────┼──────────────┼───────────┼──────────────┤
│ CNeuroMod (Friends + movies)│  264h  │    132h      │   $132    │  ~220 GB     │
│ Nature docs + TED + movies  │   50h  │     25h      │    $25    │   ~42 GB     │
│ BOLD Moments clips          │    6h  │      3h      │     $3    │    ~5 GB     │
├─────────────────────────────┼────────┼──────────────┼───────────┼──────────────┤
│ TOTAL                       │  320h  │    160h      │   $160    │  ~267 GB     │
└─────────────────────────────┴────────┴──────────────┴───────────┴──────────────┘

  ¹ At $1/h for T4 on Lambda Labs / Vast.ai spot pricing

Wall-clock time (1× T4):    160h  = 6.7 days
Wall-clock time (4× T4):     40h  = 1.7 days  ← recommended (parallelise by video)
Wall-clock time (8× T4):     20h  = 0.8 days

Parallelisation: trivially parallelisable — each video is independent.
Split the video list across N GPUs. No inter-GPU communication needed.
```

#### Storage planning for teacher cache

```
Per TR:
  predictions:  20484 vertices × 4 bytes = 82 KB
  fusion_l4:    1152  dims     × 4 bytes =  4.6 KB
  fusion_l6:    1152  dims     × 4 bytes =  4.6 KB
  Total per TR:                            91.2 KB

Per hour of video (at TR=1.49s → ~2415 TRs/hour):
  2415 TRs × 91.2 KB = ~220 MB / hour

Total for 320h:   ~70 GB  (fp32)
                  ~35 GB  (fp16, negligible quality loss for KD targets)

RECOMMENDED: Store in fp16.
  - teacher_preds: fp16  (student MSE loss is scale-invariant)
  - fusion feats:  fp16  (cosine similarity is scale-invariant)
  - Saves 50% storage, no measurable KD quality drop

File format:
  One .pt file per 100-TR segment (matching training segments).
  Filename: {dataset}_{video_id}_{tr_start:05d}.pt
  Loaded on-the-fly during training, fits in RAM for CNeuroMod.
```

---

### Step B: Tiny Backbone Feature Extraction — Cost Breakdown

The tiny backbones (67.3M total) run much faster than TRIBE v2.
Run once on all data (Phases 0+1), cache, reuse across all training phases.

#### Per-backbone throughput on T4

```
┌──────────────────────────────┬───────────────┬────────────────┬─────────────────┐
│ Backbone                     │ Params        │ Throughput     │ Speedup vs TRIBE│
├──────────────────────────────┼───────────────┼────────────────┼─────────────────┤
│ all-MiniLM-L6-v2 (text)      │ 22.7M         │ ~20x realtime  │ 40x faster      │
│   Input: word events          │               │ (with WhisperX │                 │
│   WhisperX ASR first         │               │  ASR overhead: │                 │
│   → then sentence encode      │               │  ~10x realtime)│                 │
│                               │               │                │                 │
│ Whisper-Tiny encoder (audio) │ 39M           │ ~15x realtime  │ 30x faster      │
│   Input: mel spectrogram      │               │                │                 │
│   Process 30s chunks          │               │                │                 │
│                               │               │                │                 │
│ MobileViT-S (video)          │ 5.6M          │ ~30x realtime  │ 60x faster      │
│   Input: frames at 2fps       │               │ (only 2fps,    │                 │
│   Batch frames for efficiency │               │  tiny model)   │                 │
└──────────────────────────────┴───────────────┴────────────────┴─────────────────┘

All 3 in parallel on 1 T4: limited by slowest (text+ASR ~10x realtime)
→ 1 hour of video takes ~6 minutes of GPU time
→ Effective throughput: ~10x realtime
```

#### Feature extraction cost by dataset

```
┌────────────────────────┬────────┬─────────────┬──────────────┬──────────────┐
│ Dataset                │ Hours  │ GPU-hrs (T4)│ Cost ($)     │ Storage      │
├────────────────────────┼────────┼─────────────┼──────────────┼──────────────┤
│ CNeuroMod              │  264h  │    26h      │    $26       │   ~19 GB     │
│ BOLD Moments           │    6h  │     1h      │     $1       │    ~0.4 GB   │
│ HowTo100M subset       │  500h  │    50h      │    $50       │   ~36 GB     │
│ LibriSpeech            │  960h  │    96h      │    $96       │   ~48 GB     │
│   (audio+text only,    │        │             │              │  (no video   │
│    video feat = zeros) │        │             │              │   features)  │
│ VGGSound               │  200h  │    20h      │    $20       │   ~14 GB     │
│ TED-LIUM               │  100h  │    10h      │    $10       │    ~7 GB     │
│ posted/ videos         │   10h  │     1h      │     $1       │    ~0.7 GB   │
├────────────────────────┼────────┼─────────────┼──────────────┼──────────────┤
│ TOTAL                  │ 2040h  │   204h      │   $204       │  ~125 GB     │
└────────────────────────┴────────┴─────────────┴──────────────┴──────────────┘

Per-feature storage:
  text:  (T, 384) × 2 bytes fp16 = 768 bytes/TR
  audio: (T, 384) × 2 bytes fp16 = 768 bytes/TR
  video: (T, 640) × 2 bytes fp16 = 1280 bytes/TR
  Total: 2816 bytes/TR

At 2 Hz feature rate: 2 TRs/sec
  1 hour = 3600s × 2 × 2816 bytes = ~20 MB/hour
  2040h total = ~41 GB (small — fits on one drive)
```

---

### Complete Cost Comparison: Strategy A vs Strategy C

This is the core argument for Strategy C from a budget perspective.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               STRATEGY A (Direct KD — current v2 approach)                  │
├────────────────────────┬────────────┬──────────────┬──────────────┬─────────┤
│ Step                   │ Data       │ GPU-hrs (T4) │ Wall clock   │ Cost    │
├────────────────────────┼────────────┼──────────────┼──────────────┼─────────┤
│ Teacher inference      │ 500h video │     250h     │ 10.4 days    │  $250   │
│ Backbone feat extract  │ 500h video │      50h     │  2.1 days    │   $50   │
│ Phase 1 training (KD)  │ 500h       │       5h     │  5h          │    $5   │
│ Phase 2 training (E2E) │ 270h fMRI  │      10h     │ 10h          │   $10   │
├────────────────────────┼────────────┼──────────────┼──────────────┼─────────┤
│ TOTAL                  │            │     315h     │ ~13 days     │  $315   │
│ Expected Pearson r     │            │     0.27-0.29│              │         │
└────────────────────────┴────────────┴──────────────┴──────────────┴─────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│         STRATEGY C (Self-supervised + KD — proposed v3 approach)            │
├────────────────────────┬────────────┬──────────────┬──────────────┬─────────┤
│ Step                   │ Data       │ GPU-hrs (T4) │ Wall clock   │ Cost    │
├────────────────────────┼────────────┼──────────────┼──────────────┼─────────┤
│ Teacher inference      │ 320h video │     160h     │  6.7 days    │  $160   │
│  (40% less than A)     │            │              │  (4×T4: 1.7d)│         │
│                        │            │              │              │         │
│ Backbone feat extract  │ 2040h      │     204h     │  8.5 days    │  $204   │
│  (4× more data but     │            │              │  (4×T4: 2.1d)│         │
│   10× cheaper/hr)      │            │              │              │         │
│                        │            │              │              │         │
│ Phase 1: self-sup      │ 1770h      │      20h     │ 20h          │   $20   │
│  (pre-extracted feats) │            │              │              │         │
│                        │            │              │              │         │
│ Phase 2: KD            │ 320h cached│       5h     │  5h          │    $5   │
│  (pre-extracted feats) │            │              │              │         │
│                        │            │              │              │         │
│ Phase 3: fMRI finetune │ 270h fMRI  │      10h     │ 10h          │   $10   │
├────────────────────────┼────────────┼──────────────┼──────────────┼─────────┤
│ TOTAL                  │            │     399h     │ ~16 days     │  $399   │
│  (4×T4 parallel):      │            │     120h     │  5 days      │  $120   │
│ Expected Pearson r     │            │     0.29-0.31│              │         │
└────────────────────────┴────────────┴──────────────┴──────────────┴─────────┘

KEY INSIGHT:
  Strategy C costs ~$80 more in total (single GPU) but:
  - 40% fewer teacher inferences (the slow, expensive operation)
  - 5× more training data seen by the fusion model
  - Expected +0.02-0.04 Pearson r improvement
  - Model generalizes to new subjects cheaply (FiLM: 1024 floats vs 5M params)
  - With 4× parallel T4s: 5 days wall clock vs 13 days, similar total cost

  The extra $80 comes entirely from backbone feature extraction on 2040h of
  free data (vs 500h in Strategy A). This is the cheapest GPU work possible
  — tiny models at 10× realtime. The payoff is a much better fusion model.
```

---

### Where to Cut Costs If Budget Is Tight

```
Full budget (~$400 total, ~5 days with 4×T4):
  Run everything as described above.

Medium budget (~$200 total):
  Teacher cache:       Only CNeuroMod (264h → $130)  skip extra video diversity
  Self-sup data:       Drop LibriSpeech (saves 96 GPU-hours → ~$96)
                       Use HowTo100M + VGGSound only (700h)
  Expected hit:        ~0.01-0.02 Pearson r vs full budget

Minimal budget (~$100 total):
  Teacher cache:       CNeuroMod only (264h → $130)  ← can't cut this
  Self-sup data:       HowTo100M 200h only ($20 extraction)
  Skip Phase 1:        Fall back to Strategy A direct KD
                       Lose the self-supervised gains
  This is essentially Strategy A at the same cost.
  Don't do this — just run Strategy A cleanly instead.

FREE (if you already have TRIBE v2 predictions from Algonauts competition):
  Skip Phase 0 entirely. The competition submission required running TRIBE v2
  on the test set → you already have predictions cached.
  Only cost: backbone feature extraction ($204) + training ($35)
  Total: ~$240, saves 160 GPU-hours.
```

---

### Practical Scheduling: When to Run What

```
TIMELINE (4× T4 GPU, parallel)
─────────────────────────────────────────────────────────
Day 0-1:  [GPU 1]   Teacher inference on CNeuroMod (264h → 66h on 1 T4)
          [GPU 2]   Teacher inference on extra video (56h → 14h on 1 T4)
          [GPU 3]   Backbone extraction on HowTo100M + VGGSound (70h work)
          [GPU 4]   Backbone extraction on LibriSpeech + TED-LIUM (106h work)
          → All complete within ~1.7 days

Day 2:    [GPU 1-4] Phase 1 self-supervised training
          All 4 GPUs train together (DDP, batch_size=8/GPU → effective 32)
          → 20h / 4 GPUs = 5h wall clock

Day 3:    [GPU 1]   Phase 2 KD fine-tuning (5h)
          → Can be done on 1 GPU. No DDP needed.

Day 4:    [GPU 1-4] Phase 3 fMRI fine-tuning
          → 10h / 4 GPUs = 2.5h wall clock (but fMRI data is 270h, DDP helps)

Day 5:    Evaluation + ONNX export
          → 2-3h

TOTAL WALL CLOCK: ~5 days with 4×T4
─────────────────────────────────────────────────────────

CRITICAL PATH (what you must finish before next step can start):
  Teacher inference (Day 1) → unblocks Phase 2 and Phase 3
  Backbone extraction (Day 1) → unblocks Phase 1
  Phase 1 (Day 2) → unblocks Phase 2
  Phase 2 (Day 3) → unblocks Phase 3
  Phase 3 (Day 4) → final model
```

---

### Why Teacher Inference Is the Real Distillation Bottleneck

```
This is the key insight for distillation planning:

  TRIBE v2 throughput:      2s video per GPU-second  (2× realtime)
  Tiny backbone throughput: 20s video per GPU-second (20× realtime)
  Student training:         doesn't require teacher at inference time at all

So the distillation pipeline is:
  [Slow]   Teacher inference:   1 GPU-hour produces 2h of labeled data
  [Fast]   Backbone extraction: 1 GPU-hour produces 20h of features
  [Cheap]  Student training:    reads from disk, GPU fully utilised

The bottleneck is always Step A. Every design decision should be evaluated
by how much it reduces the teacher inference cost:

  Decision: Cache teacher predictions                → ✓ run once, reuse forever
  Decision: Self-supervised pre-training             → ✓ fusion needs less teacher data
  Decision: FiLM instead of SubjectLayers           → ✓ new subjects need no teacher
  Decision: Multi-res loss (parcel-level)           → ✓ richer signal per teacher sample
  Decision: Feature KD (save fusion activations)   → ✓ more info per teacher forward pass

The best distillation strategy squeezes maximum signal from each teacher
forward pass and minimises the number of passes needed.
For 320h of video at 2× realtime, that's 160 GPU-hours — irreducible minimum.
Everything else (backbone extraction, student training) is negligible by comparison.
```

---

## Part 3c: Minimal Distillation Dataset on Free GPUs (Kaggle + Lightning + Modal)

The goal is distillation — not retraining TRIBE v2 from scratch.
This changes the calculus completely. You need far less teacher data than you think,
and you have three free platforms that together are sufficient.

---

### How Little Teacher Data Does Distillation Actually Need?

The key insight: after Phase 1 self-supervised pre-training, the fusion transformer
already knows how to combine text, audio, and video across time. Phase 2 just needs
to learn the **mapping from fused representations to brain vertices** — which is
a much simpler problem. That mapping is near-linear once the representations are good.

```
EMPIRICAL EVIDENCE FROM SIMILAR DISTILLATION WORK:

  DistilBERT:     5% of BERT's training data → 97% of BERT's performance
  Distil-Whisper: 22K hours pseudo-labelled audio, but student matched teacher
                  with only 2K hours of real teacher data for the actual KD step
  TinyCLIP:       10% of LAION used for KD → 95% of CLIP performance
  LLaVA-1.5:      ~600K instruction pairs for full KD, but projector-only fine-tune
                  achieves 90% with just 50K samples

TRIBE v2 equivalent:
  Full teacher training data:   264h × 4 subjects = ~640K TRs
  Estimated minimum for KD:     ~10-20h × 4 subjects = ~100K TRs
  Why it works: fusion pre-training (Phase 1) replaces the need for most of this data.
  The brain mapping itself is learnable from a small diverse set.
```

**The minimum viable teacher inference dataset:**

```
┌─────────────────────────────────────────────────────────────────┐
│  ABSOLUTE MINIMUM:  5h of diverse video                         │
│  Expected Pearson r: 0.20-0.23  (65-74% of TRIBE v2)           │
│  GPU-hours needed:  2.5h T4                                     │
│  Fits in: 1 Kaggle session                                      │
├─────────────────────────────────────────────────────────────────┤
│  PRACTICAL MINIMUM: 15h of diverse video                        │
│  Expected Pearson r: 0.24-0.27  (77-87% of TRIBE v2)           │
│  GPU-hours needed:  7.5h T4                                     │
│  Fits in: 1 Kaggle week                                         │
├─────────────────────────────────────────────────────────────────┤
│  COMFORTABLE:       30h of diverse video                        │
│  Expected Pearson r: 0.27-0.29  (87-94% of TRIBE v2)           │
│  GPU-hours needed:  15h T4                                      │
│  Fits in: 2 Kaggle weeks (background)                           │
├─────────────────────────────────────────────────────────────────┤
│  DIMINISHING RETURNS above 50h — Phase 1 pre-training and       │
│  Phase 3 fMRI fine-tuning compensate for more teacher data.     │
└─────────────────────────────────────────────────────────────────┘

DIVERSITY > VOLUME
  10h of 10 different content types > 100h of the same content type.
  The fusion model already generalises — the teacher cache just needs
  to show it enough variety to learn the vertex mapping.

WHAT TO RUN TEACHER ON (priority order, ~15h total):
  1. CNeuroMod clips (2h)      — has paired fMRI, most relevant
  2. Nature documentary (2h)   — rich visual + narration + ambient audio
  3. TED talk / lecture (2h)   — sustained speech, gesture, slides
  4. Drama / movie clip (2h)   — dialogue, emotion, social interaction
  5. Music video (1h)          — music + motion, non-speech audio
  6. Sports / action (1h)      — fast motion, crowd noise
  7. Cooking / tutorial (2h)   — fine motor, speech + objects
  8. Ambient / nature (1h)     — minimal speech, pure visual + audio
  9. Podcast / interview (2h)  — mostly audio + face, text-heavy
```

---

### Your Available GPU Resources

```
┌──────────────────┬──────────────┬───────────┬──────────────┬─────────────────────┐
│ Platform         │ GPU          │ Free quota │ Session limit│ Best use            │
├──────────────────┼──────────────┼───────────┼──────────────┼─────────────────────┤
│ Kaggle           │ T4 16GB      │ 30h/week  │ 9h/session   │ Teacher inference   │
│                  │ (or P100)    │ per account│              │ (reliable, scheduled)│
├──────────────────┼──────────────┼───────────┼──────────────┼─────────────────────┤
│ Lightning AI     │ T4 16GB      │ 22h/month │ varies       │ Student training    │
│                  │              │ free tier  │              │ (good for Phase 1-3)│
├──────────────────┼──────────────┼───────────┼──────────────┼─────────────────────┤
│ Modal            │ T4 / A10G    │ $30 credit │ per-second   │ Teacher inference   │
│                  │ / A100       │ on signup  │ billing      │ (burst, fast setup) │
└──────────────────┴──────────────┴───────────┴──────────────┴─────────────────────┘

COMBINED FIRST-WEEK CAPACITY:
  Kaggle:    30h GPU → 60h of video predictions
  Modal:     $30 credit ÷ $0.00056/s (T4) = 53,571s = ~14.9h GPU → 30h video
             OR $30 ÷ $0.00111/s (A10G) = ~7.5h → 15h video (but 2× faster)
  Lightning: 22h/month GPU → 44h video (but save for student training)
  ─────────────────────────────────────────────────────────────────────
  Total week 1 teacher inference: ~60-90h of video predictions
  This exceeds the "comfortable" threshold (30h) in week 1 alone.
```

---

### Platform 1: Kaggle — Teacher Inference

Use Kaggle for the bulk of teacher inference. Reliable, automated, no credit card.

**Setup once (~1h)**

```python
# kaggle_tribe_inference.py
# Run this as a Kaggle Notebook (GPU T4, Internet ON)

# ── Install ──────────────────────────────────────────────────────────
!pip install -q tribev2 pydrive2 tqdm

# ── Config ───────────────────────────────────────────────────────────
GDRIVE_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID"   # where to save outputs
VIDEO_SOURCE     = "/kaggle/input/your-video-dataset"  # Kaggle dataset mount
MANIFEST_PATH    = "/kaggle/working/manifest.json"
CHUNK_SECONDS    = 150   # 100 TRs at TR=1.49s
DEVICE           = "cuda"

# ── Load manifest (resume from checkpoint) ───────────────────────────
import json, os
from pathlib import Path

if os.path.exists(MANIFEST_PATH):
    manifest = json.load(open(MANIFEST_PATH))
else:
    # Build manifest from all video files
    videos = sorted(Path(VIDEO_SOURCE).glob("*.mp4"))
    manifest = {}
    for v in videos:
        duration = get_video_duration(v)  # ffprobe
        n_chunks  = int(duration // CHUNK_SECONDS)
        for i in range(n_chunks):
            key = f"{v.stem}_{i:04d}"
            manifest[key] = "pending"
    json.dump(manifest, open(MANIFEST_PATH, "w"))

# ── Load TRIBE v2 (once, cached in /kaggle/working/hf_cache) ─────────
from tribev2 import TribeModel
import torch

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
model = TribeModel.from_pretrained("facebook/tribev2")
model = model.to(DEVICE).eval()

# Register hooks to capture fusion layer 4 and 6 activations
fusion_activations = {}
def make_hook(name):
    def hook(module, input, output):
        fusion_activations[name] = output.detach().cpu().half()
    return hook

# Hook into TRIBE v2 fusion transformer layers 4 and 6
model.encoder.layers[3].register_forward_hook(make_hook("layer4"))
model.encoder.layers[5].register_forward_hook(make_hook("layer6"))

# ── Inference loop ───────────────────────────────────────────────────
from google.colab import auth   # for Kaggle, use PyDrive2 instead
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Auth Google Drive (requires one-time browser approval)
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

pending = [k for k,v in manifest.items() if v == "pending"]
print(f"{len(pending)} segments remaining")

for seg_key in pending:
    try:
        video_id, chunk_idx = seg_key.rsplit("_", 1)
        chunk_idx = int(chunk_idx)
        video_path = f"{VIDEO_SOURCE}/{video_id}.mp4"
        t_start = chunk_idx * CHUNK_SECONDS
        t_end   = t_start + CHUNK_SECONDS

        # Run TRIBE v2 inference on this segment
        with torch.inference_mode():
            preds, segments = model.predict(
                video_path,
                start_sec=t_start,
                end_sec=t_end,
            )
        # preds: (T, 20484) float32

        # Save: predictions + cached fusion activations
        out = {
            "predictions": preds.cpu().half(),           # fp16, ~8MB per chunk
            "fusion_l4":   fusion_activations["layer4"], # fp16, ~0.5MB
            "fusion_l6":   fusion_activations["layer6"], # fp16, ~0.5MB
            "video_id":    video_id,
            "t_start":     t_start,
            "t_end":       t_end,
        }
        out_path = f"/kaggle/working/{seg_key}.pt"
        torch.save(out, out_path)

        # Upload to Google Drive
        f = drive.CreateFile({"parents": [{"id": GDRIVE_FOLDER_ID}],
                              "title": f"{seg_key}.pt"})
        f.SetContentFile(out_path)
        f.Upload()
        os.remove(out_path)  # free local disk

        # Update manifest
        manifest[seg_key] = "done"
        json.dump(manifest, open(MANIFEST_PATH, "w"))

        print(f"✓ {seg_key}")

    except Exception as e:
        manifest[seg_key] = f"failed: {e}"
        json.dump(manifest, open(MANIFEST_PATH, "w"))
        print(f"✗ {seg_key}: {e}")
```

**Kaggle schedule setup:**
```
1. Upload the notebook to Kaggle
2. Enable GPU (T4 × 1)
3. Enable Internet access (required for HuggingFace + Drive)
4. Add your video files as a Kaggle Dataset (private)
5. Schedule → Run daily at 00:00 UTC
6. Each run processes ~18h of video (9h session × 2× realtime)
   stops automatically at session limit, resumes next day from manifest

Week 1 output: ~60h of diverse video predictions saved to Google Drive
```

---

### Platform 2: Modal — Teacher Inference (Burst, Fast)

Modal is ideal for burning the $30 free credits on teacher inference fast.
Modal bills per second, has A10G GPUs (2× faster than T4), and cold-start
in ~30s. No session time limits.

**Why Modal for inference specifically:**
```
T4  on Modal: $0.000556/s = $2.00/h → 30h video per $30 credit
A10G on Modal: $0.001110/s = $4.00/h → but 2× faster → same video/$ as T4
               TRIBE v2 runs in fp16 → A10G 24GB fits full model easily
               Effective: ~30h of video predictions from $30 credit

No session limit → run one big job, process all 15h target dataset at once
Cold start: ~30s (TRIBE v2 model load) → amortised over long runs
```

```python
# modal_tribe_inference.py
# Run locally: modal run modal_tribe_inference.py

import modal
import torch
from pathlib import Path

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("tribev2", "torch", "tqdm", "google-cloud-storage")
)

app = modal.App("tribe-inference", image=image)

# GPU: use A10G for speed (fits within $30 credits for 15h of video)
@app.function(
    gpu="A10G",                      # or "T4" to stretch credits further
    timeout=3600,                    # 1h per function call
    secrets=[modal.Secret.from_name("google-cloud-storage")],
    volumes={"/cache": modal.Volume.from_name("tribe-cache", create_if_missing=True)},
)
def run_inference_on_segment(video_path: str, t_start: float, t_end: float, seg_key: str):
    """Run TRIBE v2 on one 150s segment, save to GCS."""
    import os
    from tribev2 import TribeModel

    os.environ["HF_HOME"] = "/cache/hf"    # persisted in Modal Volume

    # Load model (cached in Volume after first call — no re-download)
    model = TribeModel.from_pretrained("facebook/tribev2")
    model = model.cuda().eval()

    # Hook fusion layers
    acts = {}
    model.encoder.layers[3].register_forward_hook(
        lambda m, i, o: acts.update({"l4": o.detach().cpu().half()})
    )
    model.encoder.layers[5].register_forward_hook(
        lambda m, i, o: acts.update({"l6": o.detach().cpu().half()})
    )

    with torch.inference_mode():
        preds, _ = model.predict(video_path, start_sec=t_start, end_sec=t_end)

    result = {
        "predictions": preds.cpu().half(),
        "fusion_l4":   acts["l4"],
        "fusion_l6":   acts["l6"],
        "seg_key":     seg_key,
    }

    # Save to Modal Volume (or GCS)
    out_path = f"/cache/predictions/{seg_key}.pt"
    torch.save(result, out_path)
    return seg_key

@app.local_entrypoint()
def main():
    # Build list of all segments to process
    segments = build_segment_list("./videos", chunk_seconds=150)

    # Run all segments in parallel on Modal (each gets its own A10G)
    # Modal handles parallelism automatically
    results = list(run_inference_on_segment.starmap(segments))
    print(f"Completed {len(results)} segments")
```

**Running it:**
```bash
# One command, Modal handles everything
modal run modal_tribe_inference.py

# Modal spins up N A10G containers in parallel (N = number of segments)
# Each container processes one 150s chunk
# All results saved to Modal Volume → download to Google Drive
# Total time for 15h of video: ~45min (parallel)
# Total cost: ~$6-8 from $30 credits
# Remaining $22 credits → save for student training if Lightning quota runs out
```

**Cost breakdown for Modal $30 credits:**
```
A10G: $0.00111/s
Per segment (150s video → ~75s inference on A10G): 75s × $0.00111 = $0.083
15h video = 360 segments × $0.083 = $30 total  ← uses all credits for 15h

T4: $0.000556/s
Per segment (150s video → ~150s inference on T4): 150s × $0.000556 = $0.083
Same cost per video-hour! A10G is faster but twice the price.

RECOMMENDATION: Use T4 on Modal to maximise video hours per dollar.
  $30 ÷ $0.083/segment = 360 segments = 360 × 150s = 15h of video
  This hits the "comfortable" threshold in one Modal job.

  Run it all at once → 15h of video processed in ~4h wall clock (parallel)
```

---

### Platform 3: Lightning AI — Student Training (Not Inference)

Lightning AI free tier (22h/month GPU) is better spent on student training,
not teacher inference. Here's why and how.

```
WHY NOT LIGHTNING FOR INFERENCE:
  22h/month is limited. At 2× realtime: only 44h of video.
  Better to use Kaggle (30h/week) for inference.
  Lightning sessions are more reliable for training (steady GPU workload).

WHY LIGHTNING FOR STUDENT TRAINING:
  Student training (Phase 1, 2, 3) runs for hours continuously.
  Lightning AI has better session stability than Kaggle for long training runs.
  22h/month covers:
    Phase 1 self-supervised: ~20h  ← uses most of monthly quota
    Phase 2 KD fine-tuning:   ~5h  ← use Kaggle for this
    Phase 3 fMRI fine-tune:  ~10h  ← split across 2 months or use Kaggle

LIGHTNING SETUP FOR PHASE 1 (self-supervised pre-training):

  1. Create a Lightning Studio (free tier)
  2. Clone your repo: git clone <your-repo>
  3. pip install -r requirements.txt
  4. Mount Google Drive (where cached features live):
       from google.colab import drive
       drive.mount('/gdrive')
  5. Run Phase 1 training:
       python train_phase1.py \
         --feature-dir /gdrive/MyDrive/tribe_features \
         --checkpoint-dir /gdrive/MyDrive/checkpoints \
         --batch-size 32 \
         --epochs 25

LIGHTNING SESSION MANAGEMENT:
  Lightning AI sessions persist between connections (unlike Colab).
  Start Phase 1, close browser, reconnect next day — training continues.
  This makes it ideal for long Phase 1 runs (20h total).

  Save checkpoints to Google Drive every epoch:
    → If Lightning session ends, resume from last checkpoint
    → 25 epochs × ~45min each = ~19h total for Phase 1
    → Fits within 22h/month Lightning quota with ~3h to spare
```

---

### The Actual Execution Plan With Your Resources

```
WEEK 0 (Day 1, ~3h setup):
  □ Modal: run inference job on 15h diverse video ($8 of $30 credits)
    → 360 segments × parallel A10G → done in ~4h, saves to Modal Volume
    → Download to Google Drive (~8GB fp16 predictions)
    → You now have your entire target teacher dataset

  □ Kaggle: set up inference notebook + daily schedule
    → Will accumulate MORE predictions in background (optional top-up)

  □ Extract tiny backbone features on Kaggle (first session, ~6h):
    python extract_features.py \
      --video-dir /kaggle/input/your-videos \
      --output-dir /kaggle/working/features \
      --models miniLM,whisper-tiny,mobilevit-s
    → Upload to Google Drive

WEEK 1 (Lightning AI, Phase 1 self-supervised):
  □ Mount Google Drive features in Lightning Studio
  □ Start Phase 1 training (20h, runs over ~3 Lightning sessions)
  □ Checkpoint to Drive every epoch
  □ Meanwhile Kaggle accumulates more teacher predictions (background)

WEEK 2 (Lightning AI / Kaggle, Phase 2 KD):
  □ Phase 2 KD fine-tuning on Modal teacher predictions (5h)
    Run on Kaggle (1 session) or Lightning (uses ~5h of monthly quota)
  □ Val Pearson r check → should be >0.22

WEEK 2-3 (Kaggle, Phase 3 fMRI fine-tuning):
  □ Download CNeuroMod fMRI data (free, requires data agreement)
  □ Phase 3 training on Kaggle (10h = 2 Kaggle sessions)
  □ Final Pearson r: target >0.27

TOTAL:
  Modal credits used:    ~$8   (15h video, first day)
  Kaggle quota used:     ~60h  (background inference + Phase 3 training)
  Lightning AI quota:    ~22h  (Phase 1 self-supervised)
  Wall clock:            ~3 weeks
  Expected Pearson r:    0.27-0.29
```

---

### Minimum Viable Run: If You Only Have 1 Kaggle Session

```
If you want to test the entire pipeline before committing weeks of time,
here is the smallest possible end-to-end distillation run:

DATA:
  Teacher predictions:  2h of diverse video (1 Kaggle session, ~4h GPU)
  Backbone features:    Same 2h + 10h of LibriSpeech audio-only (free, tiny)
  fMRI:                 CNeuroMod Friends S01E01 only (~45 min)

TRAINING:
  Phase 1 (self-sup):   5 epochs only (2-3h GPU on Kaggle)
  Phase 2 (KD):         3 epochs (1h GPU)
  Phase 3 (fMRI):       3 epochs (1h GPU)

EXPECTED RESULT:
  Pearson r: 0.15-0.20  (50-65% of TRIBE v2)
  This is not final quality — it's a smoke test of the full pipeline.
  Every component gets exercised: inference, feature extraction,
  self-supervised pre-training, KD, fMRI fine-tuning.

WHY DO THIS FIRST:
  - Catch bugs before spending 3 weeks on a broken pipeline
  - Validate that teacher predictions are correctly formatted
  - Confirm that the self-supervised losses actually decrease
  - Measure actual GPU memory usage (may need to reduce batch size)

TOTAL GPU COST: ~8h Kaggle (1 week quota), $0
TOTAL TIME: 2-3 days
```

---

## Part 3d: The 2-Day Sprint Plan

Get a trained, evaluated student model in 48 hours.
Uses Modal credits for inference, Kaggle for training.
No Lightning AI needed (save that quota for a longer run later).

Expected result: Pearson r **0.24-0.27** — a real, working distilled model.

---

### What You Accept As Tradeoffs

```
Full Strategy C target:  0.29-0.31 Pearson r  (3-5 weeks)
2-Day Sprint target:     0.24-0.27 Pearson r  (48 hours)

Tradeoffs accepted:
  ✗ No Phase 1 self-supervised pre-training  (saves ~20h)
  ✗ Only 5h of teacher predictions instead of 15-30h
  ✗ No Phase 3 fMRI fine-tuning             (saves ~10h)
  ✓ Full KD pipeline exercised end to end
  ✓ Real model you can run inference with
  ✓ Clear baseline to iterate from
  ✓ All cached artifacts reusable for the full run later

The 2-day model is not throwaway — it becomes the Phase 2 checkpoint
for the full Strategy C run. Nothing is wasted.
```

---

### Hour-by-Hour Schedule

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAY 1  |  TEACHER INFERENCE + FEATURE EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

H+0:00  YOU: Launch Modal inference job (5 min of your time)
        modal run modal_tribe_inference.py --hours 5 --gpu T4
        → Modal spins up ~120 parallel T4 containers
        → Each container processes one 150s video segment
        → All 120 segments (= 5h video) done in parallel

H+0:05  YOU: Launch Kaggle notebook for feature extraction (10 min)
        → Upload your video files to Kaggle Dataset
        → Start notebook: extract_features.py on the same 5h of video
        → Tiny backbones (67M params) run fast — done in ~30min

H+0:10  YOU: Nothing to do. Both jobs running in parallel.
        Modal:  120 containers each finishing in ~75s
        Kaggle: feature extraction churning through videos

H+0:45  Modal done. 5h of teacher predictions in Modal Volume.
        → Download to local machine: modal volume get tribe-cache predictions/
        → Upload to Google Drive: ~4GB fp16 predictions

H+1:00  Kaggle feature extraction done.
        → 5h of backbone features (text + audio + video) saved to Drive
        → ~0.5GB total

H+1:30  All data on Google Drive. Both jobs complete.
        Modal cost so far: ~$4 of $30 credits (5h × T4 rate)

        ┌─────────────────────────────────────────────┐
        │  READY FOR TRAINING                          │
        │  Google Drive now has:                       │
        │    predictions/  — teacher preds (4GB fp16)  │
        │    features/     — backbone feats (0.5GB)    │
        └─────────────────────────────────────────────┘

H+2:00  YOU: Start Kaggle training notebook (5 min setup)
        → Mount Google Drive in Kaggle notebook
        → Launch Phase 2 KD training (no Phase 1 — go straight to KD)

        Why skip Phase 1 here:
          Phase 1 pre-training takes 20h and needs 1000h of data.
          In the 2-day sprint we go directly to KD.
          The model starts with random fusion weights (not pre-trained).
          This costs ~0.03 Pearson r vs the full run — acceptable for a sprint.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAY 1 H+2 to H+11  |  PHASE 2: KD TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kaggle session: T4 16GB, 9h max

Training config for the sprint (aggressive but stable):
  Model:       TinyTribeMoE (v3 arch, n_vertices=1000 Schaefer target)
  Data:        5h teacher predictions on CNeuroMod/diverse video
  Epochs:      20  (more epochs, less data — compensates for small dataset)
  LR:          3e-4  (higher than normal — small dataset overfits slower with high LR)
  Batch size:  8
  Segment len: 50 TRs (shorter — more updates per epoch)
  Loss:        0.7 × output_KD  +  0.2 × temporal  +  0.1 × feature_KD  +  aux
  Modality dropout: 0.2  (lower than normal — small dataset, need all signal)

  Layer-by-layer LR:
    Projectors:  3e-4
    MoE fusion:  3e-4
    Output head: 1e-3  (higher — this is what maps features to vertices)

  Save checkpoint every 5 epochs to Google Drive.

H+2:00   Training starts on Kaggle
H+7:00   Epoch ~12 complete. Val loss plateauing.
         Kaggle saves checkpoint to Drive automatically.
H+9:00   Kaggle session hits 9h limit. Training stops at ~epoch 18.
         Checkpoint on Drive: sprint_phase2_e18.pt

H+9:30  YOU: Start second Kaggle session, resume from checkpoint.
        Load sprint_phase2_e18.pt, run 2 more epochs.
        Total: 20 epochs done. Training complete.

H+11:00  Phase 2 complete.
         Expected val Pearson r (on held-out video segments): 0.18-0.22
         (Lower than full Strategy C because no Phase 1 pre-training)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAY 2 H+0 to H+8  |  PHASE 3: fMRI FINE-TUNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start of Day 2. Load Phase 2 checkpoint. Fine-tune on real fMRI.

DATA: CNeuroMod Friends S01 only (1 subject, ~15h fMRI)
  Why just 1 subject + 1 season:
    - Proves the pipeline works on real fMRI
    - 15h × 1 subject = ~36K TRs — enough for meaningful fine-tuning
    - Full run later uses all 4 subjects × 6 seasons

  Download CNeuroMod S01 data:
    → Register at cneuromod.ca (academic data agreement, ~1 day)
    OR use the Algonauts 2025 training data if already downloaded
    → Upload preprocessed fMRI .pt files to Google Drive

Training config for Phase 3 sprint:
  Init:         sprint_phase2_e20.pt
  Epochs:       8
  LR:           {fusion: 5e-5, output: 1e-4}  (backbones frozen)
  Loss:         0.5 × fMRI  +  0.3 × teacher_pred  +  0.2 × temporal
  Batch size:   4  (fMRI segments are larger)
  Segment len:  100 TRs

H+0:00   Phase 3 training starts on Kaggle (new session)
H+5:00   Epoch 8 complete. Training done.
         Save: sprint_final.pt to Google Drive

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAY 2 H+5 to H+8  |  EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

H+5:00  Load sprint_final.pt on Kaggle.
        Run evaluation on CNeuroMod Friends S01E10-S01E12 (held-out).

Evaluation metrics to compute:
  1. Mean Pearson r across all 20,484 vertices (or 1,000 Schaefer parcels)
  2. Per-ROI Pearson r: early visual, auditory, language, default mode
  3. Noise ceiling fraction: Pearson r / noise_ceiling (how much signal captured)
  4. Inference speed: ms per second of video on T4

H+7:00  Results. Compare against TRIBE v2 teacher on the same clips.

H+8:00  DONE.
```

---

### What You Have After 48 Hours

```
Artifacts on Google Drive:
  predictions/          5h teacher predictions (4GB) — reuse in full run
  features/             5h backbone features (0.5GB) — reuse in full run
  sprint_phase2_e20.pt  KD checkpoint — starting point for full run Phase 2
  sprint_final.pt       Final model — usable now

Performance:
  Pearson r:            0.24-0.27  (77-87% of TRIBE v2)
  Inference speed:      ~280ms/s on T4, ~3s/s on CPU
  Model size:           ~120MB INT8 after export

Costs:
  Modal credits:        ~$4 (5h video inference)
  Kaggle quota:         ~20h (out of 30h/week)
  Lightning AI:         0h used (save for full run)
```

---

### What to Do Next (Full Run)

The 2-day sprint produces everything you need to start the full Strategy C run
immediately — nothing has to be redone.

```
After the sprint, the full run continues from where you left off:

  Sprint artifact             Full run usage
  ─────────────────────────────────────────────────────────────
  predictions/ (5h)        →  Kept as Phase 2 seed data
  features/ (5h)           →  Kept, add more via Kaggle background job
  sprint_phase2_e20.pt     →  Phase 2 starting point (skip Phase 2 warmup)
  sprint_final.pt          →  Phase 3 starting point (already partially trained)

  What the full run adds:
  Phase 1:  Self-supervised pre-training on 1000h+ data (Lightning AI, 3 weeks)
            → Load Phase 1 weights → re-run Phase 2 from scratch (better init)
  Phase 2:  More teacher predictions (Kaggle accumulates 60h/week)
            → Continue Phase 2 with 30h predictions instead of 5h
  Phase 3:  All 4 CNeuroMod subjects + Friends S01-S06 (not just S01)

  Expected improvement from sprint → full run:
    0.24-0.27  →  0.29-0.31 Pearson r
```

---

### Decision Tree: Sprint vs Full Run

```
                    Do you have 48 hours?
                           │
                    ┌──────┴──────┐
                   YES            NO
                    │              │
           Run the 2-day      Wait — do full
             sprint            run properly
                    │
         Is r > 0.22 after sprint?
                    │
             ┌──────┴──────┐
            YES             NO
             │               │
     Continue to         Debug:
     full run             □ Check teacher predictions shape
                          □ Check feature normalization
                          □ Check loss isn't NaN
                          □ Reduce LR by 3×, retry Phase 2
```

---

### Option 0: Reuse What Already Exists — Cost $0, Time 0

Before running a single inference, check these sources.

**0a. Your own Algonauts 2025 competition predictions**
```
If you submitted to Algonauts 2025, you already ran TRIBE v2 on the
test set stimuli (Friends S7 + held-out clips).

What you likely have on disk:
  - Predictions on ~16h of Friends S7 (the competition test split)
  - Possibly full CNeuroMod Friends S1-S7 if you ran validation locally

Action: Find these files. They are valid KD targets.
        Friends S7 alone covers the entire validation split for free.
        Friends S1-S6 (if cached) covers Phase 2 KD entirely.
```

**0b. Meta FAIR published artifacts — check HuggingFace**
```
TRIBE v2 weights are at 'facebook/tribev2' on HuggingFace.
Also check whether Meta released pre-computed predictions.

Search:
  huggingface.co/datasets?search=tribe
  huggingface.co/datasets?search=algonauts
  huggingface.co/datasets?search=cneuromod

Also: email the authors (d'Ascoli et al. at Meta FAIR).
Research groups routinely share cached predictions on request,
especially for academic distillation work. One email = potentially
264h of free predictions.
```

**0c. Algonauts 2025 challenge baseline predictions**
```
Challenge organizers provide baseline predictions to help participants.
Check the Algonauts 2025 GitHub and challenge forum for:
  - Baseline submission predictions on training set
  - Pre-extracted features from reference models
  - Any shared Google Drive links in the challenge forum

Even if only partial, these cover the most competition-relevant stimuli.
```

**Realistic outcome from Option 0:**
```
Best case: 264h of CNeuroMod predictions already exist → $0, start Phase 2 now
Worst case: nothing exists, but you save 2 weeks of searching before paying
```

---

### Option 1: Kaggle Free GPUs — Best Pure-Free Option

```
Platform:  kaggle.com (free account)
GPU:       T4 16GB or P100 16GB (assigned randomly)
Quota:     30 GPU-hours/week per account, resets every Monday
Cost:      $0

THROUGHPUT:
  30h GPU/week × 2 (video per GPU-second) = 60h video/week
  Minus overhead (model load, checkpointing, I/O): ~50h usable video/week

  1 account, 4 weeks:  ~200h of teacher predictions
  2 accounts, 4 weeks: ~400h — covers ALL required teacher data for free

SETUP (one-time, ~2h):
  1. Create Kaggle account (and a second one on a different email)
  2. Upload your video segments as a Kaggle Dataset (private)
     OR link directly from a Google Drive mount
  3. Create a Kaggle Notebook that:
       a. pip install tribev2 / loads 'facebook/tribev2' from HF Hub
       b. Reads a manifest file: {video_id: "pending"/"done"}
       c. Processes pending segments, saves predictions to /kaggle/working/
       d. Rsyncs /kaggle/working/ to Google Drive (via rclone or PyDrive)
       e. Updates manifest at end of run
  4. Enable "Schedule" on the notebook → runs daily automatically

PRACTICAL THROUGHPUT PER KAGGLE SESSION:
  Session length: up to 9h (Kaggle hard limit per run)
  TRIBE v2 load time: ~45s (cold start)
  Inference rate: 1s video per 0.5s GPU = 2× realtime
  9h session → 18h of video processed → ~8.3 GB predictions (fp16)

CHECKPOINTING (critical — Kaggle can preempt):
  Save a manifest.json tracking which segments are done.
  On startup: load manifest, skip completed segments.
  On crash/timeout: restart notebook, resumes from last checkpoint.

  manifest.json format:
  {
    "friends_s01e01_000": "done",
    "friends_s01e01_100": "done",
    "friends_s01e01_200": "pending",
    ...
  }
```

---

### Option 2: Google Colab Free Tier

```
Platform:  colab.research.google.com
GPU:       T4 (shared, not guaranteed — sometimes CPU only)
Quota:     Soft limit ~3-5h GPU/day, hard disconnect after 12h runtime
Cost:      $0

THROUGHPUT:
  3h usable GPU/day × 2 (realtime) = ~6h of video/day
  30 days: ~180h of teacher predictions

PRACTICAL ISSUES:
  - GPU not always available (sometimes assigned CPU)
  - Disconnects after ~90min inactivity
  - 12h hard runtime limit per session

ANTI-DISCONNECT (run in browser console):
  function ClickConnect() {
    console.log("Preventing disconnect...");
    document.querySelector("colab-connect-button").click();
  }
  setInterval(ClickConnect, 60000);

SETUP:
  Mount Google Drive in Colab.
  Load TRIBE v2 weights from HF Hub into Drive (cache once).
  Run inference notebook, save predictions to Drive directly.
  Same manifest-based checkpointing as Kaggle.

Colab Pro ($10/month) — RECOMMENDED if spending any money:
  - Guaranteed T4 or V100 access
  - 24h runtime limit (vs 12h free)
  - ~8h GPU/day → 16h video/day → 480h/month
  - At $10/month: effectively $0.02/GPU-hour
  - This is 30-50× cheaper than any cloud provider
  - One month of Colab Pro = all 320h of teacher inference for $10
```

---

### Option 3: HuggingFace ZeroGPU Spaces

```
Platform:  huggingface.co/spaces (ZeroGPU tier)
GPU:       A100 40GB (!) — better than T4
Quota:     GPU burst per API request, ~60s max per call
Cost:      $0 (with HuggingFace account)

TRICK: Deploy TRIBE v2 as a private Gradio Space, call it as an API.

How it works:
  1. Create a HuggingFace Space (private) with ZeroGPU enabled
  2. Gradio app: accepts video segment path → returns predictions tensor
  3. Call this Space's API endpoint from your laptop or Colab:
       response = requests.post(space_api_url, json={"segment_id": "..."})
  4. Each API call gets a fresh A100 burst for up to 60s

THROUGHPUT PER CALL:
  TRIBE v2 on A100: ~5× faster than T4 → ~10× realtime
  60s time limit → processes ~600s = 10 minutes of video per call
  With HRF padding: effectively ~8 minutes of usable predictions per call

  Daily limit (community rate limiting): ~50-100 API calls/day (unofficial)
  50 calls × 8min = 400min = 6.7h video/day → ~200h/month

SETUP (~3h):
  Create app.py with:
    @spaces.GPU
    def predict_segment(video_id, tr_start, tr_end):
        model = TribeModel.from_pretrained('facebook/tribev2')
        # load pre-extracted features for this segment from HF dataset
        preds, fusion = model.predict_and_cache(video_id, tr_start, tr_end)
        return preds.cpu().numpy(), fusion.cpu().numpy()

  gr.Interface(fn=predict_segment, ...).launch()

LIMITATION: 60s per call is tight. Pre-extract tiny backbone features
  locally first, upload to a private HF Dataset, then the Space only
  runs the TRIBE v2 fusion + output head (much faster — just the
  ~70M trainable params, not the 4.7B backbones).
  This makes each call ~3× faster → fits 30+ min of video per 60s.
```

---

### Option 4: NSF ACCESS (Free for US Academics)

```
Platform:  access-ci.org (formerly XSEDE)
GPU:       A100, V100, varies by cluster (Bridges-2, Expanse, Delta)
Quota:     Discovery allocation: 200,000 GPU-hours (free, no cost)
Cost:      $0 for academics
Time:      2-4 weeks to get approved

HOW TO APPLY:
  1. Go to access-ci.org → "Request Access"
  2. Choose "Explore" allocation (quickest, up to 400K GPU-hours)
  3. Write a 1-page project description:
     "Distillation of large-scale neural encoding models for accessible
      computational neuroscience. We compress TRIBE v2 (4.7B params) into
      a 45M-parameter student model for broad research accessibility."
  4. List PI (faculty sponsor required for students)
  5. Approval: typically 5-10 business days for Explore allocations

WHAT YOU GET:
  Bridges-2 GPU partition: A100 80GB nodes
  TRIBE v2 on A100 80GB: ~5× faster than T4 (fits full model in fp32)
  160 GPU-hours T4 equivalent = ~32 GPU-hours A100
  At 200K hours allocation: you can run this 6,000 times over.

This is the best option if you have any US academic affiliation.
Even students can apply with a faculty sponsor.
Apply now — the 2-4 week wait is the only cost.
```

---

### Option 5: Reduce Teacher Inference Needed (Architectural Tricks)

These reduce the teacher compute needed without hurting final quality.

**5a. Cache at lower resolution, upsample for loss**
```
Teacher outputs fsaverage5 (20,484 vertices).
Cache at fsaverage4 (5,124 vertices) — 4× fewer values to store.

Implementation:
  After teacher inference, project predictions to fsaverage4 surface
  using nearest-neighbour mapping (pre-computed, fast).
  Student trains on fsaverage4 targets (which it does anyway).

Storage: 220GB → 55GB for 320h of predictions (fp16)
Compute: Same teacher inference, no savings on GPU time.
Quality: Zero — student targets are already fsaverage4.
```

**5b. Diversity-sample which videos to run teacher on**
```
Instead of running teacher on all 320h uniformly:
  1. Extract tiny backbone features for all videos (cheap)
  2. Cluster all 30-second segments into K=300 clusters by content
  3. Run teacher on 1-2 representative segments per cluster
  4. Augment with interpolated pseudo-labels for unselected segments

This gives 300 × 2min ≈ 10h of actual teacher inference covering
the full semantic space of the 320h dataset.

Student trains on: 10h real teacher predictions + 310h pseudo-labels
generated by lightweight interpolation between nearest cached segments.

GPU cost: 10h video × 0.5 = 5 GPU-hours T4 (vs 160h baseline)
Quality hit: ~0.01-0.02 Pearson r (fusion pre-training compensates)
```

**5c. Student self-training loop (pseudo-labelling)**
```
After Phase 2 (student trained on 30-50h of teacher data):
  Student Pearson r ≈ 0.22-0.25 — not as good as teacher (0.31)
  but much better than random.

Use the student to generate predictions on new videos:
  Student inference: ~280ms/s on T4 → 10× faster than teacher
  Generate pseudo-labels on 300h of additional diverse video: ~15h T4

Mix into Phase 3 training:
  L = 0.4×fMRI + 0.3×real_teacher + 0.2×student_pseudo + 0.1×other

IMPORTANT: Weight pseudo-labels lower than real teacher predictions.
           Student errors can compound if given too much weight.
           Use only after Phase 2 gives Pearson r > 0.20.

GPU cost for 300h pseudo-labels: ~15h T4 ≈ $1.50 on spot
vs teacher cost for same data: 150h T4 ≈ $15.00 on spot
Savings: 10× on inference cost for additional data beyond Phase 2.
```

---

### The Zero-Dollar Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COMPLETE FREE EXECUTION PLAN                                           │
│  Expected total cost: $0                                                │
│  Expected wall clock: 5-6 weeks (parallel with other work)             │
│  Expected Pearson r: 0.27-0.30                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DAY 1: Setup (2-3h of your time)                                      │
│    □ Search HuggingFace + email authors for existing predictions        │
│    □ Set up 2 Kaggle accounts with inference notebook                   │
│    □ Extract tiny backbone features on local machine or Colab           │
│      (tiny models, runs on CPU too if needed — just slower)            │
│    □ Start Phase 1 self-supervised training (Kaggle or Colab)          │
│      ← This needs ZERO teacher predictions. Start immediately.         │
│                                                                         │
│  WEEKS 1-2: Background accumulation (automated, no attention needed)   │
│    □ Kaggle account 1: runs daily, processes ~50h video/week            │
│    □ Kaggle account 2: runs daily, processes ~50h video/week            │
│    □ Colab free (when Kaggle quota exhausted): adds ~30h/week          │
│    □ Phase 1 training completes (~20h GPU, run on Kaggle)              │
│    □ By end of week 2: ~200h of teacher predictions accumulated        │
│                                                                         │
│  WEEK 3: Phase 2 KD (30h of predictions is enough to start)           │
│    □ Phase 2 trains on cached predictions (5h GPU, 1 Kaggle session)  │
│    □ Kaggle continues accumulating predictions in background            │
│                                                                         │
│  WEEK 4: Phase 3 fMRI fine-tuning                                      │
│    □ Phase 3 trains on CNeuroMod fMRI (10h GPU, 2 Kaggle sessions)   │
│    □ Teacher predictions from Kaggle used as regularizer               │
│    □ Student pseudo-labels generated for additional data               │
│                                                                         │
│  WEEK 5: Evaluation + ONNX export                                      │
│    □ Run evaluation on Friends S7 / BOLD Moments                       │
│    □ Export to ONNX INT8 for browser deployment                        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  IF THIS IS TOO SLOW: Colab Pro ($10) cuts weeks 1-2 to 3-4 days     │
│  IF YOU HAVE ACADEMIC ACCESS: Apply for NSF ACCESS now (2-4 weeks)    │
│    → Once approved: complete all teacher inference in 1-2 days for $0 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Free Inference — Full Option Comparison

```
┌─────────────────────────────┬──────────┬─────────────┬────────────┬──────────┐
│ Option                      │ Cost/mo  │ Video/month │ Setup time │ Quality  │
├─────────────────────────────┼──────────┼─────────────┼────────────┼──────────┤
│ 0. Existing predictions     │ $0       │ instant     │ 0          │ ★★★★★   │
│ 1. Kaggle (2 accounts)      │ $0       │ ~400h       │ 2h         │ ★★★★★   │
│ 2. Colab free               │ $0       │ ~150h       │ 1h         │ ★★★☆☆   │
│ 2b. Colab Pro               │ $10      │ ~480h       │ 1h         │ ★★★★★   │
│ 3. HF ZeroGPU               │ $0       │ ~200h       │ 3h         │ ★★★★☆   │
│ 4. NSF ACCESS               │ $0       │ unlimited   │ 2-4 weeks  │ ★★★★★   │
│ 5b. Diversity sampling      │ $0       │ equiv. 300h │ 2h         │ ★★★★☆   │
│ 5c. Student pseudo-labels   │ $0       │ unlimited   │ after Ph2  │ ★★★☆☆   │
│ Vast.ai RTX3090 spot        │ ~$13     │ all 320h    │ 1h         │ ★★★★★   │
└─────────────────────────────┴──────────┴─────────────┴────────────┴──────────┘

RECOMMENDED STACK:
  Primary:   Kaggle 2× accounts  ($0, reliable, automated)
  Parallel:  HF ZeroGPU          ($0, faster GPU, good complement)
  Fallback:  Colab Pro           ($10, if Kaggle quotas exhausted)
  Academic:  NSF ACCESS          ($0, apply now — 2 week wait is worth it)
  Augment:   Student pseudo-labels ($0, after Phase 2)
```

---

### Phase 0: Cache the Teacher (Once, Never Again)

```
INPUT:  Raw video files
OUTPUT: teacher_preds.pt, teacher_fusion_l4.pt, teacher_fusion_l6.pt

Per video:
  1. Load TRIBE v2 from HuggingFace ('facebook/tribev2')
  2. model.predict(video_path)
  3. Also hook into layer 4 and 6 of fusion transformer to save activations
  4. Save to disk indexed by video_id + timestamp range

Storage format:
  {
    'video_id': 'friends_s01e01',
    'tr_start': 0,
    'tr_end': 100,
    'predictions': Tensor(100, 20484),    # vertex predictions per TR
    'fusion_l4':   Tensor(100, 1152),     # layer 4 fusion activations
    'fusion_l6':   Tensor(100, 1152),     # layer 6 fusion activations
  }

Cost: ~500ms/s of video on T4
  CNeuroMod (264h): ~132h T4
  Extra video (50h): ~25h T4
  Total: ~157h T4, store ~500GB
```

### Phase 1: Self-Supervised Pre-training

**Goal:** Train fusion transformer to understand cross-modal temporal dynamics.
No brain data. No teacher. Just raw multimodal patterns.

```
ARCHITECTURE (modified for pre-training):
  Same v3 model BUT:
  - Remove: HRF conv, gated pooling, output MLP, FiLM, vertex projection
  - Add: 4 pre-training heads (described below)
  - Backbones: frozen throughout

DATA LOADING:
  Features pre-extracted offline with frozen tiny backbones.
  Cached as .pt files. Dataset yields (text, audio, video) feature tensors.
  Segment length: 60 timesteps (30s at 2Hz).
  Stride: 30 timesteps (50% overlap).
  Batch size: 32 (large = better contrastive + MoE routing stability).

TRAINING TASKS:

  Task 1: Masked Modality Reconstruction (MMR) — weight 0.4
  ──────────────────────────────────────────────────────────
  For each sample, randomly mask one modality (zero out all its tokens).
    p(mask 1 modality) = 0.50
    p(mask 2 modalities) = 0.25
    p(mask 0 modalities) = 0.25

  A per-modality reconstruction head predicts the masked projector output:
    head_m = Sequential(LayerNorm(512), Linear(512,512), GELU, Linear(512,512))
  Applied to the transformer output at the masked modality token positions.

  Loss:
    L_mmr = 0.7 * MSE(predicted_feat, original_projected_feat.detach())
           + 0.3 * (1 - cosine_sim(predicted_feat, original_projected_feat).mean())

  Task 2: Cross-Modal Contrastive (CMC) — weight 0.2
  ───────────────────────────────────────────────────
  Pool modalities at each timestep → (B, T, 512)
  Project to 128D contrastive space (small MLP head).
  L2 normalize.

  Positives: same sample, same timestep (all modalities present)
  Negatives: different samples in the batch

  InfoNCE loss, temperature τ=0.07.
  With batch_size=32 and T=60: 1920 anchors, 1919 negatives each.

  Task 3: Next-TR Prediction (NTP) — weight 0.2
  ──────────────────────────────────────────────
  Run transformer with causal mask (only attend to past tokens).
  Predict the fused representation at t+1 from fused representation at t.

  Causal forward pass → pool modalities → predict_head(pool_t) → pred
  Non-causal forward pass → pool_t+1 (target, detached)

  L_ntp = 0.5 * MSE(pred, target) + 0.5 * (1 - cosine_sim(pred, target))

  Task 4: Temporal Order (TOP) — weight 0.1
  ──────────────────────────────────────────
  Split segment into 4 chunks. Shuffle. Binary classify each adjacent pair:
  "are these in the correct order?"

  L_top = CrossEntropy(order_head(pair), correct_or_swapped_label)

  MoE auxiliary loss — weight 0.01
  ─────────────────────────────────
  Always accumulated. Load-balance + z-loss.

TOTAL LOSS:
  L = 0.4*L_mmr + 0.2*L_cmc + 0.2*L_ntp + 0.1*L_top + 0.01*L_aux

OPTIMIZER & SCHEDULE:
  Optimizer:    AdamW, weight_decay=0.01
  LR:           3e-4 (higher than KD phase — self-sup has smoother landscape)
  Scheduler:    Cosine with linear warmup (5% of total steps)
  Epochs:       25 over the full 1770h dataset
  Grad clip:    max_norm=1.0

MoE STABILITY SCHEDULE:
  Steps 0-1000:    Router warmup — aux_loss_weight ramps 0.1 → 0.01
  Steps 0-1000:    Router temperature decays 2.0 → 1.0 (softer routing early)
  Steps 1000+:     Normal training

MONITORING:
  MMR reconstruction cosine sim: target >0.6 by epoch 10
  CMC R@1 within-batch:          target >60% by epoch 10
  Expert entropy:                target >1.5 (of max log(8)=2.08)
  Expert utilization balance:    all experts 10-15% of tokens

EXPECTED COST: ~20h T4 for 25 epochs over 1770h of data (pre-extracted features)
```

### Phase 2: Teacher KD Fine-tuning

**Goal:** Map the pre-trained fusion representations to brain vertex predictions.
The fusion model already understands cross-modal dynamics. This phase just adds
the brain-specific output.

```
ARCHITECTURE:
  Restore full v3 model:
  - Pre-training heads: REMOVED
  - Add: gated modality pooling, HRF conv (Gamma-initialized), shared MLP,
         FiLM vectors (initialized to γ=1, β=0 for identity), vertex projection
  - Initialize FiLM from scratch (no subject-specific knowledge yet)
  - Load pre-trained weights for everything else from Phase 1

FREEZING STRATEGY:
  Frozen:  all backbones (MiniLM, Whisper-Tiny, MobileViT-S)
  Frozen:  modality embeddings, positional embeddings
  Trainable: projectors (already pre-trained, low LR), MoE transformer,
             modality gates, HRF conv, output MLP, FiLM, vertex projection

DATA:
  Load cached teacher predictions from Phase 0.
  Inputs: pre-extracted backbone features (reuse Phase 1 cache)
  Targets: teacher_preds (T, 20484), teacher_fusion_l4 (T, 1152), teacher_fusion_l6

LOSS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  L = 0.60 * MSE(student_pred, teacher_pred.detach())                │
  │    + 0.20 * feature_loss                                             │
  │    + 0.10 * temporal_loss                                            │
  │    + 0.05 * multi_res_loss                                           │
  │    + 0.01 * aux_loss                                                 │
  │                                                                       │
  │  feature_loss:                                                        │
  │    s = feat_proj(student_fused)  # Linear(512, 1152) trainable      │
  │    t = teacher_fusion_l4.detach()                                    │
  │    feature_loss = 1 - cosine_sim(s, t, dim=-1).mean()               │
  │                                                                       │
  │  temporal_loss:                                                       │
  │    Δs = student_pred[:,:,1:] - student_pred[:,:,:-1]                │
  │    Δt = teacher_pred[:,:,1:] - teacher_pred[:,:,:-1]                │
  │    temporal_loss = SmoothL1(Δs, Δt.detach())                        │
  │                                                                       │
  │  multi_res_loss: (Schaefer-400 parcel average matching)              │
  │    student_parcel = parcel_avg(student_pred, atlas)  # (B, T, 400)  │
  │    teacher_parcel = parcel_avg(teacher_pred, atlas)                  │
  │    multi_res_loss = MSE(student_parcel, teacher_parcel.detach())     │
  └──────────────────────────────────────────────────────────────────────┘

OPTIMIZER:
  AdamW, lr=1e-3 for output head and gates
  AdamW, lr=1e-4 for projectors (already pre-trained)
  Scheduler: OneCycleLR, 10% warmup, cosine decay
  Epochs: 10
  Batch size: 8 (T=100 TRs per sample = 150s segments)
  Grad clip: max_norm=1.0

MODALITY DROPOUT: 0.3 throughout Phase 2

MONITORING:
  Val Pearson r on CNeuroMod Friends S07:   target >0.22 by epoch 5
  Feature cosine sim (student vs teacher):  target >0.7 by epoch 5
  Temporal loss:                            should decrease monotonically
  Expert entropy:                           maintain >1.5

EXPECTED COST: ~5h T4
EXPECTED PEARSON r: 0.22-0.25 (pre-training gives great initialization)
```

### Phase 3: fMRI Ground Truth Fine-tuning

**Goal:** Tune on real fMRI signal. This is the final push to close the gap
between teacher predictions and actual brain data.

```
FREEZING STRATEGY:
  Frozen:  MiniLM (text backbone — always frozen)
  Unfreeze: Whisper-Tiny encoder  (LR: 5e-6  — very low, small nudge)
  Unfreeze: MobileViT-S           (LR: 5e-6  — very low)
  Trainable: all other components (LR: 5e-5 for fusion, 1e-4 for output)

  Ratio: fusion LR ≈ 10× backbone LR
  This is the standard for fine-tuning frozen backbone + trainable head.

DATA:
  Primary:   CNeuroMod fMRI (Friends S01-S06 per subject)
             Inputs: raw video/audio/text → backbone → features
             Targets: fMRI responses in fsaverage4 (5,124 vertices)

  Secondary: Lebel 2023 stories
             Inputs: audio + text only (video modality dropout = 1.0)
             Targets: fMRI responses
             Weight: 0.3× (less data, different modality profile)

LOSS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  L = 0.40 * MSE(student_pred, fmri_target)                          │
  │    + 0.30 * MSE(student_pred, teacher_pred.detach())  # regularizer │
  │    + 0.10 * feature_loss  (cosine vs teacher fusion features)        │
  │    + 0.10 * temporal_loss                                            │
  │    + 0.05 * multi_res_loss                                           │
  │    + 0.01 * aux_loss                                                 │
  │                                                                       │
  │  Note: teacher_pred is from cached Phase 0 predictions.             │
  │  It acts as a regularizer — prevents the model from overfitting to  │
  │  subject-specific noise in the fMRI signal.                          │
  └──────────────────────────────────────────────────────────────────────┘

MODALITY DROPOUT SCHEDULE:
  Epoch 1-3:   0.3  (maintain robustness from Phase 2)
  Epoch 4-6:   0.1  (teacher signal richest with all modalities)
  Epoch 7-10:  0.0  (squeeze maximum performance at evaluation time)

CURRICULUM:
  Start with shorter segments (50 TRs = 75s).
  After epoch 3, switch to full segments (100 TRs = 150s).
  Why: shorter segments give more gradient updates early,
       full segments give better temporal context later.

OPTIMIZER:
  AdamW
  LR: {backbones: 5e-6, projectors: 1e-5, fusion: 5e-5, output: 1e-4}
  Scheduler: OneCycleLR, 5% warmup, cosine decay
  Epochs: 10
  Batch size: 4 (larger segments, less memory)

VALIDATION:
  CNeuroMod Friends S07 (held out completely)
  Metric: mean Pearson r across 20,484 vertices, averaged across 4 subjects
  Secondary: BOLD Moments test clips (out-of-domain)

MONITORING:
  Val Pearson r:       target >0.27 by epoch 5, >0.29 by epoch 10
  Teacher consistency: MSE(student, teacher) shouldn't spike
                       (if it does → overfitting to noise, increase teacher weight)
  FiLM γ norms:        should diverge across subjects (means adaptation is working)
                       if all γ ≈ 1: FiLM is not learning subject differences

EXPECTED COST: ~10h T4
EXPECTED PEARSON r: 0.29-0.31 (matching or exceeding TRIBE v2)
```

---

## Summary: Everything in One Table

| | v2 MoE | v3 Strategy C |
|--|--------|--------------|
| **Backbones** | MiniLM + Whisper-Tiny + MobileViT-S | Same |
| **Video temporal** | Per-frame only | + Depthwise Conv1D (motion) |
| **Projectors** | 3-layer, 768 intermediate | Same |
| **Positional embed** | Shared, modality-blind | Per-modality temporal embed |
| **Fusion** | 4-layer MoE, full attention | Layers 1-2 local+HRF bias, 3-4 full |
| **Modality pool** | Mean | Gated (sigmoid, learned) |
| **HRF modeling** | AdaptiveAvgPool | Depthwise Conv1D, Gamma-initialized |
| **Subject heads** | Per-subject linear (33M) | Shared MLP + FiLM (0.6M) |
| **Stochastic depth** | Uniform 10% | Linear schedule 5-20% |
| **Aux loss bug** | Skipped when layer dropped | Always accumulated |
| **Training** | Direct KD only | Self-sup (Phase1) → KD (Phase2) → fMRI (Phase3) |
| **Teacher inferences** | ~500h video | ~320h (CNeuroMod + extras) |
| **Self-sup data** | 0 | 1770h free |
| **Trainable params** | ~42M | ~14M (FiLM replaces SubjectLayers) |
| **Active params** | ~16M | ~17M |
| **Expected Pearson r** | 0.27-0.29 | **0.29-0.31** |
| **New subject cost** | Retrain 5M+ params | Learn 1024 floats (γ, β) |
| **Browser size (INT8)** | ~120MB | ~120MB |
