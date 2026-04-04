# Tiny-TRIBE v3: Final Architecture, Critique & Training Strategy

---

## Question 1: Current Architecture (v2 MoE) — The Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                    TINY-TRIBE v2 MoE (current)                  ║
╚══════════════════════════════════════════════════════════════════╝

 Text (transcript)    Audio (16kHz)       Video (2fps frames)
        │                   │                      │
        ▼                   ▼                      ▼
 ┌─────────────┐    ┌─────────────┐      ┌─────────────────┐
 │all-MiniLM   │    │Whisper-Tiny │      │  MobileViT-S    │
 │  22.7M      │    │  39M        │      │    5.6M         │
 │  (frozen)   │    │  (frozen→   │      │  (frozen→unfrz  │
 │             │    │   unfreeze  │      │   phase 2)      │
 │             │    │   phase 2)  │      │  PER-FRAME ONLY │  ← ⚠ no motion
 └──────┬──────┘    └──────┬──────┘      └───────┬─────────┘
        │                  │                     │
   (B,T,384)          (B,T,384)             (B,T,640)
        │                  │                     │
        ▼                  ▼                     ▼
 ┌────────────┐    ┌────────────┐       ┌────────────────┐
 │3-layer MLP │    │3-layer MLP │       │  3-layer MLP   │
 │384→768→512 │    │384→768→512 │       │  640→768→512   │
 └──────┬─────┘    └──────┬─────┘       └───────┬────────┘
        │                  │                     │
   (B,T,512)          (B,T,512)             (B,T,512)
        │                  │                     │
        ▼                  ▼                     ▼
  + modality_embed[0]  + modality_embed[1]  + modality_embed[2]
        │                  │                     │
        └──────────────────┴──────────────────────┘
                           │
                    INTERLEAVE                        ← ✓ good
              [t1, a1, v1, t2, a2, v2, ...]
                    (B, T×3, 512)
                           │
              + positional embeddings                 ← ⚠ modality-blind
                           │
                           ▼
         ┌─────────────────────────────────┐
         │   MoE TRANSFORMER × 4 layers    │
         │                                 │
         │  PreNorm → MHA (8h × 64D) ──┐  │
         │  PreNorm → MoEFFN            │  │
         │    Router: Linear(512→8)     │  │
         │    TopK(k=2) → 2 experts     │  │  ← ✓ good
         │    each: 512→1024→512        │  │
         │    + aux loss + z-loss       │  │
         │  Residual ───────────────────┘  │
         │  Layer dropout: 10%             │  ← ⚠ skips aux loss too (bug)
         └──────────────┬──────────────────┘
                        │
                  (B, T×3, 512)
                        │
              MEAN POOL modalities                    ← ⚠ lossy, brain-region
                .reshape(B,T,3,512)                      agnostic
                .mean(dim=2)
                        │
                  (B, T, 512)
                        │
              Linear(512 → 256)                      ← ⚠ single flat bottleneck
                        │
                  (B, T, 256)
                        │
              SubjectLayers[subject_id]               ← ⚠ 33M params for 25 subj
              W: (256 → n_vertices), linear               won't generalize
                        │
                  (B, T, n_vertices)
                        │
              AdaptiveAvgPool1d(n_TRs)
                        │
                  (B, n_vertices, n_TRs)
```

---

## Question 1: Critique of v2 MoE

### What's Good (Keep)

| Component | Why It Works |
|-----------|-------------|
| 3-layer projectors, 768 intermediate | Projector quality is the real bottleneck — validated by BLIP-2/LLaVA |
| Interleaved tokens | Same-timestep cross-modal attention in layer 1 (vs concatenation needs many layers) |
| 8 heads × 64D | Proven sweet spot. 4 heads = 128D/head is too wide |
| MoE 8 experts, top-2 | 4× capacity at same compute. Experts naturally specialize per modality-brain region |
| Expert init from shared FFN + noise | Prevents early collapse, faster specialization |
| Z-loss + aux load balancing | Stable training, uniform expert utilization |
| Modality dropout 30% | Right balance — 50% kills cross-modal learning, 0% is brittle |
| Phase 1 frozen → Phase 2 E2E | Stable, proven 2-stage KD pattern |

### What's Broken / Suboptimal (Fix)

**Flaw 1: Mean modality pooling — brain-region blind**
```
current: .mean(dim=2)  → all regions get equal text/audio/video blend
problem: visual cortex should weight video >> text
         Broca's area should weight text >> video
         Superior temporal sulcus: audio + visual integration
fix:     learned per-token gating — 3 gates per timestep, sigmoid weighted sum
```

**Flaw 2: No HRF modeling — zero neurophysical inductive bias**
```
current: AdaptiveAvgPool1d(n_TRs) — blind temporal aggregation
problem: fMRI signal peaks ~6s AFTER stimulus with known Gamma shape
         model wastes capacity learning this from data alone
fix:     learnable HRF convolution layer (init to canonical Gamma kernel)
         Conv1d(n_vertices, n_vertices, kernel_size=8, groups=n_vertices)
         initialized from double-Gamma HRF, fine-tuned during training
```

**Flaw 3: MobileViT-S is per-frame — no motion signal**
```
current: MobileViT extracts spatial features independently per frame
problem: visual motion cortex (V5/MT+) is the most fMRI-predictable region
         motion is frame differences, not frame content
fix:     temporal Conv1D after video projector
         Conv1d(512, 512, kernel_size=3, padding=1, groups=512) — depthwise
         2 params per channel, adds frame-to-frame dynamics
```

**Flaw 4: Flat single linear SubjectLayers — parameter explosion**
```
current: W[s]: 256 × n_vertices per subject = 33M params for 25 subjects
problem: purely linear, can't model nonlinear subject anatomy differences
         won't generalize to new subjects (no shared structure)
fix:     shared MLP backbone + subject-specific FiLM conditioning
         shared: Linear(512, 512) → GELU → Linear(512, n_vertices)
         per-subject: scale γ[s] (512,) and shift β[s] (512,) → FiLM
         total: 512×512 + 512×n_vertices + 2×512×n_subjects
         = 26M (same) but generalizes + shares computation
```

**Flaw 5: Positional embeddings are modality-blind**
```
current: pos_embed[position] — same for text, audio, video at same t
problem: text token at position 15 = audio token at position 15 = video at 15
         model can't tell which modality's temporal context matters
fix:     factored embedding: pos_embed[position] + modality_time_embed[mod, t]
         separate learned temporal embeddings per modality
```

**Flaw 6: Layer dropout skips aux_loss (training bug)**
```python
# current (moe_model.py line 280-283):
if self.training and torch.rand(1).item() < self.layer_dropout:
    continue  # ← also skips aux_loss accumulation
fused, aux_loss = layer(fused)
total_aux_loss = total_aux_loss + aux_loss

# fix: always compute aux_loss even when skipping forward pass
fused_out, aux_loss = layer(fused)
total_aux_loss = total_aux_loss + aux_loss  # always accumulate
if not (self.training and torch.rand(1).item() < self.layer_dropout):
    fused = fused_out
```

**Flaw 7: Uniform attention — HRF locality ignored**
```
current: full self-attention over all T×3 tokens equally
problem: brain encodes recent stimulus (0-8s ago) most strongly (HRF shape)
         distant past (>30s ago) contributes little
fix:     temporal decay attention bias in early layers
         bias[i,j] = -α × |t_i - t_j| (exponential decay with HRF timescale)
         free parameter α, learned per layer
```

---

## Question 2: Refined Architecture — Tiny-TRIBE v3

```
╔══════════════════════════════════════════════════════════════════════╗
║                    TINY-TRIBE v3 (proposed)                         ║
║              ~45M params, ~17M active, ~280ms/s video               ║
╚══════════════════════════════════════════════════════════════════════╝

 Text (transcript)    Audio (16kHz)       Video (2fps frames)
        │                   │                      │
        ▼                   ▼                      ▼
 ┌─────────────┐    ┌─────────────┐      ┌─────────────────┐
 │all-MiniLM   │    │Whisper-Tiny │      │  MobileViT-S    │
 │  22.7M      │    │  39M        │      │    5.6M         │
 │  (always    │    │  (unfreeze  │      │  (unfreeze      │
 │   frozen)   │    │   phase 3)  │      │   phase 3)      │
 └──────┬──────┘    └──────┬──────┘      └───────┬─────────┘
        │                  │                     │
   (B,T,384)          (B,T,384)             (B,T,640)
        │                  │                     │
        ▼                  ▼                     ▼
 ┌────────────┐    ┌────────────┐       ┌────────────────┐
 │3-layer MLP │    │3-layer MLP │       │  3-layer MLP   │
 │384→768→512 │    │384→768→512 │       │  640→768→512   │
 └──────┬─────┘    └──────┬─────┘       └───────┬────────┘
        │                  │                     │
        │                  │            ┌────────▼────────┐
        │                  │            │ Temporal Conv1D  │  ← NEW: motion
        │                  │            │ depthwise k=3    │        signal
        │                  │            │ captures Δframe  │
        │                  │            └────────┬────────┘
        │                  │                     │
   (B,T,512)          (B,T,512)             (B,T,512)
        │                  │                     │
        ▼                  ▼                     ▼
  + modality_time_embed[text, t]                           ← NEW: per-modality
  + modality_time_embed[audio, t]                               temporal embed
  + modality_time_embed[video, t]
        │                  │                     │
        └──────────────────┴─────────────────────┘
                           │
                    INTERLEAVE
              [t1, a1, v1, t2, a2, v2, ...]
                    (B, T×3, 512)
                           │
              + shared positional embedding
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │  MoE TRANSFORMER × 4 layers              │
         │                                          │
         │  Layers 1-2: LOCAL attention             │  ← NEW: temporal
         │    window_size = 12 tokens (±2 TRs)      │        locality bias
         │    + HRF decay bias: -α|t_i - t_j|       │        (HRF-motivated)
         │                                          │
         │  Layers 3-4: FULL attention              │  ← global context
         │    (semantic integration, narrative)     │
         │                                          │
         │  Each layer:                             │
         │    PreNorm → MHA (8h × 64D)             │
         │    PreNorm → MoEFFN (8 experts, top-2)  │  ← same as v2
         │    Residual                              │
         │    Stochastic depth (linear schedule)   │  ← NEW: deeper=more drop
         │                                          │
         │  *** aux_loss always accumulated ***     │  ← BUG FIX
         └──────────────────┬───────────────────────┘
                            │
                     (B, T×3, 512)
                            │
              GATED MODALITY POOLING                 ← NEW: brain-region aware
              gates = sigmoid(Linear(512, 3))
              per timestep: text_w, audio_w, video_w
              out = Σ gate_m × token_m
                            │
                     (B, T, 512)
                            │
              ┌─────────────────────────────────┐
              │  HRF CONVOLUTION LAYER          │  ← NEW: neurophysical prior
              │  depthwise Conv1d               │
              │  kernel_size=8 (~12s at 1.5s TR)│
              │  init: double-Gamma HRF kernel   │
              │  fine-tuned during training      │
              └─────────────────┬───────────────┘
                                │
                         (B, T, 512)
                                │
              ┌─────────────────────────────────┐
              │  SHARED OUTPUT MLP              │  ← NEW: replaces flat linear
              │  LayerNorm(512)                 │
              │  Linear(512 → 512) + GELU       │
              │  ↓                              │
              │  FiLM conditioning:             │  ← NEW: subject as scale/shift
              │  γ[subject] * x + β[subject]    │       not separate weight matrix
              │  γ, β ∈ R^512 per subject       │       generalizes to new subjects
              │  ↓                              │
              │  Linear(512 → n_vertices)       │
              └─────────────────┬───────────────┘
                                │
                         (B, T, n_vertices)
                                │
                    transpose + pool to n_TRs
                                │
                    (B, n_vertices, n_TRs)
```

### What Changed and Why

| Change | v2 | v3 | Reason |
|--------|----|----|--------|
| Video temporal | None (per-frame) | Depthwise Conv1D after projector | MT+/V5 encodes motion, not content |
| Positional embed | Shared, modality-blind | Per-modality temporal embed | Text/audio/video have different temporal structure |
| Attention (early) | Full over all tokens | Local window + HRF decay bias | HRF means recent stimulus dominates |
| Attention (late) | Full | Full | Narrative/semantic context needs global |
| Stochastic depth | Uniform 10% | Linear schedule (l/L × 0.2) | Later layers regularized more |
| Modality pooling | Mean | Gated sigmoid | Visual cortex ≠ language cortex weighting |
| HRF modeling | None (AdaptiveAvgPool) | Learnable Conv1D, Gamma init | 6s hemodynamic delay is known physics |
| Subject layers | Linear per-subject matrix | Shared MLP + FiLM per-subject | Generalizes, 10× fewer subject params |
| Layer dropout bug | Skips aux_loss | Always accumulates aux_loss | Consistent load balancing |

### Parameter Budget (v3 vs v2)

| Component | v2 | v3 | Delta |
|-----------|----|----|-------|
| Backbones | 67.3M frozen | 67.3M frozen | 0 |
| Projectors | ~2.5M | ~2.5M | 0 |
| Temporal Conv (video) | — | ~1.5K | +1.5K |
| Modality-time embeddings | — | ~0.3M | +0.3M |
| MoE Transformer | ~37.8M | ~37.8M | 0 |
| Modality gates | — | ~1.5K | +1.5K |
| HRF Conv | — | ~0.1M | +0.1M |
| Output MLP (shared) | 512×256 linear | 512×512+512×n_vert | +0.5M |
| Subject FiLM | — | 2×512×n_subj | −30M (vs 33M SubjectLayers) |
| **Total trainable** | **~42M** | **~14M** | **−28M** |

v3 is actually **smaller** (28M fewer params) and more capable. The param
reduction comes entirely from replacing per-subject weight matrices (33M)
with per-subject FiLM vectors (~0.6M for 25 subjects).

---

## Question 3: Datasets + Training Strategies

### Datasets

#### Tier A — Brain-paired (use for Phase 3 fine-tuning only)

| Dataset | Subjects | Hours | What it has | Use for |
|---------|----------|-------|------------|---------|
| Algonauts 2025 (Courtois NeuroMod) | 4 | 264h | Friends S1-7, movies, paired fMRI | Phase 3 primary |
| Lebel 2023 | 8 | 60-140h | 82 spoken stories, paired fMRI | Phase 3 |
| BOLD Moments (Lahner 2024) | 10 | 62h | 1000 3s video clips, 10 reps | Phase 3 validation |
| Wen 2017 | 3 | 35h | Video segments, paired fMRI | Phase 3 |

#### Tier B — Teacher-cached (run TRIBE v2 once, cache forever)

Run the 4.7B teacher on 100h of **diverse, curated** video.
This is the only time the big model runs.

| Content | Hours | Why |
|---------|-------|-----|
| Nature documentaries | 20h | Rich visual + narration, diverse motion |
| TED talks | 20h | Sustained speech + gesture, semantic density |
| Movie clips (various genres) | 20h | Emotional range, narrative structure |
| Algonauts stimuli | 20h | Has paired fMRI → teacher + ground truth |
| Music videos + sports | 10h | Motion extremes, non-speech audio |
| Cooking / instructional | 10h | Fine motor + speech, HowTo-like |

Cache: final predictions (T, 20484) + fusion layers 4 and 6 (T, 1152).

#### Tier C — Self-supervised (Phase 1, zero teacher cost)

| Dataset | Size | Modalities | Cost |
|---------|------|------------|------|
| HowTo100M (subset) | 500h | Video + audio + ASR text | Feature extraction only |
| LibriSpeech | 960h | Audio + text (no video) | Feature extraction only |
| VGGSound | 200h | Video + audio | Feature extraction only |
| TED-LIUM | 100h | Audio + text | Feature extraction only |

All extracted with frozen tiny backbones (67M params, 10-50× realtime on T4).
Zero teacher inferences. Zero fMRI. Just raw feature learning.

---

### Training Strategies (All 4 Compared)

---

#### Strategy A: Direct KD Only (current Tiny-TRIBE v2 approach)

```
Phase 1: Freeze backbones, train fusion on cached teacher preds (5 epochs)
Phase 2: Unfreeze Whisper+MobileViT, E2E KD + fMRI GT (10 epochs)
```

**Loss:** MSE(pred, teacher) + cosine feature KD + temporal coherence + aux

**Data needed:**
- Teacher inference: ~500h of diverse video
- fMRI ground truth: ~270h across 25 subjects

**Pros:** Simple, proven, matches existing v2 pipeline
**Cons:** Data-hungry, teacher inference is expensive, fusion must learn everything from scratch
**Expected Pearson r:** 0.27-0.29

---

#### Strategy B: Progressive Layer Distillation

```
Stage 1: Distill 8-layer TRIBE v2 teacher → 6-layer intermediate
Stage 2: Distill 6-layer intermediate → 4-layer student
Stage 3: Backbone replacement (large → tiny), E2E fine-tune
```

**Loss:** Per-stage MSE + feature KD at matched layers

**Pros:** Each compression step is smaller, easier to debug quality drops
**Cons:** 3× the training time, need to maintain intermediate models
**Expected Pearson r:** 0.27-0.28 (similar to direct KD, more effort)

---

#### Strategy C: Self-Supervised Pre-training → KD Fine-tuning (v3 strategy)

```
Phase 0: Cache teacher on 100h diverse video (one-time, ~50 GPU-hours T4)
Phase 1: Self-supervised pre-training on 1000h unlabeled data
         Tasks: masked modality reconstruction + cross-modal contrastive
                + next-TR prediction + temporal order prediction
         No teacher. No fMRI. Just backbone features.
Phase 2: KD fine-tuning on cached teacher preds (frozen backbones, 10 epochs)
         Loss: MSE(pred, teacher) + feature KD + temporal + aux
Phase 3: E2E fine-tuning on fMRI ground truth (unfreeze Whisper+MobileViT)
         Loss: 0.4×fMRI + 0.3×teacher + 0.1×feature + 0.1×temporal + aux
```

**Data needed:**
- Teacher inference: only 100h (vs 500h in Strategy A) — 5× less
- Self-supervised: 1000h free (no teacher)
- fMRI ground truth: same 270h

**Pros:**
- 5× fewer teacher inferences (huge compute saving)
- Fusion model sees far more diverse data → better cross-modal representations
- Phase 2 becomes near-linear-probe (easier, needs less data)
- FiLM conditioning generalizes to new subjects without retraining
**Cons:** Phase 1 takes 2-3 days of T4 time (feature extraction + training)
**Expected Pearson r:** 0.29-0.31 (potentially matching TRIBE v2 teacher)

---

#### Strategy D: Multi-Teacher Distillation

```
Teacher 1: TRIBE v2 (4.7B) — brain predictions
Teacher 2: CLIP-ViT-L — visual-semantic alignment
Teacher 3: Whisper-large — audio temporal features

Student learns from all 3 simultaneously:
Loss = 0.5 × MSE(pred, tribe_teacher)
     + 0.2 × cosine(video_feat, clip_feat)
     + 0.2 × cosine(audio_feat, whisper_large_feat)
     + 0.1 × feature KD (tribe fusion)
     + aux
```

**Pros:** Richer supervision signal, backbones guided toward better representations
**Cons:** Complex, 3 teachers to manage, unclear if CLIP/Whisper alignment helps brain encoding
**Expected Pearson r:** 0.27-0.29 (marginal gain over A, much more complexity)

---

### The Best Strategy: C

**Why Strategy C wins:**

| Factor | A (Direct KD) | C (Self-sup + KD) |
|--------|--------------|-------------------|
| Teacher GPU-hours | ~250h | ~50h (5× less) |
| Training data diversity | ~100-500h stimuli | 1000h+ any video |
| Fusion quality | Learns fusion + brain mapping together | Learns fusion first (better) |
| Subject generalization | Limited (per-subject matrices) | Strong (FiLM generalizes) |
| Expected Pearson r | 0.27-0.29 | 0.29-0.31 |
| Inference cost | ~300ms/s | ~280ms/s |
| Model size | ~42M | ~14M trainable |
| New subject cost | Retrain SubjectLayers | Add 1024 FiLM vectors only |

The core argument is simple: **the fusion model doesn't need brain data to learn
that speech and lip movement co-occur, or that a fire creates both visual and
audio signals**. These are properties of the world, learnable from any video.
Only the mapping FROM those fused representations TO brain activations requires
neuroscience data. Strategy C separates these two learning problems and handles
each with the right data.

---

### Recommended Training Schedule (Strategy C)

```
Week 1:
  Day 1-2: Feature extraction — 1000h video through tiny backbones
            Run in parallel: 4 workers, each handles one dataset
            Cost: ~150 GPU-hours T4

  Day 3: Cache teacher predictions on 100h curated video
         Store: predictions + fusion layers 4,6
         Cost: ~50 GPU-hours T4

Week 2:
  Day 1-3: Phase 1 — self-supervised pre-training (25 epochs)
            LR: 3e-4, batch 32, OneCycleLR
            Monitor: MMR loss, contrastive retrieval accuracy
            Cost: ~15-20 GPU-hours T4

Week 3:
  Day 1: Phase 2 — KD fine-tuning (10 epochs, frozen backbones)
          LR: 1e-3, batch 16, OneCycleLR
          Loss: 0.6×output_KD + 0.2×feature_KD + 0.1×temporal + aux
          Cost: ~3-5 GPU-hours T4

  Day 2-3: Phase 3 — E2E fine-tuning on fMRI (10 epochs)
            Unfreeze: Whisper-Tiny (LR 1e-5), MobileViT-S (LR 1e-5)
            Fusion: LR 1e-4
            Loss: 0.4×fMRI + 0.3×teacher + 0.1×feature + 0.1×temporal + aux
            Modality dropout: decay 0.3 → 0.1 over training
            Cost: ~5-10 GPU-hours T4

Total: ~3 weeks, ~230 GPU-hours T4, ~$200-300 cloud cost
       vs current approach: ~500h teacher inference alone
```

---

### Key Monitoring Metrics During Training

| Phase | Metric | Target | Action if off |
|-------|--------|--------|--------------|
| Phase 1 | MMR reconstruction loss | Decreasing by epoch 5 | Check projector LR |
| Phase 1 | CMC R@5 accuracy | >80% by epoch 10 | Check temperature |
| Phase 1 | Expert entropy | >1.5 (of max 2.08) | Increase aux loss weight |
| Phase 2 | Val Pearson r | >0.20 by epoch 5 | Check teacher cache quality |
| Phase 2 | Feature cosine sim | >0.7 by epoch 5 | Check feat_proj |
| Phase 3 | Val Pearson r | >0.27 by epoch 5 | Check LR schedule |
| All | Expert utilization | 10-15% each | Adjust aux/z-loss |
| All | Gradient norm | <5.0 | Clip at 1.0 |
