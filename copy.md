# Approaches Explored & Techniques Tried
## Tiny-TRIBE: Compressing TRIBE v2 (4.7B → ~40M)

---

## The Core Problem

TRIBE v2 is a 4.7B parameter multimodal brain encoder:
- Predicts fMRI responses (20,484 cortical vertices) from video + audio + text
- ~10GB in memory, ~500ms per second of video on T4
- Not deployable on consumer hardware or browser

Goal: Build something smaller, faster, at least 90% as accurate.

---

## Version 1: Dense Tiny-TRIBE (`tiny_tribe/model.py`)

### What It Is

The first compression attempt. Straight shrink of TRIBE v2's fusion model with tiny backbone replacements.

### Architecture

```
Text:  LLaMA 3.2-3B (3072D)     →  all-MiniLM-L6-v2 (384D)
Audio: Wav2Vec-BERT 2.0 (1024D) →  Whisper-Tiny encoder (384D)
Video: V-JEPA2-ViT-G (1536D)   →  MobileViT-S (640D)

Per-modality:
  text_proj:  384 → 170D   (2-layer MLP)
  audio_proj: 384 → 170D   (2-layer MLP)
  video_proj: 640 → 172D   (2-layer MLP)
  concat:     512D total

combiner: Linear(512, 512) + GELU

pos_embed: learned (max 1024 positions)

Transformer: 4 layers × 512D × 4 heads × dense FFN (ff_mult=4)

Low-rank head: 512 → 256
SubjectLayers: 256 → n_vertices  (per-subject linear)
AdaptiveAvgPool → (B, n_vertices, n_TRs)
```

### What Was Tried / Found

| Decision | Tried | Finding |
|----------|-------|---------|
| Backbone size | MiniLM vs DistilBERT vs TinyBERT | MiniLM best size/quality tradeoff |
| Projector depth | 1-layer vs 2-layer | 2-layer meaningfully better |
| Fusion width | 256D vs 512D | 512D necessary — 256D too narrow for attention heads |
| Num heads | 4 vs 8 | 4 heads = 128D/head (too wide). 8 heads = 64D/head (optimal) |
| Token strategy | Concatenate | All modalities flattened then concat → slow cross-modal fusion |
| Temporal align | F.interpolate linear | Works but crude |

### Critical Flaws Found

1. **2-layer projectors are undersized** — BLIP-2/LLaVA literature shows projectors are the bottleneck. Need 3 layers.
2. **Concatenation is wrong** — `[text_all | audio_all | video_all]` means same-timestep cross-modal interaction needs to propagate through many attention layers. Text at t=5 is far from audio at t=5.
3. **4 heads on 512D = 128D per head** — well above the 64D-per-head sweet spot. Most attention capacity wasted.
4. **Dense FFN** — no capacity expansion trick. Parameters are wasted uniformly.
5. **combiner is redundant** — adds a Linear after projectors that already output 512D. Unnecessary params.
6. **AdaptiveAvgPool(1) throws away temporal structure** — pools the entire segment to 1 TR prediction.

### Expected Performance
- Mean Pearson r: ~0.22-0.24 (71-77% of TRIBE v2)
- Parameters: ~47M total
- Inference: ~350ms/s of video

---

## Version 2: MoE Tiny-TRIBE (`tiny_tribe/moe_model.py`)

### What Was Changed From v1

Based on deep analysis of distillation literature (DistilBERT, TinyBERT, DeiT,
Distil-Whisper, BLIP-2, LLaVA, Mixtral, Switch Transformer):

| Issue in v1 | Fix in v2 | Rationale |
|-------------|-----------|-----------|
| 2-layer projectors | **3-layer, 768 intermediate** | Projector quality is disproportionately important |
| 4 heads | **8 heads (64D each)** | Proven sweet spot in transformer literature |
| Concatenation | **Interleaved tokens [t1,a1,v1, t2,a2,v2...]** | Same-timestep cross-modal in first attention layer |
| Dense FFN | **MoE: 8 experts, top-2 routing** | 4× capacity at same compute cost |
| Modality identity lost | **Learned modality embeddings** | Added before interleaving |
| Random expert init | **Init from shared FFN + noise σ=0.01** | Faster specialization, avoids collapse |
| No router stability | **Z-loss + aux load-balancing** | Prevents expert collapse and logit explosion |
| Uniform layer dropout | **10% per layer** | Simple stochastic depth |

### Architecture

```
Backbones (67.3M, frozen):
  all-MiniLM-L6-v2  22.7M  → (B, T, 384)
  Whisper-Tiny       39.0M  → (B, T, 384)
  MobileViT-S         5.6M  → (B, T, 640)

Projectors (3-layer, 768 intermediate, ~2.5M trainable):
  text:  384 → 768 → 768 → 512
  audio: 384 → 768 → 768 → 512
  video: 640 → 768 → 768 → 512

Modality embeddings: 3 × 512D learnable vectors
Positional embeddings: learned, max 2048

Interleave: [t1,a1,v1, t2,a2,v2, ...] → (B, T×3, 512)

MoE Transformer × 4 layers:
  Pre-LayerNorm
  MultiheadAttention: 8 heads × 64D = 512D total
  Pre-LayerNorm
  MoEFFN: Router(512→8) → Top-2 experts → each 512→1024→512
  Residual connections throughout
  Layer dropout: 10%

Final LayerNorm
Modality pool: (B, T×3, 512) → mean → (B, T, 512)

Low-rank head: Linear(512→256, no bias)
SubjectLayers: 256 → n_vertices (per-subject linear)
Transpose + AdaptiveAvgPool1d
→ (B, n_vertices, n_TRs)
```

### Distillation Loss (2 phases)

```
Phase 1 (frozen backbones, 5 epochs):
  L = 0.8 × MSE(student_pred, teacher_pred)
    + 0.1 × cosine_loss(student_fused, teacher_fused_proj)
    + 0.1 × smooth_l1(Δstudent, Δteacher)
    + 0.01 × aux_loss

Phase 2 (unfreeze Whisper+MobileViT, 10 epochs):
  L = 0.5 × MSE(student_pred, teacher_pred)
    + 0.3 × MSE(student_pred, fmri_target)
    + 0.1 × cosine_loss(student_fused, teacher_fused_proj)
    + 0.05 × smooth_l1(Δstudent, Δteacher)
    + 0.01 × aux_loss
```

### What Works Well in v2

- MoE gives meaningful capacity expansion without memory blowup
- Interleaved tokens dramatically improve early fusion
- 3-layer projectors with wide intermediate (768) noticeably better
- Expert init from shared FFN + noise prevents early collapse
- Z-loss stabilizes training, prevents router logit explosion
- Modality dropout at 30% (not 50%) is the right robustness/quality balance

### Remaining Flaws in v2

**Architectural:**

1. **Mean modality pooling is dumb** — after the transformer, all 3 modality tokens per timestep are averaged. This throws away the learned per-modality specialization. A gated/attention pool would let the model learn "visual cortex cares about video tokens, Broca's area cares about text tokens."

2. **No HRF (hemodynamic response function) modeling** — the brain's fMRI signal lags stimulus by ~6s with a known shape (Gamma function). The model has zero inductive bias for this. It must learn the delay implicitly, which wastes capacity and needs more data.

3. **MobileViT-S is per-frame** — no temporal information between frames. The visual system tracks motion, not just frame content. A simple Conv1D after video projection would capture frame-to-frame dynamics.

4. **Flat low-rank bottleneck** — 512→256→n_vertices. The SubjectLayers are a single linear map. With 256D, predicting 20,484 vertices independently per-subject is very constrained. Different brain regions have fundamentally different computational demands.

5. **Positional embeddings are modality-blind** — position 5 in the interleaved sequence gets the same embedding whether it's a text, audio, or video token. Separate per-modality positional embeddings would give better temporal grounding.

6. **Uniform attention** — full self-attention over T×3 tokens treats all timesteps equally. In brain encoding, the HRF means recent stimulus (~0-6s ago) matters most. A causal attention bias or local window for early layers would be more neurobiologically motivated.

7. **SubjectLayers are parameter-inefficient at scale** — with 25 subjects × 256 × 5124 = 33M params just for subject layers. These are entirely linear. A shared MLP backbone + subject-specific scale/shift (FiLM conditioning) would use ~1M params and generalize better to new subjects.

8. **Layer dropout skip also skips aux_loss** — when a layer is dropped (line 282 in moe_model.py), the aux_loss contribution from that layer is also missing, making load-balancing inconsistent across training steps.

**Training:**

9. **No self-supervised pre-training** — the model only trains on teacher KD data. The fusion transformer must simultaneously learn cross-modal dynamics AND brain mapping from limited labeled data. These are separable skills.

10. **Teacher data is the bottleneck** — running TRIBE v2 on training videos costs ~500ms/s on T4. This limits how much data the model sees. If we pre-trained the fusion on unlabeled data, we'd need far less teacher data.

11. **No temporal augmentation** — no time warping, no speed variation. Brain responses to 1.1× speed-up are highly predictable but never shown in training.

### Expected Performance
- Mean Pearson r: ~0.27-0.29 (87-94% of TRIBE v2)
- Parameters: ~42M total (~16M active per forward pass)
- Inference: ~300ms/s of video

---

## Version 3 Strategy: Data-Optimal Pipeline (`TRIBE_V3_STRATEGY.md`)

Explored but not yet implemented. Key idea: decouple multimodal fusion learning
from brain mapping. Self-supervised pre-training on unlabeled data, then distill.

See `TRIBE_V3_STRATEGY.md` and `SELF_SUPERVISED_PRETRAINING.md` for full design.

---

## What Else Was Explored (Research Survey)

### Backbone Alternatives Considered

| Modality | Tried / Considered | Winner | Why |
|----------|-------------------|--------|-----|
| Text | LLaMA-3.2-3B, DistilBERT, TinyBERT-4L, Qwen-0.5B, all-MiniLM | MiniLM (PoC), Qwen-0.5B (quality) | MiniLM: 85MB, fast. Qwen: better semantics |
| Audio | Wav2Vec2-Base, HuBERT-Base, AST-Tiny, Whisper-Tiny | Whisper-Tiny | Best speech quality/size, ONNX-ready |
| Video | EfficientNet-B0, TinyViT-11M, CLIP-ViT-B/16, 3D-MobileNet | MobileViT-S | Smallest, browser-deployable, good accuracy |

### Fusion Architectures Considered

| Approach | Tried | Finding |
|----------|-------|---------|
| MLP-Mixer | Research review | No quadratic attention, simpler, but weaker long-range |
| Linear Attention | Research review | O(T×H) vs O(T²×H), but still experimental |
| PoolFormer | Research review | 1-2M params, too much quality loss |
| Cross-attention Q-Former | Research review | Better than concat/interleave but more params |
| Standard full-attention transformer | v1 | Works, baseline |
| MoE transformer | v2 | Best quality/compute tradeoff |

### Output Space Compression Considered

| Approach | Resolution | Accuracy | Status |
|----------|-----------|---------|--------|
| fsaverage5 (full teacher) | 20,484 vertices | 100% | Teacher only |
| fsaverage4 | 5,124 vertices | 90-95% | Used in v2 |
| fsaverage3 | 1,284 vertices | 80-85% | Considered for ultra-tiny |
| Schaefer-400 parcels | 400 | 85% | Used for Algonauts target |
| Schaefer-1000 parcels | 1,000 | 88% | Competition target |
| PCA top-100 eigenvectors | ~100 modes | 85-90% | Proposed, not tried |

### Distillation Techniques Surveyed

From literature analysis (`DISTILLATION_PATTERNS.md`):

| Technique | Domain | Applicable to TRIBE? | Used in v2? |
|-----------|--------|---------------------|-------------|
| Soft label KD (temperature τ) | NLP classification | No — regression task, no softmax | No |
| Feature MSE matching | All | Yes | No (cosine used instead) |
| Cosine feature matching | All | Yes — more stable than MSE | Yes |
| Attention matrix matching | Transformers | No — different head counts | No |
| Value-relation transfer (MiniLM) | NLP | Possibly useful | No |
| Progressive distillation (8→6→4 layers) | All | Yes, not tried | No |
| Pseudo-labelling (Distil-Whisper) | Speech | Equivalent to caching teacher preds | Yes (Phase 0) |
| Affinity mimicking (TinyCLIP) | Vision-language | Potentially for brain similarity structure | No |
| Multi-resolution output matching | Dense prediction | Yes — parcel-level + vertex-level | Proposed, not in v2 |
| Temporal coherence loss (Δ matching) | Sequential | Yes | Yes |
| Uncertainty-weighted KD | Regression | Proposed, not tried | No |
| HRF-aware loss | Brain encoding | Strong prior, not tried | No |

---

## Summary: What's Working vs What's Not

### Working
- MoE with 8 experts, top-2: good capacity/compute tradeoff
- Interleaved tokens: much better than concatenation for early cross-modal fusion
- 3-layer projectors with wide intermediate (768D): projector quality is the real bottleneck
- Expert init from shared FFN + noise: prevents collapse, faster specialization
- Z-loss on router + aux load-balancing: training stability
- Phase 1 frozen → Phase 2 E2E: stable, works as expected
- Cosine feature KD: better signal than MSE for intermediate features
- Temporal coherence loss (Δ): helps produce smooth predictions

### Not Working / Untested
- Per-frame MobileViT: no motion information, likely the biggest single quality gap
- Mean modality pooling: probably losing region-specific modality preferences
- HRF modeling: zero explicit inductive bias for the fundamental brain physics
- Flat subject layers: parameter-inefficient, won't generalize to new subjects
- No self-supervised pre-training: constrained by labeled data volume
- Temporal interpolation: crude alignment for inherently asynchronous modalities

### The Biggest Unexploited Opportunity
Self-supervised pre-training on unlabeled multimodal data (HowTo100M, LibriSpeech, VGGSound).
This would give the fusion transformer diverse training signal for cross-modal dynamics
at zero teacher inference cost, then the brain mapping becomes a simple fine-tuning task.
