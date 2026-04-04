# Tribe v3: Data-Optimal Distillation Strategy

## Problem Statement

TRIBE v2 (4.7B params) is expensive to run inference on (~500ms/s of video on T4).
Current Tiny-TRIBE distillation requires running the teacher on every training sample.
We want to:
- **Minimize teacher inferences** (each one costs GPU time on the 4.7B model)
- **Maximize utility** of every inference we do run
- **Use extra unlabeled data** that requires zero teacher involvement
- Build a model that is **smaller, faster, and potentially better** than Tiny-TRIBE v2

---

## Core Insight

The current pipeline conflates two skills:
1. **Multimodal temporal fusion** — understanding how text, audio, and video relate across time
2. **Brain mapping** — predicting fMRI vertex activations from fused representations

Skill (1) can be learned from **any** multimodal data. No teacher, no fMRI needed.
Skill (2) requires teacher predictions or real fMRI data, but is a simpler mapping problem
once the fusion model already understands cross-modal dynamics.

**Decouple them. Pre-train (1) on massive free data. Fine-tune (2) on a small teacher cache.**

---

## 3-Phase Training Pipeline

### Phase 0: One-Time Teacher Cache (Run Once)

Run TRIBE v2 on a curated set of videos. Cache aggressively — this is the only time
the 4.7B model runs.

**What to cache per video:**
| Output | Shape | Purpose |
|--------|-------|---------|
| Final vertex predictions | (T, 20484) | Output KD target |
| Fusion layer 4 activations | (T, 1152) | Feature KD target |
| Fusion layer 6 activations | (T, 1152) | Feature KD target |
| Per-modality projected features | (T, 3, 384) | Projector alignment target |

**How much to cache:**
- Minimum: 50 hours of diverse video (covers core patterns)
- Sweet spot: 100 hours (diminishing returns after this for fine-tuning)
- Compare: current approach needs 500+ hours

**Selection strategy for what to run teacher on:**
- Prioritize **diversity** over volume: nature, dialogue, music, sports, lectures, cooking
- Include edge cases: silence-only, text-heavy, fast-motion
- Include Algonauts/BOLD Moments stimuli (have paired fMRI for validation)
- Skip redundant content (e.g., don't cache 50 hours of similar talking heads)

**Cost estimate:**
- 100 hours of video = ~50 GPU-hours on T4 (~$50-75 on cloud)
- One-time cost, amortized over all future training runs

---

### Phase 1: Self-Supervised Multimodal Pre-Training (No Teacher)

Train the tiny fusion model on **massive unlabeled multimodal data**.
Zero teacher inferences. Zero fMRI data. Just raw videos with audio and transcripts.

#### Architecture (same as Tiny-TRIBE MoE, with pre-training heads)

```
Frozen tiny backbones (MiniLM + Whisper-Tiny + MobileViT)
    → Projectors (384/384/640 → 512)
    → Interleaved tokens
    → 4-layer MoE Transformer (8 experts, top-2)
    → Pre-training heads (removed after Phase 1)
```

#### Pre-Training Tasks

**Task 1: Masked Modality Reconstruction (MMR)**

Mask one entire modality at a timestep, predict its projected features from the other two.

```python
# Randomly select modality to mask per sample
masked_modality = random.choice(['text', 'audio', 'video'])

# Zero out that modality's tokens in the interleaved sequence
interleaved_tokens[masked_modality] = 0

# Forward through fusion transformer
fused = transformer(interleaved_tokens)

# Predict the masked modality's projected features
predicted = mmr_head(fused)  # Linear(512, 512)
target = original_projected_features[masked_modality]

loss_mmr = MSE(predicted, target.detach())
```

**Why it works:** Forces the fusion model to learn cross-modal correlations.
If someone is talking about a dog, the model must predict dog-like video features
from just the audio+text. This is exactly the cross-modal understanding needed
for brain encoding (associative cortices respond to meaning regardless of modality).

**Task 2: Temporal Order Prediction (TOP)**

Shuffle temporal segments, predict the correct ordering.

```python
# Split sequence into 4-8 segments
segments = split_temporal(fused_repr, n_segments=6)

# Shuffle
perm = random.permutation(n_segments)
shuffled = segments[perm]

# Predict correct order
predicted_order = top_head(shuffled)  # Transformer + classification head
loss_top = CrossEntropy(predicted_order, inverse_perm)
```

**Why it works:** Brain responses are deeply temporal — the hemodynamic response
has a ~6s delay, and narrative context builds over minutes. A fusion model that
understands temporal ordering will produce better temporally-coherent predictions.

**Task 3: Cross-Modal Contrastive Learning (CMC)**

Aligned (text, audio, video) at the same timestamp are positive pairs.
Different timestamps or different videos are negatives.

```python
# Get fused representation at each timestep
fused_t = mean_pool_modalities(transformer_output)  # (B, T, 512)

# Positive pairs: same timestep, different modality subsets
anchor = fused_t[:, t, :]      # all modalities
positive = fused_t_masked[:, t, :]  # one modality masked

# Negatives: different timesteps or different samples
negatives = fused_t[:, other_t, :]

loss_cmc = InfoNCE(anchor, positive, negatives, temperature=0.07)
```

**Why it works:** Builds a temporally-grounded shared representation space.
The brain represents the same event similarly regardless of input modality —
this loss directly teaches that property.

**Task 4: Next-TR Feature Prediction (NTP)**

Predict the fused representation at t+1 from representations up to t.

```python
# Causal masking: only attend to past
fused_causal = transformer(interleaved_tokens, causal_mask=True)

# Predict next timestep's representation
predicted_next = ntp_head(fused_causal[:, t, :])  # MLP(512, 512)
target_next = fused_repr[:, t+1, :].detach()  # from non-causal forward

loss_ntp = MSE(predicted_next, target_next) + cosine_loss(predicted_next, target_next)
```

**Why it works:** fMRI prediction requires understanding what comes next
(hemodynamic delay means current brain state reflects past+current+anticipated stimuli).

#### Combined Pre-Training Loss

```python
L_pretrain = 0.4 * loss_mmr      # Primary: cross-modal reconstruction
           + 0.2 * loss_cmc      # Contrastive alignment
           + 0.2 * loss_ntp      # Temporal prediction
           + 0.1 * loss_top      # Temporal ordering
           + 0.01 * aux_loss     # MoE load balancing
```

#### Data Sources (All Free, No Teacher)

| Dataset | Type | Size | Content |
|---------|------|------|---------|
| HowTo100M | Video + ASR text | 136M clips | Instructional videos, diverse |
| YouTube-8M (subset) | Video + audio | 8M videos | Broad domain coverage |
| LibriSpeech | Audio + text | 960h | Clean read speech |
| Common Voice | Audio + text | 19,000h+ | Multilingual speech |
| VGGSound | Video + audio | 200K clips | Sound event recognition |
| Your posted/ videos | Video + audio + text | ~hours | Domain-specific content |

**Practical subset:** Use 500-1000 hours total. Extract features offline with frozen
tiny backbones (cheap — MiniLM + Whisper-Tiny + MobileViT total only 67M params,
runs at 10-50x realtime on T4).

#### Training Config

| Parameter | Value |
|-----------|-------|
| Duration | 20-30 epochs over full dataset |
| LR | 3e-4 (higher than KD phase — self-sup is smoother) |
| Scheduler | Cosine with linear warmup (5% steps) |
| Batch size | 32-64 (large batches help contrastive + MoE routing) |
| Backbones | Frozen throughout |
| Segment length | 30s (shorter = more diversity per batch) |
| GPU time | ~10-20 hours on T4 |
| Teacher inferences | **0** |

#### What This Phase Achieves

The fusion transformer learns:
- How text, audio, and video features relate to each other
- Temporal dynamics of multimodal streams
- Robustness to missing modalities
- A good shared representation space

It does NOT yet know anything about brains. That comes next.

---

### Phase 2: Teacher Distillation (Cached Data Only)

Now use the cached teacher predictions from Phase 0.
The fusion model already understands multimodal dynamics — it just needs to learn
the brain mapping.

#### What Changes From Phase 1

```
Remove: pre-training heads (MMR, TOP, CMC, NTP)
Add: Low-rank bottleneck (512 → 256)
Add: SubjectLayers (256 → 20,484 vertices)
Add: Temporal pooling
```

#### Loss Function

```python
L_distill = 0.6 * MSE(student_pred, teacher_pred)           # Output KD
          + 0.2 * cosine_loss(student_fused, teacher_fused)  # Feature KD
          + 0.1 * smooth_l1(delta_student, delta_teacher)    # Temporal coherence
          + 0.05 * multi_res_loss(student_pred, teacher_pred) # Parcel-level matching
          + 0.01 * aux_loss                                   # MoE load balancing
```

#### Training Config

| Parameter | Value |
|-----------|-------|
| Duration | 10 epochs |
| LR | 1e-3 (fusion), backbone still frozen |
| Scheduler | OneCycleLR (10% warmup) |
| Batch size | 8-16 |
| Data | Cached teacher predictions (50-100h of video) |
| Segment length | 100 TRs |
| GPU time | ~3-5 hours on T4 |
| Teacher inferences | **0** (all cached) |

#### Why This Works With Less Teacher Data

Without pre-training, the fusion transformer must simultaneously learn:
- Cross-modal feature interaction (hard, needs lots of data)
- Temporal dynamics (hard, needs diverse sequences)
- Brain-specific vertex mapping (straightforward linear mapping)

With pre-training, only the third task remains. This is essentially a
**linear probe on a pre-trained representation** — known to need far less data.

---

### Phase 3: End-to-End Fine-Tuning with fMRI Ground Truth

If you have access to real fMRI data (Algonauts, Lebel, BOLD Moments, Wen),
fine-tune the full model end-to-end.

#### Loss Function

```python
L_finetune = 0.4 * MSE(student_pred, fmri_target)           # Ground truth
           + 0.3 * MSE(student_pred, teacher_pred)           # Teacher regularizer
           + 0.1 * cosine_loss(student_fused, teacher_fused) # Feature KD
           + 0.1 * smooth_l1(delta_student, delta_teacher)   # Temporal coherence
           + 0.01 * aux_loss
```

#### Training Config

| Parameter | Value |
|-----------|-------|
| Duration | 5-10 epochs |
| LR | 1e-4 (fusion), 1e-5 (backbones — now unfrozen) |
| Unfreeze | Whisper-Tiny + MobileViT (text stays frozen) |
| Data | fMRI-paired datasets (~270h across 25 subjects) |
| Modality dropout | 0.3 → 0.1 (reduce over training) |
| GPU time | ~5-10 hours on T4 |
| Teacher inferences | **0** (cached from Phase 0) |

---

## Architecture Comparison: v2 vs v3

| Component | Tiny-TRIBE v2 (MoE) | Tiny-TRIBE v3 |
|-----------|---------------------|---------------|
| Text backbone | all-MiniLM-L6-v2 (22.7M) | Same |
| Audio backbone | Whisper-Tiny (39M) | Same |
| Video backbone | MobileViT-S (5.6M) | Same |
| Projectors | 2-layer MLP | **3-layer MLP** (projector quality matters) |
| Fusion | 4-layer MoE, 8 experts, top-2 | Same (proven design) |
| Pre-training | None | **Self-supervised multimodal** |
| Hidden dim | 512 | 512 |
| Total params | ~42M | ~43M (slightly larger projectors) |
| Active params | ~16M | ~17M |

The architecture is nearly identical. The difference is **how it's trained**.

---

## Expected Performance

| Metric | TRIBE v2 (teacher) | Tiny-TRIBE v2 | **Tiny-TRIBE v3** |
|--------|-------------------|---------------|-------------------|
| Mean Pearson r | 0.31 | 0.27-0.29 | **0.29-0.31** |
| Parameters | 4.7B | 42M | 43M |
| Active params | 4.7B | 16M | 17M |
| Model size (INT8) | 10GB | 120MB | 120MB |
| Inference (GPU) | ~500ms | ~300ms | ~300ms |
| Inference (CPU) | 10-15s | 3-5s | 3-5s |
| Browser deployable | No | Yes | Yes |
| Teacher GPU-hours needed | N/A | ~250h | **~50h (5x less)** |
| Unlabeled data used | 0 | 0 | **500-1000h** |

### Why v3 Could Match or Exceed TRIBE v2

1. **More diverse training data** — TRIBE v2 only saw fMRI-paired stimuli (TV shows, movies, stories). v3's fusion model sees cooking, nature, sports, lectures, music — far more diverse visual/audio/language patterns.

2. **Better cross-modal representations** — self-supervised pre-training explicitly optimizes for cross-modal understanding. TRIBE v2's fusion only implicitly learns this through the fMRI prediction objective.

3. **Brain alignment saturates early** — the literature shows 3B models match 7-14B in neural predictivity. The bottleneck isn't backbone capacity, it's fusion quality. A better-trained fusion model with tiny backbones could match a poorly-trained fusion with giant backbones.

4. **MoE capacity advantage** — 8 experts give 4x the capacity of a dense model at the same compute. With good pre-training, these experts can specialize meaningfully (e.g., speech-brain expert, visual-motion expert, narrative-integration expert).

---

## Implementation Priority

1. **Phase 0** — Cache teacher predictions on 100h of diverse video (~$50-75 cloud cost)
2. **Phase 1** — Self-supervised pre-training on 500h+ unlabeled data (~10-20h T4)
3. **Phase 2** — Teacher distillation on cached data (~3-5h T4)
4. **Phase 3** — fMRI fine-tuning (~5-10h T4)
5. **Export** — ONNX INT8 for browser deployment

Total GPU cost: ~$100-150 (vs ~$500+ for current approach that runs teacher on everything).

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Self-supervised pre-training doesn't help | Low (proven in vision/NLP) | Skip to Phase 2, fall back to v2 approach |
| Tiny backbones too weak for self-sup tasks | Medium | Use larger backbone subset for pre-training, distill down |
| Phase 2 needs more teacher data than expected | Medium | Cache more videos incrementally, prioritize diverse content |
| MoE experts don't specialize during pre-training | Low | Monitor routing entropy, adjust aux loss weight |
| Pre-training representations misalign with brain space | Low | Phase 2 KD explicitly aligns them |

---

## Key References

- MAE (He et al., 2022) — masked autoencoder pre-training for vision
- data2vec (Baevski et al., 2022) — self-supervised learning across modalities
- VideoMAE (Tong et al., 2022) — masked video pre-training
- BERT (Devlin et al., 2019) — masked language model pre-training
- TRIBE v2 (d'Ascoli et al., 2026) — the teacher model
- Brain alignment saturation studies (2024-2025) — smaller models match larger ones
