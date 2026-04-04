# Distillation Patterns Across Foundation Models
## Analysis & Recommendations for Tiny-TRIBE

---

## Table of Contents

1. [Taxonomy of Knowledge Distillation](#1-taxonomy)
2. [Language Model Distillation](#2-language-models)
3. [Vision Model Distillation](#3-vision-models)
4. [Speech/Audio Model Distillation](#4-speech-models)
5. [Multimodal Model Distillation](#5-multimodal-models)
6. [MoE-Specific Distillation](#6-moe-distillation)
7. [Regression & Dense Prediction Distillation](#7-regression-distillation)
8. [Cross-Cutting Patterns](#8-cross-cutting-patterns)
9. [Anti-Patterns & Failures](#9-anti-patterns)
10. [Recommendations for Tiny-TRIBE](#10-recommendations)

---

## 1. Taxonomy of Knowledge Distillation {#1-taxonomy}

### 1.1 By What Is Transferred

| Method | What transfers | Where it's used | Typical weight |
|--------|---------------|-----------------|----------------|
| **Output KD** | Final predictions (logits/values) | Universal | 0.5-0.8 |
| **Feature KD** | Intermediate layer activations | Deep models | 0.1-0.3 |
| **Attention KD** | Attention weight matrices | Transformers | 0.05-0.1 |
| **Relation KD** | Pairwise sample similarities | Embedding models | 0.1-0.2 |
| **Contrastive KD** | Positive/negative feature pairs | Vision, multimodal | 0.1-0.3 |
| **Task KD** | Ground truth labels alongside teacher | When GT available | 0.2-0.5 |

### 1.2 By How It Is Transferred

**Offline distillation**: Teacher runs once, outputs cached, student trains against cache.
- Pros: No teacher in GPU during student training, can use any teacher size
- Cons: Can't do online augmentation of teacher views
- Used by: DistilBERT, Distil-Whisper, most practical systems

**Online distillation**: Teacher and student run simultaneously.
- Pros: Teacher can see same augmented batch, richer signal
- Cons: Both models must fit in memory
- Used by: DeiT (distillation token), some small-to-small setups

**Self-distillation**: Model distills from its own deeper layers or earlier checkpoints.
- Pros: No external teacher needed
- Cons: Limited by own capacity
- Used by: Be Your Own Teacher (BYOT), progressive shrinking

### 1.3 By When It Happens

**Pre-training distillation**: Distill during pre-training on large corpus.
- Example: DistilBERT pre-trained with KD on BookCorpus+Wikipedia
- Most expensive but best quality

**Fine-tuning distillation**: Distill a task-specific teacher on task data.
- Example: TinyBERT Stage 2 (task-specific KD)
- Cheaper, good for specific domains

**Post-training distillation**: Teacher already trained, student trains from scratch.
- Example: Most practical deployments
- What Tiny-TRIBE will use

---

## 2. Language Model Distillation {#2-language-models}

### 2.1 DistilBERT (Sanh et al., 2019)

**Teacher**: BERT-base (110M, 12 layers, 768D)
**Student**: DistilBERT (66M, 6 layers, 768D)
**Compression**: 40% fewer params, 60% faster

**Key decisions:**
- Kept width (768D), halved depth (12→6 layers)
- Why: width preserves representational capacity per layer; depth is cheaper to cut
- Initialized student from every other teacher layer (layers 0,2,4,6,8,10)
- Triple loss: MLM + KD (soft targets with τ=8) + cosine embedding loss

**What worked:**
- Soft target KD with high temperature (τ=8) was critical — forces student to learn teacher's uncertainty distribution, not just argmax
- Cosine embedding loss on [CLS] token aligned final representations
- Retained 97% of BERT performance on GLUE

**What didn't matter much:**
- Attention transfer (tried, marginal gain)
- MSE on hidden states (cosine was better)

**Pattern**: *Width > Depth for compression. Initialize from teacher layers.*

### 2.2 TinyBERT (Jiao et al., 2020)

**Teacher**: BERT-base (110M)
**Student**: TinyBERT-4L (14.5M, 4 layers, 312D) or TinyBERT-6L (66M)
**Compression**: 7.5× smaller (4L variant)

**Key decisions:**
- Two-stage distillation: (1) general KD on pre-training data, (2) task-specific KD
- Reduced BOTH width and depth
- Feature matching with learned linear projections (student dim → teacher dim)
- Attention matrix matching: MSE on attention weights

**Loss function (most comprehensive in literature):**
```
L = L_embedding  (input embedding alignment)
  + L_attention  (attention weight matching, per layer)
  + L_hidden     (hidden state matching, per layer, with projection)
  + L_prediction (soft label KD on final output)
```

**What worked:**
- Two-stage approach: general pre-training KD + task-specific KD
- Data augmentation during task-specific KD (word replacement with GloVe neighbors)
- Layer mapping: student layer i maps to teacher layer i×(N_teacher/N_student)
- Projector matrices for dimension mismatch (critical when width differs)

**What didn't work well:**
- Single-stage distillation (general only, or task only) — need both
- Without augmentation, task-specific stage overfits quickly

**Pattern**: *Two-stage KD (general then task). Augment during distillation. Map layers proportionally.*

### 2.3 MiniLM (Wang et al., 2020)

**Teacher**: BERT-base or BERT-large
**Student**: Various sizes (6L/384D to 12L/384D)

**Key innovation: Self-attention relation transfer**
Instead of matching hidden states or attention weights, matches the
**self-attention value-relation** — the scaled dot-product of values:
```
V_relation = (V @ V^T) / sqrt(d)
```

**Why this is better than hidden state matching:**
- Hidden states are high-dimensional and brittle to match
- Value relations capture structural relationships between tokens
- Dimension-agnostic (no projector needed)

**What worked:**
- Only distilling from the LAST layer's attention relations
- Much simpler than TinyBERT's per-layer matching
- Teacher assistant: distill large→medium→small progressively

**Pattern**: *Relational knowledge > absolute activation matching. Last layer often sufficient.*

### 2.4 LLaMA/Mistral Distillation (2023-2025)

**The modern LLM distillation landscape:**

Most LLM compression now uses:
1. **Pruning + KD**: Remove attention heads/layers, then distill to recover
2. **Quantization-Aware KD**: Train quantized student against fp16 teacher
3. **Speculative decoding distillation**: Small model drafts, large model verifies

**Key finding from scaling law studies:**
- Below 3B params, KD provides diminishing returns for language modeling
- But for DOWNSTREAM TASKS, KD still helps significantly at any size
- The "brain alignment saturates early" finding from neuroscience aligns: for task-specific predictions (not general language modeling), smaller models can match larger ones

**Pattern**: *For task-specific distillation (like brain encoding), KD works even at very small scales.*

---

## 3. Vision Model Distillation {#3-vision-models}

### 3.1 DeiT (Touvron et al., 2021)

**Teacher**: RegNet-16GF (CNN, 84M)
**Student**: DeiT-S (ViT-Small, 22M)

**Key innovation: Distillation token**
Added a special [DIST] token to the input (like [CLS]).
- [CLS] trained with ground truth labels
- [DIST] trained with teacher's soft labels
- At inference, average both predictions

**Why a CNN teacher for a ViT student?**
- CNNs have strong inductive biases (locality, translation invariance)
- ViTs learn these from data, slowly
- Distilling CNN→ViT transfers the inductive bias cheaply

**What worked:**
- Hard-label distillation (argmax of teacher) > soft-label KD for vision
  (opposite of NLP! because vision labels are less ambiguous)
- CNN teacher > ViT teacher (different architecture provides complementary views)
- Distillation token mechanism (separate token for teacher signal)

**What didn't work:**
- Soft KD with temperature (unlike NLP, marginal benefit in vision)
- Same-architecture teacher (ViT teaching ViT — less gain than CNN→ViT)

**Pattern**: *Cross-architecture distillation can outperform same-architecture. Hard labels work for vision.*

### 3.2 TinyCLIP (Wu et al., 2023)

**Teacher**: CLIP ViT-B/16 (150M)
**Student**: TinyCLIP (various, down to 8M)

**Key decisions:**
- Affinity mimicking: match the image-text similarity matrix, not individual embeddings
- Weight inheritance: initialize student from pruned teacher weights
- Pre-training KD on large image-text dataset (LAION)

**What worked:**
- Affinity mimicking (preserving the STRUCTURE of the embedding space)
- Progressive compression: CLIP-B → CLIP-S → TinyCLIP
- Weight inheritance > random initialization (5-10% quality gain)

**Pattern**: *For embedding/retrieval models, preserve relational structure, not absolute values.*

### 3.3 EfficientViT / MobileViT Distillation

**Relevant because we use MobileViT-S as our video backbone.**

**Findings from mobile vision model training:**
- Feature distillation on spatial attention maps works well
- Stage-wise distillation (match each stage of the CNN/ViT separately)
- Augmentation is critical: AutoAugment + RandErasing + MixUp during KD
- The gap between teacher and student WIDENS at higher resolutions
  (implication: for video, use lower frame resolution during distillation)

**Pattern**: *For video/vision students, augmentation during KD is essential. Lower resolution helps.*

---

## 4. Speech/Audio Model Distillation {#4-speech-models}

### 4.1 Distil-Whisper (Gandhi et al., 2023)

**Teacher**: Whisper-large-v2 (1.55B, 32 decoder layers)
**Student**: Distil-Whisper (756M, 2 decoder layers)
**Compression**: 6× faster, 49% fewer params

**Directly relevant because our audio backbone IS Whisper-Tiny.**

**Key decisions:**
- Only compressed the decoder (32→2 layers), kept encoder intact
- Why: encoder features are reused by downstream tasks; decoder is task-specific
- Pseudo-labelling: ran teacher on 22K hours of audio, used transcriptions as targets
- KD loss on token-level outputs, not hidden states

**What worked phenomenally:**
- Pseudo-labelling > direct KD (by a significant margin)
- Keeping the encoder untouched (backbone features matter most)
- Massive data: 22K hours of pseudo-labelled audio
- Word-level timestamps preserved (critical for temporal alignment)

**What didn't work:**
- Feature matching on decoder hidden states (noisy, didn't help)
- Small pseudo-label datasets (<1K hours insufficient)

**Pattern**: *For sequence models, pseudo-labelling (teacher generates target text) beats feature KD. Encoders are more important than decoders.*

### 4.2 HuBERT/Wav2Vec Distillation (FitHuBERT, DistilHuBERT)

**Teacher**: HuBERT-base (95M, 12 layers)
**Student**: DistilHuBERT (23.5M, 2 layers)

**Key findings:**
- Time-domain feature matching is critical for speech (preserve temporal structure)
- Student needs to match teacher at MULTIPLE time scales
- Prediction heads help: student predicts teacher's cluster assignments
- Layer skip with regression: student layer 1 predicts teacher layers 1-6,
  student layer 2 predicts teacher layers 7-12

**Pattern**: *For temporal/sequential models, multi-scale temporal matching matters. Layer skipping with regression projectors.*

### 4.3 Implications for Tiny-TRIBE Audio Pipeline

Our audio path: raw audio → Whisper-Tiny encoder → features → fusion.

Key takeaways:
1. Whisper-Tiny is already distilled (it's the smallest Whisper). Don't over-compress.
2. The encoder features are the most valuable part — keep them as-is.
3. When fine-tuning in Stage 2, use a very low LR for the encoder (1e-5 vs 1e-4 for fusion).
4. Temporal alignment of audio features to other modalities is critical.

---

## 5. Multimodal Model Distillation {#5-multimodal-models}

### 5.1 BLIP-2 & InstructBLIP Compression

**Architecture**: Frozen image encoder + Q-Former + Frozen LLM

**Distillation patterns:**
- Q-Former (the fusion module) is the primary distillation target
- Image encoder and LLM remain frozen during distillation
- This is EXACTLY analogous to Tiny-TRIBE:
  frozen backbones + trainable fusion → distill the fusion

**What worked:**
- Two-stage: (1) vision-language alignment, (2) instruction tuning
- Projector quality matters more than fusion depth
- Feature alignment between Q-Former and LLM input space

### 5.2 LLaVA Compression

**Teacher**: LLaVA-13B (ViT-L + Vicuna-13B)
**Student**: LLaVA-1.5-7B, TinyLLaVA

**Key findings:**
- The visual projector (2-layer MLP) is disproportionately important
- Scaling the projector while shrinking the LLM > scaling the LLM with small projector
- Multi-stage training: (1) align projector, (2) instruction tune

**Implication**: in Tiny-TRIBE, the modality projectors deserve more capacity than
you might think. Our current 2-layer MLPs may be undersized.

### 5.3 CogVLM & Multimodal MoE

**Recent trend (2024-2025): Multimodal MoE**

CogVLM, Uni-MoE, and MoE-LLaVA use modality-specific expert routing:
- Some experts specialize in vision tokens
- Some experts specialize in language tokens
- Router learns this automatically from the data

**Key insight**: In our interleaved MoE architecture, we should expect:
- Some experts will specialize in text→brain mapping
- Some in audio→brain mapping
- Some in cross-modal integration
- The router handles this automatically — no need to force it

**What worked:**
- Shared experts + specialized experts (some experts used by all modalities)
- Auxiliary routing loss with per-modality load balancing
- Capacity factor > 1.0 (allow expert buffers for load balancing)

### 5.4 VideoMAE Distillation

**Relevant for our video pipeline.**

**Teacher**: VideoMAE-L (305M)
**Student**: VideoMAE-S (22M)

**Key findings:**
- Temporal attention is harder to distill than spatial attention
- Feature matching on temporal CLS tokens works better than frame-level matching
- Motion features are the first thing lost in compression
- Adding a temporal difference loss (predict frame-to-frame change) helps retain dynamics

**Implication**: Our video backbone (MobileViT-S) is per-frame only. We lose temporal
dynamics entirely. Adding a simple temporal difference module (Conv1D on frame features)
after MobileViT could recover some of this.

---

## 6. MoE-Specific Distillation {#6-moe-distillation}

### 6.1 Distilling MoE → Dense (relevant if we later compress further)

**Approaches from Switch Transformer and Mixtral literature:**

1. **Expert averaging**: Average all expert weights → single dense FFN
   - Surprisingly effective as initialization for further fine-tuning
   - Loses specialization but retains "consensus" knowledge

2. **Expert pruning**: Keep only the top-K most utilized experts
   - Monitor routing statistics, drop least-used experts
   - Then fine-tune the remaining experts

3. **Expert merging**: Cluster similar experts, merge within clusters
   - Uses weight similarity or activation similarity to find clusters
   - More principled than simple averaging

### 6.2 Training Dense → MoE (what we're doing)

**We are distilling a dense teacher (TRIBE v2) into an MoE student.**
This is less studied but has important patterns:

1. **Expert initialization matters enormously**
   - Random init: experts start identical, slow to differentiate
   - **Recommended**: Initialize all experts from the same pre-trained FFN weights,
     then add small random noise (σ=0.01) to break symmetry
   - This gives experts a good starting point and lets them specialize gradually

2. **Router warmup**
   - Start with uniform routing (all experts equal weight)
   - Gradually increase router temperature over first 1000 steps
   - This prevents early expert collapse before features are meaningful

3. **Load balancing strength schedule**
   - Start with high aux loss weight (0.1) to enforce diversity
   - Decay to 0.01 over training as experts naturally specialize
   - Without this: 1-2 experts dominate, rest go unused

### 6.3 MoE Training Stability

**Known issues and solutions:**

| Issue | Solution |
|-------|----------|
| Expert collapse (1-2 experts get all traffic) | Aux load-balancing loss + dropout on router |
| Training instability (loss spikes) | Router z-loss: penalize large router logits |
| Gradient noise from discrete routing | SmoothTop-k: use softmax approximation |
| Memory spikes from uneven load | Capacity factor (cap tokens per expert per batch) |

---

## 7. Regression & Dense Prediction Distillation {#7-regression-distillation}

### 7.1 Why Brain Encoding KD Is Different

Standard KD assumes classification (softmax outputs). Brain encoding is:
- **Regression**: continuous values, not class probabilities
- **Dense prediction**: 5,124+ output dimensions per timestep
- **Temporal**: predictions at every TR, temporal coherence matters
- **Subject-specific**: each subject has different brain anatomy
- **Multi-scale**: some brain regions are coarse (language), some fine (retinotopy)

### 7.2 Regression KD Techniques

**From dense prediction tasks (depth estimation, segmentation, pose):**

1. **Direct MSE matching**: simplest, often sufficient
   ```
   L = MSE(student_pred, teacher_pred)
   ```
   Works because the output space is continuous and the teacher's predictions
   are already smooth.

2. **Structured output matching**: preserve spatial/topological relationships
   ```
   L = MSE(pred, target) + λ × gradient_matching(pred, target)
   ```
   For brain encoding: preserve spatial gradients across the cortical surface.
   Adjacent vertices should have similar prediction errors.

3. **Uncertainty-weighted KD**: teacher's uncertain predictions get lower weight
   ```
   L = Σ_i (1/σ²_teacher_i) × (pred_i - target_i)²
   ```
   Run teacher multiple times with dropout to estimate uncertainty.
   Weight distillation loss inversely with uncertainty.

4. **Multi-resolution KD**: match at multiple spatial scales
   ```
   L = MSE(pred, target)
         + MSE(downsample(pred), downsample(target))
         + MSE(ROI_avg(pred), ROI_avg(target))
   ```
   First match at vertex level, then at parcel level, then at network level.
   The coarse-level matching guides the student toward the right large-scale
   patterns even when vertex-level matching is noisy.

### 7.3 Feature KD for Regression

**CKA (Centered Kernel Alignment)** is the standard for feature comparison:
```
CKA(X, Y) = ||X^T Y||²_F / (||X^T X||_F × ||Y^T Y||_F)
```

But CKA has issues:
- **Compute cost**: O(N² × D) for N samples, D features
- **Batch sensitivity**: very different values for small vs large batches
- **Not a true loss**: CKA ∈ [0,1], gradient signal is weak near 1.0

**Better alternatives for feature KD in regression:**

1. **Projected cosine similarity** (simpler, more stable):
   ```
   student_proj = Linear(student_feat, teacher_dim)
   L = 1 - cosine_similarity(student_proj, teacher_feat).mean()
   ```

2. **Attention pattern matching** (for transformers):
   ```
   L = MSE(student_attn_weights, teacher_attn_weights)
   ```
   Particularly useful for matching HOW modalities attend to each other.

3. **Gram matrix matching** (structural):
   ```
   G_s = student_feat @ student_feat^T  # (B×T, B×T)
   G_t = teacher_feat @ teacher_feat^T
   L = MSE(G_s, G_t) / (B×T)²
   ```
   Preserves pairwise relationships between timesteps without requiring
   same feature dimensions.

### 7.4 Temporal Distillation

**Unique to sequential prediction tasks like brain encoding:**

1. **Temporal coherence loss**: student predictions should be as smooth as teacher's
   ```
   L_smooth = MSE(Δstudent, Δteacher)  where Δ = pred[t] - pred[t-1]
   ```

2. **Hemodynamic-aware KD**: fMRI signals have a known temporal smoothness
   (hemodynamic response function, ~6s peak). Teacher predictions should
   already reflect this. Adding a temporal smoothness prior on student
   predictions helps:
   ```
   L_hrf = MSE(conv1d(student_pred, hrf_kernel), student_pred)
   ```

---

## 8. Cross-Cutting Patterns {#8-cross-cutting-patterns}

### 8.1 Universal Patterns (Appear in ALL Domains)

1. **Width > Depth for compression**
   Every successful distillation (DistilBERT, DistilHuBERT, TinyViT) keeps
   width and reduces depth. Wider layers preserve per-layer representational
   power; depth can be compensated by better training.

   **For Tiny-TRIBE**: Keep 512D hidden, use 4 layers (not 2 layers × 1024D).

2. **Teacher initialization > random initialization**
   Initializing student weights from teacher layers (even approximately)
   consistently outperforms random init by 5-15%.

   **For Tiny-TRIBE**: Initialize projector weights from teacher's projector.
   Initialize MoE expert weights from teacher's FFN weights (with noise).

3. **Progressive distillation > one-shot**
   Two-stage or multi-stage distillation consistently outperforms single-stage.
   First stage: align representations. Second stage: fine-tune for task.

   **For Tiny-TRIBE**: Already planned — Stage 1 (frozen) + Stage 2 (E2E).

4. **Data volume matters more than loss function sophistication**
   DistilBERT with simple triple loss + lots of data > TinyBERT with complex
   4-component loss + less data.

   **For Tiny-TRIBE**: Run teacher on as many videos as possible. 10+ hours minimum.

5. **Feature KD helps most in middle layers**
   Across all domains, matching intermediate layers (not first, not last)
   gives the biggest quality boost. First layers are too low-level,
   last layers are too task-specific.

   **For Tiny-TRIBE**: Match student layers [1,2] to teacher layers [3,5]
   (middle of teacher's 8-layer stack).

### 8.2 Multimodal-Specific Patterns

1. **Projector quality is disproportionately important**
   In BLIP-2, LLaVA, and TRIBE v2, the modality projector is the bottleneck.
   Underparameterized projectors lose information before fusion even starts.

   **For Tiny-TRIBE**: Consider 3-layer projectors (current: 2-layer).
   Or increase projector hidden dim (384→512 intermediate).

2. **Cross-modal feature matching > within-modal matching**
   Matching how modalities interact (cross-attention patterns, fused representations)
   transfers better than matching each modality's features independently.

   **For Tiny-TRIBE**: Match FUSED features (after transformer), not per-modality features.

3. **Modality dropout should decrease during KD**
   High dropout (50%) during pre-training helps robustness.
   During distillation, the teacher's signal is richest when all modalities
   are present. Decrease dropout during KD (30% → 10% → 0% over training).

4. **Temporal alignment errors compound across modalities**
   If audio features are misaligned by 1 TR with video features,
   the fusion model learns a wrong correspondence. This error is
   amplified during distillation because the teacher had perfectly
   aligned features (from its better backbones).

   **For Tiny-TRIBE**: Ensure backbone feature extraction uses identical
   timestamps. Consider a learned temporal alignment module.

### 8.3 MoE-Specific Patterns

1. **Expert diversity predicts quality**
   Monitor: average pairwise cosine similarity between expert outputs.
   If experts produce similar outputs, the MoE is degenerating to dense.
   Target: cosine similarity < 0.7 between experts.

2. **Router entropy should be high early, moderate late**
   Early training: router entropy should be near log(K) (uniform routing).
   Late training: entropy should decrease as experts specialize.
   If entropy drops too fast: increase aux loss weight.
   If entropy stays too high: decrease aux loss weight.

3. **Top-2 is the sweet spot for small MoE**
   - Top-1: too sparse, quality drops 5-10% vs dense baseline
   - Top-2: matches or exceeds dense baseline at same compute
   - Top-4: no quality gain over top-2, but 2× compute

---

## 9. Anti-Patterns & Failures {#9-anti-patterns}

### Things That Look Good But Don't Work

1. **Matching ALL teacher layers**
   Per-layer matching for every layer (student layer i → teacher layer j)
   creates conflicting gradients. The student can't simultaneously match
   early AND late teacher representations. Pick 2-3 key layers.

2. **Very high temperature for regression KD**
   Temperature τ is a classification concept (softmax scaling).
   For regression, there's no temperature. Using MSE directly is fine.
   Some papers apply temperature to feature matching — this is almost
   always worse than simple cosine/MSE.

3. **Distilling attention patterns when architectures differ**
   If teacher has 8 heads and student has 4 heads, matching attention
   matrices is ill-defined. Attention transfer works ONLY when the
   number of heads matches. For our case (different architectures),
   skip attention KD entirely.

4. **Too-small learning rate during KD**
   KD provides a smoother loss landscape than ground truth labels.
   Students can use HIGHER learning rates during KD than during
   standard training (1.5-2× higher). Common mistake: using the
   same LR as the teacher's training.

5. **Freezing too much for too long**
   Stage 1 (frozen backbones) should be SHORT (5 epochs max).
   The projectors and fusion need to co-adapt with backbone features.
   If you freeze for 20 epochs, the fusion overfits to frozen features
   and can't adapt when you unfreeze.

6. **Identical augmentation for teacher and student**
   The teacher was trained with specific augmentations. The student
   should use DIFFERENT augmentations during KD to learn robustness
   that the teacher didn't provide.

---

## 10. Recommendations for Tiny-TRIBE {#10-recommendations}

### 10.1 Architecture Recommendations

Based on the patterns above:

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Hidden dim | **512** (keep, don't reduce) | Width > depth for quality |
| Num layers | **4** (keep) | Sweet spot for T4 memory |
| Num heads | **8** (increase from 4) | 64-dim per head, proven in literature |
| Num experts | **8** (keep) | Standard MoE, enough diversity |
| Top-k routing | **2** (keep) | Sweet spot quality/compute |
| Expert FFN mult | **2** (512→1024→512) | Each expert small, total capacity = 4× dense |
| Projector depth | **3 layers** (increase from 2) | Projector quality is disproportionately important |
| Projector intermediate | **768** (increase from 512) | Extra capacity in the bottleneck |
| Modality tokens | **Interleaved** (keep) | Better than concatenation for cross-modal |
| Modality embeddings | **Learned** (keep) | Cheap, effective |
| Modality dropout | **0.3 → 0.1** (decrease during KD) | Full teacher signal when all modalities present |

### 10.2 Distillation Strategy Recommendations

**Phase 0: Teacher Data Generation**

```
Priority: QUANTITY of teacher data > everything else.
Target: 10+ hours of diverse video content.
Use your posted/ videos plus any public domain videos.
```

- Run TRIBE v2 on all videos, save predictions to Google Drive
- Also save intermediate fusion features (teacher layer 4 and layer 6)
- Store per-TR predictions, not averaged — temporal structure matters
- This is a one-time cost. Spend the GPU hours here.

**Phase 1: Frozen Backbone Training (5 epochs)**

```python
# Recommended loss
L = 0.8 × MSE(student_pred, teacher_pred)     # Output matching
  + 0.1 × cosine_loss(student_fused, teacher_fused_proj)  # Feature matching
  + 0.1 × smooth_l1(Δstudent, Δteacher)       # Temporal coherence
  + 0.01 × aux_loss                             # MoE load balancing
```

- Initialize MoE experts from a single pre-trained FFN (with noise)
- LR: 1e-3 with OneCycleLR
- Router warmup: uniform routing for first 500 steps
- Aux loss warmup: 0.1 → 0.01 over first 1000 steps
- Gradient clipping: max_norm=1.0

**Phase 2: End-to-End KD Fine-Tuning (10 epochs)**

```python
# Recommended loss (now add ground truth fMRI if available)
L = 0.5 × MSE(student_pred, teacher_pred)     # Output KD (reduced weight)
  + 0.3 × MSE(student_pred, fmri_target)      # Ground truth (if available)
  + 0.1 × cosine_loss(student_fused, teacher_fused_proj)
  + 0.05 × smooth_l1(Δstudent, Δteacher)
  + 0.01 × aux_loss
  + 0.04 × multi_res_loss(student_pred, teacher_pred)  # Parcel-level matching
```

- Unfreeze Whisper-Tiny encoder (LR: 1e-5, 10× lower than fusion)
- Unfreeze MobileViT-S (LR: 1e-5)
- Keep text backbone frozen (MiniLM is already well-pretrained)
- Decrease modality dropout: 0.3 → 0.1 over training
- Increase batch size if memory allows (larger batches help MoE routing stability)

### 10.3 Monitoring Recommendations

Track these during training:

| Metric | Good range | Action if out of range |
|--------|-----------|----------------------|
| Expert utilization entropy | >1.5 (of max 2.08 for 8 experts) | Increase aux loss if too low |
| Per-expert token fraction | 10-15% each (ideal 12.5%) | Check for collapse if any >30% |
| Router logit magnitude | <10.0 | Add z-loss if >20 |
| Feature cosine sim (student vs teacher) | >0.7 by epoch 5 | Check projector quality |
| Temporal smoothness ratio | Within 2× of teacher | Add temporal loss if too noisy |
| Val loss improving | Decreasing or plateau | Stop training if increasing for 3 epochs |

### 10.4 Data Recommendations

In order of priority:

1. **Your posted/ videos** (already available, diverse content)
2. **YouTube public domain** (nature documentaries, talks, cooking shows)
3. **Algonauts 2025 challenge** (has paired fMRI for validation)
4. **BOLD Moments** (3-second clips, good for augmentation)

For validation (requires real fMRI):
- Algonauts 2025 test set (Friends S7)
- BOLD Moments test clips (102 clips, 10 reps)

### 10.5 What NOT To Do

1. **Don't use synthetic/proxy features for final training** — always use real teacher outputs
2. **Don't match all 8 teacher layers** — pick 2-3 middle layers
3. **Don't freeze backbones for >5 epochs** — diminishing returns, overfitting risk
4. **Don't use the same modality dropout during KD as during pre-training** — reduce it
5. **Don't skip the aux loss** — MoE WILL collapse without it
6. **Don't initialize experts randomly** — initialize from same FFN + noise
7. **Don't use a tiny batch size** — MoE routing is unstable with BS<4

---

## Appendix A: Loss Function Summary

### Complete Recommended Loss (Phase 2)

```python
class TinyTribeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.feat_proj = nn.Linear(512, 1152)  # student → teacher dim

    def forward(self, student_pred, teacher_pred, fmri_target,
                student_feat, teacher_feat, aux_loss):

        # 1. Output KD (teacher predictions as target)
        L_output = self.mse(student_pred, teacher_pred.detach())

        # 2. Task loss (real fMRI, if available)
        L_task = self.mse(student_pred, fmri_target) if fmri_target is not None else 0

        # 3. Feature alignment (cosine similarity)
        s = self.feat_proj(student_feat.reshape(-1, 512))
        t = teacher_feat.detach().reshape(-1, 1152)
        L_feat = 1 - F.cosine_similarity(s, t, dim=-1).mean()

        # 4. Temporal coherence
        Δs = student_pred[:, :, 1:] - student_pred[:, :, :-1]
        Δt = teacher_pred[:, :, 1:] - teacher_pred[:, :, :-1]
        L_temporal = self.smooth_l1(Δs, Δt.detach())

        # 5. Multi-resolution (optional: match at ROI level)
        # Requires Schaefer parcellation mapping
        # L_multi = mse(parcel_avg(student), parcel_avg(teacher))

        # 6. MoE auxiliary loss (already computed by model)
        L_aux = aux_loss

        # Weighted sum
        L = (0.5 * L_output
           + 0.3 * L_task
           + 0.1 * L_feat
           + 0.05 * L_temporal
           + 0.01 * L_aux)

        return L
```

### Appendix B: Key References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Hinton et al., "Distilling the Knowledge" | 2015 | Original KD formulation |
| DistilBERT (Sanh et al.) | 2019 | Width>depth, triple loss |
| TinyBERT (Jiao et al.) | 2020 | Two-stage KD, attention matching |
| MiniLM (Wang et al.) | 2020 | Value-relation transfer |
| DeiT (Touvron et al.) | 2021 | Distillation token, cross-arch KD |
| Switch Transformer (Fedus et al.) | 2022 | MoE load balancing |
| Distil-Whisper (Gandhi et al.) | 2023 | Pseudo-labelling for speech |
| TinyCLIP (Wu et al.) | 2023 | Affinity mimicking |
| Mixtral (Jiang et al.) | 2024 | Sparse MoE at scale |
| TRIBE v2 (d'Ascoli et al.) | 2026 | Multimodal brain encoding |
