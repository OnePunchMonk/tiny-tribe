# TRIBE v2: A Foundation Model for In-Silico Neuroscience

## Deep Dive, Analysis, and Research Ideas

---

## 1. What is TRIBE v2?

TRIBE v2 (TRImodal Brain Encoder, version 2) is a deep multimodal brain encoding model from Meta FAIR that predicts **fMRI brain responses** to naturalistic stimuli — video, audio, and text. It maps multimodal representations from three foundation models onto the cortical surface, enabling "in-silico neuroscience" — simulating what the brain would do in response to any stimulus, without scanning anyone.

**Key facts:**

- Combines **LLaMA 3.2-3B** (text), **V-JEPA2-ViT-G** (video), and **Wav2Vec-BERT 2.0** (audio)
- Predicts **20,484 cortical vertices** on the fsaverage5 mesh (10,242 per hemisphere)
- Trained on **1,000+ hours of fMRI** across **720 subjects** (full paper; open-source code includes 4 datasets, 25 subjects)
- Won **1st place** at the Algonauts 2025 brain modeling competition (v1)
- Released under CC-BY-NC 4.0

**Paper:** d'Ascoli, Rapin, Benchetrit, Brookes, Begany, Raugel, Banville, King. *"A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience."* Meta FAIR, 2026.

---

## 2. Architecture Overview

### 2.1 Feature Extraction

Three frozen foundation models extract representations from naturalistic stimuli at 2 Hz:

| Modality | Model | Parameters | Output Dim | Layers Used | Notes |
|----------|-------|------------|------------|-------------|-------|
| **Text** | LLaMA 3.2-3B | 3B | 3,072 | [0, 0.2, 0.4, 0.6, 0.8, 1.0] | Contextualized (k=1024 words), aligned to word timestamps via WhisperX |
| **Audio** | Wav2Vec-BERT 2.0 | ~600M | 1,024 | [0.75, 1.0] | 60s chunks, bidirectional encoding |
| **Video** | V-JEPA2-ViT-G | 1.1B | 1,536 | [0.75, 1.0] | 4s clips at 2 fps, spatial average over patches (discards retinotopy) |

Multiple intermediate layers are extracted (not just the last layer), giving the model access to representations at varying levels of abstraction — a design choice grounded in neuroscience, where low-level sensory cortices correlate with early layers and higher-order areas correlate with deeper layers.

### 2.2 Fusion Transformer

```
Per-modality features (B, T, L*D)
    |
    v
Per-modality MLP projectors (LayerNorm + GELU)
    → each modality projected to hidden/3 dimensions
    |
    v
Concatenate across modalities → (B, T, 1152)
    |
    v
+ Learnable positional embeddings (up to 1024 timesteps)
    |
    v
8-layer Transformer encoder (8 heads)
    |
    v
Low-rank bottleneck: 1152 → 2048
    |
    v
SubjectLayers: 2048 → 20,484 vertices (subject-specific linear heads)
    |
    v
Adaptive temporal pooling → (B, 20484, T_output)
```

**Key design decisions:**
- **Modality dropout (p=0.3):** During training, entire modalities are randomly zeroed, forcing the model to learn robust representations from any subset of inputs
- **Subject layers with dropout (p=0.1):** 10% of the time, subject-specific weights are replaced by their average, creating a "virtual average subject" used at inference time
- **No combiner MLP in v2:** The transformer directly handles cross-modal fusion (v1 used an explicit combiner)

### 2.3 Output Space

Predictions live on the **fsaverage5** cortical mesh — a standard FreeSurfer template with 10,242 vertices per hemisphere (20,484 total). Each vertex represents a point on the cortical surface, tiling every gyrus and sulcus. This is a 20x increase in spatial resolution compared to v1's 1,000 Schaefer parcels.

---

## 3. Training

### 3.1 Datasets

| Dataset | Subjects | Hours/Subject | Stimuli Type | TR |
|---------|----------|---------------|--------------|-----|
| Algonauts2025 (Courtois NeuroMod) | 4 | ~66h | TV shows (Friends S1-7), movies | 1.49s |
| Lebel2023 | 8 | 6-18h | 82 spoken narrative stories | 2.0s |
| Lahner2024 (BOLD Moments) | 10 | ~6.2h | 1,000 brief 3s video clips | 1.75s |
| Wen2017 | 3 | ~11.7h | Video segments | 2.0s |

The full paper reportedly uses additional datasets (Vanessen2023, Aliko2020, Li2022, Nastase2020) to reach 720 subjects — these are referenced in the codebase but not included in the open-source release.

### 3.2 Preprocessing

- fMRIprep for volumetric preprocessing
- MNI152 standard space registration
- Surface projection via `vol_to_surf` (ball kernel, radius=3mm) to fsaverage5
- 5-second hemodynamic delay compensation (fMRI offset)
- Features extracted at 2 Hz, fMRI at 1 Hz

### 3.3 Optimization

- **Loss:** MSE (per-vertex, unreduced then averaged)
- **Optimizer:** Adam, lr=1e-4, no weight decay
- **Scheduler:** OneCycleLR (10% warmup, cosine decay)
- **Segments:** 100 TRs each
- **Batch size:** 8
- **Epochs:** 15

### 3.4 v1 Competition Strategy

The original TRIBE v1 that won Algonauts 2025 used an ensemble of **1,000 models** with per-parcel softmax weighting (temperature=0.3). This is a competition-specific trick — v2 is a single foundation model.

---

## 4. Key Results

### 4.1 Algonauts 2025 Competition (v1)

| Rank | Team | Mean Pearson r |
|------|------|---------------|
| **1st** | **TRIBE** | **0.2146** |
| 2nd | NCG | 0.2096 |
| 3rd | SDA | 0.2094 |

Normalized against the noise ceiling: **0.54 ± 0.1** across all parcels — meaning TRIBE captures over half the explainable variance in brain responses.

### 4.2 Modality Contributions

| Configuration | Mean r |
|---------------|--------|
| Text only | 0.22 |
| Audio only | 0.24 |
| Video only | 0.25 |
| Text + Video | 0.30 |
| All three | 0.31 |

Multimodal outperforms unimodal by up to **30% in associative cortices** (prefrontal, parieto-occipito-temporal junction) — regions that integrate information across senses.

### 4.3 Anatomical Specialization

Each modality dominates its expected cortical territory:
- **Video** → occipital and parietal cortex (visual areas)
- **Audio** → superior temporal gyrus (auditory cortex)
- **Text** → parietal and prefrontal cortex (language/semantic areas)
- **Cross-modal gains** → superior temporal (text+audio), ventral/dorsal visual (video+audio)

### 4.4 Scaling Behavior

No saturation observed in either dimension:
- More training data → better predictions (monotonic, no plateau)
- Longer LLM context (up to 1,024 words) → stronger performance, especially in language areas

---

## 5. Strengths

### 5.1 First True Multimodal Brain Foundation Model

Previous brain encoding models were unimodal (vision-only or language-only) or bimodal at best. TRIBE v2 is the first to integrate vision, audition, and language into a unified architecture that predicts whole-brain responses. This matters because real-world perception is inherently multimodal — the brain doesn't process senses in isolation.

### 5.2 Vertex-Level Resolution

Predicting 20,484 cortical vertices (vs. 1,000 parcels in v1 or ~300 ROIs in prior work) preserves fine-grained spatial information. This enables analysis of within-area gradients, boundary effects between regions, and subtle topographic organization that parcel-level models miss entirely.

### 5.3 Backbone Quality

The choice of V-JEPA2, Wav2Vec-BERT, and LLaMA 3.2 is strong. These are individually SOTA or near-SOTA in their domains, and critically, they were trained on objectives that produce hierarchical, semantically rich representations — not just classification features. V-JEPA2 in particular (joint embedding predictive architecture) learns temporal dynamics in video, which aligns well with how the brain processes dynamic scenes.

### 5.4 Multi-Layer Extraction

Using intermediate layers (not just the final layer) is neuroscientifically motivated. There's strong evidence that early visual cortex correlates with early CNN/ViT layers and higher-order areas with deeper layers. TRIBE lets the transformer learn which layers matter for which vertices.

### 5.5 In-Silico Neuroscience Capability

The ability to predict brain responses to arbitrary stimuli without scanning is transformative. It enables:
- Hypothesis testing at zero marginal cost
- Stimulus optimization before running expensive fMRI experiments
- Population-level analysis without collecting new data
- Exploring stimulus spaces that would be impractical to test empirically

### 5.6 Open Release

The code, model weights, and demo are publicly available. The architecture is clean and modular. This is rare for neuroscience models of this scale.

---

## 6. Weaknesses and Limitation

### 6.3 Temporal Resolution is Fundamentally Limited by fMRI

The BOLD signal has a hemodynamic response function with ~5-6 second lag and ~1 Hz effective bandwidth. TRIBE inherits this limitation — it predicts the sluggish hemodynamic response, not neural activity. Any phenomena faster than ~1 Hz (most of neuroscience) are invisible. EEG/MEG encoding models would be needed for temporal dynamics.

### 6.4 Only 25 Subjects in Open Release (Claims 720)

The paper claims 720 subjects and 1,000+ hours of fMRI, but the released code only includes 4 datasets with 25 subjects. The gap between what's claimed and what's reproducible is large. Key results about cross-subject generalization and population-level predictions cannot be independently verified.

### 6.5 Cognitive Scope is Narrow

TRIBE only models passive perception and comprehension of audiovisual narratives. It cannot predict:
- Working memory or episodic memory encoding
- Decision-making or reward processing
- Motor planning and execution
- Emotional regulation
- Social cognition beyond what's in narratives
- Any task-related or goal-directed cognition

This limits its utility as a general "brain simulator."

### 6.6 Average Subject Prediction is a Blunt Tool

At inference, TRIBE averages all subject-specific weights into a "virtual average subject." This erases individual differences in functional organization — which can be substantial (e.g., language lateralization varies across individuals). For any study where individual differences matter (clinical applications, neurodiverse populations), this is a significant limitation.

### 6.7 No Subcortical Coverage in v1 (Partial in v2)

The cortical surface projection excludes subcortical structures (amygdala, hippocampus, thalamus, basal ganglia, cerebellum). v2 adds partial subcortical support via a separate `MaskProjector`, but the released model weights focus on cortical prediction. Emotional processing, memory consolidation, and motor control are all heavily subcortical.

### 6.8 Training Data Homogeneity

The training stimuli are overwhelmingly Western, English-language content (Friends, Hollywood movies, English narratives). Cross-cultural or cross-linguistic generalization is untested. The "foundation model" claim is premature without evidence of transfer across languages and cultural contexts.

### 6.9 No Explicit Temporal Modeling of HRF

The model uses a fixed 5-second offset to compensate for hemodynamic delay, but the HRF varies across brain regions, individuals, and even trials. A learnable, region-specific HRF deconvolution would be more principled.

---

## 7. Experiment Ideas

### 7.1 Quick Experiments (Days)

**A. Modality ablation on your own content**
Run your 9 videos through TRIBE with each modality disabled (using `features_to_mask`). Compare which brain regions are driven by visual vs. audio vs. text content in your specific videos. This reveals what makes your content neurally engaging.

**B. Temporal engagement curve**
Plot mean cortical activation over time for each video. Identify peaks and valleys — these correspond to moments of high/low neural engagement. Cross-reference wis

### 6.1 Spatial Averaging in Video Destroys Retinotopy

V-JEPA2 patch tokens are spatially averaged before being fed to the transformer. This discards all positional information, meaning the model cannot predict retinotopic organization in V1/V2/V3 — arguably the best-understood property of the visual cortex. This is a significant gap. The authors acknowledge it but don't address it.

**Impact:** Low-level visual areas (V1-V3) are systematically underpredicted. Any experiment involving spatial visual processing (attention, saccades, scene layout) is fundamentally limited.

### 6.2 Bidirectional Audio Encoding is Non-Causal

Wav2Vec-BERT is bidirectional — it uses future audio to encode the present. The brain doesn't do this. This means the model has access to information the brain doesn't have at each moment, which could inflate predictions in auditory cortex and confound temporal analysis. This is especially problematic for studying predictive processing, surprise responses, or temporal expectations.th video content to understand what drives attention.

**C. Cross-video comparison**
Compare brain activation patterns across your 9 videos using representational similarity analysis (RSA). Which videos produce the most similar brain responses? Which are most distinct? This is a neural fingerprint of your content.

**D. ROI-focused analysis**
Extract predictions for specific HCP-MMP1 ROIs (e.g., FFA for faces, STS for social processing, V1 for visual complexity, auditory cortex for speech). Build a per-video profile of which cognitive systems are engaged.

### 7.2 Medium-Term Experiments (Weeks)

**E. Content optimization loop**
Use TRIBE as a differentiable objective — backpropagate through the model to find what input features maximize activation in specific brain regions. This could guide content creation: "what audio/visual elements maximize engagement in attention networks?"

**F. Fine-tune on new subjects**
The codebase supports `resize_subject_layer` for transfer learning. Collect a small amount of fMRI data from a new subject watching your content, fine-tune the subject layer, and get personalized predictions. This tests whether the foundation model transfers.

**G. Add retinotopic features**
Address weakness 6.1: Instead of spatial averaging, extract V-JEPA2 patch tokens with positional encoding intact. Add a spatial feature pathway that preserves retinotopic information. Measure improvement in V1-V3 prediction.

**H. Causal audio encoder swap**
Replace Wav2Vec-BERT with a causal audio model (e.g., a causal Whisper variant or AudioMAE). Compare temporal prediction profiles in auditory cortex — the causal model should better capture surprise/prediction-error signals.

### 7.3 Research-Grade Experiments (Months)

**I. Cross-linguistic brain encoding**
Run TRIBE on non-English content (Hindi, Mandarin, etc.). The text encoder (LLaMA 3.2) is multilingual. Compare predicted language-area activations across languages. Does the model predict universal or language-specific processing?

**J. Temporal dynamics with EEG**
Train a TRIBE-like model that predicts EEG instead of fMRI. Replace the fsaverage5 output with electrode-level predictions. Use the same multimodal backbone but with millisecond-resolution targets. This addresses weakness 6.3.

**K. Clinical brain encoding**
Collect fMRI from clinical populations (ADHD, autism, depression) watching naturalistic stimuli. Fine-tune TRIBE's subject layers on these subjects. Compare their learned subject weights to neurotypical subjects — deviations could be biomarkers.

**L. Memory encoding prediction**
Extend TRIBE to predict not just perceptual responses but subsequent memory. Train a secondary head that predicts whether a scene will be remembered (using post-scan recall tests). This would extend beyond passive perception (addressing 6.5).

---

## 8. Framework and Product Ideas

### 8.1 Neuro-Informed Content Scoring API

Build an API that takes a video URL and returns a "neural engagement score" — a summary of predicted brain activation across key networks (attention, emotion, language, visual). Applications:
- **Ad testing:** Score advertisements before airing
- **Film editing:** Identify scenes that underperform neurally
- **Education:** Optimize lecture videos for learning-related brain regions

### 8.2 Real-Time Brain Prediction Dashboard

Stream a video through TRIBE frame-by-frame and display a live 3D brain visualization showing predicted activations. This could be a powerful demonstration tool for:
- Neuroscience education
- Content creator tools
- UX research presentations

### 8.3 Stimulus Design Framework

Build a framework that uses TRIBE as a differentiable reward model for stimulus design. Given a target brain activation pattern (e.g., "maximize fusiform face area while minimizing amygdala"), use gradient-based optimization to guide content creation. This inverts the typical neuroscience workflow.

### 8.4 Brain-Similarity Search Engine

Given a query video, find other videos in a library that produce the most similar predicted brain responses. This is "search by neural fingerprint" — two videos might look completely different but engage the same cognitive processes. Applications in content recommendation.

### 8.5 Multimodal Brain-Computer Interface Decoder

Invert TRIBE: given fMRI data, decode what the person was watching/hearing. The model's learned mapping from features→brain could be inverted (or a separate decoder trained using TRIBE's representations) for brain-to-content reconstruction.

### 8.6 Neuroscience Experiment Simulator

A platform where researchers define experimental paradigms (stimulus sequences, conditions, contrasts) and TRIBE predicts the expected brain activation patterns. This could:
- Power-analyze experiments before collecting data
- Generate pilot results for grant applications
- Test hypotheses computationally before committing to expensive scanning

### 8.7 Personalized Brain Model Marketplace

A platform where individuals contribute small amounts of fMRI data, get a personalized TRIBE subject layer, and can then predict their own brain responses to any content. Privacy-preserving (only the linear layer is stored, not raw brain data).

---

## 9. Comparison: TRIBE v1 vs v2

| Aspect | TRIBE v1 | TRIBE v2 |
|--------|----------|----------|
| Output space | 1,000 Schaefer parcels | 20,484 fsaverage5 vertices |
| Training data | 4 subjects, ~265 hours | 720 subjects, 1,000+ hours (claimed) |
| Hidden dim | 1,024 | 1,152 |
| Combiner | MLP | None (transformer handles fusion) |
| Modality dropout | 0.2 | 0.3 |
| Subject adaptation | Subject embedding | SubjectLayers with dropout + low-rank head |
| Ensembling | 1,000 models | Single model |
| Subcortical | No | Yes (partial) |
| Open weights | Competition submission | HuggingFace release |

---

## 10. Technical Setup for Experimentation

### Quick Start (GPU required, ~16GB VRAM)

```python
from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain
from pathlib import Path

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
plotter = PlotBrain(mesh="fsaverage5")

# Predict brain responses to a video
df = model.get_events_dataframe(video_path="your_video.mp4")
preds, segments = model.predict(events=df)
# preds.shape = (n_timesteps, 20484)

# Visualize
fig = plotter.plot_timesteps(
    preds[:15], segments=segments[:15],
    cmap="fire", norm_percentile=99, vmin=0.5,
    alpha_cmap=(0, 0.2), show_stimuli=True,
)
```

### Requirements

- Python 3.10+
- PyTorch 2.5-2.6
- CUDA GPU (T4 minimum, A100 recommended)
- HuggingFace account with LLaMA 3.2 access
- `pip install -e ".[plotting]"` from the tribev2 repo

### Key Config Overrides

```python
# Disable text encoder (if no LLaMA access)
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    config_update={"data.features_to_mask": ["text"]},
)

# Use lighter backbones
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    config_update={
        "data.text_feature.model_name": "Qwen/Qwen3-0.6B",
    },
)
```

---

## 11. Key References

- **TRIBE v1 (Algonauts 2025):** arXiv 2507.22229
- **TRIBE v2 (Foundation Model):** ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- **Code:** github.com/facebookresearch/tribev2
- **Weights:** huggingface.co/facebook/tribev2
- **Demo:** aidemos.atmeta.com/tribev2/
- **V-JEPA2:** Bardes et al., 2024
- **Wav2Vec-BERT 2.0:** Babu et al., 2024
- **LLaMA 3.2:** Grattafiori et al., 2024
- **fsaverage5:** FreeSurfer, Fischl 2012
- **HCP-MMP1 Parcellation:** Glasser et al., 2016

---

*Document prepared March 2026. Based on the publicly released codebase and available paper materials.*
