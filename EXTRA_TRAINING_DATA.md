# Extra Training Data Sources for Tiny-TRIBE v3

## Principle

The v3 strategy decouples **multimodal fusion learning** from **brain mapping**.
Fusion learning needs diverse multimodal data — no fMRI, no teacher required.
This massively expands the usable training data.

---

## 1. Data Sources Ranked by Value

### Tier 1: High Value (Diverse, Multimodal, Free)

| Dataset | Size | Modalities | Why Valuable | Access |
|---------|------|------------|-------------|--------|
| **HowTo100M** | 136M clips, 1.2M videos | Video + ASR text + audio | Massive, diverse, instructional (cooking, repair, crafts) | Public, YouTube-based |
| **VGGSound** | 200K 10s clips | Video + audio | Diverse sound events tied to visual scenes | Public download |
| **AudioSet** | 2M 10s clips | Audio (+ YouTube video) | Most diverse audio source, 527 categories | Public, YouTube-based |
| **Spoken Moments in Time** | 500K 3s clips | Video + audio + captions | Dense captions per clip, good for CMC task | Public |

**Recommended: HowTo100M subset (500h) + VGGSound (200h)**

### Tier 2: Moderate Value (Audio + Text, No Video)

| Dataset | Size | Modalities | Why Valuable | Access |
|---------|------|------------|-------------|--------|
| **LibriSpeech** | 960h | Audio + text | Clean speech, perfect for audio↔text alignment | Public download |
| **Common Voice** | 19,000h+ | Audio + text | Multilingual, diverse speakers | Public download |
| **GigaSpeech** | 10,000h | Audio + text | Diverse: YouTube, podcast, audiobook | Public |
| **TED-LIUM** | 452h | Audio + text | Lectures, clean recording | Public download |

These lack video, but that's fine — modality dropout during pre-training means
the model also needs to handle text+audio only. Using these for the 2-modality
masking condition is ideal.

**Recommended: LibriSpeech (960h) for audio+text pre-training**

### Tier 3: Neuroscience-Adjacent (Have fMRI Pairing)

These are already used in TRIBE v2 training. Their value is for Phase 3 (fMRI fine-tuning),
not Phase 1 (self-supervised).

| Dataset | Subjects | Hours/Subj | Stimuli | fMRI? |
|---------|----------|-----------|---------|-------|
| Algonauts 2025 (Courtois NeuroMod) | 4 | ~66h | Friends S1-7, movies | Yes |
| Lebel 2023 | 8 | 6-18h | 82 spoken stories | Yes |
| BOLD Moments (Lahner 2024) | 10 | ~6.2h | 1,000 3s video clips | Yes |
| Wen 2017 | 3 | ~11.7h | Video segments | Yes |

**Use these only in Phase 3. Don't waste self-supervised pre-training time on them.**

### Tier 4: Large-Scale Video (Use Sparingly)

| Dataset | Size | Why Not Tier 1 |
|---------|------|---------------|
| YouTube-8M | 8M videos | Only video-level labels, no temporal alignment |
| Kinetics-700 | 650K clips | Action-focused, less diverse than HowTo100M |
| WebVid-10M | 10M clips | Noisy, web-scraped, quality varies |

These are too large and noisy for the compute budget. HowTo100M is a better choice.

---

## 2. Data Selection Strategy

### Diversity Over Volume

Brain responses are driven by **semantic content**, not pixel-level variation.
10 videos of different activities > 100 videos of similar cooking scenes.

**Selection criteria:**
1. **Content diversity**: cover as many semantic categories as possible
2. **Temporal dynamics**: include fast (sports), slow (lectures), and mixed (movies)
3. **Audio diversity**: speech, music, environmental sounds, silence
4. **Visual diversity**: indoor, outdoor, close-up, wide-angle, animation

### Recommended Data Mix (for 1000h total)

```
Phase 1 Pre-Training Data (1000h)
├── HowTo100M subset          500h   (50%)  — diverse instructional
├── LibriSpeech                200h   (20%)  — clean speech+text (no video)
├── VGGSound                   150h   (15%)  — sound events + video
├── TED-LIUM                   100h   (10%)  — lectures
└── Domain-specific (posted/)   50h    (5%)  — your own content
```

### Why This Mix

- **50% HowTo100M**: broadest coverage of real-world multimodal scenes
- **20% LibriSpeech**: text backbone needs clean speech alignment; trains audio↔text
  pathway in 2-modality mode (video modality dropout)
- **15% VGGSound**: strong audio↔video correspondence (e.g., drums → person drumming)
- **10% TED-LIUM**: sustained narrative, builds long-range temporal understanding
- **5% Domain-specific**: ensures the model sees content similar to evaluation stimuli

---

## 3. Feature Extraction Cost

### Per-Dataset Cost (on T4 GPU)

| Dataset | Hours | Text Extract | Audio Extract | Video Extract | Total GPU-h |
|---------|-------|-------------|---------------|---------------|-------------|
| HowTo100M (500h) | 500 | 25h | 33h | 17h | ~75h |
| LibriSpeech (200h) | 200 | 10h | 13h | 0 (no video) | ~23h |
| VGGSound (150h) | 150 | 8h | 10h | 5h | ~23h |
| TED-LIUM (100h) | 100 | 5h | 7h | 3h | ~15h |
| Domain (50h) | 50 | 3h | 3h | 2h | ~8h |
| **Total** | **1000h** | **51h** | **66h** | **27h** | **~144h** |

**With 4 parallel T4s: ~36h wall time (~1.5 days)**

### Storage Requirements

| Component | Per Hour | Total (1000h) |
|-----------|---------|---------------|
| Text features (384-dim @ 2Hz) | ~5.5 MB | ~5.5 GB |
| Audio features (384-dim @ 2Hz) | ~5.5 MB | ~5.5 GB |
| Video features (640-dim @ 2Hz) | ~9.2 MB | ~9.2 GB |
| **Total** | **~20 MB/h** | **~20 GB** |

Very manageable. Fits on a single Google Drive account.

---

## 4. Data Quality Filters

### Automatic Filters

```python
def should_include(video_meta):
    """Filter out low-quality training samples."""

    # Minimum duration: 10 seconds
    if video_meta['duration_s'] < 10:
        return False

    # Must have at least 2 modalities with content
    n_active = sum([
        video_meta.get('has_speech', False),
        video_meta.get('has_meaningful_audio', False),
        video_meta.get('has_meaningful_video', False),
    ])
    if n_active < 2:
        return False

    # Audio quality: reject if SNR too low
    if video_meta.get('snr_db', 20) < 5:
        return False

    # Video quality: reject if mostly static (e.g., slideshow with voiceover)
    if video_meta.get('optical_flow_mean', 1.0) < 0.01:
        return False

    return True
```

### Content Diversity Sampling

```python
def sample_diverse_subset(all_videos, target_hours=500):
    """
    Sample a diverse subset using k-means on content embeddings.
    Ensures broad coverage of semantic space.
    """
    # Get content embedding per video (average of text features)
    embeddings = [v['text_features'].mean(0) for v in all_videos]

    # Cluster into N groups (N = target_hours / avg_video_length)
    n_clusters = target_hours * 3600 // 180  # ~180s avg video
    kmeans = KMeans(n_clusters=min(n_clusters, len(all_videos)))
    labels = kmeans.fit_predict(embeddings)

    # Sample 1 video per cluster (maximizes diversity)
    selected = []
    for i in range(kmeans.n_clusters):
        cluster_videos = [v for v, l in zip(all_videos, labels) if l == i]
        selected.append(random.choice(cluster_videos))

    return selected
```

---

## 5. Data Augmentation During Pre-Training

### Temporal Augmentations

| Augmentation | Implementation | Purpose |
|--------------|---------------|---------|
| Time warping | Resample features at 0.9-1.1x speed | Robustness to tempo changes |
| Random crop | Random start offset within segment | Positional invariance |
| Segment reversal | 10% chance, reverse a 5s window | Temporal understanding |

### Feature-Level Augmentations

| Augmentation | Implementation | Purpose |
|--------------|---------------|---------|
| Gaussian noise | σ=0.02 on backbone features | Robustness to feature noise |
| Feature dropout | Zero 5% of feature dimensions | Avoid over-reliance on specific dims |
| Modality delay | Shift one modality by ±1 timestep | Robustness to sync errors |

### Modality-Level Augmentations

| Augmentation | Implementation | Purpose |
|--------------|---------------|---------|
| Modality dropout | 30% during pre-training | Already in the masking strategy |
| Modality swap | 5% chance, swap audio from different sample | Hard negative for contrastive |
| Modality noise scale | 2x noise on one modality | Robustness to noisy inputs |

---

## 6. How to Download and Prepare

### HowTo100M

```bash
# Download video IDs and captions
git clone https://github.com/antoine77340/howto100m
cd howto100m

# Download a 500h subset using yt-dlp
# Filter for videos that still exist and have good audio
python download_subset.py --target-hours 500 --min-duration 30 --max-duration 600

# Extract features with tiny backbones
python extract_features.py \
    --video-dir ./videos \
    --output-dir ./features \
    --text-model sentence-transformers/all-MiniLM-L6-v2 \
    --audio-model openai/whisper-tiny \
    --video-model apple/mobilevit-small \
    --sample-rate 2  # 2 Hz feature extraction
```

### LibriSpeech

```bash
# Download (already hosted as direct downloads)
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-other-500.tar.gz

# Extract audio features (no video)
python extract_features.py \
    --audio-dir ./LibriSpeech \
    --output-dir ./features/librispeech \
    --text-from transcripts \
    --audio-model openai/whisper-tiny \
    --text-model sentence-transformers/all-MiniLM-L6-v2 \
    --no-video
```

### VGGSound

```bash
# Download from official source
# https://www.robots.ox.ac.uk/~vgg/data/vggsound/

python extract_features.py \
    --video-dir ./vggsound \
    --output-dir ./features/vggsound \
    --text-model sentence-transformers/all-MiniLM-L6-v2 \
    --audio-model openai/whisper-tiny \
    --video-model apple/mobilevit-small \
    --sample-rate 2
```

---

## 7. Data for Phase 0 (Teacher Cache)

Separate from the self-supervised data, we need a curated set to run
TRIBE v2 teacher on. This should be **high-quality and diverse**.

### Recommended Teacher Cache Videos (100h)

```
Teacher Cache Data (100h)
├── Algonauts stimuli       30h   — Has fMRI ground truth for validation
├── BOLD Moments stimuli     6h   — Short clips, diverse
├── Lebel story stimuli      5h   — Spoken narratives
├── Nature documentaries    15h   — Visual richness + narration
├── TED talks               15h   — Sustained speech + slides/gestures
├── Movie clips              15h  — Emotional, narrative-heavy
├── Music videos             5h   — Music + visual motion
├── Sports highlights        5h   — Fast motion, crowd noise
└── Silent/ambient           4h   — Edge case: visual only
```

**Total teacher GPU cost: ~50h on T4 (~$50-75)**

### Teacher Cache Format

```python
# Per video, save:
{
    'video_id': str,
    'duration_s': float,
    'predictions': tensor(T, 20484),       # vertex-level predictions
    'fusion_layer4': tensor(T, 1152),      # intermediate fusion features
    'fusion_layer6': tensor(T, 1152),      # intermediate fusion features
    'modality_projected': {
        'text': tensor(T, 384),
        'audio': tensor(T, 384),
        'video': tensor(T, 384),
    },
}
```

---

## 8. Summary: Data Budget

| Phase | Data Source | Hours | Teacher Needed? | GPU Cost |
|-------|-----------|-------|----------------|----------|
| Phase 0 | Curated diverse videos | 100h | Yes (one time) | ~50h T4 |
| Phase 1 | HowTo100M + LibriSpeech + VGGSound + TED | 1000h | No | ~160h T4 (extraction + training) |
| Phase 2 | Cached teacher predictions | 100h | No (cached) | ~5h T4 |
| Phase 3 | fMRI-paired datasets | 270h | No (cached) | ~10h T4 |
| **Total** | | **1470h** | **50h teacher** | **~225h T4** |

Compare current approach: ~500h teacher inference + ~270h data = much less diverse training.

**v3 uses 5x more data with 10x fewer teacher inferences.**
