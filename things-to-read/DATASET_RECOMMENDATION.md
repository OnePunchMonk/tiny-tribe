# Which Dataset to Use — Concrete Recommendation

Three questions determine the answer: what phase, what do you have access to now,
and what is the goal (sprint vs full run).

---

## TL;DR — Use These Four, In This Order

```
1. CNeuroMod (Algonauts 2025)     fMRI + video  Teacher inference + Phase 3
2. BOLD Moments                   fMRI + video  Validation + Phase 3 supplement
3. HowTo100M (200h subset)        video only    Phase 1 self-supervised
4. LibriSpeech                    audio + text  Phase 1 self-supervised
```

Everything else is optional. These four cover all three training phases.

---

## Dataset 1: CNeuroMod / Algonauts 2025 — The Core Dataset

```
┌──────────────────────────────────────────────────────────────────────┐
│  What it is                                                          │
│  4 subjects watched Friends S1-7 + 3 movies in an MRI scanner.      │
│  ~66 hours of fMRI per subject. Whole brain, 2mm, fsaverage5.        │
│                                                                      │
│  What you get                                                        │
│  - Raw videos: Friends episodes + Raiders + Forrest Gump             │
│  - Preprocessed fMRI: surface-projected, HRF-shifted, clean          │
│  - Word-level transcripts with timestamps                            │
│  - Everything TRIBE v2 was trained on                                │
│                                                                      │
│  Why it's the right choice                                           │
│  - The teacher model (TRIBE v2) was trained on this data             │
│    → teacher predictions are maximally informative on this content   │
│  - Has ground-truth fMRI for Phase 3 fine-tuning                    │
│  - Friends is the best-studied naturalistic fMRI stimulus in the     │
│    world — decades of neuroscience research validate it              │
│  - 264h is far more fMRI data than any other open dataset           │
│                                                                      │
│  How to get it                                                       │
│  1. Go to cneuromod.ca                                               │
│  2. Sign the data sharing agreement (academic use, free, ~1 day)    │
│  3. Download via DataLad:                                            │
│       datalad install https://github.com/courtois-neuromod/cneuromod │
│       datalad get cneuromod/movie10/                                 │
│       datalad get cneuromod/friends/                                 │
│  4. Videos: Friends episodes are NOT distributed (copyright).        │
│     You source these yourself (the dataset provides fMRI only).      │
│     The episode filenames + timestamps are in the dataset metadata.  │
│                                                                      │
│  What to run teacher on                                              │
│  For 2-day sprint (5h budget):  Friends S01E01-S01E06  (~2h video)  │
│                                 + 1 movie clip (~30min)             │
│                                 + 2h diverse content (see below)    │
│  For full run (30h budget):     Friends S01-S06 all episodes (~48h) │
│                                 Use all 4 subjects' viewing sessions │
│                                                                      │
│  Train / val split                                                   │
│  Train: Friends S01-S06                                              │
│  Val:   Friends S07  (held out — used in Algonauts 2025 test set)   │
│  Test:  BOLD Moments (out-of-domain)                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dataset 2: BOLD Moments — Validation + Supplement

```
┌──────────────────────────────────────────────────────────────────────┐
│  What it is                                                          │
│  10 subjects watched 1,000 unique 3-second video clips,             │
│  10 repetitions each. Clips from MIT Moments in Time dataset.        │
│  ~62h total. TR=1.75s. Whole brain.                                  │
│                                                                      │
│  Why use it                                                          │
│  - 10 repetitions per clip → very accurate noise ceiling            │
│    Noise ceiling tells you: how predictable is this fMRI signal?    │
│    If your model hits 80% of noise ceiling, it's doing very well.   │
│  - 10 subjects (more than CNeuroMod) → better average-subject model │
│  - Out-of-domain from Friends → tests real generalisation           │
│  - Clips are only 3s → tests fast visual responses                  │
│                                                                      │
│  How to get it                                                       │
│  OSF: osf.io/2h3fq (openly available, no agreement required)        │
│  Videos: the MIT Moments in Time dataset provides the source clips  │
│  moments.csail.mit.edu — download the subset used in BOLD Moments   │
│                                                                      │
│  Use for                                                             │
│  - Validation during Phase 3 (not training — too little data)       │
│  - Computing noise ceiling to contextualise Pearson r scores        │
│  - Teacher inference: 6h video → 3h GPU on Modal ($1.50 of credits) │
│    → Gives diverse short-clip teacher predictions for Phase 2       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dataset 3: HowTo100M (200h subset) — Phase 1 Self-Supervised

```
┌──────────────────────────────────────────────────────────────────────┐
│  What it is                                                          │
│  1.2M instructional YouTube videos with ASR transcripts.            │
│  Cooking, home repair, crafts, sport tutorials, etc.                │
│  No fMRI. No labels. Just video + audio + ASR text.                 │
│                                                                      │
│  Why use it for Phase 1                                              │
│  - Maximum content diversity: 23,000 different activity categories  │
│  - Strong audio-visual correspondence: person explains what they do │
│  - ASR transcripts give word-level text alignment                   │
│  - Instructional content has clear cause-effect temporal structure  │
│    → ideal for next-TR prediction task                              │
│  - 200h subset still covers enormous semantic breadth               │
│                                                                      │
│  How to get it                                                       │
│  Paper: arxiv.org/abs/1906.03327                                    │
│  CSV of YouTube IDs + timestamps: github.com/antoine77340/howto100m │
│  Download with yt-dlp:                                              │
│    yt-dlp --batch-file video_ids.txt --format "bestvideo+bestaudio" │
│    (some videos will be unavailable — expect 60-70% success rate)   │
│                                                                      │
│  Practical subset strategy                                           │
│  Don't download 200h randomly. Sample for diversity:                │
│    - Pick 20 videos from each of 10 categories (cooking, repair,    │
│      gardening, sport, music, art, science, language, tech, nature) │
│    - 200 videos × ~1h avg = ~200h                                   │
│    - Run tiny backbone feature extraction (not teacher inference)   │
│    - Takes ~20h Kaggle GPU for all features                         │
│                                                                      │
│  For 2-day sprint: skip this entirely. Only needed for Phase 1.     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dataset 4: LibriSpeech — Audio + Text Pathway

```
┌──────────────────────────────────────────────────────────────────────┐
│  What it is                                                          │
│  1,000 hours of read English audiobooks with transcripts.           │
│  Clean studio-quality speech. No video.                             │
│                                                                      │
│  Why use it                                                          │
│  - Strengthens the audio↔text pathway in the fusion model           │
│  - Whisper-Tiny was trained on this data → features are great       │
│  - 960h gives massive text+audio coverage for masked modality tasks  │
│  - During training: set video modality dropout = 1.0                │
│    (forces model to handle audio+text only — matches real use case  │
│     when video is unavailable or low quality)                       │
│                                                                      │
│  How to get it                                                       │
│  Direct download (no signup):                                       │
│    openslr.org/12                                                    │
│    train-clean-360.tar.gz  (360h, clean speech, ~23GB)             │
│    train-other-500.tar.gz  (500h, varied, ~30GB)                   │
│                                                                      │
│  Feature extraction cost:                                            │
│    audio + text only (no video frames)                              │
│    ~96h Kaggle GPU for 960h of audio                               │
│    But: run in background over 3-4 Kaggle weeks, automated          │
│                                                                      │
│  For 2-day sprint: skip this entirely.                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What to Use For Each Phase

### 2-Day Sprint

```
Phase 0 — Teacher inference (5h budget, Modal):
  Priority 1: CNeuroMod Friends S01E01-S01E06   ~2h video
  Priority 2: BOLD Moments clips (diverse)       ~1h video
  Priority 3: Fill remaining 2h with:
    - 30min nature documentary (YouTube free)
    - 30min TED talk (ted.com, free download)
    - 30min sports highlight
    - 30min podcast/interview

  WHY THIS MIX:
    Friends:       matches the fMRI fine-tuning data exactly
    BOLD Moments:  diverse 3s clips, covers semantic breadth
    Nature/TED:    adds sustained narration + clean audio
    Sports:        fast motion, non-speech audio variety

Phase 2 — KD training:
  Same 5h of cached teacher predictions above

Phase 3 — fMRI fine-tuning:
  CNeuroMod Friends S01, subject 1 only (~15h fMRI)
  Val: Friends S01E10-E13 (hold out last 3 episodes)

Validation:
  BOLD Moments test clips (102 clips, pre-defined test set)
```

### Full Strategy C Run

```
Phase 1 — Self-supervised (no teacher, no fMRI):
  HowTo100M 200h subset
  LibriSpeech 960h
  VGGSound 200h (osf download, audiovisual events)
  → Total: ~1360h

Phase 2 — KD:
  CNeuroMod Friends S01-S06 teacher predictions (Kaggle accumulation)
  BOLD Moments teacher predictions
  → Target: 30h diverse predictions

Phase 3 — fMRI fine-tuning:
  CNeuroMod all 4 subjects, Friends S01-S06
  Lebel 2023 (82 spoken stories, 8 subjects) — audio+text pathway
  → Total: ~270h fMRI

Validation:
  CNeuroMod Friends S07 (primary)
  BOLD Moments test set (secondary, out-of-domain)
```

---

## What to Download Right Now (Today)

In order of priority:

```
┌────┬──────────────────────────────┬──────────┬────────────────────────────────┐
│ #  │ Dataset                      │ Size     │ Command / URL                  │
├────┼──────────────────────────────┼──────────┼────────────────────────────────┤
│ 1  │ CNeuroMod data agreement     │ N/A      │ cneuromod.ca → sign agreement  │
│    │ (takes ~1 day to process)    │          │ Do this FIRST, it gates fMRI   │
├────┼──────────────────────────────┼──────────┼────────────────────────────────┤
│ 2  │ BOLD Moments videos          │ ~50GB    │ moments.csail.mit.edu          │
│    │ (no agreement needed)        │          │ + osf.io/2h3fq for fMRI        │
├────┼──────────────────────────────┼──────────┼────────────────────────────────┤
│ 3  │ LibriSpeech train-clean-360  │ 23GB     │ openslr.org/12                 │
│    │ (immediate, no signup)       │          │ wget https://openslr.org/      │
│    │                              │          │  resources/12/                 │
│    │                              │          │  train-clean-360.tar.gz        │
├────┼──────────────────────────────┼──────────┼────────────────────────────────┤
│ 4  │ HowTo100M 200-video subset   │ ~20GB    │ github.com/antoine77340/       │
│    │ (download via yt-dlp)        │          │  howto100m → sample CSV        │
│    │                              │          │ yt-dlp --batch-file ids.txt    │
├────┼──────────────────────────────┼──────────┼────────────────────────────────┤
│ 5  │ Friends episodes             │ ~50GB    │ Source yourself (copyright)    │
│    │ (S01-S07, needed for fMRI)   │          │ Must match CNeuroMod metadata  │
└────┴──────────────────────────────┴──────────┴────────────────────────────────┘

For the 2-day sprint you need: #2 (BOLD Moments videos) + Friends S01 videos
Everything else can wait.

Total download for 2-day sprint: ~10-15GB
Total download for full run:     ~150GB
```

---

## Why Not These Other Datasets

```
Kinetics-700        Action classification labels. No temporal narrative.
                    Videos are 10s clips — too short for fMRI temporal structure.
                    Skip.

WebVid-10M          Noisy web-scraped stock footage. Heavily watermarked.
                    Poor audio quality. Not worth the 10M download.
                    Skip.

AudioSet            Audio-only event labels. No transcripts, no video narrative.
                    Useful only for audio backbone fine-tuning specifically.
                    Low priority — LibriSpeech is better for this pipeline.

VGGSound            200K 10s clips, audio + video. Good audio-visual pairing.
                    Useful but not essential if HowTo100M + LibriSpeech already included.
                    Add it if you have storage (35GB).

Wen 2017            3 subjects, ~12h fMRI, video stimuli. Small.
                    Use if you want more fMRI subjects in Phase 3.
                    Not worth prioritising over CNeuroMod (264h vs 35h).

Lebel 2023          82 spoken stories, 8 subjects, audio+text only (no video).
                    Very good for language cortex predictions specifically.
                    Add in full run Phase 3 (audio+text with video dropout=1.0).
                    Skip for 2-day sprint.
```
