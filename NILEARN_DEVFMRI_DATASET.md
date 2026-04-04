# Using nilearn fetch_development_fmri for Distillation

## What This Dataset Actually Is

```python
from nilearn import datasets
data = datasets.fetch_development_fmri(n_subjects=3)
```

```
Dataset:   Richardson et al. 2018 — Development fMRI
Stimulus:  "Partly Cloudy" — Pixar short film (~6 minutes)
Subjects:  155 total (children + adults). fetch() returns n_subjects.
TRs:       168 per subject (~252s at TR=1.5s)
Space:     MNI volumetric → parcellated to Schaefer-400 by nilearn
Output:    (168, 400) per subject — 168 timepoints, 400 parcels
fMRI:      Real BOLD signal, real subjects, real naturalistic movie watching
```

**The critical fact:** The stimulus is "Partly Cloudy" by Pixar.
It is publicly available on YouTube (~6 minutes).
This means you CAN run TRIBE v2 on the actual movie and get real teacher predictions.

---

## What You Can and Can't Do With It

```
┌─────────────────────────────────────────────────────────────────────┐
│  CAN DO                                                             │
│                                                                     │
│  ✓ Run TRIBE v2 on "Partly Cloudy" video (6 min → 3 GPU-minutes)  │
│    Cost on Modal: ~$0.02 of credits. Essentially free.             │
│                                                                     │
│  ✓ Map TRIBE v2 fsaverage5 predictions → Schaefer-400              │
│    (pre-computed atlas mapping, one numpy operation)               │
│                                                                     │
│  ✓ Use real fMRI (168 TRs × 3 subjects) as Phase 3 ground truth   │
│                                                                     │
│  ✓ Validate the full pipeline end-to-end in < 1 hour              │
│                                                                     │
│  ✓ Use as the smoke test dataset before touching CNeuroMod         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CAN'T DO                                                           │
│                                                                     │
│  ✗ Train a production model on it — 504 TRs total is too small     │
│    (3 subjects × 168 TRs = 504 samples)                            │
│                                                                     │
│  ✗ Predict at vertex resolution — data is already parcellated      │
│    (400 Schaefer parcels, not 20,484 vertices or 5,124 fsaverage4) │
│                                                                     │
│  ✗ Separate children from adults cleanly without metadata          │
│    (mixed developmental stages in the default fetch)               │
│                                                                     │
│  ✗ Use "proxy" input features — you must use the real movie         │
│    Proxy features (random/sine waves) teach the model nothing       │
│    real about brain-stimulus mapping                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Right Way to Use It: Smoke Test + Bootstrap

### Step 1: Get the actual stimulus video (5 min)

```bash
# Download "Partly Cloudy" from YouTube
pip install yt-dlp
yt-dlp "https://www.youtube.com/watch?v=ImfhfFrC_YE" \
  -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  -o "partly_cloudy.mp4"

# Trim to exact stimulus duration used in the dataset
# Richardson et al. used 168 TRs × 1.5s = 252s = 4min 12s
ffmpeg -i partly_cloudy.mp4 -t 252 -c copy partly_cloudy_stimulus.mp4
```

### Step 2: Run TRIBE v2 on it (3 GPU-minutes on Modal)

```python
# modal_smoke_test.py
import modal

app = modal.App("tribe-smoke-test")
image = modal.Image.debian_slim().pip_install("tribev2", "torch")

@app.function(gpu="T4", timeout=600,
              volumes={"/cache": modal.Volume.from_name("tribe-cache",
                                                        create_if_missing=True)})
def get_teacher_predictions(video_path: str):
    import os, torch
    from tribev2 import TribeModel

    os.environ["HF_HOME"] = "/cache/hf"
    model = TribeModel.from_pretrained("facebook/tribev2")
    model = model.cuda().eval()

    # Hook fusion layers for feature KD
    acts = {}
    model.encoder.layers[3].register_forward_hook(
        lambda m, i, o: acts.update({"l4": o.detach().cpu().half()})
    )

    with torch.inference_mode():
        preds, _ = model.predict(video_path)
    # preds: (168, 20484) — one prediction per TR

    return {
        "predictions_fsaverage5": preds.cpu().half(),  # (168, 20484)
        "fusion_l4": acts["l4"],                        # (168, 1152)
    }

@app.local_entrypoint()
def main():
    import torch
    result = get_teacher_predictions.remote("partly_cloudy_stimulus.mp4")
    torch.save(result, "teacher_partly_cloudy.pt")
    print(f"Predictions shape: {result['predictions_fsaverage5'].shape}")
    # Expected: torch.Size([168, 20484])
    # Cost: ~3 GPU-minutes = ~$0.10 on T4
```

### Step 3: Map teacher predictions to Schaefer-400

```python
import numpy as np
import torch
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker

# Load the Schaefer-400 atlas (same one nilearn uses internally)
from nilearn.datasets import fetch_atlas_schaefer_2018
atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)

# Load pre-computed fsaverage5 → Schaefer-400 mapping
# (TRIBE v2 outputs fsaverage5 vertices — map to parcels)
# Option A: use pre-computed vertex-to-parcel lookup
#   Download from: github.com/ThomasYeoLab/CBIG
#   File: Schaefer2018_400Parcels_7Networks_order.annot (fsaverage5)

from nilearn.surface import load_surf_data
annot_lh = load_surf_data("lh.Schaefer400.annot")  # (10242,) parcel labels
annot_rh = load_surf_data("rh.Schaefer400.annot")  # (10242,) parcel labels
labels = np.concatenate([annot_lh, annot_rh])       # (20484,) vertex→parcel map

def fsaverage5_to_schaefer400(vertex_preds):
    """
    vertex_preds: (T, 20484) — TRIBE v2 output
    returns:      (T, 400)   — Schaefer-400 parcel averages
    """
    T = vertex_preds.shape[0]
    parcel_preds = np.zeros((T, 400))
    for p in range(1, 401):  # parcels 1-400
        mask = labels == p
        if mask.sum() > 0:
            parcel_preds[:, p-1] = vertex_preds[:, mask].mean(axis=1)
    return parcel_preds

# Apply mapping
teacher_preds_v5 = torch.load("teacher_partly_cloudy.pt")["predictions_fsaverage5"]
teacher_preds_s400 = fsaverage5_to_schaefer400(teacher_preds_v5.float().numpy())
# Shape: (168, 400) — matches nilearn fMRI output exactly
```

### Step 4: Load nilearn fMRI and train

```python
import torch
import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from torch.utils.data import Dataset, DataLoader

# Load fMRI data
fmri_data = datasets.fetch_development_fmri(n_subjects=3)
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

fmri_timeseries = []
for func_img in fmri_data.func:
    ts = masker.fit_transform(func_img)  # (168, 400)
    fmri_timeseries.append(ts)

# Now you have:
# fmri_timeseries: list of 3 arrays, each (168, 400) — real fMRI
# teacher_preds_s400: (168, 400) — TRIBE v2 predictions on same stimulus

# The fMRI has a 5-TR hemodynamic delay — shift teacher preds to align
HRF_SHIFT = 5  # TRs
teacher_aligned = teacher_preds_s400[:-HRF_SHIFT]          # (163, 400)
fmri_aligned    = [ts[HRF_SHIFT:] for ts in fmri_timeseries]  # each (163, 400)

# Training dataset
class SmokeFMRIDataset(Dataset):
    def __init__(self, fmri_list, teacher_preds, seq_len=20):
        self.fmri = fmri_list        # list of (163, 400)
        self.teacher = teacher_preds # (163, 400)
        self.seq_len = seq_len
        self.samples = []
        for subj_idx, ts in enumerate(fmri_list):
            T = ts.shape[0]
            for start in range(0, T - seq_len, seq_len // 2):
                self.samples.append((subj_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_idx, start = self.samples[idx]
        end = start + self.seq_len
        return {
            "fmri":    torch.tensor(self.fmri[subj_idx][start:end], dtype=torch.float32),
            "teacher": torch.tensor(self.teacher[start:end],         dtype=torch.float32),
            "subject": torch.tensor(subj_idx, dtype=torch.long),
        }

dataset = SmokeFMRIDataset(fmri_aligned, teacher_aligned, seq_len=20)
loader  = DataLoader(dataset, batch_size=4, shuffle=True)

# Student model with Schaefer-400 output
from tiny_tribe.moe_model import TinyTribeMoE

model = TinyTribeMoE(
    n_vertices=400,    # Schaefer-400, not fsaverage
    n_subjects=3,
    low_rank_dim=128,  # smaller bottleneck for tiny dataset
)

# But: you don't have backbone features yet —
# the model expects (B, T, 384) text, (B, T, 384) audio, (B, T, 640) video
# SEE NEXT SECTION for how to handle this
```

---

## The Proxy Features Problem — And How to Fix It

The notebook you referenced uses proxy features because it doesn't have the movie.
**Do not use proxy features for distillation.** Here's why and the fix:

```
PROXY FEATURES (bad):
  Random vectors or sine waves as text/audio/video input
  → The model learns to map noise → brain activity
  → The teacher predictions are computed from real video
  → Proxy features and teacher predictions are UNCORRELATED
  → The student learns nothing useful
  → Pearson r will be ~0 or random

REAL FEATURES (correct):
  1. Download "Partly Cloudy" (5 min)
  2. Run tiny backbones on it: MiniLM + Whisper-Tiny + MobileViT-S
  3. These 67M frozen models are free to run — no teacher inference
  4. Result: (168, 384) text features, (168, 384) audio, (168, 640) video
  5. These are correlated with the teacher predictions (both come from the same video)
  6. Now the student can learn the mapping
```

```python
# extract_partly_cloudy_features.py
# Run this on Kaggle (free, fast — tiny models only)

from sentence_transformers import SentenceTransformer
import whisper
import torch
from transformers import AutoFeatureExtractor, MobileViTModel
import cv2
import numpy as np

VIDEO_PATH = "partly_cloudy_stimulus.mp4"
FEATURE_RATE = 2  # Hz — match TRIBE v2's feature extraction rate

# ── Text features (MiniLM) ───────────────────────────────────────────
# Step 1: transcribe with WhisperX to get word timestamps
import whisperx
model_wx = whisperx.load_model("tiny", device="cuda")
result = model_wx.transcribe(VIDEO_PATH)
aligned = whisperx.align(result["segments"], ...)

# Step 2: for each 0.5s window (2Hz), encode the current sentence
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
T = 168  # number of TRs
text_features = []
for t in range(T):
    t_sec = t * 1.5  # TR = 1.5s
    words_in_window = get_words_at_time(aligned, t_sec, window=2.0)
    sentence = " ".join(words_in_window) if words_in_window else "[silence]"
    emb = text_model.encode(sentence)  # (384,)
    text_features.append(emb)
text_features = np.stack(text_features)  # (168, 384)

# ── Audio features (Whisper-Tiny encoder) ────────────────────────────
import librosa
audio, sr = librosa.load(VIDEO_PATH, sr=16000)
# Process in 30s chunks, extract encoder features at 2Hz
audio_model = whisper.load_model("tiny")
audio_features = extract_whisper_features_at_2hz(audio, audio_model)  # (168, 384)

# ── Video features (MobileViT-S) ─────────────────────────────────────
feat_extractor = AutoFeatureExtractor.from_pretrained("apple/mobilevit-small")
video_model = MobileViTModel.from_pretrained("apple/mobilevit-small").cuda().eval()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
video_features = []
for t in range(T):
    frame_idx = int(t * 1.5 * fps)  # frame at this TR
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = feat_extractor(images=frame_rgb, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = video_model(**inputs)
    feat = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # (640,)
    video_features.append(feat)
video_features = np.stack(video_features)  # (168, 640)

# Save
torch.save({
    "text":  torch.tensor(text_features,  dtype=torch.float16),  # (168, 384)
    "audio": torch.tensor(audio_features, dtype=torch.float16),  # (168, 384)
    "video": torch.tensor(video_features, dtype=torch.float16),  # (168, 640)
}, "partly_cloudy_features.pt")
# Total time: ~5 min on Kaggle T4
# Total size: ~0.5MB
```

---

## What This Dataset Is Good For In the Pipeline

```
┌──────────────────────┬────────────────────────────────────────────────┐
│ Use                  │ Details                                        │
├──────────────────────┼────────────────────────────────────────────────┤
│ Smoke test           │ Best use. Validates entire pipeline            │
│                      │ in <1h: inference → features → train → eval   │
├──────────────────────┼────────────────────────────────────────────────┤
│ Phase 2 KD seed      │ Usable as one of many videos in the teacher    │
│                      │ prediction cache. 6 min = 0.1h contribution   │
│                      │ (tiny — pair with other videos)               │
├──────────────────────┼────────────────────────────────────────────────┤
│ Phase 3 fMRI         │ Can use but is very small (504 TRs total).    │
│                      │ Supplements CNeuroMod, doesn't replace it.    │
│                      │ Value: 3 extra subjects, different age range  │
├──────────────────────┼────────────────────────────────────────────────┤
│ Schaefer-400 target  │ Good match — lets you validate at parcel      │
│                      │ resolution before moving to fsaverage         │
├──────────────────────┼────────────────────────────────────────────────┤
│ Primary training     │ Too small. 504 TRs will overfit in 1 epoch.  │
│                      │ Use CNeuroMod (640K TRs) for this.           │
└──────────────────────┴────────────────────────────────────────────────┘
```

---

## Recommended: Use It As Your Day-0 Smoke Test

```
COMPLETE SMOKE TEST — runs in under 2 hours, costs ~$0.10

H+0:00  Download "Partly Cloudy" video (5 min)
H+0:05  Modal: run TRIBE v2 on it ($0.10, ~3 GPU-minutes)
H+0:10  Kaggle: extract tiny backbone features (5 min GPU)
H+0:20  Kaggle: nilearn downloads fMRI automatically (no agreement needed)
H+0:25  Kaggle: map teacher predictions to Schaefer-400
H+0:30  Kaggle: train student model for 5 epochs on 3 subjects
H+1:30  Evaluate: compute Pearson r on held-out TRs

Expected result:
  If pipeline is correct: Pearson r > 0.10  (above chance on 400 parcels)
  If pipeline has a bug:  Pearson r ≈ 0.00  (tells you what to fix)

This costs ~$0.10 and tells you whether your entire setup is working
before spending any real compute on teacher inference or training.
```

---

## Bottom Line

```
fetch_development_fmri:  ✓ use it, but use the REAL movie as input
                         ✓ perfect smoke test dataset
                         ✓ real fMRI, real subjects, Schaefer-400 output
                         ✗ too small for production training
                         ✗ do NOT use proxy features

"Partly Cloudy" video:   6 min, free on YouTube, $0.10 teacher inference
nilearn fMRI:            automatic download, no agreement, works immediately
Feature extraction:      5 min on Kaggle with tiny backbones

This is your Day 0 validation run. Do it before anything else.
```