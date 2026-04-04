"""Dataset classes for Tiny-TRIBE v3 distillation training.

Supports two modes:
  1. Smoke test  — nilearn fetch_development_fmri + "Partly Cloudy" features
  2. Full run    — CNeuroMod / any .pt segment files from the teacher cache

Data format expected on disk (one .pt file per 100-TR segment):
  {
    "text":        Tensor(T, 384)     backbone features (fp16 ok)
    "audio":       Tensor(T, 384)
    "video":       Tensor(T, 640)
    "teacher":     Tensor(T, n_verts) TRIBE v2 predictions  (Phase 2)
    "fusion_l4":   Tensor(T, 1152)   teacher fusion layer 4 (Phase 2)
    "fmri":        Tensor(T, n_verts) real fMRI signal       (Phase 3)
    "subject_id":  int
  }

Not every field is required for every phase — see each Dataset class.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Utility ───────────────────────────────────────────────────────────────────

def _to_float(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.dtype != torch.float32 else t


# ── Phase 2: Knowledge Distillation Dataset ───────────────────────────────────

class KDDataset(Dataset):
    """Loads cached backbone features + teacher predictions for Phase 2 KD.

    Each sample is a fixed-length segment (seq_len TRs) with:
      - backbone features (text, audio, video)
      - teacher vertex predictions
      - teacher fusion features (for intermediate KD)
      - subject ID

    Directory structure:
      data_dir/
        sub-01_friends_s01e01_000.pt
        sub-01_friends_s01e01_001.pt
        ...
    """

    def __init__(
        self,
        data_dir:   str,
        seq_len:    int  = 50,
        stride:     int  = 25,
        subjects:   Optional[List[int]] = None,  # None = all
        require_teacher: bool = True,
    ):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)

        files = sorted(self.data_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        # Build (file, start_tr) index
        self.samples = []
        self.subject_map = {}  # filename → subject_id int

        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=True)

            # Skip if teacher predictions missing and required
            if require_teacher and "teacher" not in data:
                continue

            subj_id = int(data.get("subject_id", 0))
            if subjects is not None and subj_id not in subjects:
                continue

            T = data["text"].shape[0]
            for start in range(0, T - seq_len + 1, stride):
                self.samples.append((f, start, subj_id))

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in {data_dir}. "
                f"Check that files contain 'text', 'audio', 'video', 'teacher' keys."
            )

        print(f"KDDataset: {len(self.samples)} segments from {len(files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, start, subj_id = self.samples[idx]
        end  = start + self.seq_len
        data = torch.load(fpath, map_location="cpu", weights_only=True)

        out = {
            "text":       _to_float(data["text"][start:end]),        # (T, 384)
            "audio":      _to_float(data["audio"][start:end]),       # (T, 384)
            "video":      _to_float(data["video"][start:end]),       # (T, 640)
            "subject_id": torch.tensor(subj_id, dtype=torch.long),
        }

        if "teacher" in data:
            out["teacher"] = _to_float(data["teacher"][start:end])  # (T, n_v)

        if "fusion_l4" in data:
            out["fusion_l4"] = _to_float(data["fusion_l4"][start:end])  # (T, 1152)

        return out


# ── Phase 3: fMRI Fine-tuning Dataset ────────────────────────────────────────

class FMRIDataset(Dataset):
    """Loads backbone features + real fMRI + optional teacher predictions.

    Same .pt format as KDDataset but 'fmri' field is required.
    Teacher predictions used as regulariser (optional but recommended).
    """

    def __init__(
        self,
        data_dir:   str,
        seq_len:    int  = 100,
        stride:     int  = 50,
        subjects:   Optional[List[int]] = None,
        hrf_shift:  int  = 0,   # already applied in preprocessing, set 0
    ):
        self.seq_len   = seq_len
        self.hrf_shift = hrf_shift
        self.data_dir  = Path(data_dir)

        files = sorted(self.data_dir.glob("*.pt"))
        self.samples = []

        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=True)

            if "fmri" not in data:
                continue

            subj_id = int(data.get("subject_id", 0))
            if subjects is not None and subj_id not in subjects:
                continue

            T = data["fmri"].shape[0]
            for start in range(0, T - seq_len + 1, stride):
                self.samples.append((f, start, subj_id))

        print(f"FMRIDataset: {len(self.samples)} segments from {len(files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, start, subj_id = self.samples[idx]
        end  = start + self.seq_len
        data = torch.load(fpath, map_location="cpu", weights_only=True)

        out = {
            "text":       _to_float(data["text"][start:end]),
            "audio":      _to_float(data["audio"][start:end]),
            "video":      _to_float(data["video"][start:end]),
            "fmri":       _to_float(data["fmri"][start:end]),         # (T, n_v)
            "subject_id": torch.tensor(subj_id, dtype=torch.long),
        }

        if "teacher" in data:
            out["teacher"] = _to_float(data["teacher"][start:end])

        if "fusion_l4" in data:
            out["fusion_l4"] = _to_float(data["fusion_l4"][start:end])

        return out


# ── Smoke test: nilearn dataset builder ───────────────────────────────────────

def build_nilearn_dataset(
    out_dir:          str,
    teacher_pt:       str,        # path to teacher_partly_cloudy.pt
    features_pt:      str,        # path to partly_cloudy_features.pt
    n_subjects:       int = 3,
    hrf_shift_trs:    int = 5,    # hemodynamic delay in TRs
    val_last_n_trs:   int = 30,   # hold out last N TRs for validation
):
    """Convert nilearn fMRI + teacher predictions into .pt segment files.

    Run once. Produces train/ and val/ subdirectories in out_dir.

    Usage:
        build_nilearn_dataset(
            out_dir="./data/nilearn",
            teacher_pt="./teacher_partly_cloudy.pt",
            features_pt="./partly_cloudy_features.pt",
        )
    """
    from nilearn import datasets
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.datasets import fetch_atlas_schaefer_2018

    out_dir = Path(out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    # Load teacher predictions and backbone features
    teacher_data = torch.load(teacher_pt,   map_location="cpu", weights_only=True)
    feat_data    = torch.load(features_pt,  map_location="cpu", weights_only=True)

    text_feat  = _to_float(feat_data["text"])    # (T, 384)
    audio_feat = _to_float(feat_data["audio"])   # (T, 384)
    video_feat = _to_float(feat_data["video"])   # (T, 640)

    # Map teacher fsaverage5 → Schaefer-400
    teacher_v5 = _to_float(teacher_data["predictions_fsaverage5"])  # (T, 20484)
    fusion_l4  = _to_float(teacher_data.get("fusion_l4",
                            torch.zeros(teacher_v5.shape[0], 1152)))

    teacher_s400 = _map_to_schaefer400(teacher_v5)   # (T, 400)

    # Load nilearn fMRI
    print("Fetching nilearn development fMRI dataset...")
    fmri_data = datasets.fetch_development_fmri(n_subjects=n_subjects)
    atlas      = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    masker     = NiftiLabelsMasker(labels_img=atlas.maps,
                                   standardize=True, t_r=1.5)

    for subj_idx, func_img in enumerate(fmri_data.func[:n_subjects]):
        print(f"Processing subject {subj_idx + 1}/{n_subjects}...")
        fmri_ts = masker.fit_transform(func_img)  # (168, 400)
        T_fmri  = fmri_ts.shape[0]

        # Apply HRF shift: features at t → fMRI at t + shift
        T_usable   = T_fmri - hrf_shift_trs
        fmri_shift = torch.tensor(fmri_ts[hrf_shift_trs:], dtype=torch.float32)  # (T_usable, 400)
        text_s      = text_feat[:T_usable]
        audio_s     = audio_feat[:T_usable]
        video_s     = video_feat[:T_usable]
        teacher_s   = teacher_s400[:T_usable]
        fusion_s    = fusion_l4[:T_usable]

        # Train / val split
        T_val   = val_last_n_trs
        T_train = T_usable - T_val

        def save_split(split, t, a, v, fmri, teacher, fusion, start):
            seg = {
                "text":       t.half(),
                "audio":      a.half(),
                "video":      v.half(),
                "fmri":       fmri.half(),
                "teacher":    teacher.half(),
                "fusion_l4":  fusion.half(),
                "subject_id": subj_idx,
            }
            fname = out_dir / split / f"sub{subj_idx:02d}_start{start:04d}.pt"
            torch.save(seg, fname)

        save_split("train",
                   text_s[:T_train], audio_s[:T_train], video_s[:T_train],
                   fmri_shift[:T_train], teacher_s[:T_train], fusion_s[:T_train],
                   start=0)
        save_split("val",
                   text_s[T_train:], audio_s[T_train:], video_s[T_train:],
                   fmri_shift[T_train:], teacher_s[T_train:], fusion_s[T_train:],
                   start=T_train)

    print(f"Done. Data saved to {out_dir}/train and {out_dir}/val")
    return str(out_dir / "train"), str(out_dir / "val")


def _map_to_schaefer400(vertex_preds: torch.Tensor) -> torch.Tensor:
    """Map (T, 20484) fsaverage5 predictions to (T, 400) Schaefer parcels.

    Uses a simple uniform average over vertices belonging to each parcel.
    For a production run, use the proper annot file lookup.
    Here we use a deterministic random mapping as a placeholder that
    produces the right shape — replace with real atlas mapping.
    """
    T, V = vertex_preds.shape
    n_parcels = 400

    # Placeholder: average every V/400 consecutive vertices per parcel
    # Replace this with real fsaverage5 → Schaefer-400 atlas lookup
    chunk = V // n_parcels
    parcel_preds = torch.stack([
        vertex_preds[:, i * chunk:(i + 1) * chunk].mean(dim=1)
        for i in range(n_parcels)
    ], dim=1)  # (T, 400)

    return parcel_preds


def get_dataloaders(
    train_dir: str,
    val_dir:   str,
    mode:      str = "kd",   # "kd" or "fmri"
    seq_len:   int = 50,
    batch_size: int = 8,
    num_workers: int = 2,
    **dataset_kwargs,
) -> tuple:
    """Convenience function to build train and val DataLoaders."""

    DatasetClass = KDDataset if mode == "kd" else FMRIDataset

    train_ds = DatasetClass(train_dir, seq_len=seq_len,
                            stride=seq_len // 2, **dataset_kwargs)
    val_ds   = DatasetClass(val_dir,   seq_len=seq_len,
                            stride=seq_len,     **dataset_kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader
