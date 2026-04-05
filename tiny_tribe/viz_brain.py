"""Brain surface visualization: TRIBE v2 (teacher) vs Tiny TRIBE v3 (student).

Uses tribev2's PlotBrain (PyVista backend) — same renderer as generate_paper_brain.py.
Produces side-by-side cortical heatmaps for 3 validation clips across all 5 TRs.

Output (all in checkpoints/plots/):
    brain_viz_clip{i}_{stem}_teacher_vs_student.png  — 4-view grid per clip
    brain_viz_clip{i}_{stem}_timesteps_teacher.png   — temporal sequence (teacher)
    brain_viz_clip{i}_{stem}_timesteps_student.png   — temporal sequence (student)

Usage:
    source venv/bin/activate
    python tiny_tribe/viz_brain.py \\
        --checkpoint checkpoints/best-epoch=052-val/pearson_r=0.7278.ckpt \\
        --features_dir ./features \\
        --distillation_dir ./distillation_dataset \\
        --save_dir checkpoints/plots
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# PyVista must be off-screen before import
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from tiny_tribe.train_lightning import TinyTribeKD, parcellate


# ── Reverse parcellation: 400 parcels → 20484 fsaverage5 vertices ─────────────

def unpack_parcels(parcel_vals: np.ndarray, n_vertices: int = 20484) -> np.ndarray:
    """Map (400,) → (20484,) by reversing uniform-chunk parcellation."""
    n_parcels = parcel_vals.shape[0]
    chunk = n_vertices // n_parcels
    out = np.zeros(n_vertices, dtype=np.float32)
    for i in range(n_parcels):
        out[i * chunk:(i + 1) * chunk] = parcel_vals[i]
    out[n_parcels * chunk:] = parcel_vals[-1]
    return out


# ── Load checkpoint and run val inference ─────────────────────────────────────

def get_val_predictions(ckpt_path, features_dir, n_clips=3, seed=42):
    from torch.utils.data import random_split
    device = torch.device("cpu")

    module = TinyTribeKD.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    n_parcels = module.hparams.n_vertices

    all_files = sorted(Path(features_dir).glob("*.pt"))
    n_val = max(1, int(len(all_files) * 0.20))
    n_train = len(all_files) - n_val
    gen = torch.Generator().manual_seed(seed)
    _, val_files = random_split(all_files, [n_train, n_val], generator=gen)
    val_files = list(val_files)[:n_clips]

    student_preds, teacher_preds_400, stems = [], [], []

    with torch.no_grad():
        for fpath in val_files:
            data = torch.load(fpath, map_location="cpu", weights_only=True)
            text  = data["text"].float().unsqueeze(0)
            audio = data["audio"].float().unsqueeze(0)
            video = data["video"].float().unsqueeze(0)
            subj  = torch.zeros(1, dtype=torch.long)

            out  = module.model(text, audio, video, subj)
            pred = out["prediction"].squeeze(0).permute(1, 0).numpy()  # (T, 400)

            teacher = data["teacher"].float().numpy()
            if teacher.shape[1] != n_parcels:
                teacher = parcellate(torch.from_numpy(teacher), n_parcels).numpy()

            student_preds.append(pred)
            teacher_preds_400.append(teacher)
            stems.append(Path(fpath).stem)

    return student_preds, teacher_preds_400, stems


def load_teacher_full_res(distillation_dir, stems):
    """Load raw (T, 20484) teacher preds from distillation_dataset/preds.npy."""
    out = []
    for stem in stems:
        p = Path(distillation_dir) / stem / "preds.npy"
        arr = np.load(str(p)).astype(np.float32) if p.exists() else np.zeros((5, 20484))
        if arr.shape[0] == 20484:
            arr = arr.T  # ensure (T, 20484)
        out.append(arr)
    return out


# ── Plotting using tribev2 PlotBrain (PyVista) ────────────────────────────────

def plot_4view_comparison(plotter, teacher_avg, student_avg, stem, save_dir, cmap="fire"):
    """4-view (left/right/medial_left/medial_right) side-by-side: teacher vs student.

    Mirrors generate_paper_brain.py Plot 1 style.
    """
    from tribev2.plotting.utils import plot_colorbar

    vmax = float(np.percentile(np.abs(teacher_avg), 99))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    views = ["left", "right", "medial_left", "medial_right"]

    sm = None
    for col, view in enumerate(views):
        sm = plotter.plot_surf(
            teacher_avg,
            axes=axes[0, col],
            views=view,
            cmap=cmap,
            norm_percentile=99,
        )
        plotter.plot_surf(
            student_avg,
            axes=axes[1, col],
            views=view,
            cmap=cmap,
            norm_percentile=99,
        )
        axes[0, col].set_title(view, fontsize=9)
        axes[1, col].set_title(view, fontsize=9)

    axes[0, 0].set_ylabel("TRIBE v2\n(teacher)", fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("Tiny TRIBE v3\n(student)", fontsize=10, labelpad=8)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    plot_colorbar(cbar_ax, sm=sm, label="Activation (norm.)")

    fig.suptitle(f"Average Activation — clip {stem}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(right=0.90, wspace=0.02, hspace=0.05)

    out = save_dir / f"brain_viz_{stem}_avg_teacher_vs_student.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [brain] Saved {out}")


def plot_timesteps_both(plotter, teacher_full, student_full, stem, save_dir,
                        cmap="fire", norm_percentile=99):
    """Temporal sequence (all TRs) for teacher and student separately.

    Uses tribev2's plot_timesteps — identical to generate_paper_brain.py Plot 3.
    """
    T = teacher_full.shape[0]

    # ── Teacher ───────────────────────────────────────────────────────────────
    fig_t = plotter.plot_timesteps(
        teacher_full,
        views="left",
        norm_percentile=norm_percentile,
        cmap=cmap,
    )
    fig_t.suptitle(f"TRIBE v2 — clip {stem} — all {T} TRs", fontsize=12, fontweight="bold")
    out_t = save_dir / f"brain_viz_{stem}_timesteps_teacher.png"
    fig_t.savefig(out_t, dpi=200, bbox_inches="tight")
    plt.close(fig_t)
    print(f"  [brain] Saved {out_t}")

    # ── Student ───────────────────────────────────────────────────────────────
    fig_s = plotter.plot_timesteps(
        student_full,
        views="left",
        norm_percentile=norm_percentile,
        cmap=cmap,
    )
    fig_s.suptitle(f"Tiny TRIBE v3 — clip {stem} — all {T} TRs", fontsize=12, fontweight="bold")
    out_s = save_dir / f"brain_viz_{stem}_timesteps_student.png"
    fig_s.savefig(out_s, dpi=200, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  [brain] Saved {out_s}")


def plot_signed_comparison(plotter, teacher_avg, student_avg, stem, save_dir):
    """Seismic (diverging) side-by-side — mirrors generate_paper_brain.py Plot 5."""
    from tribev2.plotting.utils import plot_colorbar

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    views = ["left", "right"]

    sm = None
    for col, view in enumerate(views):
        sm = plotter.plot_surf(
            teacher_avg,
            axes=axes[0, col],
            views=view,
            cmap="seismic",
            norm_percentile=99,
            symmetric_cbar=True,
        )
        plotter.plot_surf(
            student_avg,
            axes=axes[1, col],
            views=view,
            cmap="seismic",
            norm_percentile=99,
            symmetric_cbar=True,
        )
        axes[0, col].set_title(view, fontsize=9)
        axes[1, col].set_title(view, fontsize=9)

    axes[0, 0].set_ylabel("TRIBE v2 (teacher)", fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("Tiny TRIBE v3 (student)", fontsize=10, labelpad=8)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    plot_colorbar(cbar_ax, sm=sm, label="Signed Activation")

    fig.suptitle(f"Signed Activation (Seismic) — clip {stem}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(right=0.90, wspace=0.02, hspace=0.05)

    out = save_dir / f"brain_viz_{stem}_seismic_teacher_vs_student.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [brain] Saved {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--features_dir",     type=str, default="./features")
    parser.add_argument("--distillation_dir", type=str, default="./distillation_dataset")
    parser.add_argument("--save_dir",         type=str, default="./checkpoints/plots")
    parser.add_argument("--n_clips",          type=int, default=3)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Instantiate plotter once (loads fsaverage5 mesh) ─────────────────────
    from tribev2.plotting import PlotBrain
    print("Loading PlotBrain (PyVista backend)...")
    plotter = PlotBrain(mesh="fsaverage5")

    # ── Get val predictions ───────────────────────────────────────────────────
    print("Running student inference on val clips...")
    student_400, _, stems = get_val_predictions(
        args.checkpoint, args.features_dir, n_clips=args.n_clips, seed=args.seed
    )
    print(f"Loading full-res teacher preds (20484 vertices)...")
    teacher_20484 = load_teacher_full_res(args.distillation_dir, stems)

    # ── Per-clip plots ────────────────────────────────────────────────────────
    for i, (stem, s400, t20484) in enumerate(zip(stems, student_400, teacher_20484)):
        T = min(s400.shape[0], t20484.shape[0])
        print(f"\nClip {i+1}/{len(stems)}: {stem}  (T={T})")

        # Unpack student 400 parcels → 20484 vertices for all TRs
        student_full = np.stack([unpack_parcels(s400[t]) for t in range(T)])  # (T, 20484)
        teacher_full = t20484[:T]                                               # (T, 20484)

        # Average across TRs for the 4-view + seismic plots
        teacher_avg = teacher_full.mean(axis=0)   # (20484,)
        student_avg = student_full.mean(axis=0)

        # Plot 1: 4-view average activation (fire cmap) — teacher row vs student row
        plot_4view_comparison(plotter, teacher_avg, student_avg, stem, save_dir, cmap="fire")

        # Plot 2: Temporal sequence left-hemisphere (fire cmap)
        plot_timesteps_both(plotter, teacher_full, student_full, stem, save_dir,
                            cmap="fire", norm_percentile=99)

        # Plot 3: Signed seismic comparison
        plot_signed_comparison(plotter, teacher_avg, student_avg, stem, save_dir)

    print(f"\nAll brain plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
