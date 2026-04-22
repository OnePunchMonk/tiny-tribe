"""Generate brain visualizations from Caro Pasta TF! predictions."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv

pv.OFF_SCREEN = True

from tribev2.plotting import PlotBrain

PRED_PATH = Path("Caro_Pasta_TF!_preds.npy")
JSON_PATH = Path("brain_mapping_results.json")
OUT_DIR = Path("results/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

preds = np.load(PRED_PATH)
with open(JSON_PATH) as f:
    meta = json.load(f)["Caro Pasta TF!.mp4"]

print(f"Predictions: {preds.shape}, peak t={meta['peak_timestep']}")

plotter = PlotBrain(mesh="fsaverage5", bg_map="sulcal")

# 1. Peak timestep heatmap
print("[1/4] Peak heatmap...")
peak = preds[meta["peak_timestep"]]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax in axes:
    ax.set_axis_off()
plotter.plot_surf(peak, axes=[axes[0]], views="left", cmap="hot", norm_percentile=99)
plotter.plot_surf(peak, axes=[axes[1]], views="right", cmap="hot", norm_percentile=99)
plotter.plot_surf(peak, axes=[axes[2]], views="medial_left", cmap="hot", norm_percentile=99)
plotter.plot_surf(peak, axes=[axes[3]], views="medial_right", cmap="hot", norm_percentile=99)
fig.suptitle(f"Caro Pasta TF! — Peak Activity (t={meta['peak_timestep']}s)", fontsize=16, y=0.98)
fig.savefig(OUT_DIR / "peak_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'peak_heatmap.png'}")

# 2. Mean activation
print("[2/4] Mean activation...")
mean_act = preds.mean(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes:
    ax.set_axis_off()
plotter.plot_surf(mean_act, axes=[axes[0]], views="left", cmap="hot", norm_percentile=99)
plotter.plot_surf(mean_act, axes=[axes[1]], views="right", cmap="hot", norm_percentile=99)
fig.suptitle("Caro Pasta TF! — Mean Activity (49s)", fontsize=16, y=0.98)
fig.savefig(OUT_DIR / "mean_activation.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'mean_activation.png'}")

# 3. Temporal grid
print("[3/4] Temporal grid...")
step = 4
timesteps = list(range(0, preds.shape[0], step))
n_cols = len(timesteps)
fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
for row in axes:
    for ax in row:
        ax.set_axis_off()
for i, t in enumerate(timesteps):
    frame = preds[t]
    plotter.plot_surf(frame, axes=[axes[0, i]], views="left", cmap="hot", norm_percentile=99)
    plotter.plot_surf(frame, axes=[axes[1, i]], views="right", cmap="hot", norm_percentile=99)
    axes[0, i].set_title(f"t={t}s", fontsize=10)
fig.suptitle("Caro Pasta TF! — Activity Over Time", fontsize=16, y=1.02)
fig.savefig(OUT_DIR / "temporal_grid.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'temporal_grid.png'}")

# 4. Timeline plot
print("[4/4] Timeline...")
fig, ax = plt.subplots(figsize=(14, 4))
lh_mean = preds[:, :10242].mean(axis=1)
rh_mean = preds[:, 10242:].mean(axis=1)
overall_max = preds.max(axis=1)
t = np.arange(preds.shape[0])
ax.plot(t, lh_mean, label="Left Hemi (mean)", color="#E74C3C", linewidth=1.5)
ax.plot(t, rh_mean, label="Right Hemi (mean)", color="#3498DB", linewidth=1.5)
ax.plot(t, overall_max, label="Max activation", color="#2ECC71", linewidth=1, alpha=0.7)
ax.axvline(meta["peak_timestep"], color="gray", linestyle="--", alpha=0.7, label=f"Peak (t={meta['peak_timestep']}s)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Predicted Activation")
ax.set_title("Caro Pasta TF! — Activation Timeline")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig(OUT_DIR / "activation_timeline.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'activation_timeline.png'}")

print(f"\nDone! All outputs in {OUT_DIR}/")
