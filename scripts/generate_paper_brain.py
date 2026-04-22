"""
Generate paper-quality brain visualizations using tribev2 PlotBrain (PyVista backend).

Usage:
    source /Users/avaya.aggarwal@zomato.com/tribe/venv/bin/activate
    cd /Users/avaya.aggarwal@zomato.com/tribe
    python generate_paper_brain.py
"""

import os
import sys

# Ensure tribev2 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tribev2"))

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

# Must set pyvista to off-screen before importing it
os.environ["PYVISTA_OFF_SCREEN"] = "true"

from tribev2.plotting import PlotBrain
from tribev2.plotting.utils import plot_colorbar

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_PATH = os.path.join(OUT_DIR, "Caro_Pasta_TF!_preds.npy")

# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------
preds = np.load(PREDS_PATH)
print(f"Predictions shape: {preds.shape}  dtype: {preds.dtype}")
print(f"  min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}")

assert preds.ndim == 2 and preds.shape[1] == 20484, (
    f"Expected (n_timesteps, 20484), got {preds.shape}"
)

# ---------------------------------------------------------------------------
# Instantiate plotter (loads fsaverage5 mesh once)
# ---------------------------------------------------------------------------
plotter = PlotBrain(mesh="fsaverage5")

# ---------------------------------------------------------------------------
# Plot 1: Average activation across all timesteps -- 4 canonical views
# ---------------------------------------------------------------------------
avg = preds.mean(axis=0)  # (20484,)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
views_grid = [
    ("left", axes[0, 0]),
    ("right", axes[0, 1]),
    ("medial_left", axes[1, 0]),
    ("medial_right", axes[1, 1]),
]
sm = None
for view, ax in views_grid:
    sm = plotter.plot_surf(avg, axes=ax, views=view, cmap="fire", norm_percentile=99)

# Add a single colorbar using the ScalarMappable returned by plot_surf
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
plot_colorbar(cbar_ax, sm=sm, label="Activation (norm.)")

fig.suptitle("Average Brain Activation \u2014 Caro Pasta TF!", fontsize=14, fontweight="bold")
fig.subplots_adjust(right=0.90, wspace=0.05, hspace=0.05)
out1 = os.path.join(OUT_DIR, "brain_avg_activation.png")
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Peak timestep (highest mean absolute activation)
# ---------------------------------------------------------------------------
peak_t = int(np.argmax(np.abs(preds).mean(axis=1)))
peak = preds[peak_t]
print(f"Peak timestep: t={peak_t}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sm = plotter.plot_surf(peak, axes=axes[0], views="left", cmap="fire", norm_percentile=99)
plotter.plot_surf(peak, axes=axes[1], views="right", cmap="fire", norm_percentile=99)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
plot_colorbar(cbar_ax, sm=sm, label=f"Activation (t={peak_t})")

fig.suptitle(f"Peak Brain Activation (t={peak_t})", fontsize=14, fontweight="bold")
fig.subplots_adjust(right=0.90, wspace=0.05)
out2 = os.path.join(OUT_DIR, "brain_peak_activation.png")
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")

# ---------------------------------------------------------------------------
# Plot 3: Temporal sequence via plot_timesteps
# ---------------------------------------------------------------------------
n_show = min(10, preds.shape[0])
step = max(1, preds.shape[0] // n_show)
# plot_timesteps expects (n_timesteps, n_vertices) 2-D array
subset = preds[::step][:n_show]
print(f"Temporal sequence: {n_show} frames, step={step}, subset shape={subset.shape}")

fig = plotter.plot_timesteps(
    subset,
    views="left",
    norm_percentile=99,
    cmap="fire",
)
fig.suptitle("Brain Activation Over Time", fontsize=14, fontweight="bold")
out3 = os.path.join(OUT_DIR, "brain_temporal_sequence.png")
fig.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out3}")

# ---------------------------------------------------------------------------
# Plot 4: Fire colormap with alpha transparency (fading low activations)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sm = plotter.plot_surf(
    avg, axes=axes[0], views="left", cmap="fire",
    norm_percentile=99, vmin=0.5, alpha_cmap=(0, 0.2),
)
plotter.plot_surf(
    avg, axes=axes[1], views="right", cmap="fire",
    norm_percentile=99, vmin=0.5, alpha_cmap=(0, 0.2),
)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
plot_colorbar(cbar_ax, sm=sm, label="Activation")

fig.suptitle("Brain Activation (Fire Colormap with Alpha)", fontsize=14, fontweight="bold")
fig.subplots_adjust(right=0.90, wspace=0.05)
out4 = os.path.join(OUT_DIR, "brain_fire_colormap.png")
fig.savefig(out4, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out4}")

# ---------------------------------------------------------------------------
# Plot 5 (bonus): Diverging colormap (seismic) for signed activations
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sm = plotter.plot_surf(
    avg, axes=axes[0], views="left", cmap="seismic",
    norm_percentile=99, symmetric_cbar=True,
)
plotter.plot_surf(
    avg, axes=axes[1], views="right", cmap="seismic",
    norm_percentile=99, symmetric_cbar=True,
)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
plot_colorbar(cbar_ax, sm=sm, label="Activation")

fig.suptitle("Brain Activation (Signed, Seismic Colormap)", fontsize=14, fontweight="bold")
fig.subplots_adjust(right=0.90, wspace=0.05)
out5 = os.path.join(OUT_DIR, "brain_seismic_colormap.png")
fig.savefig(out5, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out5}")

print("\nAll done! Generated 5 brain visualization PNGs.")
