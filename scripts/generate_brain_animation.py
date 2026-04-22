"""Generate brain activity animation from Caro Pasta TF! predictions."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv
pv.OFF_SCREEN = True

from tribev2.plotting import PlotBrain

preds = np.load("Caro_Pasta_TF!_preds.npy")  # (49, 20484)
print(f"Loaded: {preds.shape} — generating animation...")

plotter = PlotBrain(mesh="fsaverage5", bg_map="sulcal")

plotter.plot_timesteps_mp4(
    preds,
    filepath="results/visualizations/caro_pasta_brain_activity.mp4",
    views="left",
    cmap="hot",
    norm_percentile=99,
    suptitle=None,
    interpolated_fps=10,
)

print("Done! Saved to results/visualizations/caro_pasta_brain_activity.mp4")
