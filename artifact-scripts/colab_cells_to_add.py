"""
Cells to add to Colab notebook when connection is restored.
Copy-paste these into new code cells.
"""

# ═══════════════════════════════════════════════════════════════════
# CELL: Download Moments in Time Mini for teacher inference
# Add BEFORE Phase 0 (teacher inference)
# ═══════════════════════════════════════════════════════════════════

CELL_DOWNLOAD_DATASET = """
# Download Moments in Time Mini — 100K diverse 3-second video clips with audio (9.4 GB)
# This is the teacher inference dataset. Direct download, no auth needed.

import os
from pathlib import Path

DATASET_DIR = Path('/content/mit_mini')

if not DATASET_DIR.exists():
    print("Downloading Moments in Time Mini (9.4 GB)...")
    !wget -q --show-progress http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip -O /content/mit_mini.zip
    !unzip -q /content/mit_mini.zip -d /content/mit_mini
    !rm /content/mit_mini.zip
    print("Done!")
else:
    print("Dataset already downloaded.")

# Count videos
video_exts = {'.mp4', '.avi', '.webm', '.mkv'}
all_videos = [f for f in DATASET_DIR.rglob('*') if f.suffix.lower() in video_exts]
print(f"Found {len(all_videos)} videos")

# For teacher inference, sample a manageable subset (e.g., 500-2000 videos)
# More = better distillation data, but slower teacher inference
import random
random.seed(42)
SAMPLE_SIZE = 1000  # Adjust: 500 for quick test, 2000+ for production
sampled_videos = random.sample(all_videos, min(SAMPLE_SIZE, len(all_videos)))

# Copy sampled videos to working directory
os.makedirs('videos', exist_ok=True)
for v in sampled_videos:
    dst = Path('videos') / v.name
    if not dst.exists():
        os.symlink(v, dst)

print(f"Sampled {len(sampled_videos)} videos for teacher inference")
est_hours = len(sampled_videos) * 3 / 3600  # 3 seconds each
est_inference_min = est_hours * 60 / 0.5  # ~0.5 real-time on T4
print(f"Estimated teacher inference time: ~{est_inference_min:.0f} minutes on T4")
"""


# ═══════════════════════════════════════════════════════════════════
# CELL: Brain visualization using tribev2's PlotBrain
# Add AFTER the training visualization cell
# ═══════════════════════════════════════════════════════════════════

CELL_BRAIN_VIZ = """
# Brain Visualization — using TRIBE v2's high-quality PlotBrain
# PyVista-based rendering with proper fsaverage mesh, sulcal shading, colormaps

from tribev2.plotting import PlotBrain

plotter = PlotBrain(mesh='fsaverage5')

# Get predictions from trained model
model.eval()
batch = next(iter(val_loader))
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.no_grad():
    out = model(batch['text_feat'], batch['audio_feat'],
                batch['video_feat'], batch['subject_id'])

student_preds = out['prediction'][0].cpu().numpy()    # (V, T)
teacher_preds = batch['teacher_pred'][0].cpu().numpy() # (V, T)

student_avg = student_preds.mean(axis=1)
teacher_avg = teacher_preds.mean(axis=1)

print(f"Student: {student_avg.shape}, [{student_avg.min():.3f}, {student_avg.max():.3f}]")
print(f"Teacher: {teacher_avg.shape}, [{teacher_avg.min():.3f}, {teacher_avg.max():.3f}]")

# ── Teacher vs Student comparison ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

plotter.plot_surf(teacher_avg, views='left', axes=axes[0,0],
                  cmap='cold_hot', norm_percentile=99,
                  colorbar=True, colorbar_title='Teacher')
plotter.plot_surf(teacher_avg, views='right', axes=axes[0,1],
                  cmap='cold_hot', norm_percentile=99, colorbar=False)
axes[0,0].set_title('Teacher — Left', fontsize=12)
axes[0,1].set_title('Teacher — Right', fontsize=12)

plotter.plot_surf(student_avg, views='left', axes=axes[1,0],
                  cmap='cold_hot', norm_percentile=99,
                  colorbar=True, colorbar_title='Student')
plotter.plot_surf(student_avg, views='right', axes=axes[1,1],
                  cmap='cold_hot', norm_percentile=99, colorbar=False)
axes[1,0].set_title('Student — Left', fontsize=12)
axes[1,1].set_title('Student — Right', fontsize=12)

plt.suptitle('Teacher vs Student Brain Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(DRIVE_ROOT / 'brain_comparison.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── Temporal sequence ──
n_show = min(8, student_preds.shape[1])
fig = plotter.plot_timesteps(
    student_preds[:, :n_show].T,
    cmap='cold_hot', norm_percentile=99, views='left',
)
fig.suptitle('Student Predictions Across Time', fontsize=14, fontweight='bold')
fig.savefig(str(DRIVE_ROOT / 'brain_temporal.png'), dpi=150, bbox_inches='tight')
fig.show()

# ── Difference map (where does student diverge from teacher?) ──
diff = student_avg - teacher_avg
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plotter.plot_surf(diff, views='left', axes=axes[0],
                  cmap='seismic', symmetric_cbar=True, norm_percentile=95,
                  colorbar=True, colorbar_title='Student - Teacher')
plotter.plot_surf(diff, views='right', axes=axes[1],
                  cmap='seismic', symmetric_cbar=True, norm_percentile=95,
                  colorbar=False)
axes[0].set_title('Difference — Left', fontsize=12)
axes[1].set_title('Difference — Right', fontsize=12)
plt.suptitle('Student - Teacher Difference Map', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(DRIVE_ROOT / 'brain_difference.png'), dpi=200, bbox_inches='tight')
plt.show()

from scipy import stats
r, _ = stats.pearsonr(student_avg, teacher_avg)
print(f"\\nPearson r (student vs teacher): {r:.4f}")
print(f"Saved: brain_comparison.png, brain_temporal.png, brain_difference.png")
"""

print("Cells ready. Copy-paste from this file when Colab reconnects.")
print(f"\nCELL_DOWNLOAD_DATASET: Moments in Time Mini (9.4 GB, 100K clips)")
print(f"CELL_BRAIN_VIZ: High-quality brain visualization using tribev2.plotting.PlotBrain")
