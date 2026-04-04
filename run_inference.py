"""Run TRIBE v2 inference on all videos in the posted/ directory."""

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

from tribev2.demo_utils import TribeModel

CACHE_FOLDER = Path("./cache")
VIDEO_DIR = Path("./posted")
OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading TRIBE v2 model...")
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_FOLDER,
    device="cpu",
    config_update={
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
    },
)

videos = sorted(VIDEO_DIR.glob("*.mp4"))
print(f"\nFound {len(videos)} videos:\n")
for v in videos:
    print(f"  - {v.name}")

results_summary = {}

for i, video_path in enumerate(videos):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(videos)}] Processing: {video_path.name}")
    print(f"{'='*60}")

    try:
        df = model.get_events_dataframe(video_path=str(video_path))
        preds, segments = model.predict(events=df)
        print(f"  Predictions shape: {preds.shape}")

        safe_name = video_path.stem.replace(" ", "_")
        np.save(OUTPUT_DIR / f"{safe_name}_preds.npy", preds)

        n_timesteps, n_vertices = preds.shape
        n_hemi = n_vertices // 2
        lh_preds = preds[:, :n_hemi]
        rh_preds = preds[:, n_hemi:]

        summary = {
            "video": video_path.name,
            "n_timesteps": int(n_timesteps),
            "n_vertices": int(n_vertices),
            "duration_seconds": int(n_timesteps),
            "overall_mean_activation": float(np.mean(preds)),
            "overall_std_activation": float(np.std(preds)),
            "overall_max_activation": float(np.max(preds)),
            "left_hemisphere_mean": float(np.mean(lh_preds)),
            "right_hemisphere_mean": float(np.mean(rh_preds)),
            "left_hemisphere_max": float(np.max(lh_preds)),
            "right_hemisphere_max": float(np.max(rh_preds)),
            "peak_timestep": int(np.argmax(np.mean(preds, axis=1))),
            "top_10_vertex_indices": np.argsort(np.mean(preds, axis=0))[-10:][::-1].tolist(),
            "temporal_mean_activation": np.mean(preds, axis=1).tolist(),
        }

        results_summary[video_path.name] = summary
        print(f"  Mean activation: {summary['overall_mean_activation']:.4f}")
        print(f"  Max activation: {summary['overall_max_activation']:.4f}")
        print(f"  LH mean: {summary['left_hemisphere_mean']:.4f} | RH mean: {summary['right_hemisphere_mean']:.4f}")
        print(f"  Peak activity at timestep: {summary['peak_timestep']}s")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results_summary[video_path.name] = {"error": str(e)}

with open(OUTPUT_DIR / "brain_mapping_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, res in results_summary.items():
    if "error" in res:
        print(f"  {name}: FAILED - {res['error']}")
    else:
        print(f"  {name}: {res['n_timesteps']}s, mean={res['overall_mean_activation']:.4f}, max={res['overall_max_activation']:.4f}")

print(f"\nResults saved to {OUTPUT_DIR}/")
