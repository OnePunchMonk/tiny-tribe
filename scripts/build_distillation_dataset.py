"""
Build distillation_dataset/ from outputs_inference/ and clips/.

Structure:
  distillation_dataset/
    <video_stem>/
      <video_stem>.mp4
      preds.npy
"""

import shutil
from pathlib import Path

CLIPS_DIR   = Path("~/Downloads/clips").expanduser()
INFER_DIR   = Path("./outputs_inference")
OUTPUT_DIR  = Path("./distillation_dataset")

stems = [p.name for p in INFER_DIR.iterdir() if (p / "preds.npy").exists()]
print(f"Found {len(stems)} processed videos")

ok, missing = 0, []

for stem in sorted(stems):
    dst = OUTPUT_DIR / stem
    dst.mkdir(parents=True, exist_ok=True)

    # copy preds.npy
    shutil.copy2(INFER_DIR / stem / "preds.npy", dst / "preds.npy")

    # copy mp4
    mp4_src = CLIPS_DIR / f"{stem}.mp4"
    if mp4_src.exists():
        shutil.copy2(mp4_src, dst / f"{stem}.mp4")
        ok += 1
    else:
        missing.append(stem)

print(f"Done: {ok} complete  |  {len(missing)} missing mp4s")
if missing:
    print("Missing mp4s:", missing[:10])
