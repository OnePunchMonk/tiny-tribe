"""
TRIBE v2 inference on 8100 videos via Modal GPU.

Setup (one-time):
  1. modal token new
  2. modal secret create huggingface HF_TOKEN=<your_token>
  3. modal.com → Settings → Spending Limit → $30

Run:
  modal run modal_inference.py

Results: ./outputs_inference/<video_stem>/preds.npy
"""

import modal
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CLIPS_DIR    = Path("~/Downloads/clips").expanduser()
OUTPUT_DIR   = Path("./outputs_inference")
VOLUME_PATH  = "/outputs"
BUDGET_USD   = 5.0
T4_RATE      = 0.59 / 3600   # $/sec
BATCH_SIZE   = 50             # videos per container — amortises model load (~60s)

# ── Modal primitives ──────────────────────────────────────────────────────────

app = modal.App("tribe-v2-inference")

output_vol = modal.Volume.from_name("tribe-inference-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("numpy<2")
    .pip_install(
        "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
    )
)

# ── GPU worker ────────────────────────────────────────────────────────────────


@app.cls(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2400,   # 40 min: enough for 50 videos + model load
)
class TribeInference:
    @modal.enter()
    def load_model(self):
        import os
        from tribev2.demo_utils import TribeModel
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder="/hf_cache",
        )
        print("Model loaded on", next(self.model._model.parameters()).device)

    @modal.method()
    def process_batch(self, batch: list) -> list:
        """Process a batch of (video_stem, video_bytes) pairs.
        Returns one result dict per video.
        """
        import tempfile, os, time
        import numpy as np

        results = []
        for video_stem, video_bytes in batch:
            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                tmp_path = f.name
            try:
                df = self.model.get_events_dataframe(video_path=tmp_path)
                preds, _ = self.model.predict(events=df, verbose=False)

                out_dir = Path(VOLUME_PATH) / video_stem
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "preds.npy", preds)

                results.append({
                    "stem": video_stem,
                    "shape": list(preds.shape),
                    "gpu_seconds": time.perf_counter() - t0,
                    "ok": True,
                })
            except Exception as e:
                results.append({
                    "stem": video_stem,
                    "error": str(e),
                    "gpu_seconds": time.perf_counter() - t0,
                    "ok": False,
                })
            finally:
                os.unlink(tmp_path)

        output_vol.commit()
        return results


# ── Local entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main():
    import json

    videos = sorted(CLIPS_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} videos in {CLIPS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [v for v in videos if not (OUTPUT_DIR / v.stem / "preds.npy").exists()]
    print(f"Pending: {len(pending)}  |  Batches of {BATCH_SIZE}: {-(-len(pending)//BATCH_SIZE)}")

    if not pending:
        print("All videos already processed.")
        return

    # Split into batches
    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    def _batch_args():
        for batch in batches:
            yield [{"stem": v.stem, "bytes": v.read_bytes()} for v in batch]

    # starmap sends all batches to Modal — containers run in parallel
    # Each batch item is a list; unpack into (batch,) arg
    def _args():
        for batch in batches:
            yield [(v.stem, v.read_bytes()) for v in batch]

    inferencer = TribeInference()
    all_results = []
    total_gpu_seconds = 0.0
    budget_hit = False

    for batch_results in inferencer.process_batch.map(
        list(_args()), order_outputs=False
    ):
        for r in batch_results:
            total_gpu_seconds += r.get("gpu_seconds", 0)
            all_results.append(r)

        estimated_cost = total_gpu_seconds * T4_RATE
        print(f"  {len(all_results)}/{len(pending)} done  |  ~${estimated_cost:.2f} spent")

        if estimated_cost >= BUDGET_USD:
            print(f"\nBudget ${BUDGET_USD} reached — stopping. Re-run to resume.")
            budget_hit = True
            break

    ok     = [r for r in all_results if r.get("ok")]
    failed = [r for r in all_results if not r.get("ok")]
    print(f"\nDone: {len(ok)} ok  |  {len(failed)} failed  |  ~${total_gpu_seconds * T4_RATE:.2f} spent")

    if failed:
        print("Failed:")
        for r in failed:
            print(f"  {r['stem']}: {r['error']}")

    # Download from volume to local disk
    print("\nDownloading results from Modal volume...")
    for entry in output_vol.listdir("/"):
        stem = entry.path.rstrip("/")
        local_dir = OUTPUT_DIR / stem
        local_dir.mkdir(parents=True, exist_ok=True)
        preds_path = local_dir / "preds.npy"
        if not preds_path.exists():
            with open(preds_path, "wb") as f:
                for chunk in output_vol.read_file(f"{stem}/preds.npy"):
                    f.write(chunk)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({r["stem"]: r for r in all_results}, f, indent=2)

    print(f"\nResults in {OUTPUT_DIR}/  →  outputs_inference/<video_stem>/preds.npy")
