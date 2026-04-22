"""Coreset selection on extracted feature files.

Selects the most informative and diverse subset of clips from features/
using k-Center greedy selection on per-clip video+audio feature vectors.
Run this after extract_features_v3.py and before train_lightning.py.

Usage:
    # Select 15% of 8100 clips (≈1215 clips)
    python select_coreset.py --features_dir ./features --ratio 0.15

    # Select exactly 800 clips
    python select_coreset.py --features_dir ./features --k 800

    # Use herding instead of k-center (better for regression tasks)
    python select_coreset.py --features_dir ./features --ratio 0.15 --method herding

Output:
    coreset.txt  — one .pt file path per line (selected clips)

Then train on coreset:
    python tiny_tribe/train_lightning.py --features_dir ./features \\
        --coreset_file ./coreset.txt

Algorithm:
    k-Center: greedily pick the clip farthest from the current selected set.
    Guarantees uniform coverage of feature space. O(n*k) in time, O(n) in space.

    Herding: picks clips closest to the running mean prototype.
    Better for preserving marginal distributions (good for regression targets).

Feature representation per clip:
    video_feat (T, 640) → mean over T → (640,)
    audio_feat (T, 384) → mean over T → (384,)
    concat → (1024,) L2-normalised
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ── Feature loading ───────────────────────────────────────────────────────────

def load_features(features_dir: Path, verbose: bool = True) -> tuple[np.ndarray, list[Path]]:
    """Load all .pt files and return (feature_matrix, file_list).

    feature_matrix: (N, 1024) float32, L2-normalised
    file_list:      N paths in same order as rows
    """
    pt_files = sorted(features_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: No .pt files found in {features_dir}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Loading features from {len(pt_files)} clips in {features_dir} ...")

    feats = []
    valid_files = []

    for i, path in enumerate(pt_files):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            video = data["video"].float()   # (T, 640)
            audio = data["audio"].float()   # (T, 384)

            # Mean-pool over time → clip-level descriptor
            v = video.mean(0).numpy()   # (640,)
            a = audio.mean(0).numpy()   # (384,)
            clip_feat = np.concatenate([v, a])  # (1024,)
            feats.append(clip_feat)
            valid_files.append(path)
        except Exception as e:
            print(f"  WARNING: skipping {path.name}: {e}", file=sys.stderr)

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(pt_files)} loaded")

    X = np.stack(feats, axis=0).astype(np.float32)  # (N, 1024)

    # L2 normalise so Euclidean distance ≈ angular distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = X / norms

    if verbose:
        print(f"Feature matrix: {X.shape}  (L2-normalised)")

    return X, valid_files


# ── Selectors ─────────────────────────────────────────────────────────────────

def kcenter(X: np.ndarray, k: int, seed: int = 42) -> list[int]:
    """Greedy k-Center: maximise minimum distance to nearest selected point.

    Returns list of k indices into X.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)

    first = int(rng.integers(n))
    selected = [first]
    # Distance of each point to its nearest selected centre
    dists = np.linalg.norm(X - X[first], axis=1)  # (N,)

    for step in range(k - 1):
        farthest = int(np.argmax(dists))
        selected.append(farthest)
        new_dists = np.linalg.norm(X - X[farthest], axis=1)
        dists = np.minimum(dists, new_dists)

        if (step + 1) % 100 == 0:
            print(f"  k-Center: {step + 1}/{k - 1} centres selected  "
                  f"(max-min-dist={dists.max():.4f})")

    return selected


def herding(X: np.ndarray, k: int) -> list[int]:
    """Herding: iteratively select the point closest to the residual mean.

    Preserves marginal distribution of features — good for regression.
    Returns list of k indices into X.
    """
    n = X.shape[0]
    k = min(k, n)

    selected = []
    remaining_mean = X.mean(axis=0).copy()   # prototype to match
    cumulative_sum = np.zeros(X.shape[1])

    remaining = set(range(n))

    for step in range(k):
        target = (step + 1) * remaining_mean - cumulative_sum
        # Find unselected point closest to target
        idx_list = list(remaining)
        dists = np.linalg.norm(X[idx_list] - target, axis=1)
        best_local = int(np.argmin(dists))
        best = idx_list[best_local]

        selected.append(best)
        cumulative_sum += X[best]
        remaining.discard(best)

        if (step + 1) % 100 == 0:
            print(f"  Herding: {step + 1}/{k} selected")

    return selected


# ── Diagnostics ───────────────────────────────────────────────────────────────

def coverage_stats(X_all: np.ndarray, X_coreset: np.ndarray) -> dict:
    """For each point in X_all, find its distance to nearest coreset point."""
    # Compute pairwise distances in chunks to avoid OOM
    n_all = X_all.shape[0]
    min_dists = np.full(n_all, np.inf)
    chunk = 256
    for i in range(0, X_all.shape[0], chunk):
        Xi = X_all[i:i + chunk]
        d = np.linalg.norm(Xi[:, None, :] - X_coreset[None, :, :], axis=2)  # (chunk, k)
        min_dists[i:i + chunk] = d.min(axis=1)

    return {
        "max_coverage_dist":  float(min_dists.max()),
        "mean_coverage_dist": float(min_dists.mean()),
        "median_coverage_dist": float(np.median(min_dists)),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Coreset selection for distillation training")
    p.add_argument("--features_dir", type=str, default="./features",
                   help="Directory of .pt feature files (from extract_features_v3.py)")
    p.add_argument("--output",       type=str, default="./coreset.txt",
                   help="Output file — one selected .pt path per line")
    p.add_argument("--ratio",        type=float, default=None,
                   help="Fraction of clips to keep (e.g. 0.15 = 15%%)")
    p.add_argument("--k",            type=int,   default=None,
                   help="Exact number of clips to select (overrides --ratio)")
    p.add_argument("--method",       type=str,   default="kcenter",
                   choices=["kcenter", "herding"],
                   help="Selection algorithm (default: kcenter)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--diagnostics",  action="store_true",
                   help="Compute coverage stats after selection (slower)")
    return p.parse_args()


def main():
    args = parse_args()

    features_dir = Path(args.features_dir)
    output_path  = Path(args.output)

    # ── Load ──────────────────────────────────────────────────────────────────
    X, files = load_features(features_dir)
    n = len(files)

    # ── Resolve k ─────────────────────────────────────────────────────────────
    if args.k is not None:
        k = args.k
    elif args.ratio is not None:
        k = max(1, int(n * args.ratio))
    else:
        print("ERROR: specify --ratio or --k", file=sys.stderr)
        sys.exit(1)

    print(f"\nSelecting {k} / {n} clips ({k/n:.1%}) using {args.method} ...")

    # ── Select ────────────────────────────────────────────────────────────────
    if args.method == "kcenter":
        selected_idx = kcenter(X, k, seed=args.seed)
    else:
        selected_idx = herding(X, k)

    selected_files = [files[i] for i in selected_idx]

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if args.diagnostics:
        print("\nComputing coverage stats ...")
        X_coreset = X[selected_idx]
        stats = coverage_stats(X, X_coreset)
        print(f"  Max coverage distance:    {stats['max_coverage_dist']:.4f}")
        print(f"  Mean coverage distance:   {stats['mean_coverage_dist']:.4f}")
        print(f"  Median coverage distance: {stats['median_coverage_dist']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in selected_files:
            f.write(str(p.resolve()) + "\n")

    print(f"\nCoreset saved → {output_path}  ({len(selected_files)} clips)")
    print(f"Next step:")
    print(f"  python tiny_tribe/train_lightning.py \\")
    print(f"      --features_dir {features_dir} \\")
    print(f"      --coreset_file {output_path}")


if __name__ == "__main__":
    main()
