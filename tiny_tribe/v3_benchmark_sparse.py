"""Benchmark: Dense vs Sparse architectures for Tiny-TRIBE v3.

Compares accuracy, FLOPs, memory, and training speed across all
architecture modes on synthetic data.  Useful for picking the best
sparsity recipe for your GPU budget.

Usage:
    python v3_benchmark_sparse.py --architectures dense mot spark hetero full \
        --n_vertices 50 --seq_len 20 --epochs 5 --batch_size 4
"""

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tiny_tribe.v3_sparse import TinyTribeV3Sparse
from tiny_tribe.v3_pipeline import SyntheticDataset
from tiny_tribe.v3_train import DistillationLoss, evaluate


def benchmark_architecture(name: str, args) -> dict:
    """Train one architecture for a few epochs and collect metrics."""
    device = torch.device(args.device)

    # Build model
    kw = {"num_layers": args.num_layers, "num_experts": args.num_experts}
    if name in ("spark", "full"):
        kw["spark_k_ratio"] = args.spark_k_ratio

    model = TinyTribeV3Sparse(
        architecture=name,
        n_vertices=args.n_vertices,
        n_subjects=args.n_subjects,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_seq_len=args.seq_len * 3,
        **kw,
    ).to(device)

    param_info = model.count_params()
    flops_info = model.estimate_flops(batch_size=args.batch_size, seq_len=args.seq_len)

    # Datasets
    train_ds = SyntheticDataset(
        n_samples=args.n_train, seq_len=args.seq_len,
        n_vertices=args.n_vertices, n_subjects=args.n_subjects,
        mode="kd", seed=42,
    )
    val_ds = SyntheticDataset(
        n_samples=args.n_val, seq_len=args.seq_len,
        n_vertices=args.n_vertices, n_subjects=args.n_subjects,
        mode="kd", seed=99,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DistillationLoss()

    # Warmup + timing
    model.train()
    for _ in range(2):
        batch = next(iter(train_loader))
        out = model(
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["video"].to(device),
            batch["subject_id"].to(device),
        )
        pred = out["prediction"]                    # (B, n_v, T)
        teacher_t = batch["teacher"].to(device).transpose(1, 2)  # (B, n_v, T)
        T_min = min(pred.shape[2], teacher_t.shape[2])
        pred = pred[:, :, :T_min]
        teacher_t = teacher_t[:, :, :T_min]
        loss = criterion(pred, teacher_t, out["aux_loss"])
        loss["total"].backward()
        opt.zero_grad()

    # Training loop
    epoch_times = []
    epoch_losses = []
    epoch_val_rs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0

        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            teacher = batch["teacher"].to(device)
            subj = batch["subject_id"].to(device)

            opt.zero_grad()
            out = model(text, audio, video, subj)

            # Align shapes: model output is (B, n_v, T), teacher is (B, T, n_v)
            pred = out["prediction"]
            teacher_t = teacher.transpose(1, 2)  # (B, n_v, T)
            T_min = min(pred.shape[2], teacher_t.shape[2])
            pred = pred[:, :, :T_min]
            teacher_t = teacher_t[:, :, :T_min]

            losses = criterion(pred, teacher_t, out["aux_loss"])
            losses["total"].backward()
            opt.step()

            train_loss += losses["total"].item()
            n += 1

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        epoch_losses.append(train_loss / max(n, 1))

        # Validation
        val_metrics = evaluate(model, val_loader, device, mode="kd")
        epoch_val_rs.append(val_metrics["pearson_r"])

    # Peak memory
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_mem = 0.0

    return {
        "architecture": name,
        "total_params_M": param_info["total"] / 1e6,
        "active_params_M": param_info.get("active_per_forward", param_info["total"]) / 1e6,
        "flops_per_forward_M": flops_info["total"] / 1e6,
        "final_train_loss": epoch_losses[-1],
        "final_val_r": epoch_val_rs[-1],
        "best_val_r": max(epoch_val_rs),
        "mean_epoch_time_s": sum(epoch_times) / len(epoch_times),
        "peak_memory_MB": peak_mem,
        "param_info": {k: int(v) for k, v in param_info.items()},
        "epoch_history": [
            {"epoch": i + 1, "loss": l, "val_r": r, "time": t}
            for i, (l, r, t) in enumerate(zip(epoch_losses, epoch_val_rs, epoch_times))
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--architectures", nargs="+", default=["dense", "mot", "spark", "hetero", "full"])
    p.add_argument("--n_vertices", type=int, default=50)
    p.add_argument("--n_subjects", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--ff_mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--n_val", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--spark_k_ratio", type=float, default=0.15)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default="benchmark_results.json")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Benchmarking: {args.architectures}\n")

    results = []
    for name in args.architectures:
        print(f"\n{'='*60}")
        print(f"  Architecture: {name}")
        print(f"{'='*60}")
        try:
            result = benchmark_architecture(name, args)
            results.append(result)
            print(f"  Total params:  {result['total_params_M']:.2f}M")
            print(f"  Active params: {result['active_params_M']:.2f}M")
            print(f"  FLOPs/forward: {result['flops_per_forward_M']:.1f}M")
            print(f"  Final loss:    {result['final_train_loss']:.4f}")
            print(f"  Best val r:    {result['best_val_r']:.4f}")
            print(f"  Avg epoch:     {result['mean_epoch_time_s']:.2f}s")
            if device.type == "cuda":
                print(f"  Peak memory:   {result['peak_memory_MB']:.1f}MB")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Arch':>10} | {'Total':>7} | {'Active':>7} | {'FLOPs':>9} | "
          f"{'Best Val r':>10} | {'Epoch(s)':>8} | {'Mem(MB)':>8}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['architecture']:>10} | {r['total_params_M']:>6.2f}M | "
              f"{r['active_params_M']:>6.2f}M | {r['flops_per_forward_M']:>8.1f}M | "
              f"{r['best_val_r']:>10.4f} | {r['mean_epoch_time_s']:>7.2f}s | "
              f"{r['peak_memory_MB']:>7.1f}")
    print(f"{'='*80}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
