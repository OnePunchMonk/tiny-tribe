"""Unified end-to-end Tiny-TRIBE v3 training pipeline.

This script orchestrates all three phases:
  Phase 1: Self-supervised pre-training (no teacher, no fMRI)
  Phase 2: Knowledge distillation (frozen backbones, teacher targets)
  Phase 3: fMRI fine-tuning (real brain data, teacher as regularizer)

Usage:
    # Run all phases end-to-end
    python v3_pipeline.py \
        --phase1_dir ./pretrain_features \
        --phase2_train ./data/kd/train --phase2_val ./data/kd/val \
        --phase3_train ./data/fmri/train --phase3_val ./data/fmri/val \
        --n_vertices 400 --n_subjects 4

    # Or run just one phase
    python v3_pipeline.py --phase 2 --phase2_train ./data/kd/train --phase2_val ./data/kd/val

    # Quick synthetic demo (no data needed)
    python v3_pipeline.py --demo
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tiny_tribe.v3_model import TinyTribeV3
from tiny_tribe.v3_dataset import get_dataloaders
from tiny_tribe.v3_pretrain import PretrainDataset, train_phase1, TinyTribePretrain
from tiny_tribe.v3_train import train_phase2, train_phase3


# ── Synthetic Demo Dataset ─────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """Generates random synthetic data for quick pipeline testing.

    All features and targets are random noise. Useful for:
      - Verifying the pipeline runs without errors
      - Profiling GPU memory and throughput
      - Checking that shapes and keys align
    """

    def __init__(self, n_samples=100, seq_len=20, n_vertices=100, n_subjects=4,
                 mode="kd", seed=42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_vertices = n_vertices
        self.n_subjects = n_subjects
        self.mode = mode
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Deterministic per-index randomness
        s = idx + 42
        np.random.seed(s)
        T = self.seq_len
        V = self.n_vertices

        out = {
            "text":  torch.randn(T, 384, dtype=torch.float32),
            "audio": torch.randn(T, 384, dtype=torch.float32),
            "video": torch.randn(T, 640, dtype=torch.float32),
            "subject_id": torch.tensor(self.rng.randint(0, self.n_subjects), dtype=torch.long),
        }

        if self.mode == "kd":
            out["teacher"] = torch.randn(T, V, dtype=torch.float32)
            out["fusion_l4"] = torch.randn(T, 1152, dtype=torch.float32)
        elif self.mode == "fmri":
            out["fmri"] = torch.randn(T, V, dtype=torch.float32)
            out["teacher"] = torch.randn(T, V, dtype=torch.float32)
            out["fusion_l4"] = torch.randn(T, 1152, dtype=torch.float32)

        return out


def make_synthetic_dataloaders(mode="kd", n_samples=100, seq_len=20,
                                n_vertices=100, n_subjects=4,
                                batch_size=4, num_workers=0):
    train_ds = SyntheticDataset(
        n_samples=n_samples, seq_len=seq_len, n_vertices=n_vertices,
        n_subjects=n_subjects, mode=mode, seed=42
    )
    val_ds = SyntheticDataset(
        n_samples=n_samples // 5, seq_len=seq_len, n_vertices=n_vertices,
        n_subjects=n_subjects, mode=mode, seed=99
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ── Demo Run ──────────────────────────────────────────────────────────────────

def run_demo(device: torch.device, n_vertices: int = 50, n_subjects: int = 2,
             epochs: int = 3):
    """Run a quick end-to-end demo on synthetic data.

    Verifies all three phases execute without errors.
    """
    print(f"\n{'='*60}")
    print("  Tiny-TRIBE v3 — End-to-End Demo")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  n_vertices: {n_vertices}, n_subjects: {n_subjects}")
    print(f"  Epochs per phase: {epochs}")

    seq_len = 10
    batch_size = 2

    # ── Phase 1: Pre-training ──
    print(f"\n{'─'*60}")
    print("  Phase 1: Self-Supervised Pre-Training")
    print(f"{'─'*60}")

    pretrain_base = TinyTribeV3(
        n_vertices=1, n_subjects=1, hidden_dim=128, num_layers=2,
        num_heads=2, num_experts=2, top_k=1, ff_mult=2, dropout=0.1,
        max_seq_len=seq_len * 3,
    ).to(device)
    pretrain_model = TinyTribePretrain(pretrain_base).to(device)

    pretrain_loader, pretrain_val = make_synthetic_dataloaders(
        mode="kd", n_samples=20, seq_len=seq_len,
        n_vertices=n_vertices, n_subjects=n_subjects,
        batch_size=batch_size
    )

    train_phase1(
        pretrain_model, pretrain_loader, pretrain_val, device,
        epochs=epochs, lr=1e-3, save_dir="./checkpoints/demo/phase1",
        log_every=5, use_amp=False,
    )

    # ── Phase 2: KD ──
    print(f"\n{'─'*60}")
    print("  Phase 2: Knowledge Distillation")
    print(f"{'─'*60}")

    kd_model = TinyTribeV3(
        n_vertices=n_vertices, n_subjects=n_subjects,
        hidden_dim=128, num_layers=2, num_heads=2,
        num_experts=2, top_k=1, ff_mult=2, dropout=0.1,
        max_seq_len=seq_len * 3,
    ).to(device)

    kd_loader, kd_val = make_synthetic_dataloaders(
        mode="kd", n_samples=20, seq_len=seq_len,
        n_vertices=n_vertices, n_subjects=n_subjects,
        batch_size=batch_size
    )

    train_phase2(
        kd_model, kd_loader, kd_val, device,
        epochs=epochs, lr=1e-3, save_dir="./checkpoints/demo/phase2",
        log_every=5, use_amp=False,
    )

    # ── Phase 3: fMRI fine-tuning ──
    print(f"\n{'─'*60}")
    print("  Phase 3: fMRI Fine-Tuning")
    print(f"{'─'*60}")

    fmri_model = TinyTribeV3(
        n_vertices=n_vertices, n_subjects=n_subjects,
        hidden_dim=128, num_layers=2, num_heads=2,
        num_experts=2, top_k=1, ff_mult=2, dropout=0.1,
        max_seq_len=seq_len * 3,
    ).to(device)

    fmri_loader, fmri_val = make_synthetic_dataloaders(
        mode="fmri", n_samples=20, seq_len=seq_len,
        n_vertices=n_vertices, n_subjects=n_subjects,
        batch_size=batch_size
    )

    train_phase3(
        fmri_model, fmri_loader, fmri_val, device,
        epochs=epochs, lr=5e-4, save_dir="./checkpoints/demo/phase3",
        log_every=5, use_amp=False,
    )

    print(f"\n{'='*60}")
    print("  Demo complete! All 3 phases ran successfully.")
    print(f"{'='*60}")
    return True


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Phase 1 ──
    if args.phase in (0, 1) and args.phase1_dir is not None:
        print(f"\n{'='*60}")
        print("Phase 1: Self-Supervised Pre-Training")
        print(f"{'='*60}")

        train_ds = PretrainDataset(args.phase1_dir, seq_len=args.seq_len,
                                   stride=args.seq_len // 2)
        val_size = max(1, int(0.1 * len(train_ds)))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        base = TinyTribeV3(
            n_vertices=1, n_subjects=1,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            num_heads=args.num_heads, num_experts=args.num_experts,
            top_k=args.top_k, ff_mult=args.ff_mult, dropout=args.dropout,
            max_seq_len=args.seq_len * 3,
        ).to(device)
        pretrain = TinyTribePretrain(base).to(device)

        train_phase1(
            pretrain, train_loader, val_loader, device,
            epochs=args.phase1_epochs, lr=args.lr, save_dir=args.phase1_save,
            use_amp=(device.type == "cuda"),
        )

        phase1_ckpt = Path(args.phase1_save) / "phase1_best.pt"
        if phase1_ckpt.exists():
            print(f"\nPhase 1 checkpoint: {phase1_ckpt}")
    else:
        phase1_ckpt = None

    # ── Phase 2 ──
    if args.phase in (0, 2) and args.phase2_train is not None:
        print(f"\n{'='*60}")
        print("Phase 2: Knowledge Distillation")
        print(f"{'='*60}")

        train_loader, val_loader = get_dataloaders(
            train_dir=args.phase2_train,
            val_dir=args.phase2_val,
            mode="kd",
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Infer n_vertices from first batch
        sample = next(iter(train_loader))
        n_vertices = sample["teacher"].shape[-1]

        model = TinyTribeV3(
            n_vertices=n_vertices, n_subjects=args.n_subjects,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            num_heads=args.num_heads, num_experts=args.num_experts,
            top_k=args.top_k, ff_mult=args.ff_mult, dropout=args.dropout,
            max_seq_len=args.seq_len * 3,
        ).to(device)

        # Optionally load Phase 1 weights into projectors
        if phase1_ckpt is not None and phase1_ckpt.exists():
            print(f"Loading Phase 1 projector weights from {phase1_ckpt}")
            state = torch.load(phase1_ckpt, map_location=device, weights_only=True)
            # Load only matching keys (projectors, embeddings)
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in state["model"].items()
                          if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict)
            print(f"  Loaded {len(pretrained)} keys from Phase 1")

        train_phase2(
            model, train_loader, val_loader, device,
            epochs=args.phase2_epochs, lr=args.lr,
            save_dir=args.phase2_save,
            use_amp=(device.type == "cuda"),
        )

        phase2_ckpt = Path(args.phase2_save) / "phase2_best.pt"
    else:
        phase2_ckpt = args.phase2_checkpoint

    # ── Phase 3 ──
    if args.phase in (0, 3) and args.phase3_train is not None:
        print(f"\n{'='*60}")
        print("Phase 3: fMRI Fine-Tuning")
        print(f"{'='*60}")

        train_loader, val_loader = get_dataloaders(
            train_dir=args.phase3_train,
            val_dir=args.phase3_val,
            mode="fmri",
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        sample = next(iter(train_loader))
        n_vertices = sample["fmri"].shape[-1]

        model = TinyTribeV3(
            n_vertices=n_vertices, n_subjects=args.n_subjects,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            num_heads=args.num_heads, num_experts=args.num_experts,
            top_k=args.top_k, ff_mult=args.ff_mult, dropout=args.dropout,
            max_seq_len=args.seq_len * 3,
        ).to(device)

        train_phase3(
            model, train_loader, val_loader, device,
            checkpoint=phase2_ckpt,
            epochs=args.phase3_epochs, lr=args.lr * 0.1,
            backbone_lr=args.lr * 0.01,
            save_dir=args.phase3_save,
            use_amp=(device.type == "cuda"),
        )

    print("\nPipeline complete!")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tiny-TRIBE v3 Unified Pipeline")

    # What to run
    p.add_argument("--demo", action="store_true",
                   help="Run quick synthetic demo (no data needed)")
    p.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                   help="0=all, 1=pretrain, 2=KD, 3=fMRI")

    # Phase 1 (Pre-training)
    p.add_argument("--phase1_dir", type=str, default=None,
                   help="Directory with .pt feature files for pre-training")
    p.add_argument("--phase1_epochs", type=int, default=25)
    p.add_argument("--phase1_save", type=str, default="./checkpoints/phase1")

    # Phase 2 (KD)
    p.add_argument("--phase2_train", type=str, default=None)
    p.add_argument("--phase2_val", type=str, default=None)
    p.add_argument("--phase2_epochs", type=int, default=60)
    p.add_argument("--phase2_save", type=str, default="./checkpoints/phase2")
    p.add_argument("--phase2_checkpoint", type=str, default=None,
                   help="Resume Phase 2 from checkpoint (skip Phase 1)")

    # Phase 3 (fMRI)
    p.add_argument("--phase3_train", type=str, default=None)
    p.add_argument("--phase3_val", type=str, default=None)
    p.add_argument("--phase3_epochs", type=int, default=30)
    p.add_argument("--phase3_save", type=str, default="./checkpoints/phase3")

    # Model
    p.add_argument("--n_vertices", type=int, default=400)
    p.add_argument("--n_subjects", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--ff_mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=2)

    # Infra
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    if args.demo:
        run_demo(device, n_vertices=50, n_subjects=2, epochs=3)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
