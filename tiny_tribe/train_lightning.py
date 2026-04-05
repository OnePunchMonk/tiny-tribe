"""PyTorch Lightning training script for Tiny-TRIBE v3 Knowledge Distillation.

Trains TinyTribeV3 to mimic TRIBE v2 fMRI predictions on the 200-clip
distillation dataset extracted via extract_features_v3.py.

Prerequisites:
    pip install lightning torchmetrics matplotlib

Usage:
    python tiny_tribe/train_lightning.py \\
        --features_dir ./features \\
        --save_dir ./checkpoints \\
        --epochs 100 \\
        --batch_size 16

For Lightning AI (A10G):
    python tiny_tribe/train_lightning.py \\
        --features_dir ./features \\
        --save_dir ./checkpoints \\
        --epochs 100 \\
        --batch_size 32 \\
        --precision bf16-mixed
"""

import argparse
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tiny_tribe.v3_model import TinyTribeV3


# ── Dataset ───────────────────────────────────────────────────────────────────

def parcellate(teacher: torch.Tensor, n_parcels: int) -> torch.Tensor:
    """Map (T, 20484) fsaverage5 → (T, n_parcels) by uniform chunk averaging.

    Replace with a real atlas lookup (e.g. Schaefer-400 annot file) for
    production. This is a valid placeholder: each parcel = mean of
    20484/n_parcels consecutive vertices.
    """
    T, V = teacher.shape
    if V == n_parcels:
        return teacher
    chunk = V // n_parcels
    return torch.stack([
        teacher[:, i * chunk:(i + 1) * chunk].mean(dim=1)
        for i in range(n_parcels)
    ], dim=1)  # (T, n_parcels)


class ClipDataset(Dataset):
    """Loads pre-extracted .pt feature files from extract_features_v3.py.

    Each file contains:
        text:       (T, 384)   float16
        audio:      (T, 384)   float16
        video:      (T, 640)   float16
        teacher:    (T, 20484) float16 — TRIBE v2 predictions
        subject_id: int

    If n_parcels < 20484, teacher is parcellated on load (no re-extraction needed).
    """

    def __init__(self, pt_files: List[Path], n_parcels: int = 400):
        self.files = pt_files
        self.n_parcels = n_parcels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        teacher = data["teacher"].float()  # (T, 20484)
        if self.n_parcels != teacher.shape[1]:
            teacher = parcellate(teacher, self.n_parcels)
        return {
            "text":       data["text"].float(),
            "audio":      data["audio"].float(),
            "video":      data["video"].float(),
            "teacher":    teacher,
            "subject_id": torch.tensor(
                int(data.get("subject_id", 0)), dtype=torch.long
            ),
        }


class ClipDataModule(L.LightningDataModule):
    def __init__(
        self,
        features_dir: str,
        batch_size: int = 16,
        val_fraction: float = 0.20,
        num_workers: int = 2,
        seed: int = 42,
        n_parcels: int = 400,
    ):
        super().__init__()
        self.features_dir = Path(features_dir)
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.n_parcels = n_parcels

    def setup(self, stage=None):
        all_files = sorted(self.features_dir.glob("*.pt"))
        if not all_files:
            raise FileNotFoundError(f"No .pt files in {self.features_dir}. "
                                    "Run extract_features_v3.py first.")
        n_val = max(1, int(len(all_files) * self.val_fraction))
        n_train = len(all_files) - n_val

        gen = torch.Generator().manual_seed(self.seed)
        train_files, val_files = random_split(all_files, [n_train, n_val],
                                              generator=gen)
        self.train_ds = ClipDataset(list(train_files), n_parcels=self.n_parcels)
        self.val_ds   = ClipDataset(list(val_files),   n_parcels=self.n_parcels)
        print(f"Dataset: {n_train} train / {n_val} val | n_parcels={self.n_parcels}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


# ── Loss ──────────────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    """Knowledge distillation loss (Phase 2).

    Works on shape (B, n_v, T) — the model's native output format.
    """

    def __init__(
        self,
        w_output:   float = 0.60,
        w_temporal: float = 0.10,
        w_multires: float = 0.05,
        w_aux:      float = 0.01,
    ):
        super().__init__()
        self.w_output   = w_output
        self.w_temporal = w_temporal
        self.w_multires = w_multires
        self.w_aux      = w_aux

    def temporal_coherence(self, pred: torch.Tensor) -> torch.Tensor:
        # pred: (B, n_v, T) — penalise TR-to-TR jitter
        diff = pred[:, :, 1:] - pred[:, :, :-1]
        return (diff ** 2).mean()

    def multi_resolution(self, pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for s in (2,):  # T=5 is too short for stride 4
            p = pred[:, :, ::s]
            t = teacher[:, :, ::s]
            T_min = min(p.shape[2], t.shape[2])
            loss = loss + F.mse_loss(p[:, :, :T_min], t[:, :, :T_min])
        return loss

    def forward(
        self,
        pred:     torch.Tensor,  # (B, n_v, T)
        teacher:  torch.Tensor,  # (B, n_v, T)
        aux_loss: torch.Tensor,
    ) -> dict:
        losses = {
            "output_kd": F.mse_loss(pred, teacher),
            "temporal":  self.temporal_coherence(pred),
            "multi_res": self.multi_resolution(pred, teacher),
            "aux":       aux_loss,
        }
        losses["total"] = (
            self.w_output   * losses["output_kd"]
          + self.w_temporal * losses["temporal"]
          + self.w_multires * losses["multi_res"]
          + self.w_aux      * losses["aux"]
        )
        return losses


# ── Pearson r metric (full val set) ──────────────────────────────────────────

class PearsonRMetric:
    """Accumulates predictions and targets across validation batches.
    Computes per-vertex Pearson r at the end of the epoch.
    """

    def __init__(self):
        self.preds   = []
        self.targets = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred/target: (B, n_v, T)
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())

    def compute(self) -> dict:
        if not self.preds:
            return {"mean": 0.0, "std": 0.0, "top10pct": 0.0}

        preds   = torch.cat(self.preds,   dim=0)   # (N, n_v, T)
        targets = torch.cat(self.targets, dim=0)

        N, V, T = preds.shape
        # Flatten (N, T) into one axis, keep vertices
        p = preds.permute(1, 0, 2).reshape(V, N * T)    # (V, N*T)
        t = targets.permute(1, 0, 2).reshape(V, N * T)

        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)

        num   = (p * t).sum(dim=1)
        denom = p.norm(dim=1) * t.norm(dim=1) + 1e-8
        r_per_vertex = (num / denom).numpy()  # (V,)

        return {
            "mean":    float(r_per_vertex.mean()),
            "std":     float(r_per_vertex.std()),
            "top10pct": float(np.percentile(r_per_vertex, 90)),
            "per_vertex": r_per_vertex,  # full array for plotting
        }

    def reset(self):
        self.preds   = []
        self.targets = []


# ── Lightning Module ──────────────────────────────────────────────────────────

class TinyTribeKD(L.LightningModule):
    def __init__(
        self,
        n_vertices:  int   = 400,    # POC: Schaefer-400 parcels
        n_subjects:  int   = 1,
        hidden_dim:  int   = 256,    # POC: smaller model
        num_layers:  int   = 2,      # POC: 2 transformer layers
        num_heads:   int   = 4,
        num_experts: int   = 4,      # POC: 4 experts
        top_k:       int   = 2,
        ff_mult:     int   = 2,
        dropout:     float = 0.3,    # POC: heavier dropout
        modality_dropout: float = 0.5,  # POC: aggressive modality dropout
        lr:          float = 1e-3,
        wd:          float = 0.1,    # POC: heavy weight decay
        warmup_epochs: int = 3,
        max_epochs:  int   = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TinyTribeV3(
            n_vertices=n_vertices,
            n_subjects=n_subjects,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            ff_mult=ff_mult,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"TinyTribeV3 — {n_params/1e6:.1f}M params")

        self.criterion   = KDLoss()
        self.val_pearson = PearsonRMetric()

        # Storage for end-of-val plot data
        self._val_pred_sample   = None
        self._val_teacher_sample = None

    def forward(self, text, audio, video, subject_id):
        return self.model(text, audio, video, subject_id)

    def _step(self, batch):
        text       = batch["text"]      # (B, T, 384)
        audio      = batch["audio"]     # (B, T, 384)
        video      = batch["video"]     # (B, T, 640)
        teacher    = batch["teacher"]   # (B, T, 20484)
        subject_id = batch["subject_id"]

        out  = self.model(text, audio, video, subject_id)
        pred = out["prediction"]        # (B, n_v, T)
        aux  = out["aux_loss"]

        # Teacher from dataset is (B, T, n_v); transpose to match model output
        teacher_t = teacher.permute(0, 2, 1)  # (B, n_v, T)

        # Align T dimension (HRF conv may shift length slightly)
        T_min   = min(pred.shape[2], teacher_t.shape[2])
        pred_t  = pred[:, :, :T_min]
        teach_t = teacher_t[:, :, :T_min]

        losses = self.criterion(pred_t, teach_t, aux)
        return losses, pred_t, teach_t

    def training_step(self, batch, batch_idx):
        losses, _, _ = self._step(batch)
        self.log_dict({
            "train/loss":      losses["total"],
            "train/kd_mse":    losses["output_kd"],
            "train/temporal":  losses["temporal"],
            "train/aux":       losses["aux"],
        }, on_step=False, on_epoch=True, prog_bar=True)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        losses, pred, teacher = self._step(batch)
        self.val_pearson.update(pred, teacher)

        # Save one sample for plotting
        if batch_idx == 0:
            self._val_pred_sample    = pred[:3].detach().cpu()
            self._val_teacher_sample = teacher[:3].detach().cpu()

        self.log_dict({
            "val/loss":   losses["total"],
            "val/kd_mse": losses["output_kd"],
        }, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.val_pearson.compute()
        self.log_dict({
            "val/pearson_r":      metrics["mean"],
            "val/pearson_r_std":  metrics["std"],
            "val/pearson_top10":  metrics["top10pct"],
        }, prog_bar=True)
        self.val_pearson.reset()

        # Generate plots every 10 epochs and at end
        if (self.current_epoch + 1) % 10 == 0 or self.current_epoch == self.hparams.max_epochs - 1:
            self._save_plots(metrics)

    def _save_plots(self, metrics):
        try:
            _save_pearson_histogram(
                metrics["per_vertex"],
                epoch=self.current_epoch + 1,
                out_dir=Path(self.trainer.default_root_dir) / "plots",
            )
            if self._val_pred_sample is not None:
                _save_temporal_profile(
                    self._val_pred_sample,
                    self._val_teacher_sample,
                    epoch=self.current_epoch + 1,
                    out_dir=Path(self.trainer.default_root_dir) / "plots",
                )
        except Exception as e:
            print(f"  [plot] Warning: {e}")

    def configure_optimizers(self):
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.wd,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr,
                                      betas=(0.9, 0.98), eps=1e-8)

        # Cosine with warmup
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_epochs * (
            total_steps // self.hparams.max_epochs
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
        }
        return [optimizer], [scheduler]


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save_pearson_histogram(r_per_vertex: np.ndarray, epoch: int, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r_per_vertex, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(r_per_vertex.mean(), color="red", linestyle="--",
               label=f"mean r = {r_per_vertex.mean():.3f}")
    ax.axvline(np.percentile(r_per_vertex, 90), color="orange", linestyle="--",
               label=f"90th pct = {np.percentile(r_per_vertex, 90):.3f}")
    ax.set_xlabel("Pearson r (student vs teacher)")
    ax.set_ylabel("# vertices")
    ax.set_title(f"Per-vertex Pearson r — epoch {epoch}")
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"pearson_hist_ep{epoch:04d}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] Saved {path}")


def _save_temporal_profile(
    pred:    torch.Tensor,   # (B, n_v, T) — up to 3 clips
    teacher: torch.Tensor,
    epoch:   int,
    out_dir: Path,
    n_clips: int = 3,
    n_vertices_shown: int = 5,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    B = min(pred.shape[0], n_clips)
    # Pick vertices with highest teacher variance
    teacher_var = teacher.var(dim=2).mean(dim=0)  # (n_v,)
    top_verts = teacher_var.topk(n_vertices_shown).indices.tolist()

    fig, axes = plt.subplots(B, n_vertices_shown, figsize=(16, 3 * B))
    if B == 1:
        axes = axes[None, :]

    for i in range(B):
        for j, v in enumerate(top_verts):
            ax = axes[i, j]
            T = pred.shape[2]
            ax.plot(range(T), teacher[i, v].numpy(), "k-o", label="teacher", markersize=4)
            ax.plot(range(T), pred[i, v].numpy(),    "b--s", label="student", markersize=4)
            if i == 0:
                ax.set_title(f"vertex {v}")
            if j == 0:
                ax.set_ylabel(f"clip {i}")
            if i == B - 1:
                ax.set_xlabel("TR")
            ax.legend(fontsize=6)

    plt.suptitle(f"Student vs Teacher predictions — epoch {epoch}", y=1.01)
    plt.tight_layout()
    path = out_dir / f"temporal_ep{epoch:04d}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {path}")


def _save_activation_comparison(pred_all: torch.Tensor, teacher_all: torch.Tensor,
                                 out_dir: Path, n_sample=200):
    """Bar chart: mean activation per sampled vertex, student vs teacher."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    # Sample n_sample vertices uniformly
    V = pred_all.shape[1]
    step = max(1, V // n_sample)
    idx = list(range(0, V, step))[:n_sample]

    pred_mean    = pred_all[:, idx, :].mean(dim=(0, 2)).numpy()
    teacher_mean = teacher_all[:, idx, :].mean(dim=(0, 2)).numpy()

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(idx))
    ax.bar(x - 0.2, teacher_mean, 0.4, label="teacher", alpha=0.7, color="gray")
    ax.bar(x + 0.2, pred_mean,    0.4, label="student", alpha=0.7, color="steelblue")
    ax.set_xlabel("vertex index (sampled)")
    ax.set_ylabel("mean predicted activation")
    ax.set_title("Student vs Teacher — mean activation per vertex")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "activation_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] Saved {path}")


# ── Final summary plots (run after training) ──────────────────────────────────

def generate_final_plots(ckpt_path: str, dm: ClipDataModule, save_dir: Path):
    """Load best checkpoint and generate final diagnostic plots."""
    print("\nGenerating final plots...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = TinyTribeKD.load_from_checkpoint(ckpt_path)
    module.eval().to(device)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_preds   = []
    all_teachers = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            text       = batch["text"].to(device)
            audio      = batch["audio"].to(device)
            video      = batch["video"].to(device)
            subject_id = batch["subject_id"].to(device)
            teacher    = batch["teacher"].permute(0, 2, 1)  # (B, n_v, T)

            out  = module.model(text, audio, video, subject_id)
            pred = out["prediction"]
            T_min = min(pred.shape[2], teacher.shape[2])
            all_preds.append(pred[:, :, :T_min].cpu())
            all_teachers.append(teacher[:, :, :T_min].cpu())

    preds_all    = torch.cat(all_preds,    dim=0)
    teachers_all = torch.cat(all_teachers, dim=0)

    # Pearson r histogram
    V = preds_all.shape[1]
    N, _, T = preds_all.shape
    p = preds_all.permute(1, 0, 2).reshape(V, N * T)
    t = teachers_all.permute(1, 0, 2).reshape(V, N * T)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    r_per_vertex = ((p * t).sum(dim=1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-8)).numpy()

    _save_pearson_histogram(r_per_vertex, epoch=9999, out_dir=plots_dir)
    # rename to "final"
    (plots_dir / "pearson_hist_ep9999.png").rename(plots_dir / "pearson_hist_final.png")

    _save_activation_comparison(preds_all, teachers_all, plots_dir)
    _save_temporal_profile(preds_all[:3], teachers_all[:3], epoch=9999, out_dir=plots_dir)
    (plots_dir / "temporal_ep9999.png").rename(plots_dir / "temporal_final.png")

    print(f"Final plots saved to {plots_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=str, default="./features")
    p.add_argument("--save_dir",     type=str, default="./checkpoints")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--wd",           type=float, default=0.1)
    p.add_argument("--n_subjects",   type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.20)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--precision",    type=str, default="32",
                   help="e.g. 32, 16-mixed, bf16-mixed")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--n_parcels",    type=int,   default=400,
                   help="Teacher vertices to use (400=POC, 20484=full)")
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--num_heads",    type=int,   default=4)
    p.add_argument("--num_experts",  type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--modality_dropout", type=float, default=0.5)
    p.add_argument("--fast_dev_run", action="store_true",
                   help="Smoke test: 1 batch train + val")
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    dm = ClipDataModule(
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        n_parcels=args.n_parcels,
    )
    dm.setup()

    # ── Model ─────────────────────────────────────────────────────────────────
    module = TinyTribeKD(
        n_vertices=args.n_parcels,
        n_subjects=args.n_subjects,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        lr=args.lr,
        wd=args.wd,
        max_epochs=args.epochs,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename="best-{epoch:03d}-{val/pearson_r:.4f}",
            monitor="val/pearson_r",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/pearson_r",
            mode="max",
            patience=20,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=str(save_dir), name="tb_logs")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(save_dir),
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(module, dm)

    # ── Final plots ───────────────────────────────────────────────────────────
    if not args.fast_dev_run:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            generate_final_plots(best_ckpt, dm, save_dir)

    print(f"\nTraining complete. Checkpoints in {save_dir}/")
    print(f"TensorBoard: tensorboard --logdir {save_dir}/tb_logs")


if __name__ == "__main__":
    main()
