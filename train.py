"""Training script for Tiny-TRIBE.

Stage 1: Frozen backbones — train projectors + fusion + output head
Stage 2: End-to-end fine-tuning with knowledge distillation
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from tiny_tribe.config import TinyTribeConfig
from tiny_tribe.model import TinyTribe
from tiny_tribe.distillation import DistillationLoss, TaskOnlyLoss

logger = logging.getLogger(__name__)


class TinyTribeDataset(Dataset):
    """Dataset that loads pre-extracted features or raw stimuli + fMRI targets.

    Expected data directory structure:
        data_dir/
            subject_XX/
                text_ids.pt       # (N, T_text)
                text_mask.pt      # (N, T_text)
                audio_features.pt # (N, 80, T_mel)
                video_frames.pt   # (N, T_vid, 3, H, W)
                fmri.pt           # (N, n_vertices, n_trs)
                metadata.json     # {"subject_id": int, ...}
    """

    def __init__(self, data_dir: str, subjects: list[str] | None = None):
        self.data_dir = Path(data_dir)
        self.samples = []

        subject_dirs = sorted(self.data_dir.iterdir()) if subjects is None else [
            self.data_dir / s for s in subjects
        ]

        for sdir in subject_dirs:
            if not sdir.is_dir():
                continue
            meta = json.loads((sdir / "metadata.json").read_text())
            n_samples = torch.load(sdir / "fmri.pt", weights_only=True).shape[0]
            for i in range(n_samples):
                self.samples.append({
                    "subject_dir": sdir,
                    "sample_idx": i,
                    "subject_id": meta["subject_id"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        sdir = info["subject_dir"]
        i = info["sample_idx"]

        return {
            "text_ids": torch.load(sdir / "text_ids.pt", weights_only=True)[i],
            "text_mask": torch.load(sdir / "text_mask.pt", weights_only=True)[i],
            "audio_features": torch.load(sdir / "audio_features.pt", weights_only=True)[i],
            "video_frames": torch.load(sdir / "video_frames.pt", weights_only=True)[i],
            "fmri": torch.load(sdir / "fmri.pt", weights_only=True)[i],
            "subject_id": info["subject_id"],
        }


def collate_fn(batch):
    """Collate with padding for variable-length sequences."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k == "subject_id":
            collated[k] = torch.tensor([b[k] for b in batch])
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated


def train_stage1(
    model: TinyTribe,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: TinyTribeConfig,
    device: torch.device,
    save_dir: Path,
):
    """Stage 1: Frozen backbones, train fusion components only."""
    logger.info("=== Stage 1: Frozen backbone training ===")

    # Freeze backbones, only train projectors + fusion + output
    for param in model.backbones.parameters():
        param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = Adam(trainable, lr=config.training.stage1_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.stage1_lr,
        epochs=config.training.stage1_epochs,
        steps_per_epoch=len(train_loader),
    )
    loss_fn = TaskOnlyLoss()

    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(config.training.stage1_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = model(
                text_ids=batch["text_ids"],
                text_mask=batch["text_mask"],
                audio_features=batch["audio_features"],
                video_frames=batch["video_frames"],
                subject_id=batch["subject_id"],
            )

            losses = loss_fn(output["prediction"], batch["fmri"])
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{config.training.stage1_epochs} "
                    f"Step {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            logger.info(f"Epoch {epoch+1} val loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "stage1_best.pt")

    torch.save(model.state_dict(), save_dir / "stage1_final.pt")
    return model


def train_stage2(
    model: TinyTribe,
    teacher_model: nn.Module | None,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: TinyTribeConfig,
    device: torch.device,
    save_dir: Path,
):
    """Stage 2: End-to-end fine-tuning with knowledge distillation."""
    logger.info("=== Stage 2: End-to-end fine-tuning with KD ===")

    # Unfreeze audio and video backbones (text stays frozen)
    model.backbones.unfreeze_all()

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = Adam(trainable, lr=config.training.stage2_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.stage2_lr,
        epochs=config.training.stage2_epochs,
        steps_per_epoch=len(train_loader),
    )

    if teacher_model is not None:
        loss_fn = DistillationLoss(
            task_weight=config.training.kd_task_weight,
            output_kd_weight=config.training.kd_output_weight,
            feature_kd_weight=config.training.kd_feature_weight,
            student_hidden=config.fusion.hidden_dim,
            teacher_hidden=1152,  # TRIBE v2 hidden dim
        )
        teacher_model.to(device)
        teacher_model.eval()
    else:
        loss_fn = TaskOnlyLoss()

    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(config.training.stage2_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = model(
                text_ids=batch["text_ids"],
                text_mask=batch["text_mask"],
                audio_features=batch["audio_features"],
                video_frames=batch["video_frames"],
                subject_id=batch["subject_id"],
            )

            # Get teacher predictions if available
            teacher_pred = None
            teacher_intermediates = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(batch)
                    teacher_pred = teacher_output["prediction"] if isinstance(teacher_output, dict) else teacher_output
                    teacher_intermediates = teacher_output.get("intermediates")

            if isinstance(loss_fn, DistillationLoss):
                losses = loss_fn(
                    student_pred=output["prediction"],
                    fmri_target=batch["fmri"],
                    teacher_pred=teacher_pred,
                    student_intermediates=output.get("intermediates"),
                    teacher_intermediates=teacher_intermediates,
                )
            else:
                losses = loss_fn(output["prediction"], batch["fmri"])

            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                loss_str = " ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
                logger.info(
                    f"  Epoch {epoch+1}/{config.training.stage2_epochs} "
                    f"Step {batch_idx}/{len(train_loader)} {loss_str}"
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, TaskOnlyLoss(), device)
            logger.info(f"Epoch {epoch+1} val loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "stage2_best.pt")

    torch.save(model.state_dict(), save_dir / "stage2_final.pt")
    return model


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(
            text_ids=batch["text_ids"],
            text_mask=batch["text_mask"],
            audio_features=batch["audio_features"],
            video_frames=batch["video_frames"],
            subject_id=batch["subject_id"],
        )
        losses = loss_fn(output["prediction"], batch["fmri"])
        total_loss += losses["total"].item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train Tiny-TRIBE")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/tiny_tribe")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Teacher model for KD")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = TinyTribeConfig()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import dataclasses
    with open(save_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Build model
    model = TinyTribe(config)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Build dataloaders
    train_dataset = TinyTribeDataset(args.data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if args.stage == 1:
        train_stage1(model, train_loader, None, config, device, save_dir)
    elif args.stage == 2:
        teacher_model = None
        if args.teacher_checkpoint:
            logger.info(f"Loading teacher from {args.teacher_checkpoint}")
            # Teacher model loading depends on your TRIBE v2 setup
            # teacher_model = load_tribe_v2_teacher(args.teacher_checkpoint)
        train_stage2(model, teacher_model, train_loader, None, config, device, save_dir)


if __name__ == "__main__":
    main()
