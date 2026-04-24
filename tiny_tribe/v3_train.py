"""Training script for Tiny-TRIBE v3 distillation pipeline.

Phase 2: Knowledge Distillation
  - Frozen backbone features (cached .pt files)
  - Student trained against teacher predictions + intermediate features
  - Modality dropout 0.3

Phase 3: fMRI Fine-tuning
  - Real fMRI signal as primary supervision
  - Teacher predictions as regulariser
  - Unfreeze Whisper + MobileViT at very low LR

Usage:
    # Smoke test (nilearn dataset)
    python v3_train.py --mode kd --train_dir ./data/nilearn/train --val_dir ./data/nilearn/val

    # Full Phase 2 KD
    python v3_train.py --mode kd --train_dir ./data/cneuromod/train --val_dir ./data/cneuromod/val \
        --n_subjects 4 --n_vertices 20484 --batch_size 16 --epochs 60

    # Full Phase 3 fMRI
    python v3_train.py --mode fmri --train_dir ./data/cneuromod/train --val_dir ./data/cneuromod/val \
        --checkpoint ./checkpoints/phase2_best.pt --n_subjects 4 --n_vertices 20484
"""

import os
import math
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

from tiny_tribe.v3_model import TinyTribeV3
from tiny_tribe.v3_dataset import get_dataloaders


# ── Loss Functions ─────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """Phase 2 KD loss.

    Total = 0.6 * output_kd
          + 0.2 * feature_kd
          + 0.1 * temporal_coherence
          + 0.05 * multi_res
          + aux_weight * aux_loss
    """

    def __init__(
        self,
        w_output:   float = 0.60,
        w_feature:  float = 0.20,
        w_temporal: float = 0.10,
        w_multires: float = 0.05,
        w_aux:      float = 0.01,
        temperature: float = 2.0,
        student_dim: int = 512,
        teacher_dim: int = 1152,
    ):
        super().__init__()
        self.w_output   = w_output
        self.w_feature  = w_feature
        self.w_temporal = w_temporal
        self.w_multires = w_multires
        self.w_aux      = w_aux
        self.T          = temperature
        # Project student fusion features to teacher feature dim (lazy init)
        self.feat_proj = nn.LazyLinear(teacher_dim, bias=False)

    def output_kd(
        self,
        pred:    torch.Tensor,   # (B, T, n_v)
        teacher: torch.Tensor,   # (B, T, n_v)
    ) -> torch.Tensor:
        """Soft-label KD: smooth teacher predictions with temperature."""
        # Use MSE for regression (vertex predictions are continuous)
        return F.mse_loss(pred, teacher)

    def feature_kd(
        self,
        student_feat: torch.Tensor,   # (B, T, D_student)
        teacher_feat: torch.Tensor,   # (B, T, D_teacher)
    ) -> torch.Tensor:
        """Cosine similarity in feature space — teacher layer 4 vs student fusion.
        
        If dimensions don't match, projects student to teacher dim first.
        """
        if student_feat is None or teacher_feat is None:
            return torch.tensor(0.0, device=student_feat.device if student_feat is not None else 'cpu')
        B, T_s, D_s = student_feat.shape
        _, T_t, D_t = teacher_feat.shape
        # Flatten batch+time for linear projection if needed
        s = student_feat.reshape(B * T_s, D_s)
        if D_s != D_t:
            s = self.feat_proj(s)  # (B*T_s, D_t)
            s = s.reshape(B, T_s, D_t)
        else:
            s = s.reshape(B, T_s, D_t)
        # Normalise along feature dim
        s = F.normalize(s, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        return 1.0 - (s * t).sum(dim=-1).mean()

    def temporal_coherence(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalise high-frequency jitter between adjacent TRs."""
        diff = pred[:, 1:, :] - pred[:, :-1, :]   # (B, T-1, n_v)
        return (diff ** 2).mean()

    def multi_resolution(
        self,
        pred:    torch.Tensor,   # (B, T, n_v)
        teacher: torch.Tensor,   # (B, T, n_v)
        scales:  Tuple[int, ...] = (2, 4),
    ) -> torch.Tensor:
        """MSE at temporally downsampled resolutions."""
        loss = torch.tensor(0.0, device=pred.device)
        for s in scales:
            # Average-pool along time axis
            p = pred[:, ::s, :]
            t = teacher[:, ::s, :]
            T_min = min(p.shape[1], t.shape[1])
            loss = loss + F.mse_loss(p[:, :T_min], t[:, :T_min])
        return loss / len(scales)

    def forward(
        self,
        pred:         torch.Tensor,
        teacher:      torch.Tensor,
        aux_loss:     torch.Tensor,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        losses = {}
        losses["output_kd"]  = self.output_kd(pred, teacher)
        losses["temporal"]   = self.temporal_coherence(pred)
        losses["multi_res"]  = self.multi_resolution(pred, teacher)
        losses["aux"]        = aux_loss

        if student_feat is not None and teacher_feat is not None:
            losses["feature_kd"] = self.feature_kd(student_feat, teacher_feat)
        else:
            losses["feature_kd"] = torch.tensor(0.0, device=pred.device)

        total = (
            self.w_output   * losses["output_kd"]
          + self.w_feature  * losses["feature_kd"]
          + self.w_temporal * losses["temporal"]
          + self.w_multires * losses["multi_res"]
          + self.w_aux      * losses["aux"]
        )
        losses["total"] = total
        return losses


class FMRILoss(nn.Module):
    """Phase 3 fMRI fine-tuning loss.

    Total = 0.40 * fmri_mse
          + 0.30 * teacher_reg
          + 0.10 * feature_kd
          + 0.10 * temporal_coherence
          + 0.01 * aux
    """

    def __init__(
        self,
        w_fmri:     float = 0.40,
        w_teacher:  float = 0.30,
        w_feature:  float = 0.10,
        w_temporal: float = 0.10,
        w_aux:      float = 0.01,
        student_dim: int = 512,
        teacher_dim: int = 1152,
    ):
        super().__init__()
        self.w_fmri    = w_fmri
        self.w_teacher = w_teacher
        self.w_feature = w_feature
        self.w_temporal= w_temporal
        self.w_aux     = w_aux
        self.feat_proj = nn.LazyLinear(teacher_dim, bias=False)

    def pearson_loss(
        self,
        pred: torch.Tensor,   # (B, T, n_v)
        fmri: torch.Tensor,   # (B, T, n_v)
    ) -> torch.Tensor:
        """1 - mean Pearson r across vertices (differentiable)."""
        # Flatten B×T into batch dim, keep vertices separate
        B, T, V = pred.shape
        p = pred.reshape(B * T, V)
        f = fmri.reshape(B * T, V)

        p = p - p.mean(dim=0, keepdim=True)
        f = f - f.mean(dim=0, keepdim=True)

        num = (p * f).sum(dim=0)
        denom = p.norm(dim=0) * f.norm(dim=0) + 1e-8
        r = num / denom   # (V,)
        return 1.0 - r.mean()

    def temporal_coherence(self, pred: torch.Tensor) -> torch.Tensor:
        diff = pred[:, 1:, :] - pred[:, :-1, :]
        return (diff ** 2).mean()

    def feature_kd(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        if student_feat is None or teacher_feat is None:
            return torch.tensor(0.0, device=student_feat.device if student_feat is not None else 'cpu')
        B, T_s, D_s = student_feat.shape
        _, T_t, D_t = teacher_feat.shape
        s = student_feat.reshape(B * T_s, D_s)
        if D_s != D_t:
            s = self.feat_proj(s)
            s = s.reshape(B, T_s, D_t)
        else:
            s = s.reshape(B, T_s, D_t)
        s = F.normalize(s, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        return 1.0 - (s * t).sum(dim=-1).mean()

    def forward(
        self,
        pred:         torch.Tensor,
        fmri:         torch.Tensor,
        aux_loss:     torch.Tensor,
        teacher_pred: Optional[torch.Tensor] = None,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        losses = {}
        losses["fmri_pearson"] = self.pearson_loss(pred, fmri)
        losses["fmri_mse"]     = F.mse_loss(pred, fmri)
        # Combine MSE + Pearson as fMRI loss
        losses["fmri"]         = 0.5 * losses["fmri_pearson"] + 0.5 * losses["fmri_mse"]
        losses["temporal"]     = self.temporal_coherence(pred)
        losses["aux"]          = aux_loss

        if teacher_pred is not None:
            losses["teacher_reg"] = F.mse_loss(pred, teacher_pred)
        else:
            losses["teacher_reg"] = torch.tensor(0.0, device=pred.device)

        if student_feat is not None and teacher_feat is not None:
            losses["feature_kd"] = self.feature_kd(student_feat, teacher_feat)
        else:
            losses["feature_kd"] = torch.tensor(0.0, device=pred.device)

        total = (
            self.w_fmri    * losses["fmri"]
          + self.w_teacher * losses["teacher_reg"]
          + self.w_feature * losses["feature_kd"]
          + self.w_temporal* losses["temporal"]
          + self.w_aux     * losses["aux"]
        )
        losses["total"] = total
        return losses


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      TinyTribeV3,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    mode:       str = "kd",   # "kd" or "fmri"
) -> Dict[str, float]:
    """Compute Pearson r and MSE on the validation set."""
    model.eval()

    all_pred  = []
    all_target= []
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        text  = batch["text"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        subj  = batch["subject_id"].to(device)

        model.set_modality_dropout(0.0)
        out = model(text, audio, video, subj)
        pred = out["prediction"]   # (B, n_v, T)

        if mode == "fmri" and "fmri" in batch:
            target = batch["fmri"].to(device)
        elif "teacher" in batch:
            target = batch["teacher"].to(device)
        else:
            continue

        # Align T if mismatch (HRF conv may change length)
        # pred: (B, n_v, T), target: (B, T, n_v)
        T_min = min(pred.shape[2], target.shape[1])
        pred   = pred[:, :, :T_min]
        target = target[:, :T_min, :].transpose(1, 2)  # (B, n_v, T)

        total_loss += F.mse_loss(pred, target).item()
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
        n_batches += 1

    if not all_pred:
        return {"pearson_r": 0.0, "mse": float("inf")}

    preds   = torch.cat(all_pred,   dim=0)   # (N, n_v, T)
    targets = torch.cat(all_target, dim=0)

    # Pearson r per vertex, averaged
    N, V, T = preds.shape
    p = preds.permute(1, 0, 2).reshape(V, N * T)
    t = targets.permute(1, 0, 2).reshape(V, N * T)

    p = p - p.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)

    num   = (p * t).sum(dim=0)
    denom = p.norm(dim=0) * t.norm(dim=0) + 1e-8
    r_per_vertex = (num / denom).numpy()

    return {
        "pearson_r":      float(r_per_vertex.mean()),
        "pearson_r_std":  float(r_per_vertex.std()),
        "pearson_r_top10": float(np.percentile(r_per_vertex, 90)),
        "mse":            total_loss / max(n_batches, 1),
    }


# ── LR Warmup + Cosine Schedule ───────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model:      nn.Module,
    lr:         float,
    wd:         float,
    epochs:     int,
    steps_per_epoch: int,
    warmup_epochs: int = 3,
    phase3_backbone_lr: Optional[float] = None,
) -> Tuple:
    """AdamW with weight-decay separation and cosine LR with warmup.

    In Phase 3, backbone params get a 10× lower LR (fine-tuning).
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    if phase3_backbone_lr is not None:
        # Separate backbone params (projectors) from fusion params
        backbone_params = []
        fusion_params   = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "projector" in name:
                backbone_params.append((name, p))
            else:
                fusion_params.append((name, p))

        param_groups = [
            {"params": [p for n, p in fusion_params   if not any(nd in n for nd in no_decay)],
             "lr": lr, "weight_decay": wd},
            {"params": [p for n, p in fusion_params   if     any(nd in n for nd in no_decay)],
             "lr": lr, "weight_decay": 0.0},
            {"params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
             "lr": phase3_backbone_lr, "weight_decay": wd},
            {"params": [p for n, p in backbone_params if     any(nd in n for nd in no_decay)],
             "lr": phase3_backbone_lr, "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
             "weight_decay": wd},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and     any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

    total_steps   = epochs * steps_per_epoch
    warmup_steps  = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ── Training Loop: Phase 2 KD ─────────────────────────────────────────────────

def train_phase2(
    model:       TinyTribeV3,
    train_loader,
    val_loader,
    device:      torch.device,
    epochs:      int   = 60,
    lr:          float = 3e-4,
    wd:          float = 1e-2,
    clip_grad:   float = 1.0,
    save_dir:    str   = "./checkpoints",
    log_every:   int   = 20,
    use_amp:     bool  = True,
):
    """Phase 2: Knowledge Distillation from cached teacher predictions.

    Backbones are FROZEN. Only the fusion transformer and output heads train.
    Modality dropout 0.3 is applied during training for robustness.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Freeze projectors — they were pre-trained in Phase 1
    # (If skipping Phase 1, leave them trainable)
    # for name, p in model.named_parameters():
    #     if "projector" in name:
    #         p.requires_grad_(False)

    criterion = DistillationLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr, wd, epochs, len(train_loader), warmup_epochs=3
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_r  = -float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"Phase 2: Knowledge Distillation")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  AMP: {use_amp}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in
                        ["total", "output_kd", "feature_kd", "temporal", "multi_res", "aux"]}
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            text    = batch["text"].to(device)
            audio   = batch["audio"].to(device)
            video   = batch["video"].to(device)
            teacher = batch["teacher"].to(device)
            subj    = batch["subject_id"].to(device)

            teacher_feat = batch.get("fusion_l4")
            if teacher_feat is not None:
                teacher_feat = teacher_feat.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                model.set_modality_dropout(0.3)
                out  = model(text, audio, video, subj)
                pred = out["prediction"]          # (B, n_v, T)
                aux  = out["aux_loss"]

                # Align temporal dimension (model output: B,n_v,T; teacher: B,T,n_v)
                T_min   = min(pred.shape[2], teacher.shape[1])
                pred_t  = pred[:, :, :T_min]
                teach_t = teacher[:, :T_min, :].transpose(1, 2)  # (B, n_v, T)

                student_feat = out.get("fusion_feat")
                if student_feat is not None:
                    student_feat = student_feat[:, :T_min, :]
                if teacher_feat is not None:
                    teacher_feat = teacher_feat[:, :T_min, :]

                losses = criterion(
                    pred_t, teach_t, aux,
                    student_feat=student_feat,
                    teacher_feat=teacher_feat,
                )

            optimizer.zero_grad()
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d} | Step {step:4d}/{len(train_loader)} "
                      f"| loss={losses['total']:.4f} "
                      f"| kd={losses['output_kd']:.4f} "
                      f"| aux={losses['aux']:.4f} "
                      f"| lr={lr_now:.2e}")

        # Epoch summary
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device, mode="kd")
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{epochs} | "
              f"train_loss={epoch_losses['total']:.4f} | "
              f"val_r={val_metrics['pearson_r']:.4f} | "
              f"val_mse={val_metrics['mse']:.4f} | "
              f"time={elapsed:.0f}s")

        history.append({"epoch": epoch, **epoch_losses, **val_metrics})

        # Save best checkpoint
        if val_metrics["pearson_r"] > best_r:
            best_r = val_metrics["pearson_r"]
            ckpt_path = Path(save_dir) / "phase2_best.pt"
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "val_pearson": best_r,
                "history":     history,
            }, ckpt_path)
            print(f"  ✓ New best val Pearson r = {best_r:.4f} → saved to {ckpt_path}")

        # Always save latest
        torch.save({
            "epoch":    epoch,
            "model":    model.state_dict(),
            "history":  history,
        }, Path(save_dir) / "phase2_latest.pt")

    print(f"\nPhase 2 complete. Best val Pearson r = {best_r:.4f}")
    return history


# ── Training Loop: Phase 3 fMRI ───────────────────────────────────────────────

def train_phase3(
    model:          TinyTribeV3,
    train_loader,
    val_loader,
    device:         torch.device,
    checkpoint:     Optional[str] = None,
    epochs:         int   = 30,
    lr:             float = 5e-5,
    wd:             float = 1e-2,
    backbone_lr:    float = 5e-6,
    clip_grad:      float = 0.5,
    save_dir:       str   = "./checkpoints",
    log_every:      int   = 10,
    use_amp:        bool  = True,
):
    """Phase 3: fMRI Fine-tuning.

    Loads Phase 2 checkpoint. Unfreezes everything.
    Projectors get backbone_lr (10× lower than fusion lr).
    Primary signal: real fMRI. Teacher predictions used as regulariser.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        phase2_r = state.get("val_pearson", 0.0)
        print(f"Loaded Phase 2 checkpoint (val_r={phase2_r:.4f}) from {checkpoint}")

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    criterion = FMRILoss()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr, wd, epochs, len(train_loader),
        warmup_epochs=2, phase3_backbone_lr=backbone_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_r  = -float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"Phase 3: fMRI Fine-tuning")
    print(f"  Epochs: {epochs}  |  Fusion LR: {lr}  |  Backbone LR: {backbone_lr}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in
                        ["total", "fmri", "teacher_reg", "feature_kd", "temporal", "aux"]}
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            text  = batch["text"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            fmri  = batch["fmri"].to(device)
            subj  = batch["subject_id"].to(device)

            teacher_pred = batch.get("teacher")
            if teacher_pred is not None:
                teacher_pred = teacher_pred.to(device)

            teacher_feat = batch.get("fusion_l4")
            if teacher_feat is not None:
                teacher_feat = teacher_feat.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                model.set_modality_dropout(0.1)
                out  = model(text, audio, video, subj)
                pred = out["prediction"]
                aux  = out["aux_loss"]

                T_min = min(pred.shape[2], fmri.shape[1])
                pred_t  = pred[:, :, :T_min]
                fmri_t  = fmri[:, :T_min, :].transpose(1, 2)  # (B, n_v, T)

                teacher_t = None
                if teacher_pred is not None:
                    teacher_t = teacher_pred[:, :T_min, :].transpose(1, 2)

                student_feat = out.get("fusion_feat")
                if student_feat is not None:
                    student_feat = student_feat[:, :T_min, :]
                if teacher_feat is not None:
                    teacher_feat = teacher_feat[:, :T_min, :]

                losses = criterion(
                    pred_t, fmri_t, aux,
                    teacher_pred=teacher_t,
                    student_feat=student_feat,
                    teacher_feat=teacher_feat,
                )

            optimizer.zero_grad()
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d} | Step {step:4d}/{len(train_loader)} "
                      f"| loss={losses['total']:.4f} "
                      f"| fmri={losses['fmri']:.4f} "
                      f"| teacher={losses['teacher_reg']:.4f} "
                      f"| lr={lr_now:.2e}")

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device, mode="fmri")
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{epochs} | "
              f"train_loss={epoch_losses['total']:.4f} | "
              f"val_r={val_metrics['pearson_r']:.4f} | "
              f"val_r_top10={val_metrics['pearson_r_top10']:.4f} | "
              f"time={elapsed:.0f}s")

        history.append({"epoch": epoch, **epoch_losses, **val_metrics})

        if val_metrics["pearson_r"] > best_r:
            best_r = val_metrics["pearson_r"]
            ckpt_path = Path(save_dir) / "phase3_best.pt"
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "val_pearson": best_r,
                "history":     history,
            }, ckpt_path)
            print(f"  ✓ New best val Pearson r = {best_r:.4f} → saved to {ckpt_path}")

        torch.save({
            "epoch":   epoch,
            "model":   model.state_dict(),
            "history": history,
        }, Path(save_dir) / "phase3_latest.pt")

    print(f"\nPhase 3 complete. Best val Pearson r = {best_r:.4f}")
    return history


# ── Smoke Test ────────────────────────────────────────────────────────────────

def run_smoke_test(args):
    """Quick sanity check on nilearn dataset.

    Expects data built via:
        from v3_dataset import build_nilearn_dataset
        build_nilearn_dataset(
            out_dir=args.out_dir,
            teacher_pt=args.teacher_pt,
            features_pt=args.features_pt,
        )
    """
    device = torch.device(args.device)

    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        mode=args.mode,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=0,   # nilearn dataset is small, avoid worker overhead
    )

    # Infer n_vertices from first batch
    sample = next(iter(train_loader))
    if "teacher" in sample:
        n_vertices = sample["teacher"].shape[-1]
    elif "fmri" in sample:
        n_vertices = sample["fmri"].shape[-1]
    else:
        n_vertices = 400  # Schaefer-400 default

    print(f"n_vertices = {n_vertices}")
    print(f"text shape = {sample['text'].shape}")
    print(f"audio shape = {sample['audio'].shape}")
    print(f"video shape = {sample['video'].shape}")

    # Build model
    model = TinyTribeV3(
        n_vertices=n_vertices,
        n_subjects=args.n_subjects,
        hidden_dim=args.dim,
        num_layers=args.depth,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        num_experts=args.num_experts,
        top_k=args.top_k,
        max_seq_len=args.seq_len,
        stoch_depth_max=args.stoch_depth_max,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    if args.mode == "kd":
        train_phase2(
            model, train_loader, val_loader, device,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
            use_amp=(device.type == "cuda"),
        )
    else:
        train_phase3(
            model, train_loader, val_loader, device,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
            use_amp=(device.type == "cuda"),
        )


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tiny-TRIBE v3 training")

    # Data
    p.add_argument("--train_dir",   type=str, required=True)
    p.add_argument("--val_dir",     type=str, required=True)
    p.add_argument("--mode",        type=str, default="kd", choices=["kd", "fmri"])
    p.add_argument("--seq_len",     type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)

    # Model
    p.add_argument("--n_vertices",      type=int,   default=400)
    p.add_argument("--n_subjects",      type=int,   default=4)
    p.add_argument("--dim",             type=int,   default=512)
    p.add_argument("--depth",           type=int,   default=4)
    p.add_argument("--heads",           type=int,   default=8)
    p.add_argument("--ff_mult",         type=int,   default=4)
    p.add_argument("--num_experts",     type=int,   default=8)
    p.add_argument("--top_k",           type=int,   default=2)
    p.add_argument("--stoch_depth_max", type=float, default=0.2)

    # Training
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--backbone_lr",  type=float, default=5e-6)
    p.add_argument("--wd",           type=float, default=1e-2)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--checkpoint",   type=str,   default=None,
                   help="Phase 2 checkpoint to resume from (for Phase 3)")

    # Infra
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir",   type=str, default="./checkpoints")
    p.add_argument("--log_every",  type=int, default=20)
    p.add_argument("--no_amp",     action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        mode=args.mode,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Infer n_vertices
    sample = next(iter(train_loader))
    if args.mode == "fmri" and "fmri" in sample:
        n_vertices = sample["fmri"].shape[-1]
    elif "teacher" in sample:
        n_vertices = sample["teacher"].shape[-1]
    else:
        n_vertices = args.n_vertices
    print(f"n_vertices = {n_vertices}")

    model = TinyTribeV3(
        n_vertices=n_vertices,
        n_subjects=args.n_subjects,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ff_mult=args.ff_mult,
        num_experts=args.num_experts,
        top_k=args.top_k,
        seq_len=args.seq_len,
        stoch_depth_max=args.stoch_depth_max,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M\n")

    use_amp = not args.no_amp and device.type == "cuda"

    if args.mode == "kd":
        train_phase2(
            model, train_loader, val_loader, device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            clip_grad=args.clip_grad,
            save_dir=args.save_dir,
            log_every=args.log_every,
            use_amp=use_amp,
        )
    else:
        train_phase3(
            model, train_loader, val_loader, device,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            backbone_lr=args.backbone_lr,
            clip_grad=args.clip_grad,
            save_dir=args.save_dir,
            log_every=args.log_every,
            use_amp=use_amp,
        )


if __name__ == "__main__":
    main()
