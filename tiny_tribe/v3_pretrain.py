"""Tiny-TRIBE v3 Phase 1: Self-Supervised Multimodal Pre-Training.

Pre-trains the fusion transformer on massive unlabeled multimodal data
using four tasks: MMR, CMC, NTP, TOP. No teacher. No fMRI.

Usage:
    python v3_pretrain.py \
        --features_dir ./pretrain_features \
        --save_dir ./checkpoints/phase1 \
        --epochs 25 --batch_size 32
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from tiny_tribe.v3_model import TinyTribeV3, ModalityProjector, TemporalMotionModule


# ── Pre-training Task Modules ──────────────────────────────────────────────────

class MMRPredictionHead(nn.Module):
    """Predict masked modality projected features from fusion output."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, T, 3, D) — transformer output reshaped back to modalities
        Returns:
            (B, T, D) — predicted features for the masked modality
        """
        return self.head(fused)


class CrossModalContrastiveLoss(nn.Module):
    """InfoNCE: temporally-aligned representations should be similar."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: (B*T, D)
            positive: (B*T, D)
            negatives: (M, D) where M is pool of negatives
        Returns:
            scalar loss
        """
        # Normalize
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negatives, dim=-1)

        pos_sim = (a * p).sum(dim=-1) / self.temperature  # (B*T,)
        neg_sim = (a.unsqueeze(1) @ n.T).squeeze(1) / self.temperature  # (B*T, M)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B*T, 1+M)
        labels = torch.zeros(len(a), dtype=torch.long, device=a.device)
        return F.cross_entropy(logits, labels)


class NTPHead(nn.Module):
    """Predict next timestep's fused representation from causal history."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, T, D) — pooled fused representation
        Returns:
            (B, T-1, D) — predicted next timestep
        """
        return self.head(fused[:, :-1, :])


# ── Phase 1 Pre-training Loss ─────────────────────────────────────────────────

class PretrainingLoss(nn.Module):
    """Combined Phase 1 self-supervised loss.

    L = 0.4 * MMR + 0.2 * CMC + 0.2 * NTP + 0.1 * TOP + 0.01 * aux
    """

    def __init__(self, hidden_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.mmr_head = MMRPredictionHead(hidden_dim)
        self.cmc_loss = CrossModalContrastiveLoss(temperature)
        self.ntp_head = NTPHead(hidden_dim)

    def forward(
        self,
        transformer_out: torch.Tensor,      # (B, T*3, D)
        modality_tokens: torch.Tensor,      # (B, T, 3, D) — original projected
        masked_idx: int,                    # which modality was masked (0,1,2)
        aux_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            transformer_out: output from MoE transformer (before pooling)
            modality_tokens: original projected tokens before interleaving
            masked_idx: which modality was zeroed out during forward
            aux_loss: MoE auxiliary loss
        Returns:
            dict with 'total' and component losses
        """
        B, T3, D = transformer_out.shape
        T = T3 // 3

        # Reshape transformer output back to per-modality
        reshaped = transformer_out.reshape(B, T, 3, D)

        losses = {}

        # ── MMR: predict masked modality ──
        masked_positions = reshaped[:, :, masked_idx, :]  # (B, T, D)
        mmr_pred = self.mmr_head(masked_positions)
        mmr_target = modality_tokens[:, :, masked_idx, :].detach()
        mmr_loss = F.mse_loss(mmr_pred, mmr_target)
        mmr_cos = 1.0 - F.cosine_similarity(
            mmr_pred.reshape(-1, D), mmr_target.reshape(-1, D), dim=-1
        ).mean()
        losses["mmr"] = mmr_loss + mmr_cos

        # ── CMC: contrastive on pooled representations ──
        pooled = reshaped.mean(dim=2)  # (B, T, D)
        flat = pooled.reshape(B * T, D)

        # Positive: same timestep, adjacent positions (within ±2)
        anchor = flat
        positive = flat  # self-positive within same sample

        # Negatives: random from batch (different samples or far timesteps)
        neg_pool = torch.cat([
            flat.roll(shifts=1, dims=0),
            flat.roll(shifts=-1, dims=0),
            flat.roll(shifts=B * T // 4, dims=0),
        ], dim=0)

        losses["cmc"] = self.cmc_loss(anchor, positive, neg_pool)

        # ── NTP: predict next timestep ──
        if T > 1:
            ntp_pred = self.ntp_head(pooled)  # (B, T-1, D)
            ntp_target = pooled[:, 1:, :].detach()
            ntp_loss = F.mse_loss(ntp_pred, ntp_target)
            ntp_cos = 1.0 - F.cosine_similarity(
                ntp_pred.reshape(-1, D), ntp_target.reshape(-1, D), dim=-1
            ).mean()
            losses["ntp"] = ntp_loss + ntp_cos
        else:
            losses["ntp"] = torch.tensor(0.0, device=transformer_out.device)

        # ── TOP: temporal order (simplified — shuffle detection) ──
        # We use a simple contrastive loss: original order vs reversed
        if T > 2:
            seg_len = max(2, T // 4)
            orig_order = pooled[:, :seg_len * 2, :].mean(dim=1)  # (B, D)
            rev_order = pooled[:, :seg_len * 2, :].flip(dims=[1]).mean(dim=1)
            # Simple binary: original should have higher self-similarity
            orig_sim = F.cosine_similarity(orig_order, orig_order, dim=-1)  # =1
            rev_sim = F.cosine_similarity(orig_order, rev_order, dim=-1)
            losses["top"] = (1.0 - orig_sim + rev_sim).mean()
        else:
            losses["top"] = torch.tensor(0.0, device=transformer_out.device)

        # ── Weighted total ──
        total = (
            0.4 * losses["mmr"]
            + 0.2 * losses["cmc"]
            + 0.2 * losses["ntp"]
            + 0.1 * losses["top"]
            + 0.01 * aux_loss
        )
        losses["total"] = total
        losses["aux"] = aux_loss
        return losses


# ── Pre-training Dataset ────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """Dataset for self-supervised pre-training.

    Loads cached backbone features from disk. Each file contains:
        text:  (T, 384)
        audio: (T, 384)
        video: (T, 640)
    """

    def __init__(self, feature_dir: str, seq_len: int = 60, stride: int = 30):
        self.seq_len = seq_len
        self.files = sorted(Path(feature_dir).glob("*.pt"))

        if not self.files:
            raise FileNotFoundError(f"No .pt files in {feature_dir}")

        # Build index: (file_idx, start_timestep)
        self.index = []
        for i, f in enumerate(self.files):
            data = torch.load(f, map_location="cpu", weights_only=True)
            T = data["text"].shape[0]
            for start in range(0, T - seq_len + 1, stride):
                self.index.append((i, start))

        print(f"PretrainDataset: {len(self.index)} segments from {len(self.files)} files")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        end = start + self.seq_len
        data = torch.load(self.files[file_idx], map_location="cpu", weights_only=True)

        text = data["text"][start:end].float()    # (T, 384)
        audio = data["audio"][start:end].float()   # (T, 384)
        video = data["video"][start:end].float()   # (T, 640)

        return {
            "text": text,
            "audio": audio,
            "video": video,
        }


# ── Pre-training Model Wrapper ─────────────────────────────────────────────────

class TinyTribePretrain(nn.Module):
    """TinyTribeV3 for pre-training (without output head).

    Projects modalities → interleaves → runs transformer → returns features.
    The output head (vertex projection) is not needed for pre-training.
    """

    def __init__(self, base_model: TinyTribeV3):
        super().__init__()
        self.model = base_model

    def forward(self, text, audio, video, masked_idx=None):
        """
        Args:
            text:  (B, T, 384)
            audio: (B, T, 384)
            video: (B, T, 640)
            masked_idx: which modality to mask (0=text, 1=audio, 2=video)
        Returns:
            dict with transformer output and original projected tokens
        """
        B = text.shape[0]
        device = text.device
        D = self.model.hidden_dim

        # Project each modality
        tp = self.model.text_proj(text)       # (B, T, D)
        ap = self.model.audio_proj(audio)     # (B, T, D)
        vp = self.model.video_proj(video)     # (B, T, D)
        vp = self.model.video_motion(vp)

        # Temporal alignment
        T = max(tp.shape[1], ap.shape[1], vp.shape[1])
        tp = self.model._align(tp, T)
        ap = self.model._align(ap, T)
        vp = self.model._align(vp, T)

        # Save original for MMR target
        original_projected = torch.stack([tp, ap, vp], dim=2)  # (B, T, 3, D)

        # Apply masking if specified
        if masked_idx is not None and self.training:
            if masked_idx == 0:
                tp = torch.zeros_like(tp)
            elif masked_idx == 1:
                ap = torch.zeros_like(ap)
            else:
                vp = torch.zeros_like(vp)

        # Add per-modality time embeddings
        t_idx = torch.arange(T, device=device)
        tp = tp + self.model.text_time_embed(t_idx)
        ap = ap + self.model.audio_time_embed(t_idx)
        vp = vp + self.model.video_time_embed(t_idx)

        # Add modality type embeddings
        mod = self.model.modality_embed(torch.arange(3, device=device))
        tp = tp + mod[0]
        ap = ap + mod[1]
        vp = vp + mod[2]

        # Interleave
        x = torch.stack([tp, ap, vp], dim=2)
        x = x.reshape(B, T * 3, D)
        x = x + self.model.pos_embed[:, :x.shape[1]]

        # Transformer
        total_aux = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.model.layers):
            x_out, aux = layer(x)
            total_aux = total_aux + aux

            # Stochastic depth
            drop_prob = (i / (self.model.num_layers - 1)) * self.model.stoch_depth_max
            if self.training and drop_prob > 0 and torch.rand(1).item() < drop_prob:
                pass
            else:
                x = x_out

        x = self.model.norm(x)

        return {
            "transformer_out": x,                 # (B, T*3, D)
            "original_projected": original_projected,  # (B, T, 3, D)
            "aux_loss": total_aux * self.model.aux_loss_weight,
        }


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_phase1(
    model: TinyTribePretrain,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 3e-4,
    wd: float = 1e-2,
    clip_grad: float = 1.0,
    save_dir: str = "./checkpoints/phase1",
    log_every: int = 50,
    use_amp: bool = True,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    criterion = PretrainingLoss(hidden_dim=model.model.hidden_dim)

    # Build optimizer
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": wd},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

    # Cosine with warmup
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"Phase 1: Self-Supervised Pre-Training")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  AMP: {use_amp}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in
                        ["total", "mmr", "cmc", "ntp", "top", "aux"]}
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)

            # Random masking strategy
            # 50% mask 1 modality, 25% mask 2, 25% none
            r = torch.rand(1).item()
            if r < 0.5:
                masked_idx = torch.randint(3, (1,)).item()
            elif r < 0.75:
                # Mask 2 modalities — keep 1
                keep_idx = torch.randint(3, (1,)).item()
                masked_idx = None  # handle outside
                # Zero out 2 modalities in forward
                B, T = text.shape[0], text.shape[1]
                if keep_idx != 0:
                    text = torch.zeros_like(text)
                if keep_idx != 1:
                    audio = torch.zeros_like(audio)
                if keep_idx != 2:
                    video = torch.zeros_like(video)
            else:
                masked_idx = None  # no masking

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(text, audio, video, masked_idx=masked_idx)
                losses = criterion(
                    out["transformer_out"],
                    out["original_projected"],
                    masked_idx=masked_idx if masked_idx is not None else 0,
                    aux_loss=out["aux_loss"],
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
                      f"| mmr={losses['mmr']:.4f} "
                      f"| cmc={losses['cmc']:.4f} "
                      f"| ntp={losses['ntp']:.4f} "
                      f"| lr={lr_now:.2e}")

        # Epoch summary
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation (just MMR reconstruction)
        val_loss = 0.0
        val_n = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                text = batch["text"].to(device)
                audio = batch["audio"].to(device)
                video = batch["video"].to(device)
                masked_idx = torch.randint(3, (1,)).item()

                out = model(text, audio, video, masked_idx=masked_idx)
                losses = criterion(
                    out["transformer_out"],
                    out["original_projected"],
                    masked_idx=masked_idx,
                    aux_loss=out["aux_loss"],
                )
                val_loss += losses["total"].item()
                val_n += 1

        val_loss /= max(val_n, 1)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{epochs} | "
              f"train_loss={epoch_losses['total']:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"time={elapsed:.0f}s")

        history.append({"epoch": epoch, **epoch_losses, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "history": history,
            }, Path(save_dir) / "phase1_best.pt")
            print(f"  ✓ New best val loss = {best_val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "history": history,
        }, Path(save_dir) / "phase1_latest.pt")

    print(f"\nPhase 1 complete. Best val loss = {best_val_loss:.4f}")
    return history


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tiny-TRIBE v3 Phase 1 Pre-training")
    p.add_argument("--features_dir", type=str, required=True,
                   help="Directory with .pt files containing text/audio/video features")
    p.add_argument("--save_dir",     type=str, default="./checkpoints/phase1")
    p.add_argument("--epochs",       type=int, default=25)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--seq_len",      type=int, default=60,
                   help="Segment length in timesteps (at 2Hz = 30 seconds)")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--wd",           type=float, default=1e-2)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--hidden_dim",   type=int, default=512)
    p.add_argument("--num_layers",   type=int, default=4)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--num_experts",  type=int, default=8)
    p.add_argument("--top_k",        type=int, default=2)
    p.add_argument("--ff_mult",      type=int, default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")

    # Build dataset
    train_ds = PretrainDataset(args.features_dir, seq_len=args.seq_len, stride=args.seq_len // 2)
    val_ds = PretrainDataset(args.features_dir, seq_len=args.seq_len, stride=args.seq_len)
    # Simple split: take last 10% for val
    val_size = max(1, int(0.1 * len(train_ds)))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Build base model (output head not used in pre-training)
    base = TinyTribeV3(
        n_vertices=1,   # dummy, not used
        n_subjects=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(device)

    model = TinyTribePretrain(base).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.1f}M")

    use_amp = not args.no_amp and device.type == "cuda"
    train_phase1(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        clip_grad=args.clip_grad,
        save_dir=args.save_dir,
        log_every=args.log_every,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
