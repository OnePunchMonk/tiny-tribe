"""Knowledge distillation losses for Tiny-TRIBE.

Combined loss from research doc:
    L_total = 0.7 * MSE(student_pred, fmri_target)      # Task loss
            + 0.2 * MSE(student_pred, teacher_pred)       # Output KD
            + 0.1 * CKA(student_fusion, teacher_fusion)   # Feature KD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear Centered Kernel Alignment between two feature matrices.

    Args:
        X: (B*T, D1) student features
        Y: (B*T, D2) teacher features
    Returns:
        scalar CKA similarity (higher = more similar)
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = X.T @ X  # (D1, D1)
    YtY = Y.T @ Y  # (D2, D2)
    XtY = X.T @ Y  # (D1, D2)

    # Frobenius norms
    hsic_xy = (XtY ** 2).sum()
    hsic_xx = (XtX ** 2).sum()
    hsic_yy = (YtY ** 2).sum()

    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)
    return cka


class FeatureProjector(nn.Module):
    """Bottleneck linear to match student features to teacher dimension for KD."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationLoss(nn.Module):
    """Combined distillation loss for Tiny-TRIBE training.

    Components:
        1. Task loss: MSE(student_pred, fmri_target)
        2. Output KD: MSE(student_pred, teacher_pred)
        3. Feature KD: CKA(student_fusion, teacher_fusion) per layer
    """

    def __init__(
        self,
        task_weight: float = 0.7,
        output_kd_weight: float = 0.2,
        feature_kd_weight: float = 0.1,
        student_hidden: int = 512,
        teacher_hidden: int = 1152,
        num_student_layers: int = 4,
        num_teacher_layers: int = 8,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.output_kd_weight = output_kd_weight
        self.feature_kd_weight = feature_kd_weight

        self.mse = nn.MSELoss()

        # Feature projectors for intermediate layer matching
        # Map student layers to teacher layers (block-wise matching)
        self.feature_projectors = nn.ModuleList([
            FeatureProjector(student_hidden, teacher_hidden)
            for _ in range(num_student_layers)
        ])

        # Map student layers to teacher layers
        # e.g., 4 student layers → match to teacher layers [1, 3, 5, 7] (0-indexed)
        step = num_teacher_layers / num_student_layers
        self.layer_mapping = [int(round((i + 1) * step)) - 1 for i in range(num_student_layers)]

    def forward(
        self,
        student_pred: torch.Tensor,
        fmri_target: torch.Tensor,
        teacher_pred: torch.Tensor | None = None,
        student_intermediates: list[torch.Tensor] | None = None,
        teacher_intermediates: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            student_pred: (B, n_vertices, T) student output
            fmri_target: (B, n_vertices, T) ground truth fMRI
            teacher_pred: (B, n_vertices, T) teacher output (optional)
            student_intermediates: list of (B, T, D_student) per layer
            teacher_intermediates: list of (B, T, D_teacher) per layer
        Returns:
            dict with 'total', 'task', 'output_kd', 'feature_kd'
        """
        losses = {}

        # 1. Task loss
        task_loss = self.mse(student_pred, fmri_target)
        losses["task"] = task_loss
        total = self.task_weight * task_loss

        # 2. Output KD loss
        if teacher_pred is not None:
            output_kd = self.mse(student_pred, teacher_pred.detach())
            losses["output_kd"] = output_kd
            total = total + self.output_kd_weight * output_kd

        # 3. Feature KD loss (CKA on intermediate layers)
        if (
            student_intermediates is not None
            and teacher_intermediates is not None
            and len(student_intermediates) > 0
        ):
            feature_kd = torch.tensor(0.0, device=student_pred.device)
            n_matched = 0
            for s_idx, t_idx in enumerate(self.layer_mapping):
                if s_idx >= len(student_intermediates) or t_idx >= len(teacher_intermediates):
                    continue
                s_feat = student_intermediates[s_idx]  # (B, T, D_s)
                t_feat = teacher_intermediates[t_idx].detach()  # (B, T, D_t)

                # Flatten batch and time
                s_flat = s_feat.reshape(-1, s_feat.shape[-1])
                t_flat = t_feat.reshape(-1, t_feat.shape[-1])

                # Project student to teacher dim
                s_proj = self.feature_projectors[s_idx](s_flat)

                # CKA loss (1 - CKA, so lower = more similar)
                cka = linear_cka(s_proj, t_flat)
                feature_kd = feature_kd + (1.0 - cka)
                n_matched += 1

            if n_matched > 0:
                feature_kd = feature_kd / n_matched
            losses["feature_kd"] = feature_kd
            total = total + self.feature_kd_weight * feature_kd

        losses["total"] = total
        return losses


class TaskOnlyLoss(nn.Module):
    """Simple MSE loss for Stage 1 training (no teacher available yet)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_pred: torch.Tensor,
        fmri_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss = self.mse(student_pred, fmri_target)
        return {"total": loss, "task": loss}
