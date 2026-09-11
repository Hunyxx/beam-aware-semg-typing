# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class LinearWarmupCosineAnnealingLR(SequentialLR):
    """Linearly warm up, then apply cosine annealing using native PyTorch."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-6,
    ) -> None:
        warmup_epochs = int(warmup_epochs)
        max_epochs = int(max_epochs)
        warmup_start_lr = float(warmup_start_lr)
        eta_min = float(eta_min)
        if warmup_epochs < 1:
            raise ValueError("warmup_epochs must be at least one")
        if max_epochs <= warmup_epochs:
            raise ValueError("max_epochs must be greater than warmup_epochs")
        if warmup_start_lr <= 0:
            raise ValueError("warmup_start_lr must be positive")
        if eta_min < 0:
            raise ValueError("eta_min must be non-negative")

        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if not base_lrs or any(lr <= 0 for lr in base_lrs):
            raise ValueError("optimizer learning rates must be positive")
        if len(set(base_lrs)) != 1:
            raise ValueError(
                "all optimizer parameter groups must use the same learning rate"
            )
        base_lr = base_lrs[0]
        if warmup_start_lr > base_lr:
            raise ValueError(
                "warmup_start_lr must not exceed the optimizer learning rate"
            )

        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_lr / base_lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        super().__init__(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )


def instantiate_optimizer_and_scheduler(
    params: Iterable[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def cpus_per_task(gpus_per_node: int, tasks_per_node: int, num_workers: int) -> int:
    """Number of CPUs to request per task per node taking into account
    the number of GPUs and dataloading workers."""
    gpus_per_task = gpus_per_node // tasks_per_node
    if gpus_per_task <= 0:
        return num_workers + 1
    else:
        return (num_workers + 1) * gpus_per_task
