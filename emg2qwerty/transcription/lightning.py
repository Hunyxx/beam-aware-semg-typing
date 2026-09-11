# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 sEMG Beam-Aware Transcription contributors.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""Lightning wrappers for the raw-sEMG transcription frontends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.transcription.frontend import (
    ConformerCTC,
    FrontendOutput,
    SplitShareConformerCTC,
)


def side_position_losses(
    outputs: FrontendOutput,
    left_targets: torch.Tensor,
    right_targets: torch.Tensor,
    activity_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute side-position CE and shared activity BCE losses."""

    if (
        outputs.left_position_logits is None
        or outputs.right_position_logits is None
        or outputs.activity_logits is None
        or outputs.output_lengths is None
    ):
        raise ValueError("side-position outputs and output lengths are required")

    time = min(
        outputs.left_position_logits.size(0),
        outputs.right_position_logits.size(0),
        outputs.activity_logits.size(0),
        left_targets.size(0),
        right_targets.size(0),
        activity_targets.size(0),
    )
    left_logits = outputs.left_position_logits[:time]
    right_logits = outputs.right_position_logits[:time]
    activity_logits = outputs.activity_logits[:time].squeeze(-1)
    left_targets = left_targets[:time]
    right_targets = right_targets[:time]
    activity_targets = activity_targets[:time]

    valid_time = (
        torch.arange(time, device=outputs.output_lengths.device)[:, None]
        < outputs.output_lengths[None, :]
    )
    left_valid = valid_time & left_targets.ne(-1)
    right_valid = valid_time & right_targets.ne(-1)
    left_count = left_valid.sum()
    right_count = right_valid.sum()

    zero = left_logits.sum() * 0.0
    left_loss = (
        F.cross_entropy(left_logits[left_valid], left_targets[left_valid])
        if bool(left_count)
        else zero
    )
    right_loss = (
        F.cross_entropy(right_logits[right_valid], right_targets[right_valid])
        if bool(right_count)
        else zero
    )
    supervised = left_count + right_count
    position_loss = (
        (left_loss * left_count + right_loss * right_count) / supervised
        if bool(supervised)
        else zero
    )
    activity_loss = (
        F.binary_cross_entropy_with_logits(
            activity_logits[valid_time], activity_targets[valid_time]
        )
        if bool(valid_time.any())
        else zero
    )
    return position_loss, activity_loss


class _FrontendCTCModule(pl.LightningModule):
    model: nn.Module

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["hyper_parameters"] = OmegaConf.to_container(
            OmegaConf.create(dict(self.hparams)), resolve=True
        )

    def _finish_initialization(
        self,
        *,
        decoder: DictConfig | None,
        optimizer: DictConfig | None,
        lr_scheduler: DictConfig | None,
        use_side_position_aux: bool,
        position_weight: float,
        activity_weight: float,
    ) -> None:
        if position_weight < 0 or activity_weight < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        self.use_side_position_aux = use_side_position_aux
        self.position_weight = position_weight
        self.activity_weight = activity_weight
        self.ctc_loss = nn.CTCLoss(
            blank=charset().null_class,
            zero_infinity=True,
        )
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ("train", "val", "test")
            }
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

    @staticmethod
    def _batch_first(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            return inputs.permute(1, 2, 3, 0)
        if inputs.ndim == 3:
            return inputs.permute(1, 2, 0)
        raise ValueError("expected time-first raw sEMG with three or four dimensions")

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return cast(
            tuple[torch.Tensor, torch.Tensor | None],
            self.model(self._batch_first(inputs), input_lengths),
        )

    def _model_outputs(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> FrontendOutput:
        log_probs, output_lengths = self.forward(inputs, input_lengths)
        return FrontendOutput(log_probs=log_probs, output_lengths=output_lengths)

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        outputs = self._model_outputs(batch["inputs"], batch["input_lengths"])
        if outputs.output_lengths is None:
            raise RuntimeError(
                "the transcription models require explicit input lengths"
            )

        character_loss = self.ctc_loss(
            outputs.log_probs,
            targets.transpose(0, 1),
            outputs.output_lengths,
            target_lengths,
        )
        loss = character_loss
        batch_size = len(outputs.output_lengths)
        self.log(
            f"{phase}/char_loss",
            character_loss,
            batch_size=batch_size,
            sync_dist=True,
        )

        if self.use_side_position_aux:
            position_loss, activity_loss = side_position_losses(
                outputs,
                batch["left_position_labels"],
                batch["right_position_labels"],
                batch["position_activity_labels"],
            )
            loss = (
                loss
                + self.position_weight * position_loss
                + self.activity_weight * activity_loss
            )
            self.log(
                f"{phase}/position_loss",
                position_loss,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"{phase}/activity_loss",
                activity_loss,
                batch_size=batch_size,
                sync_dist=True,
            )

        predictions = self.decoder.decode_batch(
            emissions=outputs.log_probs.detach().float().cpu().numpy(),
            emission_lengths=outputs.output_lengths.detach().cpu().numpy(),
        )
        metric = self.metrics[f"{phase}_metrics"]
        targets_numpy = targets.detach().cpu().numpy()
        lengths_numpy = target_lengths.detach().cpu().numpy()
        for index, prediction in enumerate(predictions):
            target = LabelData.from_labels(targets_numpy[: lengths_numpy[index], index])
            metric.update(prediction=prediction, target=target)

        self.log(f"{phase}/loss", loss, batch_size=batch_size, sync_dist=True)
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        return self._step("test", batch)

    def _epoch_end(self, phase: str) -> None:
        metric = self.metrics[f"{phase}_metrics"]
        self.log_dict(metric.compute(), sync_dist=True)
        metric.reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
        )


class ConformerCTCModule(_FrontendCTCModule):
    """Base or multi-scale raw-sEMG Conformer ablation."""

    def __init__(
        self,
        input_channels: int = 32,
        hidden_dim: int = 256,
        conv_channels: Sequence[int] = (128, 64, 64),
        conv_kernels: Sequence[int] = (11, 3, 3),
        conv_strides: Sequence[int] = (5, 2, 2),
        num_layers: int = 6,
        num_heads: int = 16,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 15,
        dropout_rate: float = 0.2,
        use_multiscale_frontend: bool = False,
        multiscale_branch_kernels: Sequence[int] = (7, 11, 15),
        multiscale_branch_channels: int = 48,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ConformerCTC(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
            use_multiscale_frontend=use_multiscale_frontend,
            multiscale_branch_kernels=multiscale_branch_kernels,
            multiscale_branch_channels=multiscale_branch_channels,
        )
        self._finish_initialization(
            decoder=decoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_side_position_aux=False,
            position_weight=0.0,
            activity_weight=0.0,
        )


class SplitShareConformerCTCModule(_FrontendCTCModule):
    """Split-and-share model with concat fusion and optional side-position loss."""

    def __init__(
        self,
        input_channels_per_hand: int = 16,
        hidden_dim: int = 256,
        conv_channels: Sequence[int] = (128, 64, 64),
        conv_kernels: Sequence[int] = (11, 3, 3),
        conv_strides: Sequence[int] = (5, 2, 2),
        num_layers: int = 6,
        num_heads: int = 16,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 15,
        dropout_rate: float = 0.2,
        use_multiscale_frontend: bool = True,
        multiscale_branch_kernels: Sequence[int] = (7, 11, 15),
        multiscale_branch_channels: int = 48,
        use_side_position_aux: bool = False,
        position_weight: float = 0.0,
        activity_weight: float = 0.0,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SplitShareConformerCTC(
            input_channels_per_hand=input_channels_per_hand,
            hidden_dim=hidden_dim,
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
            use_multiscale_frontend=use_multiscale_frontend,
            multiscale_branch_kernels=multiscale_branch_kernels,
            multiscale_branch_channels=multiscale_branch_channels,
            use_side_position_aux=use_side_position_aux,
        )
        self._finish_initialization(
            decoder=decoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_side_position_aux=use_side_position_aux,
            position_weight=position_weight,
            activity_weight=activity_weight,
        )

    def _model_outputs(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> FrontendOutput:
        return cast(
            FrontendOutput,
            self.model.forward_with_aux(self._batch_first(inputs), input_lengths),
        )
