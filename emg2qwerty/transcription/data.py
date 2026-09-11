# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 sEMG Beam-Aware Transcription contributors.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""Raw-sEMG datasets with optional, isolated side-position supervision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from emg2qwerty.data import EMGSessionData, WindowedEMGDataset
from emg2qwerty.transcription.frontend import conv_stack_out_length
from emg2qwerty.transcription.position_labels import (
    key_to_side_position_names,
    LEFT_POSITION_CHARSET,
    RIGHT_POSITION_CHARSET,
)
from emg2qwerty.transforms import Transform


@dataclass
class ToTensorRaw:
    """Stack left/right raw EMG as ``(time, hands, channels)``."""

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [
                torch.as_tensor(data[field], dtype=torch.float32)
                for field in self.fields
            ],
            dim=self.stack_dim,
        )


@dataclass(frozen=True)
class SidePositionTargetBuilder:
    """Generate time-aligned per-hand position and activity targets."""

    event_times: np.ndarray
    left_event_classes: np.ndarray
    right_event_classes: np.ndarray
    window_radius: int = 6
    gaussian_sigma: float = 5.0

    def __post_init__(self) -> None:
        arrays = (
            self.event_times,
            self.left_event_classes,
            self.right_event_classes,
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("side-position event arrays must be one-dimensional")
        if not (
            len(self.event_times)
            == len(self.left_event_classes)
            == len(self.right_event_classes)
        ):
            raise ValueError("side-position event arrays must have equal lengths")
        if not np.isfinite(self.event_times).all():
            raise ValueError("side-position event times must be finite")
        if len(self.event_times) > 1 and np.any(np.diff(self.event_times) < 0):
            raise ValueError("side-position events must be sorted by time")
        if self.window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        if not np.isfinite(self.gaussian_sigma) or self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be finite and non-negative")

    @classmethod
    def from_keystrokes(
        cls,
        keystrokes: Sequence[dict[str, object]],
        *,
        time_shift: float = -0.02,
        window_radius: int = 6,
        gaussian_sigma: float = 5.0,
    ) -> "SidePositionTargetBuilder":
        event_times: list[float] = []
        left_classes: list[int] = []
        right_classes: list[int] = []
        for keystroke in keystrokes:
            left_name, right_name = key_to_side_position_names(str(keystroke["key"]))
            event_times.append(float(str(keystroke["start"])) + time_shift)
            left_classes.append(
                LEFT_POSITION_CHARSET.vocab[left_name] if left_name is not None else -1
            )
            right_classes.append(
                RIGHT_POSITION_CHARSET.vocab[right_name]
                if right_name is not None
                else -1
            )
        return cls(
            event_times=np.asarray(event_times, dtype=np.float64),
            left_event_classes=np.asarray(left_classes, dtype=np.int16),
            right_event_classes=np.asarray(right_classes, dtype=np.int16),
            window_radius=window_radius,
            gaussian_sigma=gaussian_sigma,
        )

    def build(
        self,
        *,
        window_start: float,
        window_end: float,
        output_length: int,
        valid_start: float,
        valid_end: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if output_length < 0:
            raise ValueError("output_length must be non-negative")
        left = torch.full((output_length,), -1, dtype=torch.long)
        right = torch.full((output_length,), -1, dtype=torch.long)
        activity = torch.zeros(output_length, dtype=torch.float32)
        duration = window_end - window_start
        if output_length <= 0 or duration <= 0:
            return left, right, activity

        left_distance = torch.full((output_length,), float("inf"))
        right_distance = torch.full((output_length,), float("inf"))
        start_event = int(np.searchsorted(self.event_times, window_start, side="left"))
        end_event = int(np.searchsorted(self.event_times, window_end, side="right"))

        for event_index in range(start_event, end_event):
            event_time = float(self.event_times[event_index])
            if event_time < valid_start or event_time > valid_end:
                continue
            center = int(((event_time - window_start) / duration) * (output_length - 1))
            center = min(max(center, 0), output_length - 1)

            if self.gaussian_sigma > 0:
                gaussian_radius = int(3 * self.gaussian_sigma) + 1
                start = max(0, center - gaussian_radius)
                end = min(output_length, center + gaussian_radius + 1)
                indices = torch.arange(start, end, dtype=torch.float32)
                values = torch.exp(
                    -0.5 * ((indices - center) / self.gaussian_sigma).square()
                )
                activity[start:end] = torch.maximum(activity[start:end], values)

            start = max(0, center - self.window_radius)
            end = min(output_length, center + self.window_radius + 1)
            distances = (torch.arange(start, end) - center).abs().float()

            def assign(
                labels: torch.Tensor,
                nearest: torch.Tensor,
                class_index: int,
            ) -> None:
                if class_index < 0:
                    return
                label_slice = labels[start:end]
                nearest_slice = nearest[start:end]
                replace = distances < nearest_slice
                label_slice[replace] = class_index
                nearest_slice[replace] = distances[replace]
                labels[start:end] = label_slice
                nearest[start:end] = nearest_slice

            assign(left, left_distance, int(self.left_event_classes[event_index]))
            assign(right, right_distance, int(self.right_event_classes[event_index]))

        return left, right, activity


@dataclass
class SidePositionWindowedEMGDataset(WindowedEMGDataset):
    """Meta's window dataset plus isolated side-position auxiliary targets."""

    conv_kernels: Sequence[int] = (11, 3, 3)
    conv_strides: Sequence[int] = (5, 2, 2)
    position_label_window_radius: int = 6
    position_label_time_shift: float = -0.02
    position_activity_gaussian_sigma: float = 5.0
    _target_builder: SidePositionTargetBuilder | None = field(
        default=None, init=False, repr=False
    )

    def _ensure_target_builder(self) -> SidePositionTargetBuilder:
        if self._target_builder is None:
            self._target_builder = SidePositionTargetBuilder.from_keystrokes(
                self.session.keystrokes,
                time_shift=self.position_label_time_shift,
                window_radius=self.position_label_window_radius,
                gaussian_sigma=self.position_activity_gaussian_sigma,
            )
        return self._target_builder

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self, "session"):
            self.session = EMGSessionData(self.hdf5_path)

        offset = index * self.stride
        remaining = len(self.session) - offset
        if remaining <= 0:
            raise IndexError(f"index {index} out of bounds")
        effective_window_length = min(self.window_length, remaining)
        leftover = remaining - effective_window_length
        if effective_window_length < self.window_length and not self.include_tail:
            raise IndexError(f"index {index} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        window_start_index = max(offset - self.left_padding, 0)
        window_end_index = offset + effective_window_length + self.right_padding
        window = self.session[window_start_index:window_end_index]
        inputs = self.transform(window)
        if not torch.is_tensor(inputs):
            raise TypeError("transform must return a torch.Tensor")

        timestamps = window[EMGSessionData.TIMESTAMPS]
        valid_start = float(timestamps[offset - window_start_index])
        valid_end = float(
            timestamps[(offset + effective_window_length - 1) - window_start_index]
        )
        targets = torch.as_tensor(
            self.session.ground_truth(valid_start, valid_end).labels
        )

        output_length = int(
            conv_stack_out_length(len(window), self.conv_kernels, self.conv_strides)
        )
        left, right, activity = self._ensure_target_builder().build(
            window_start=float(timestamps[0]),
            window_end=float(timestamps[-1]),
            output_length=output_length,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        return inputs, targets, left, right, activity

    @staticmethod
    def collate(  # type: ignore[override]
        samples: Sequence[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> dict[str, torch.Tensor]:
        base_batch = WindowedEMGDataset.collate(
            [(inputs, targets) for inputs, targets, *_ in samples]
        )
        base_batch.update(
            {
                "left_position_labels": nn.utils.rnn.pad_sequence(
                    [sample[2] for sample in samples], padding_value=-1
                ),
                "right_position_labels": nn.utils.rnn.pad_sequence(
                    [sample[3] for sample in samples], padding_value=-1
                ),
                "position_activity_labels": nn.utils.rnn.pad_sequence(
                    [sample[4] for sample in samples], padding_value=0.0
                ),
            }
        )
        return base_batch


class RawEMGDataModule(pl.LightningDataModule):
    """Windowed raw-sEMG data for training and window-level diagnostics.

    Full-session metrics use the separate session evaluation entry point.
    """

    dataset_type = WindowedEMGDataset

    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sessions = {
            "train": train_sessions,
            "val": val_sessions,
            "test": test_sessions,
        }
        self.transforms = {
            "train": train_transform,
            "val": val_transform,
            "test": test_transform,
        }

    def _dataset_kwargs(self) -> dict[str, Any]:
        return {}

    def setup(self, stage: str | None = None) -> None:
        for phase in ("train", "val", "test"):
            datasets = [
                self.dataset_type(
                    path,
                    transform=self.transforms[phase],
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=phase == "train",
                    **self._dataset_kwargs(),
                )
                for path in self.sessions[phase]
            ]
            setattr(self, f"{phase}_dataset", ConcatDataset(datasets))

    def _loader(self, phase: str) -> DataLoader:
        return DataLoader(
            getattr(self, f"{phase}_dataset"),
            batch_size=self.batch_size,
            shuffle=phase == "train",
            num_workers=self.num_workers,
            collate_fn=self.dataset_type.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")


class SidePositionDataModule(RawEMGDataModule):
    """Windowed raw-sEMG data for the final side-position model only."""

    dataset_type = SidePositionWindowedEMGDataset

    def __init__(
        self,
        *args,
        conv_kernels: Sequence[int] = (11, 3, 3),
        conv_strides: Sequence[int] = (5, 2, 2),
        position_label_window_radius: int = 6,
        position_label_time_shift: float = -0.02,
        position_activity_gaussian_sigma: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv_kernels = tuple(conv_kernels)
        self.conv_strides = tuple(conv_strides)
        self.position_label_window_radius = position_label_window_radius
        self.position_label_time_shift = position_label_time_shift
        self.position_activity_gaussian_sigma = position_activity_gaussian_sigma

    def _dataset_kwargs(self) -> dict[str, object]:
        return {
            "conv_kernels": self.conv_kernels,
            "conv_strides": self.conv_strides,
            "position_label_window_radius": self.position_label_window_radius,
            "position_label_time_shift": self.position_label_time_shift,
            "position_activity_gaussian_sigma": self.position_activity_gaussian_sigma,
        }
