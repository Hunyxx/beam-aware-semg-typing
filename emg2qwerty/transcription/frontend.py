# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 sEMG Beam-Aware Transcription contributors.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""Raw-sEMG Conformer frontends for transcription."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from emg2qwerty.charset import charset
from emg2qwerty.transcription.position_labels import (
    LEFT_POSITION_CHARSET,
    RIGHT_POSITION_CHARSET,
)


def conv_stack_out_length(
    input_length: int | torch.Tensor,
    kernels: Sequence[int] = (11, 3, 3),
    strides: Sequence[int] = (5, 2, 2),
) -> int | torch.Tensor:
    """Return the exact temporal length after the same-padded CNN stack."""

    if len(kernels) != len(strides):
        raise ValueError("kernels and strides must have the same length")
    length = input_length
    for kernel, stride in zip(kernels, strides):
        padding = kernel // 2
        length = (length + 2 * padding - kernel) // stride + 1
    return length


class MultiScaleRawEMGFrontEnd(nn.Module):
    """Parallel temporal convolutions followed by a learned 1x1 fusion."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        branch_out_channels: int = 48,
        branch_kernels: Sequence[int] = (7, 11, 15),
        stride: int = 5,
    ) -> None:
        super().__init__()
        if not branch_kernels:
            raise ValueError("branch_kernels must not be empty")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in branch_kernels):
            raise ValueError("branch kernels must be positive odd integers")

        self.branch_kernels = tuple(branch_kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        branch_out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                    ),
                    nn.InstanceNorm1d(branch_out_channels),
                    nn.ReLU(),
                )
                for kernel in self.branch_kernels
            ]
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(
                len(self.branch_kernels) * branch_out_channels,
                output_channels,
                kernel_size=1,
            ),
            nn.InstanceNorm1d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branches = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return self.fusion(branches)


class CNNFeaturizer(nn.Module):
    """Three-stage raw-sEMG CNN with 20x temporal downsampling."""

    def __init__(
        self,
        input_channels: int,
        conv_channels: Sequence[int] = (128, 64, 64),
        conv_kernels: Sequence[int] = (11, 3, 3),
        conv_strides: Sequence[int] = (5, 2, 2),
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        use_multiscale_frontend: bool = False,
        multiscale_branch_kernels: Sequence[int] = (7, 11, 15),
        multiscale_branch_channels: int = 48,
    ) -> None:
        super().__init__()
        if not (len(conv_channels) == len(conv_kernels) == len(conv_strides) == 3):
            raise ValueError(
                "the transcription frontend requires exactly three CNN stages"
            )

        conv_channels = tuple(conv_channels)
        self.conv_kernels = tuple(conv_kernels)
        self.conv_strides = tuple(conv_strides)
        self.total_downsampling = math.prod(self.conv_strides)

        layers: list[nn.Module] = []
        if use_multiscale_frontend:
            layers.append(
                MultiScaleRawEMGFrontEnd(
                    input_channels=input_channels,
                    output_channels=conv_channels[0],
                    branch_out_channels=multiscale_branch_channels,
                    branch_kernels=multiscale_branch_kernels,
                    stride=self.conv_strides[0],
                )
            )
            in_channels = conv_channels[0]
        else:
            layers.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        conv_channels[0],
                        kernel_size=self.conv_kernels[0],
                        stride=self.conv_strides[0],
                        padding=self.conv_kernels[0] // 2,
                    ),
                    nn.InstanceNorm1d(conv_channels[0]),
                    nn.ReLU(),
                ]
            )
            in_channels = conv_channels[0]

        for out_channels, kernel, stride in zip(
            conv_channels[1:], self.conv_kernels[1:], self.conv_strides[1:]
        ):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.feature_projection = nn.Sequential(
            nn.Linear(conv_channels[-1], hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return conv_stack_out_length(
            input_lengths, self.conv_kernels, self.conv_strides
        )

    def get_output_length(self, input_length: int) -> int:
        return int(
            conv_stack_out_length(input_length, self.conv_kernels, self.conv_strides)
        )

    def get_total_downsampling(self) -> int:
        return self.total_downsampling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            batch, bands, channels, time = inputs.shape
            inputs = inputs.reshape(batch, bands * channels, time)
        elif inputs.ndim != 3:
            raise ValueError(
                "expected (batch, channels, time) or "
                "(batch, hands, channels, time), got " + str(tuple(inputs.shape))
            )
        features = self.conv_layers(inputs).transpose(1, 2)
        return self.feature_projection(features).transpose(0, 1)


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    """Convolutional positional embedding used by the reported encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_conv_pos_embeddings: int = 128,
        num_conv_pos_embedding_groups: int = 16,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
        )
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time = hidden_states.size(0)
        outputs = self.conv(hidden_states.permute(1, 2, 0))
        outputs = self.activation(outputs)[..., :time]
        if outputs.size(-1) < time:
            outputs = F.pad(outputs, (0, time - outputs.size(-1)))
        return outputs.permute(2, 0, 1)


class ConformerEncoder(nn.Module):
    """Non-causal torchaudio Conformer encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 16,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 15,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos_embedding = Wav2Vec2PositionalConvEmbedding(hidden_dim)
        self.conformer_layers = nn.ModuleList(
            [
                torchaudio.models.Conformer(
                    input_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    num_layers=1,
                    depthwise_conv_kernel_size=conv_kernel_size,
                    dropout=dropout_rate,
                ).conformer_layers[0]
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        sequence_length, batch_size = inputs.shape[:2]
        outputs = self.dropout(inputs + self.pos_embedding(inputs))
        padding_mask = None
        if lengths is not None:
            padding_mask = (
                torch.arange(sequence_length, device=inputs.device)
                .expand(batch_size, sequence_length)
                .ge(lengths.unsqueeze(1))
            )
        for layer in self.conformer_layers:
            outputs = layer(outputs, padding_mask)
        return self.final_layer_norm(outputs)


@dataclass
class FrontendOutput:
    log_probs: torch.Tensor
    output_lengths: torch.Tensor | None
    left_position_logits: torch.Tensor | None = None
    right_position_logits: torch.Tensor | None = None
    activity_logits: torch.Tensor | None = None


class ConformerCTC(nn.Module):
    """Unsplit raw-sEMG Conformer for the base and multi-scale ablations."""

    def __init__(
        self,
        input_channels: int = 32,
        hidden_dim: int = 256,
        num_classes: int | None = None,
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
    ) -> None:
        super().__init__()
        self.cnn_featurizer = CNNFeaturizer(
            input_channels=input_channels,
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            use_multiscale_frontend=use_multiscale_frontend,
            multiscale_branch_kernels=multiscale_branch_kernels,
            multiscale_branch_channels=multiscale_branch_channels,
        )
        self.conformer_encoder = ConformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
        )
        output_classes = charset().num_classes if num_classes is None else num_classes
        if output_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.output_projection = nn.Linear(hidden_dim, output_classes)

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.cnn_featurizer(inputs)
        output_lengths = (
            self.cnn_featurizer.get_output_lengths(input_lengths)
            if input_lengths is not None
            else None
        )
        encoded = self.conformer_encoder(features, output_lengths)
        return F.log_softmax(self.output_projection(encoded), dim=-1), output_lengths


class SplitShareConformerCTC(nn.Module):
    """Shared per-hand Conformer with fixed concatenation fusion.

    ``use_side_position_aux`` is the only optional supervision path.
    """

    def __init__(
        self,
        input_channels_per_hand: int = 16,
        hidden_dim: int = 256,
        num_classes: int | None = None,
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
    ) -> None:
        super().__init__()
        self.input_channels_per_hand = input_channels_per_hand
        self.use_side_position_aux = use_side_position_aux
        self.cnn_featurizer = CNNFeaturizer(
            input_channels=input_channels_per_hand,
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            use_multiscale_frontend=use_multiscale_frontend,
            multiscale_branch_kernels=multiscale_branch_kernels,
            multiscale_branch_channels=multiscale_branch_channels,
        )
        self.total_downsampling = self.cnn_featurizer.get_total_downsampling()
        self.conformer_encoder = ConformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
        )
        output_classes = charset().num_classes if num_classes is None else num_classes
        if output_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.output_projection = nn.Linear(hidden_dim * 2, output_classes)

        self.left_position_head: nn.Linear | None = None
        self.right_position_head: nn.Linear | None = None
        self.position_activity_head: nn.Linear | None = None
        if use_side_position_aux:
            self.left_position_head = nn.Linear(
                hidden_dim, LEFT_POSITION_CHARSET.num_classes
            )
            self.right_position_head = nn.Linear(
                hidden_dim, RIGHT_POSITION_CHARSET.num_classes
            )
            self.position_activity_head = nn.Linear(hidden_dim, 1)

    def encode(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if inputs.ndim == 4:
            if inputs.size(1) != 2 or inputs.size(2) != self.input_channels_per_hand:
                raise ValueError(
                    "expected two hand streams with configured channel count"
                )
            left, right = inputs[:, 0], inputs[:, 1]
        elif inputs.ndim == 3:
            expected = 2 * self.input_channels_per_hand
            if inputs.size(1) != expected:
                raise ValueError(f"expected {expected} concatenated channels")
            left, right = inputs.split(self.input_channels_per_hand, dim=1)
        else:
            raise ValueError("expected a three- or four-dimensional raw-sEMG tensor")

        batch_size = left.size(0)
        both = torch.cat([left, right], dim=0)
        output_lengths = None
        both_lengths = None
        if input_lengths is not None:
            if input_lengths.ndim != 1 or input_lengths.numel() != batch_size:
                raise ValueError("input_lengths must contain one length per batch item")
            output_lengths = self.cnn_featurizer.get_output_lengths(
                input_lengths
            ).clamp_min(1)
            both_lengths = torch.cat([output_lengths, output_lengths])
        encoded = self.conformer_encoder(self.cnn_featurizer(both), both_lengths)
        encoded_left, encoded_right = encoded.split(batch_size, dim=1)
        return encoded_left, encoded_right, output_lengths

    @staticmethod
    def _concat_hands(
        left: torch.Tensor,
        right: torch.Tensor,
        output_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        fused = torch.cat([left, right], dim=-1)
        if output_lengths is not None:
            valid = (
                torch.arange(fused.size(0), device=fused.device)[:, None]
                < output_lengths[None, :]
            ).unsqueeze(-1)
            fused = fused * valid.to(fused.dtype)
        return fused

    def forward_with_aux(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> FrontendOutput:
        left, right, output_lengths = self.encode(inputs, input_lengths)
        fused = self._concat_hands(left, right, output_lengths)

        left_logits = right_logits = activity_logits = None
        if self.use_side_position_aux:
            assert self.left_position_head is not None
            assert self.right_position_head is not None
            assert self.position_activity_head is not None
            left_logits = self.left_position_head(left)
            right_logits = self.right_position_head(right)
            activity_logits = self.position_activity_head(0.5 * (left + right))

        return FrontendOutput(
            log_probs=F.log_softmax(self.output_projection(fused), dim=-1),
            output_lengths=output_lengths,
            left_position_logits=left_logits,
            right_position_logits=right_logits,
            activity_logits=activity_logits,
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        left, right, output_lengths = self.encode(inputs, input_lengths)
        fused = self._concat_hands(left, right, output_lengths)
        log_probs = F.log_softmax(self.output_projection(fused), dim=-1)
        return log_probs, output_lengths
