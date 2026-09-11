"""Loading helpers for Lightning checkpoints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from torch import nn


FRONTEND_HPARAMS = (
    "input_channels_per_hand",
    "hidden_dim",
    "conv_channels",
    "conv_kernels",
    "conv_strides",
    "use_multiscale_frontend",
    "multiscale_branch_kernels",
    "multiscale_branch_channels",
    "num_layers",
    "num_heads",
    "ffn_dim",
    "conv_kernel_size",
    "dropout_rate",
    "use_side_position_aux",
    "position_weight",
    "activity_weight",
)
FRONTEND_VARIANTS = (
    "conformer_base",
    "conformer_multiscale",
    "split_share",
    "split_share_position_aux",
)


def lightning_checkpoint_context():
    """Allow only the legacy configuration types used by our training files."""

    return torch.serialization.safe_globals(
        [
            DictConfig,
            ListConfig,
            AnyNode,
            Metadata,
            ContainerMetadata,
            defaultdict,
            list,
            dict,
            int,
            Any,
        ]
    )


def read_lightning_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read tensor weights and the configuration saved by Lightning."""
    with lightning_checkpoint_context():
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise ValueError("Lightning checkpoint must contain a state_dict mapping")
    return payload


def load_lightning_weights(module: nn.Module, path: str | Path) -> None:
    """Restore weights into the configured model with strict tensor checks."""
    payload = read_lightning_checkpoint(path)
    module.load_state_dict(payload["state_dict"], strict=True)


def _plain_hparam(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_hparam(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_hparam(item) for item in value]
    return value


def load_frontend_checkpoint(module: nn.Module, path: str | Path) -> None:
    """Load a frontend .ckpt and check its saved architecture settings."""
    payload = read_lightning_checkpoint(path)
    recorded = payload.get("hyper_parameters", {})
    configured = getattr(module, "hparams", {})
    if not isinstance(recorded, Mapping):
        raise ValueError("checkpoint hyper_parameters must be a mapping")
    keys = set(FRONTEND_HPARAMS) | {"input_channels"}
    for key in keys & set(recorded) & set(configured):
        if _plain_hparam(recorded[key]) != _plain_hparam(configured[key]):
            raise ValueError(
                f"checkpoint hyperparameter does not match configured frontend: {key}"
            )
    module.load_state_dict(payload["state_dict"], strict=True)
