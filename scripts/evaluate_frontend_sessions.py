#!/usr/bin/env python3
"""Evaluate a transcription frontend checkpoint on complete recording sessions.

This mirrors Meta's test protocol: one unpadded session per batch, followed by
one decoder call for the reassembled session emissions. CPU is the default.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from emg2qwerty import transforms
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.transcription.artifacts import portable_value, sha256_file
from emg2qwerty.transcription.checkpoint import load_frontend_checkpoint
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


FIELDS = (
    "split",
    "session",
    "prediction",
    "target",
    "CER",
    "IER",
    "DER",
    "SER",
    "input_samples",
    "emission_frames",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--config-name", default="frontend")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override; may be repeated (for example decoder=ctc_beam).",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--overlap-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _session_paths(config: DictConfig, split: str) -> list[Path]:
    root = Path(config.dataset.root)
    return [root / f"{entry['session']}.hdf5" for entry in config.dataset[split]]


def _build_transform(config: DictConfig, split: str) -> Any:
    return transforms.Compose([instantiate(item) for item in config.transforms[split]])


def _load_module(config: DictConfig, weights: Path, device: torch.device) -> Any:
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    load_frontend_checkpoint(module, weights)
    return module.eval().to(device)


def _plain_metric_values(values: Mapping[str, Any]) -> dict[str, float]:
    required = ("CER", "IER", "DER", "SER")
    if set(values) != set(required):
        raise ValueError("character-error metrics have an unexpected schema")
    return {
        key: float(value.detach().cpu().item() if torch.is_tensor(value) else value)
        for key, value in values.items()
    }


def _json_config(config: Any) -> Any:
    if isinstance(config, DictConfig) and "hydra" in config:
        keys = [key for key in config if key != "hydra"]
        config = OmegaConf.masked_copy(config, keys)
    value = OmegaConf.to_container(config, resolve=True)
    json.dumps(value)
    return value


def decoder_lm_artifact(config: DictConfig) -> dict[str, str] | None:
    """Return a portable identity for the configured decoder language model."""

    raw_path = config.decoder.get("lm_path")
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"decoder language model not found: {path}")
    return {"file": path.name, "sha256": sha256_file(path)}


def forward_session_in_chunks(
    module: Any,
    inputs: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    chunk_size: int,
    overlap_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run batch-size-one session inference with the overlap policy."""

    if inputs.ndim < 2 or inputs.size(1) != 1 or input_lengths.numel() != 1:
        raise ValueError("full-session evaluation requires batch size one")
    if chunk_size <= 0 or not 0 <= overlap_size < chunk_size:
        raise ValueError("require chunk_size > overlap_size >= 0")

    total_length = int(input_lengths[0])
    if total_length <= chunk_size:
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            module.forward(inputs, input_lengths),
        )

    featurizer = getattr(module.model, "cnn_featurizer", None)
    if featurizer is None or not hasattr(featurizer, "get_total_downsampling"):
        raise TypeError("transcription frontend must expose its CNN featurizer")
    downsampling = int(featurizer.get_total_downsampling())
    if downsampling <= 0:
        raise ValueError("frontend downsampling factor must be positive")
    output_overlap = overlap_size // downsampling
    step = chunk_size - overlap_size
    emissions: list[torch.Tensor] = []
    start = 0
    while start < total_length:
        end = min(start + chunk_size, total_length)
        chunk = inputs[start:end]
        chunk_length = input_lengths.new_tensor([chunk.size(0)])
        chunk_emissions, _ = module.forward(chunk, chunk_length)
        emissions.append(
            chunk_emissions if start == 0 else chunk_emissions[output_overlap:]
        )
        if end == total_length:
            break
        start += step

    joined = torch.cat(emissions, dim=0)
    return joined, input_lengths.new_tensor([joined.size(0)])


def _evaluate_session(
    *,
    module: Any,
    session_path: Path,
    split: str,
    transform: Any,
    chunk_size: int,
    overlap_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, object], LabelData, LabelData]:
    if not session_path.is_file():
        raise FileNotFoundError(f"session file not found: {session_path}")
    dataset = WindowedEMGDataset(
        session_path,
        window_length=None,
        padding=(0, 0),
        jitter=False,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=WindowedEMGDataset.collate,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    batch = next(iter(loader))
    inputs = batch["inputs"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    with torch.inference_mode():
        emissions, emission_lengths = forward_session_in_chunks(
            module,
            inputs,
            input_lengths,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )
    predictions = module.decoder.decode_batch(
        emissions=emissions.detach().float().cpu().numpy(),
        emission_lengths=emission_lengths.detach().cpu().numpy(),
    )
    target_length = int(batch["target_lengths"][0])
    target = LabelData.from_labels(batch["targets"][:target_length, 0].numpy())
    prediction = predictions[0]
    metric = CharacterErrorRates()
    metric.update(prediction=prediction, target=target)
    rates = _plain_metric_values(metric.compute())
    row: dict[str, object] = {
        "split": split,
        "session": session_path.stem,
        "prediction": prediction.text,
        "target": target.text,
        **rates,
        "input_samples": int(input_lengths[0]),
        "emission_frames": int(emission_lengths[0]),
    }
    if set(row) != set(FIELDS):
        raise RuntimeError("full-session CSV row does not match FIELDS")
    return row, prediction, target


def main() -> None:
    args = _parse_args()
    if args.chunk_size <= 0 or args.num_workers < 0:
        raise ValueError("chunk-size must be positive and num-workers non-negative")
    overlap_size = (
        max(800, int(args.chunk_size * 0.1))
        if args.overlap_size is None
        else args.overlap_size
    )
    if not 0 <= overlap_size < args.chunk_size:
        raise ValueError("require chunk_size > overlap_size >= 0")

    weights = args.weights.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name=args.config_name,
            overrides=args.override,
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(config)
    decoder_lm = decoder_lm_artifact(config)

    device = torch.device(args.device)
    module = _load_module(config, weights, device)
    transform = _build_transform(config, args.split)
    sessions = _session_paths(config, args.split)
    if not sessions:
        raise ValueError(f"no sessions configured for split {args.split!r}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = portable_value(
        _json_config(config),
        roots={
            "${EMG2QWERTY_DATA_ROOT}": Path(str(config.dataset.root)),
            "${REPOSITORY_ROOT}": config_dir.parent,
        },
    )
    if not isinstance(resolved_config, dict):
        raise TypeError("resolved frontend configuration must be a mapping")
    aggregate = CharacterErrorRates()
    rows: list[dict[str, object]] = []
    for session_path in sessions:
        row, prediction, target = _evaluate_session(
            module=module,
            session_path=session_path,
            split=args.split,
            transform=transform,
            chunk_size=args.chunk_size,
            overlap_size=overlap_size,
            num_workers=args.num_workers,
            device=device,
        )
        rows.append(row)
        aggregate.update(prediction=prediction, target=target)

    predictions_path = output_dir / "full_session_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "protocol": {
            "error_convention": "prediction_relative_to_reference_v2",
            "evaluation_unit": "full_session",
            "batch_size": 1,
            "window_length": None,
            "padding": [0, 0],
            "chunk_size": args.chunk_size,
            "overlap_size": overlap_size,
            "decoder": resolved_config["decoder"],
        },
        "metrics": _plain_metric_values(aggregate.compute()),
        "num_sessions": len(rows),
        "split": args.split,
        "frontend_checkpoint": {"file": weights.name, "sha256": sha256_file(weights)},
        "decoder_language_model": decoder_lm,
        "device": str(device),
        "resolved_config": resolved_config,
        "predictions": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
    }
    report_path = output_dir / "full_session_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
