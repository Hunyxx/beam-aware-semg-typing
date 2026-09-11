#!/usr/bin/env python3
"""Benchmark batch-one latency of the released transcription frontend.

HDF5 reads are outside the timed region. The timed preprocessing component
contains the configured test transform and host-to-device copy. The benchmark
defaults to CPU; another device is used only when passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emg2qwerty import transforms
from emg2qwerty.data import EMGSessionData
from emg2qwerty.transcription.artifacts import sha256_file
from emg2qwerty.transcription.checkpoint import load_frontend_checkpoint
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--config-name", default="frontend")
    parser.add_argument("--user", default="generic")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; may be repeated.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-windows", type=int, default=100)
    parser.add_argument("--warmup-windows", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--max-labels-per-timestep", type=int, default=10)
    lm_options = parser.add_mutually_exclusive_group()
    lm_options.add_argument(
        "--lm-path",
        type=Path,
        help=("Character language model override; defaults to the configured model."),
    )
    lm_options.add_argument(
        "--no-lm",
        action="store_true",
        help="Explicitly benchmark beam search without a language model.",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty latency sample")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def autocast_context(device: torch.device, precision: str):
    enabled = precision != "fp32"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def load_config(args: argparse.Namespace) -> DictConfig:
    config_dir = args.config_dir.expanduser().resolve()
    overrides = ["decoder=ctc_beam", f"user={args.user}", *args.override]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name=args.config_name, overrides=overrides, return_hydra_config=True
        )
    HydraConfig.instance().set_config(config)
    raw_lm = (
        None
        if args.no_lm
        else (
            args.lm_path if args.lm_path is not None else config.decoder.get("lm_path")
        )
    )
    lm_path = None
    if raw_lm is not None:
        lm_path = Path(str(raw_lm)).expanduser().resolve()
        if not lm_path.is_file():
            raise FileNotFoundError(lm_path)
    with open_dict(config):
        config.decoder.beam_size = int(args.beam_size)
        config.decoder.max_labels_per_timestep = int(args.max_labels_per_timestep)
        config.decoder.lm_path = None if lm_path is None else str(lm_path)
    return config


def session_paths(config: DictConfig, split: str) -> list[Path]:
    root = Path(config.dataset.root)
    return [root / f"{entry['session']}.hdf5" for entry in config.dataset[split]]


def build_transform(config: DictConfig, split: str) -> Any:
    del split
    return transforms.Compose([instantiate(item) for item in config.transforms.val])


def lm_artifact_manifest(path: Path | None) -> dict[str, str] | None:
    """Build a portable identity for an optional character language model."""

    if path is None:
        return None
    return {
        "file": path.name,
        "sha256": sha256_file(path),
    }


def validate_device(device: torch.device, precision: str) -> None:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this benchmark supports CPU or explicitly requested CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested explicitly but is unavailable")
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("fp16 CPU inference is unsupported; use fp32 or bf16")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_windows <= 0 or args.warmup_windows < 0:
        raise ValueError("--num-windows must be positive and warmup non-negative")
    if args.beam_size <= 0 or args.max_labels_per_timestep <= 0:
        raise ValueError("beam parameters must be positive")
    if args.sample_rate_hz <= 0:
        raise ValueError("--sample-rate-hz must be positive")

    weights = args.weights.expanduser().resolve()
    device = torch.device(args.device)
    validate_device(device, args.precision)
    config = load_config(args)
    lm_path = None if config.decoder.lm_path is None else Path(config.decoder.lm_path)
    if args.beam_size != int(config.decoder.beam_size):
        raise AssertionError("resolved decoder beam size does not match CLI")

    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    load_frontend_checkpoint(module, weights)
    module = module.eval().to(device)

    window_length = int(config.datamodule.window_length)
    padding = tuple(int(value) for value in config.datamodule.padding)
    signal_seconds = window_length / args.sample_rate_hz
    transform = build_transform(config, args.split)

    candidates: list[tuple[Path, int]] = []
    for path in session_paths(config, args.split):
        if not path.is_file():
            raise FileNotFoundError(path)
        with EMGSessionData(path) as session:
            count = max((len(session) - window_length) // window_length + 1, 0)
        candidates.extend((path, index) for index in range(count))
    if not candidates:
        raise RuntimeError("the configured split has no complete frontend windows")
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: min(args.num_windows, len(candidates))]

    def load_raw(path: Path, index: int) -> np.ndarray:
        offset = index * window_length
        start = max(offset - padding[0], 0)
        end = offset + window_length + padding[1]
        with EMGSessionData(path) as session:
            return session[start:end]

    timings: dict[str, list[float]] = {
        "preprocessing": [],
        "forward": [],
        "beam_search": [],
        "total": [],
    }

    @torch.inference_mode()
    def run_one(raw: np.ndarray, *, record: bool) -> None:
        total_start = time.perf_counter_ns()
        inputs_cpu = transform(raw)
        inputs = inputs_cpu.unsqueeze(1).to(device, non_blocking=False)
        input_lengths = torch.tensor([len(inputs_cpu)], dtype=torch.long, device=device)
        sync(device)
        preprocessing_end = time.perf_counter_ns()

        with autocast_context(device, args.precision):
            emissions, output_lengths = module.forward(inputs, input_lengths)
        sync(device)
        forward_end = time.perf_counter_ns()

        if output_lengths is None:
            raise RuntimeError("frontend did not return output lengths")
        if output_lengths.numel() != 1:
            raise RuntimeError(
                "batch-one benchmark expected exactly one frontend output length"
            )
        valid_length = int(output_lengths[0].item())
        if valid_length <= 0 or valid_length > int(emissions.shape[0]):
            raise RuntimeError(
                "frontend returned an invalid output length: "
                f"{valid_length} for {int(emissions.shape[0])} emission steps"
            )
        emission = emissions[:valid_length, 0].detach().float().cpu().numpy()
        module.decoder.reset()
        module.decoder.decode(
            emission,
            np.arange(valid_length, dtype=np.float64),
            finish=True,
        )
        beam_end = time.perf_counter_ns()

        if record:
            timings["preprocessing"].append((preprocessing_end - total_start) / 1e6)
            timings["forward"].append((forward_end - preprocessing_end) / 1e6)
            timings["beam_search"].append((beam_end - forward_end) / 1e6)
            timings["total"].append((beam_end - total_start) / 1e6)

    warmup = min(args.warmup_windows, len(selected))
    for path, index in selected[:warmup]:
        run_one(load_raw(path, index), record=False)
    for number, (path, index) in enumerate(selected, start=1):
        run_one(load_raw(path, index), record=True)
        if number % 25 == 0 or number == len(selected):
            print(f"Measured {number}/{len(selected)} windows", flush=True)

    stats = {name: summarize_ms(values) for name, values in timings.items()}
    stats["rtf"] = {
        "median": stats["total"]["median_ms"] / (signal_seconds * 1000.0),
        "p95": stats["total"]["p95_ms"] / (signal_seconds * 1000.0),
    }
    result: dict[str, Any] = {
        "frontend_checkpoint": {"file": weights.name, "sha256": sha256_file(weights)},
        "config_name": args.config_name,
        "user_config": args.user,
        "split": args.split,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "precision": args.precision,
        "batch_size": 1,
        "num_windows": len(selected),
        "warmup_windows": warmup,
        "window_samples": window_length,
        "window_seconds": signal_seconds,
        "initial_window_seconds": signal_seconds,
        "update_interval_seconds": signal_seconds,
        "timing_scope": "preprocessing_forward_and_beam_search_excluding_disk_io",
        "context_padding_samples": list(padding),
        "beam_size": args.beam_size,
        "max_labels_per_timestep": args.max_labels_per_timestep,
        "decoder_language_model": lm_artifact_manifest(lm_path),
        "latency": stats,
    }

    print("\nComponent                         Median latency    p95 latency")
    labels = (
        ("preprocessing", "Preprocessing / transfer"),
        ("forward", "Frontend forward"),
        ("beam_search", "CTC beam search"),
        ("total", "Total frontend latency"),
    )
    for key, label in labels:
        print(
            f"{label:<34}{stats[key]['median_ms']:>9.2f} ms"
            f"{stats[key]['p95_ms']:>14.2f} ms"
        )
    print(
        f"RTF                               {stats['rtf']['median']:.4f} median, "
        f"{stats['rtf']['p95']:.4f} p95"
    )
    print(
        f"First provisional output requires one complete {signal_seconds:.1f} s "
        f"window; updates occur every {signal_seconds:.1f} s."
    )
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
