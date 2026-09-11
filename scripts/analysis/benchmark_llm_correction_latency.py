#!/usr/bin/env python3
"""Benchmark batch-one latency of the released beam-aware corrector.

The input chunks are built by the transcription dataset with consecutive windows.
History duration is reported separately from post-segment correction latency.
The benchmark defaults to CPU; another device is used only when explicit.
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

from emg2qwerty.transcription.artifacts import sha256_file
from emg2qwerty.transcription.beam_corrector import load_beam_checkpoint
from emg2qwerty.transcription.beam_dataset import (
    BeamCorrectionDataset,
    resolve_topk_files,
)
from emg2qwerty.transcription.beam_protocol import (
    build_structural_bad_words_ids,
    parse_model_output,
    shared_output_filter,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument(
        "--include-empty-targets",
        action="store_true",
        help="Retain windows with empty targets.",
    )
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype")
    parser.add_argument("--num-chunks", type=int, default=100)
    parser.add_argument("--warmup-chunks", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty latency sample")
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def validate_device(device: torch.device, dtype: str) -> None:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this benchmark supports CPU or explicitly requested CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested explicitly but is unavailable")
    normalized_dtype = dtype.lower().replace("torch.", "")
    if device.type == "cpu" and normalized_dtype in {"fp16", "float16"}:
        raise ValueError("fp16 CPU inference is unsupported; use fp32 or bf16")


def topk_input_manifest(root: Path, files: Sequence[Path]) -> list[dict[str, str]]:
    root = root.resolve()
    entries: list[dict[str, str]] = []
    for path in files:
        path = path.resolve()
        if root.is_file():
            if path != root:
                raise ValueError("resolved top-k file does not match top-k root")
            name = path.name
        else:
            try:
                name = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "resolved top-k file is outside the declared top-k root"
                ) from exc
        entries.append({"name": name, "sha256": sha256_file(path)})
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_chunks <= 0 or args.warmup_chunks < 0:
        raise ValueError("--num-chunks must be positive and warmup non-negative")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")

    topk_root = args.topk_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    top_k, chunk_size = args.top_k, args.chunk_size
    if top_k <= 0 or chunk_size <= 0:
        raise ValueError("top-k and chunk-size must be positive")
    requested_filter_empty = not args.include_empty_targets
    device = torch.device(args.device)
    topk_files = resolve_topk_files(topk_root)
    dataset = BeamCorrectionDataset(
        topk_files,
        split=args.split,
        chunk_size=chunk_size,
        top_k=top_k,
        filter_empty_targets=requested_filter_empty,
        require_consecutive_windows=True,
        drop_incomplete=True,
        shuffle_candidates=False,
    )
    if not dataset:
        raise RuntimeError("no complete consecutive chunks were found")
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    measured = [dataset[index] for index in indices[: args.num_chunks]]

    module = load_beam_checkpoint(
        checkpoint,
        device=device,
        dtype=args.dtype,
        model_id=args.model_id,
        base_model_revision=args.base_model_revision,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_source_tokens,
    )
    dtype = module.dtype_name
    validate_device(device, dtype)
    max_source_tokens = module.max_seq_length
    max_new_tokens = module.max_new_tokens
    timings: dict[str, list[float]] = {
        "prompt": [],
        "generation": [],
        "filter": [],
        "total": [],
    }
    output_token_counts: list[float] = []

    @torch.inference_mode()
    def run_one(sample: dict[str, object], *, record: bool) -> None:
        total_start = time.perf_counter_ns()
        prompt = module.format_prompt(str(sample["payload"]))
        original_padding_side = module.tokenizer.padding_side
        module.tokenizer.padding_side = "left"
        try:
            encoded = module.tokenizer(
                [prompt],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            module.tokenizer.padding_side = original_padding_side
        source_tokens = int(encoded["attention_mask"].sum().item())
        if source_tokens > max_source_tokens:
            raise ValueError(
                f"prompt has {source_tokens} tokens, exceeding "
                f"the recorded max_seq_length={max_source_tokens}"
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        sync(device)
        prompt_end = time.perf_counter_ns()

        generation_options = dict(module.generation_kwargs)
        generation_options["do_sample"] = False
        generation_options["max_new_tokens"] = module.max_new_tokens
        generation_options["bad_words_ids"] = build_structural_bad_words_ids(
            module.tokenizer
        )
        generation_options.setdefault(
            "pad_token_id", int(module.tokenizer.pad_token_id)
        )
        generation_options.setdefault(
            "eos_token_id", int(module.tokenizer.eos_token_id)
        )
        generated = module.model.generate(**encoded, **generation_options)
        sync(device)
        generation_end = time.perf_counter_ns()

        prompt_width = encoded["input_ids"].shape[1]
        continuation = generated[:, prompt_width:]
        raw = module.tokenizer.decode(continuation[0], skip_special_tokens=False)
        parsed = parse_model_output(raw, tokenizer=module.tokenizer)
        shared_output_filter(
            str(sample["current_prediction"]),
            sample["candidates_per_window"],  # type: ignore[arg-type]
            parsed,
        )
        filter_end = time.perf_counter_ns()

        if record:
            timings["prompt"].append((prompt_end - total_start) / 1e6)
            timings["generation"].append((generation_end - prompt_end) / 1e6)
            timings["filter"].append((filter_end - generation_end) / 1e6)
            timings["total"].append((filter_end - total_start) / 1e6)
            output_token_counts.append(float(continuation.shape[1]))

    warmup = min(args.warmup_chunks, len(measured))
    for sample in measured[:warmup]:
        run_one(sample, record=False)
    for number, sample in enumerate(measured, start=1):
        run_one(sample, record=True)
        if number % 10 == 0 or number == len(measured):
            print(f"Measured {number}/{len(measured)} chunks", flush=True)

    stats = {key: summarize_ms(values) for key, values in timings.items()}
    history_seconds = chunk_size * args.window_seconds
    result: dict[str, Any] = {
        "checkpoint": {"file": checkpoint.name, "sha256": sha256_file(checkpoint)},
        "topk_inputs": topk_input_manifest(topk_root, topk_files),
        "base_model_id": module.model_id,
        "base_model_revision": module.base_model_revision,
        "resolved_model_revision": module.resolved_model_revision,
        "prompt_sha256": module.prompt_digest,
        "split": args.split,
        "protocol": ("filtered_targets" if requested_filter_empty else "full_coverage"),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "dtype": str(next(module.model.parameters()).dtype),
        "recorded_dtype": dtype,
        "batch_size": 1,
        "num_chunks": len(measured),
        "warmup_chunks": warmup,
        "windows_per_chunk": chunk_size,
        "window_seconds": args.window_seconds,
        "history_context_seconds": history_seconds,
        "top_k": top_k,
        "max_source_tokens": max_source_tokens,
        "max_new_tokens": max_new_tokens,
        "output_tokens": summarize_values(output_token_counts),
        "latency": stats,
    }

    print("\nComponent                         Median latency    p95 latency")
    for key, label in (
        ("prompt", "Prompt construction"),
        ("generation", "Corrector generation"),
        ("filter", "Shared output filter"),
        ("total", "Total correction latency"),
    ):
        print(
            f"{label:<34}{stats[key]['median_ms']:>9.2f} ms"
            f"{stats[key]['p95_ms']:>14.2f} ms"
        )
    print(
        f"\nHistory context: {history_seconds:.1f} s; "
        "it is not added to post-segment correction latency."
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
