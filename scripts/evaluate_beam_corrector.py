#!/usr/bin/env python3
"""Evaluate a beam-aware Qwen checkpoint with session CER/WER."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch

from emg2qwerty.transcription.artifacts import sha256_file
from emg2qwerty.transcription.beam_corrector import load_beam_checkpoint
from emg2qwerty.transcription.beam_dataset import (
    BeamCorrectionDataset,
    resolve_topk_files,
)
from emg2qwerty.transcription.beam_protocol import compute_cer_wer, reconstruct_sessions
from emg2qwerty.transcription.text import collate_text_batch
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-id")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype")
    parser.add_argument(
        "--include-empty-targets",
        action="store_true",
        help=(
            "Use the optional full-coverage protocol instead of the "
            "filtered-target protocol."
        ),
    )
    return parser.parse_args()


def _topk_input_manifest(
    topk_root: Path,
    topk_files: Sequence[Path],
) -> list[dict[str, str]]:
    root = topk_root.expanduser().resolve()
    entries: list[dict[str, str]] = []
    for path in topk_files:
        resolved = path.resolve()
        if root.is_file():
            if resolved != root:
                raise ValueError("resolved top-k file does not match top-k root")
            name = resolved.name
        else:
            try:
                name = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "resolved top-k file is outside the declared top-k root"
                ) from exc
        entries.append({"name": name, "sha256": sha256_file(resolved)})
    return entries


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _prediction_table(
    module: Any,
    batches: Iterable[Mapping[str, Sequence[Any]]],
    *,
    user: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        results = module.correct(
            batch["payload"],
            batch["current_prediction"],
            batch["candidates_per_window"],
        )
        if len(results) != len(batch["payload"]):
            raise ValueError("corrector returned a different batch size")
        for index, result in enumerate(results):
            rows.append(
                {
                    "user": user,
                    "session": batch["session"][index],
                    "split": batch["split"][index],
                    "window_index": int(batch["window_index"][index]),
                    "window_indices": json.dumps(batch["window_indices"][index]),
                    "current_prediction": batch["current_prediction"][index],
                    "target": batch["target"][index],
                    "llm_raw": result["raw"],
                    "llm_parsed": result["parsed"],
                    "llm_corrected": result["corrected"],
                    "llm_diag": json.dumps(
                        result["diagnostics"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _evaluation_tables(
    chunk_table: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    if chunk_table.empty:
        raise ValueError("evaluation dataset produced no chunks")
    raw_sessions = reconstruct_sessions(
        chunk_table,
        prediction_col="current_prediction",
        target_col="target",
    )
    corrected_sessions = reconstruct_sessions(
        chunk_table,
        prediction_col="llm_corrected",
        target_col="target",
    )
    unfiltered_sessions = reconstruct_sessions(
        chunk_table,
        prediction_col="llm_parsed",
        target_col="target",
    )
    raw_metrics, raw_summary = compute_cer_wer(raw_sessions, "prediction", "target")
    corrected_metrics, corrected_summary = compute_cer_wer(
        corrected_sessions, "prediction", "target"
    )
    unfiltered_metrics, unfiltered_summary = compute_cer_wer(
        unfiltered_sessions, "prediction", "target"
    )
    return (
        {
            "window_llm_predictions.csv": chunk_table,
            "window_raw_sessions.csv": raw_sessions,
            "window_llm_sessions.csv": corrected_sessions,
            "window_raw_session_metrics.csv": raw_metrics,
            "window_llm_session_metrics.csv": corrected_metrics,
            "window_unfiltered_sessions.csv": unfiltered_sessions,
            "window_unfiltered_session_metrics.csv": unfiltered_metrics,
        },
        {
            "raw": raw_summary,
            "corrected": corrected_summary,
            "unfiltered": unfiltered_summary,
        },
    )


def _write_summary(
    path: Path,
    *,
    user: str,
    stage: str,
    summary: dict[str, Any],
) -> None:
    pd.DataFrame([{"user": user, "stage": stage, **summary}]).to_csv(path, index=False)


def _write_outputs(
    output_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    summaries: Mapping[str, dict[str, Any]],
    user: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = output_dir / "predictions"
    summaries_dir = output_dir / "summaries"
    predictions_dir.mkdir()
    summaries_dir.mkdir()
    paths = {
        "window_llm_predictions.csv": predictions_dir / "window_llm_predictions.csv",
        "window_raw_sessions.csv": summaries_dir / "window_raw_sessions.csv",
        "window_llm_sessions.csv": summaries_dir / "window_llm_sessions.csv",
        "window_raw_session_metrics.csv": (
            summaries_dir / "window_raw_session_metrics.csv"
        ),
        "window_llm_session_metrics.csv": (
            summaries_dir / "window_llm_session_metrics.csv"
        ),
        "window_unfiltered_sessions.csv": (
            summaries_dir / "window_unfiltered_sessions.csv"
        ),
        "window_unfiltered_session_metrics.csv": (
            summaries_dir / "window_unfiltered_session_metrics.csv"
        ),
    }
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    summary_paths = {
        "window_raw_session_summary.csv": (
            summaries_dir / "window_raw_session_summary.csv"
        ),
        "window_llm_session_summary.csv": (
            summaries_dir / "window_llm_session_summary.csv"
        ),
        "window_unfiltered_session_summary.csv": (
            summaries_dir / "window_unfiltered_session_summary.csv"
        ),
    }
    _write_summary(
        summary_paths["window_raw_session_summary.csv"],
        user=user,
        stage="raw_session",
        summary=summaries["raw"],
    )
    _write_summary(
        summary_paths["window_llm_session_summary.csv"],
        user=user,
        stage="llm_session",
        summary=summaries["corrected"],
    )
    _write_summary(
        summary_paths["window_unfiltered_session_summary.csv"],
        user=user,
        stage="unfiltered_session",
        summary=summaries["unfiltered"],
    )
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in [*paths.values(), *summary_paths.values()]
    }


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    top_k, chunk_size = args.top_k, args.chunk_size
    requested_filter_empty = not args.include_empty_targets
    topk_files = resolve_topk_files(args.topk_root)
    topk_inputs = _topk_input_manifest(args.topk_root, topk_files)
    dataset = BeamCorrectionDataset(
        topk_files,
        split=args.split,
        chunk_size=chunk_size,
        top_k=top_k,
        filter_empty_targets=requested_filter_empty,
        require_consecutive_windows=True,
        drop_incomplete=False,
        shuffle_candidates=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_text_batch,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    module = load_beam_checkpoint(
        checkpoint,
        device=device,
        dtype=args.dtype,
        model_id=args.model_id,
        base_model_revision=args.base_model_revision,
    )
    chunk_table = _prediction_table(module, loader, user=args.user)
    tables, summaries = _evaluation_tables(chunk_table)
    output_dir = args.output_dir.expanduser().resolve()
    output_hashes = _write_outputs(
        output_dir,
        tables=tables,
        summaries=summaries,
        user=args.user,
    )
    run_manifest = {
        "format_version": 1,
        "user": args.user,
        "split": args.split,
        "protocol": "filtered_targets" if requested_filter_empty else "full_coverage",
        "top_k": top_k,
        "chunk_size": chunk_size,
        "device": str(device),
        "dtype": module.dtype_name,
        "topk_inputs": topk_inputs,
        "checkpoint": {"file": checkpoint.name, "sha256": sha256_file(checkpoint)},
        "base_model_id": module.model_id,
        "base_model_revision": module.resolved_model_revision,
        "prompt_sha256": module.prompt_digest,
        "outputs_sha256": output_hashes,
        "raw_summary": summaries["raw"],
        "corrected_summary": summaries["corrected"],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            _json_safe(run_manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe({"raw": summaries["raw"], "corrected": summaries["corrected"]}),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
