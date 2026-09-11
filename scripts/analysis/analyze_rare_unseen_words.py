#!/usr/bin/env python3
"""Evaluate exact-word accuracy by LLM-training-reference frequency."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from emg2qwerty.transcription.beam_dataset import BeamCorrectionDataset
from emg2qwerty.transcription.text import (
    normalize_text,
    read_transcription_csv,
    resolve_backspaces,
)


def clean(value: object) -> str:
    return resolve_backspaces(normalize_text(value))


def words(value: object) -> list[str]:
    """Match the shared transcription protocol's whitespace word tokenization."""
    return clean(value).replace("⏎", " ").replace("\n", " ").split()


def align_reference(
    prediction: Sequence[str], reference: Sequence[str]
) -> list[str | None]:
    """Map each reference word to its aligned prediction word (or deletion)."""
    n, m = len(prediction), len(reference)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (prediction[i - 1] != reference[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    mapped: list[str | None] = [None] * m
    i, j = n, m
    while i or j:
        if i and j:
            cost = prediction[i - 1] != reference[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                mapped[j - 1] = prediction[i - 1]
                i -= 1
                j -= 1
                continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mapped


def training_counts(
    root: Path, *, chunk_size: int = 3, top_k: int = 10
) -> Counter[str]:
    dataset = BeamCorrectionDataset(
        root,
        split="train",
        chunk_size=chunk_size,
        top_k=top_k,
        filter_empty_targets=True,
        require_consecutive_windows=True,
        drop_incomplete=True,
        shuffle_candidates=False,
    )
    counts: Counter[str] = Counter()
    for sample in dataset.samples:
        counts.update(words(sample.target))
    print(
        f"[training vocabulary] {sum(counts.values())} tokens, "
        f"{len(counts)} types, {len(dataset)} training chunks",
        flush=True,
    )
    return counts


def prediction_files(root: Path) -> list[Path]:
    per_user = sorted(root.glob("user*/predictions/window_llm_predictions.csv"))
    if per_user:
        return per_user
    direct = root / "predictions" / "window_llm_predictions.csv"
    return [direct] if direct.exists() else []


def frequency_bin(count: int, rare_max: int) -> str:
    if count == 0:
        return "unseen"
    if count <= rare_max:
        return "rare"
    return "frequent"


def analyze_dataset(
    name: str,
    root: Path,
    counts: Counter[str],
    rare_max: int,
    llm_column: str,
) -> pd.DataFrame:
    files = prediction_files(root)
    if not files:
        raise FileNotFoundError(f"No prediction CSVs under {root}")
    frame = pd.concat(
        [read_transcription_csv(path) for path in files],
        ignore_index=True,
    )
    required = {
        "user",
        "session",
        "split",
        "window_index",
        "current_prediction",
        "target",
        llm_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{root} is missing columns: {sorted(missing)}")

    output: list[dict[str, object]] = []
    for (user, session, split), group in frame.groupby(
        ["user", "session", "split"], sort=False
    ):
        ordered = group.sort_values("window_index")
        reference = words("".join(clean(v) for v in ordered["target"]))
        rank1 = align_reference(
            words("".join(clean(v) for v in ordered["current_prediction"])), reference
        )
        llm = align_reference(
            words("".join(clean(v) for v in ordered[llm_column])), reference
        )
        for position, (ref_word, rank1_word, llm_word) in enumerate(
            zip(reference, rank1, llm)
        ):
            ref_frequency = counts[ref_word]
            rank1_correct = rank1_word == ref_word
            llm_correct = llm_word == ref_word
            llm_frequency = counts[llm_word] if llm_word is not None else 0
            output.append(
                {
                    "dataset": name,
                    "user": user,
                    "session": session,
                    "split": split,
                    "reference_word_position": position,
                    "reference_word": ref_word,
                    "training_frequency": ref_frequency,
                    "frequency_bin": frequency_bin(ref_frequency, rare_max),
                    "rank1_word": rank1_word,
                    "llm_word": llm_word,
                    "rank1_exact": rank1_correct,
                    "llm_exact": llm_correct,
                    "llm_frequency": llm_frequency,
                    "rank1_correct_llm_wrong": rank1_correct and not llm_correct,
                    "normalized_to_more_frequent": (
                        not llm_correct
                        and llm_word is not None
                        and llm_frequency > ref_frequency
                    ),
                    "rank1_correct_normalized_to_more_frequent": (
                        rank1_correct
                        and not llm_correct
                        and llm_word is not None
                        and llm_frequency > ref_frequency
                    ),
                }
            )
    return pd.DataFrame(output)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, bin_name), group in details.groupby(
        ["dataset", "frequency_bin"], sort=False
    ):
        total = len(group)
        rank1_correct = int(group["rank1_exact"].sum())
        llm_correct = int(group["llm_exact"].sum())
        regressions = int(group["rank1_correct_llm_wrong"].sum())
        commonized = int(group["rank1_correct_normalized_to_more_frequent"].sum())
        rows.append(
            {
                "dataset": dataset,
                "frequency_bin": bin_name,
                "reference_words": total,
                "rank1_exact_words": rank1_correct,
                "rank1_exact_word_accuracy": ratio(rank1_correct, total),
                "llm_exact_words": llm_correct,
                "llm_exact_word_accuracy": ratio(llm_correct, total),
                "llm_minus_rank1_accuracy": (llm_correct - rank1_correct) / total,
                "rank1_correct_llm_wrong": regressions,
                "regression_rate_given_rank1_correct": ratio(
                    regressions, rank1_correct
                ),
                "rank1_correct_normalized_to_more_frequent": commonized,
                "commonization_rate_given_rank1_correct": ratio(
                    commonized, rank1_correct
                ),
            }
        )
    order = {"unseen": 0, "rare": 1, "frequent": 2}
    result = pd.DataFrame(rows)
    result["_order"] = result["frequency_bin"].map(order)
    return result.sort_values(["dataset", "_order"]).drop(columns="_order")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        "--evaluation_root",
        dest="evaluation_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--training-topk-root",
        "--training_topk_root",
        dest="training_topk_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--datasets", nargs="+", default=["generic", "finetune"])
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--rare-max", "--rare_max", dest="rare_max", type=int, default=5
    )
    parser.add_argument(
        "--llm-column", "--llm_column", dest="llm_column", default="llm_corrected"
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)
    if args.rare_max < 1:
        parser.error("--rare_max must be at least 1")

    counts = training_counts(
        args.training_topk_root, chunk_size=args.chunk_size, top_k=args.top_k
    )
    details = pd.concat(
        [
            analyze_dataset(
                dataset,
                args.evaluation_root / dataset,
                counts,
                args.rare_max,
                args.llm_column,
            )
            for dataset in args.datasets
        ],
        ignore_index=True,
    )
    summary = summarize(details)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "word_frequency_details.csv", index=False)
    summary.to_csv(args.output_dir / "word_frequency_summary.csv", index=False)
    failures = details[
        details["rank1_correct_llm_wrong"] | details["normalized_to_more_frequent"]
    ].copy()
    failures.to_csv(args.output_dir / "rare_word_failure_examples.csv", index=False)
    metadata = {
        "training_topk_root": str(args.training_topk_root),
        "training_split": "train",
        "frequency_protocol": "training_chunk_targets_v2",
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "filter_empty_targets": True,
        "require_consecutive_windows": True,
        "drop_incomplete": True,
        "training_token_count": sum(counts.values()),
        "training_type_count": len(counts),
        "tokenization": (
            "whitespace after replacing enter/newline with spaces; "
            "case and punctuation preserved"
        ),
        "bins": {
            "unseen": "frequency = 0",
            "rare": f"1 <= frequency <= {args.rare_max}",
            "frequent": f"frequency > {args.rare_max}",
        },
        "llm_column": args.llm_column,
        "filter_status": (
            "with_filter"
            if args.llm_column == "llm_corrected"
            else "without_filter_or_custom"
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
