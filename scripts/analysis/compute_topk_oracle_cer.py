#!/usr/bin/env python3
"""Evaluate a per-window top-k oracle with session-level CER and WER.

The oracle selects the lowest-edit-distance candidate independently in each
window (ties prefer the lower rank). Both rank-1 and oracle selections are
then reconstructed into complete sessions before metrics are computed. This
is important because edit distance is not additive across window boundaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import Levenshtein
import pandas as pd

from emg2qwerty.transcription.beam_dataset import resolve_topk_files
from emg2qwerty.transcription.beam_protocol import (
    _word_edit_counts,
    compute_cer_wer,
    reconstruct_sessions,
    tokenize_words,
)
from emg2qwerty.transcription.text import (
    normalize_text,
    read_transcription_csv,
    resolve_backspaces,
)


REQUIRED_COLUMNS = {
    "split",
    "session",
    "window_index",
    "rank",
    "prediction",
    "target",
}


def _resolved_text(value: object) -> str:
    return resolve_backspaces(normalize_text(value))


def load_candidates(
    inputs: Sequence[str | Path],
    *,
    splits: Sequence[str] | None = None,
    top_k: int | None = None,
    filter_empty_targets: bool = True,
) -> pd.DataFrame:
    """Load and validate exported candidates from files or directories."""

    frames: list[pd.DataFrame] = []
    for path in resolve_topk_files(inputs):
        frame = read_transcription_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame[list(REQUIRED_COLUMNS)].copy()
        frame["source_file"] = str(path)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)

    table["split"] = table["split"].map(normalize_text)
    table["session"] = table["session"].map(normalize_text)
    for column in ("window_index", "rank"):
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain integer values")
        table[column] = values.astype(int)
    if (table["rank"] <= 0).any():
        raise ValueError("rank values must be positive")

    if splits:
        requested = {str(value) for value in splits}
        table = table[table["split"].isin(requested)].copy()
        missing_splits = requested - set(table["split"])
        if missing_splits:
            raise ValueError(
                f"no candidates found for splits: {sorted(missing_splits)}"
            )
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        table = table[table["rank"] <= top_k].copy()
    if table.empty:
        raise ValueError("candidate selection is empty")

    table["prediction"] = table["prediction"].map(_resolved_text)
    table["target"] = table["target"].map(_resolved_text)
    identities = ["split", "session", "window_index"]
    duplicated_ranks = table.duplicated(identities + ["rank"], keep=False)
    if duplicated_ranks.any():
        rows = table.loc[duplicated_ranks, identities + ["rank", "source_file"]]
        raise ValueError(f"duplicate candidate ranks:\n{rows.to_string(index=False)}")
    for identity, group in table.groupby(identities, sort=False):
        ranks = sorted(group["rank"].tolist())
        expected = list(range(1, (top_k if top_k is not None else max(ranks)) + 1))
        if ranks != expected:
            raise ValueError(f"incomplete top-k ranks for {identity}: {ranks}")
        if group["target"].nunique(dropna=False) != 1:
            raise ValueError(f"inconsistent targets for {identity}")
    if filter_empty_targets:
        table = table[table["target"].str.strip().ne("")].copy()
    if table.empty:
        raise ValueError("no candidate windows remain after target filtering")
    return table.sort_values(identities + ["rank"], kind="stable").reset_index(
        drop=True
    )


def select_window_oracle(candidates: pd.DataFrame) -> pd.DataFrame:
    """Choose rank-1 and minimum-distance candidates for every window."""

    rows: list[dict[str, Any]] = []
    identities = ["split", "session", "window_index"]
    for identity, group in candidates.groupby(identities, sort=False):
        targets = set(group["target"])
        if len(targets) != 1:
            raise ValueError(
                "inconsistent targets for "
                f"split={identity[0]!r}, session={identity[1]!r}, "
                f"window_index={identity[2]}"
            )
        target = targets.pop()
        ranks = group["rank"].astype(int).tolist()
        if 1 not in ranks:
            raise ValueError(
                "rank 1 is missing for "
                f"split={identity[0]!r}, session={identity[1]!r}, "
                f"window_index={identity[2]}"
            )
        distances = group["prediction"].map(
            lambda prediction, reference=target: int(
                Levenshtein.distance(prediction, reference)
            )
        )
        scored = group.assign(_window_errors=distances)
        top1 = scored.loc[scored["rank"].eq(1)].iloc[0]
        oracle = scored.sort_values(["_window_errors", "rank"], kind="stable").iloc[0]
        word_distances = group["prediction"].map(
            lambda prediction: sum(
                _word_edit_counts(tokenize_words(prediction), tokenize_words(target))
            )
        )
        word_oracle = (
            group.assign(_word_errors=word_distances)
            .sort_values(["_word_errors", "rank"], kind="stable")
            .iloc[0]
        )
        rows.append(
            {
                "split": identity[0],
                "session": identity[1],
                "window_index": int(identity[2]),
                "target": target,
                "top1_prediction": top1["prediction"],
                "top1_window_errors": int(top1["_window_errors"]),
                "oracle_prediction": oracle["prediction"],
                "oracle_rank": int(oracle["rank"]),
                "word_oracle_prediction": word_oracle["prediction"],
                "word_oracle_rank": int(word_oracle["rank"]),
                "word_oracle_window_errors": int(word_oracle["_word_errors"]),
                "oracle_window_errors": int(oracle["_window_errors"]),
                "target_window_chars": len(target),
                "candidate_count": int(len(group)),
                "oracle_improves_window": int(
                    oracle["_window_errors"] < top1["_window_errors"]
                ),
            }
        )
    if not rows:
        raise ValueError("no candidate windows were selected")
    return (
        pd.DataFrame(rows)
        .sort_values(["split", "session", "window_index"], kind="stable")
        .reset_index(drop=True)
    )


def reconstruct_by_split(
    windows: pd.DataFrame,
    *,
    prediction_col: str,
) -> pd.DataFrame:
    """Use the shared protocol without merging equal session IDs across splits."""

    frames: list[pd.DataFrame] = []
    for split, group in windows.groupby("split", sort=False):
        sessions = reconstruct_sessions(
            group,
            prediction_col=prediction_col,
            target_col="target",
        )
        sessions["split"] = str(split)
        frames.append(sessions)
    return pd.concat(frames, ignore_index=True)


def compute_session_oracle(
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct rank-1/oracle sessions and compute shared transcription metrics."""

    top1_sessions = reconstruct_by_split(windows, prediction_col="top1_prediction")
    oracle_sessions = reconstruct_by_split(windows, prediction_col="oracle_prediction")
    word_oracle_sessions = reconstruct_by_split(
        windows, prediction_col="word_oracle_prediction"
    )
    top1_metrics, top1_summary = compute_cer_wer(top1_sessions, "prediction", "target")
    oracle_metrics, oracle_summary = compute_cer_wer(
        oracle_sessions, "prediction", "target"
    )
    word_oracle_metrics, word_oracle_summary = compute_cer_wer(
        word_oracle_sessions, "prediction", "target"
    )

    keys = ["split", "session"]
    session_predictions = top1_sessions.rename(
        columns={"prediction": "top1_prediction"}
    ).merge(
        oracle_sessions[keys + ["prediction"]].rename(
            columns={"prediction": "oracle_prediction"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    session_predictions = session_predictions.merge(
        word_oracle_sessions[keys + ["prediction"]].rename(
            columns={"prediction": "word_oracle_prediction"}
        ),
        on=keys,
        validate="one_to_one",
    )
    metric_identity = pd.DataFrame(
        {
            "split": top1_sessions["split"],
            "session": top1_sessions["session"],
        }
    )
    session_metrics = pd.concat(
        [
            metric_identity.reset_index(drop=True),
            top1_metrics.add_prefix("top1_").reset_index(drop=True),
            oracle_metrics.add_prefix("oracle_").reset_index(drop=True),
            word_oracle_metrics.add_prefix("word_oracle_").reset_index(drop=True),
        ],
        axis=1,
    )

    rank_counts = Counter(int(value) for value in windows["oracle_rank"])
    top1_cer = float(top1_summary["CER"])
    oracle_cer = float(oracle_summary["CER"])
    top1_wer = float(top1_summary["WER"])
    oracle_wer = float(word_oracle_summary["WER"])
    summary: dict[str, Any] = {
        "oracle_selection": "independent_character_and_word_paths_v2",
        "windows": int(len(windows)),
        "sessions": int(len(session_predictions)),
        "splits": ",".join(sorted(set(windows["split"]))),
        "observed_top_k": int(windows["candidate_count"].max()),
        "target_chars": int(top1_summary["target_chars"]),
        "target_words": int(top1_summary["target_words"]),
        "top1_CER": top1_cer,
        "oracle_topk_CER": oracle_cer,
        "absolute_CER_gain": top1_cer - oracle_cer,
        "relative_CER_gain_percent": (
            0.0 if top1_cer == 0.0 else (top1_cer - oracle_cer) / top1_cer * 100.0
        ),
        "top1_WER": top1_wer,
        "oracle_topk_WER": oracle_wer,
        "absolute_WER_gain": top1_wer - oracle_wer,
        "oracle_improves_window_count": int(windows["oracle_improves_window"].sum()),
        "oracle_improves_window_rate_percent": float(
            windows["oracle_improves_window"].mean() * 100.0
        ),
    }
    for rank, count in sorted(rank_counts.items()):
        summary[f"best_rank_{rank}_count"] = count
        summary[f"best_rank_{rank}_rate_percent"] = count / len(windows) * 100.0
    return session_predictions, session_metrics, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more exported *_topk.csv files or directories.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Limit evaluation to this split; may be repeated (default: all).",
    )
    parser.add_argument("--top-k", type=int, help="Ignore ranks above this value.")
    parser.add_argument(
        "--include-empty-targets",
        action="store_true",
        help="Use the optional full-coverage target policy instead of paper filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for CSV and JSON artifacts.",
    )
    parser.add_argument("--prefix", default="topk_oracle")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    candidates = load_candidates(
        args.inputs,
        splits=args.splits,
        top_k=args.top_k,
        filter_empty_targets=not args.include_empty_targets,
    )
    windows = select_window_oracle(candidates)
    sessions, session_metrics, summary = compute_session_oracle(windows)
    summary["filter_empty_targets"] = not args.include_empty_targets
    print(json.dumps(summary, indent=2, allow_nan=False))

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        windows.to_csv(output_dir / f"{args.prefix}_window_selections.csv", index=False)
        sessions.to_csv(
            output_dir / f"{args.prefix}_session_predictions.csv", index=False
        )
        session_metrics.to_csv(
            output_dir / f"{args.prefix}_session_metrics.csv", index=False
        )
        (output_dir / f"{args.prefix}_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
