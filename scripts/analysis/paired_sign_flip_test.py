#!/usr/bin/env python3
"""Run an exact paired sign-flip test on per-participant CER summaries.

Each participant contributes baseline CER minus corrected CER. Inputs may be two
aggregate summary tables or an evaluation root containing per-user evaluator
outputs. The exact null distribution is enumerated without materializing the
2**n sign matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from emg2qwerty.transcription.text import normalize_text


@dataclass(frozen=True)
class SignFlipResult:
    observed_mean: float
    p_value: float
    alternative: str
    n_pairs: int
    n_nonzero_differences: int
    n_zero_differences: int
    n_permutations: int


def exact_paired_sign_flip(
    differences: Sequence[float] | np.ndarray,
    *,
    alternative: str = "two-sided",
    max_exact_pairs: int = 20,
) -> SignFlipResult:
    """Enumerate the exact randomization distribution in constant memory.

    Zero differences are omitted from enumeration because flipping their signs
    leaves every null statistic unchanged. The reported observed statistic and
    every null statistic retain the original total-pair denominator.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("differences must be finite")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unsupported alternative: {alternative!r}")
    if max_exact_pairs < 0:
        raise ValueError("max_exact_pairs must be non-negative")

    magnitudes = np.abs(values[values != 0.0])
    magnitude_values = [float(value) for value in magnitudes]
    effective_pairs = int(magnitudes.size)
    if effective_pairs > max_exact_pairs:
        raise ValueError(
            "exact enumeration would require "
            f"2**{effective_pairs} sign flips; increase --max-exact-pairs "
            "only after checking the runtime"
        )

    observed = float(values.mean())
    permutations = 1 << effective_pairs
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 32
    extreme = 0

    for permutation in range(permutations):
        signed_sum = math.fsum(
            magnitude if permutation & (1 << index) else -magnitude
            for index, magnitude in enumerate(magnitude_values)
        )
        statistic = signed_sum / int(values.size)
        if alternative == "greater":
            is_extreme = statistic >= observed - tolerance
        elif alternative == "less":
            is_extreme = statistic <= observed + tolerance
        else:
            is_extreme = abs(statistic) >= abs(observed) - tolerance
        extreme += int(is_extreme)

    return SignFlipResult(
        observed_mean=observed,
        p_value=extreme / permutations,
        alternative=alternative,
        n_pairs=int(values.size),
        n_nonzero_differences=effective_pairs,
        n_zero_differences=int(values.size) - effective_pairs,
        n_permutations=permutations,
    )


def load_stage(
    path: Path,
    *,
    expected_stage: str,
    pair_column: str,
    metric_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Read one summary table and enforce one finite value per pair."""

    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {pair_column, "stage", metric_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = table.loc[
        table["stage"].astype(str).eq(expected_stage),
        [pair_column, metric_column],
    ].copy()
    if selected.empty:
        raise ValueError(f"{path} has no rows with stage={expected_stage!r}")
    selected[pair_column] = selected[pair_column].map(normalize_text)
    if selected[pair_column].eq("").any():
        raise ValueError(f"{path} contains an empty {pair_column} value")
    if selected[pair_column].duplicated().any():
        duplicates = selected.loc[
            selected[pair_column].duplicated(keep=False), pair_column
        ].tolist()
        raise ValueError(f"{path} contains duplicate pairs: {duplicates}")
    selected[metric_column] = pd.to_numeric(selected[metric_column], errors="raise")
    if not np.isfinite(selected[metric_column].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite {metric_column} values")
    return selected.rename(columns={metric_column: output_column})


def _pair_stage_frames(
    raw: pd.DataFrame,
    corrected: pd.DataFrame,
    *,
    pair_column: str,
    metric_column: str,
) -> pd.DataFrame:
    raw_name = f"baseline_{metric_column}"
    corrected_name = f"corrected_{metric_column}"
    paired = raw.merge(
        corrected,
        on=pair_column,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        unmatched = paired.loc[paired["_merge"].ne("both"), [pair_column, "_merge"]]
        raise ValueError(
            "baseline and corrected summaries are not fully paired:\n"
            + unmatched.to_string(index=False)
        )
    paired = (
        paired.drop(columns="_merge")
        .sort_values(pair_column, kind="stable")
        .reset_index(drop=True)
    )
    paired[f"{metric_column}_improvement_points"] = (
        paired[raw_name] - paired[corrected_name]
    )
    paired["improved"] = paired[f"{metric_column}_improvement_points"] > 0
    return paired


def pair_summaries(
    raw_path: Path,
    corrected_path: Path,
    *,
    raw_stage: str,
    corrected_stage: str,
    pair_column: str,
    metric_column: str,
) -> pd.DataFrame:
    raw_name = f"baseline_{metric_column}"
    corrected_name = f"corrected_{metric_column}"
    raw = load_stage(
        raw_path,
        expected_stage=raw_stage,
        pair_column=pair_column,
        metric_column=metric_column,
        output_column=raw_name,
    )
    corrected = load_stage(
        corrected_path,
        expected_stage=corrected_stage,
        pair_column=pair_column,
        metric_column=metric_column,
        output_column=corrected_name,
    )
    return _pair_stage_frames(
        raw,
        corrected,
        pair_column=pair_column,
        metric_column=metric_column,
    )


def _discover_evaluation_summaries(
    evaluation_root: Path,
    filename: str,
) -> list[Path]:
    root = evaluation_root.expanduser()
    if not root.is_dir():
        raise FileNotFoundError("evaluation root is not a directory")
    return sorted(
        path
        for path in root.rglob(filename)
        if path.is_file() and path.parent.name == "summaries"
    )


def _load_discovered_stage(
    paths: Sequence[Path],
    *,
    expected_stage: str,
    pair_column: str,
    metric_column: str,
    output_column: str,
) -> pd.DataFrame:
    if not paths:
        raise ValueError(f"no {output_column} evaluator summaries were found")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = load_stage(
            path,
            expected_stage=expected_stage,
            pair_column=pair_column,
            metric_column=metric_column,
            output_column=output_column,
        )
        if len(frame) != 1:
            raise ValueError(
                "each discovered evaluator summary must contain exactly one "
                f"{pair_column} row for stage={expected_stage!r}"
            )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined[pair_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(set(combined.loc[duplicate_mask, pair_column].tolist()))
        raise ValueError(f"duplicate {output_column} summaries for pairs: {duplicates}")
    return combined


def pair_evaluation_root(
    evaluation_root: Path,
    *,
    raw_stage: str,
    corrected_stage: str,
    pair_column: str,
    metric_column: str,
) -> pd.DataFrame:
    """Pair per-user evaluator summaries discovered below one result root."""

    raw_name = f"baseline_{metric_column}"
    corrected_name = f"corrected_{metric_column}"
    raw_paths = _discover_evaluation_summaries(
        evaluation_root, "window_raw_session_summary.csv"
    )
    corrected_paths = _discover_evaluation_summaries(
        evaluation_root, "window_llm_session_summary.csv"
    )
    raw = _load_discovered_stage(
        raw_paths,
        expected_stage=raw_stage,
        pair_column=pair_column,
        metric_column=metric_column,
        output_column=raw_name,
    )
    corrected = _load_discovered_stage(
        corrected_paths,
        expected_stage=corrected_stage,
        pair_column=pair_column,
        metric_column=metric_column,
        output_column=corrected_name,
    )
    return _pair_stage_frames(
        raw,
        corrected,
        pair_column=pair_column,
        metric_column=metric_column,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-summary", type=Path)
    parser.add_argument("--corrected-summary", type=Path)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        help="Recursively discover per-user evaluator summaries below this directory.",
    )
    parser.add_argument("--raw-stage", default="raw_session")
    parser.add_argument("--corrected-stage", default="llm_session")
    parser.add_argument("--pair-column", default="user")
    parser.add_argument("--metric-column", default="CER")
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=0,
        help="Fail unless this many complete pairs are present; 0 disables the check.",
    )
    parser.add_argument(
        "--max-exact-pairs",
        type=int,
        default=20,
        help="Maximum non-zero pairs allowed for exact 2**n enumeration.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Optionally write PREFIX_per_pair.csv and PREFIX_summary.json.",
    )
    args = parser.parse_args(argv)
    has_raw = args.raw_summary is not None
    has_corrected = args.corrected_summary is not None
    if args.evaluation_root is not None:
        if has_raw or has_corrected:
            parser.error(
                "--evaluation-root cannot be combined with --raw-summary or "
                "--corrected-summary"
            )
    elif has_raw != has_corrected:
        parser.error("--raw-summary and --corrected-summary must be provided together")
    elif not has_raw:
        parser.error(
            "provide --evaluation-root or both --raw-summary and --corrected-summary"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.expected_pairs < 0:
        raise ValueError("--expected-pairs must be non-negative")
    if args.evaluation_root is not None:
        paired = pair_evaluation_root(
            args.evaluation_root,
            raw_stage=args.raw_stage,
            corrected_stage=args.corrected_stage,
            pair_column=args.pair_column,
            metric_column=args.metric_column,
        )
    else:
        assert args.raw_summary is not None
        assert args.corrected_summary is not None
        paired = pair_summaries(
            args.raw_summary,
            args.corrected_summary,
            raw_stage=args.raw_stage,
            corrected_stage=args.corrected_stage,
            pair_column=args.pair_column,
            metric_column=args.metric_column,
        )
    if args.expected_pairs and len(paired) != args.expected_pairs:
        raise ValueError(f"expected {args.expected_pairs} pairs, found {len(paired)}")

    difference_column = f"{args.metric_column}_improvement_points"
    result = exact_paired_sign_flip(
        paired[difference_column].to_numpy(dtype=float),
        alternative=args.alternative,
        max_exact_pairs=args.max_exact_pairs,
    )
    raw_column = f"baseline_{args.metric_column}"
    corrected_column = f"corrected_{args.metric_column}"
    summary = {
        "metric": args.metric_column,
        "n_improved": int(paired["improved"].sum()),
        f"mean_baseline_{args.metric_column}": float(paired[raw_column].mean()),
        f"mean_corrected_{args.metric_column}": float(paired[corrected_column].mean()),
        f"mean_{args.metric_column}_improvement_points": result.observed_mean,
        "test": "exact paired sign-flip randomization test",
        **asdict(result),
    }

    print(paired.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.output_prefix is not None:
        prefix = args.output_prefix.expanduser().resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paired.to_csv(prefix.with_name(prefix.name + "_per_pair.csv"), index=False)
        prefix.with_name(prefix.name + "_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
