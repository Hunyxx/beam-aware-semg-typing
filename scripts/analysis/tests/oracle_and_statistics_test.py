"""CPU-only tests for transcription analysis and statistics tools."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..audit_beam_supported_edits import parse_window_indices
from ..compute_topk_oracle_cer import compute_session_oracle, select_window_oracle
from ..paired_sign_flip_test import (
    _parse_args as parse_sign_flip_args,
    exact_paired_sign_flip,
    main as sign_flip_main,
    pair_evaluation_root,
)


ANALYSIS_DIR = Path(__file__).resolve().parents[1]


def _candidate(
    *,
    split: str,
    session: str,
    window: int,
    rank: int,
    prediction: str,
    target: str,
) -> dict[str, object]:
    return {
        "split": split,
        "session": session,
        "window_index": window,
        "rank": rank,
        "prediction": prediction,
        "target": target,
    }


def test_oracle_selects_per_window_but_scores_reconstructed_session() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                split="test",
                session="s1",
                window=0,
                rank=1,
                prediction="a",
                target="ab",
            ),
            _candidate(
                split="test",
                session="s1",
                window=0,
                rank=2,
                prediction="ab",
                target="ab",
            ),
            _candidate(
                split="test",
                session="s1",
                window=1,
                rank=1,
                prediction="bcd",
                target="cd",
            ),
            _candidate(
                split="test",
                session="s1",
                window=1,
                rank=2,
                prediction="cd",
                target="cd",
            ),
        ]
    )
    windows = select_window_oracle(candidates)
    sessions, metrics, summary = compute_session_oracle(windows)

    # The two rank-1 window errors cancel at their boundary: "a" + "bcd".
    assert int(windows["top1_window_errors"].sum()) == 2
    assert windows["oracle_rank"].tolist() == [2, 2]
    assert sessions.loc[0, "top1_prediction"] == "abcd"
    assert sessions.loc[0, "oracle_prediction"] == "abcd"
    assert sessions.loc[0, "target"] == "abcd"
    assert summary["top1_CER"] == pytest.approx(0.0)
    assert summary["oracle_topk_CER"] == pytest.approx(0.0)
    assert metrics.loc[0, "top1_cer_percent"] == pytest.approx(0.0)


def test_oracle_keeps_equal_session_ids_separate_across_splits() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                split="val",
                session="same",
                window=0,
                rank=1,
                prediction="x",
                target="a",
            ),
            _candidate(
                split="val",
                session="same",
                window=0,
                rank=2,
                prediction="a",
                target="a",
            ),
            _candidate(
                split="test",
                session="same",
                window=0,
                rank=1,
                prediction="b",
                target="b",
            ),
            _candidate(
                split="test",
                session="same",
                window=0,
                rank=2,
                prediction="z",
                target="b",
            ),
        ]
    )
    windows = select_window_oracle(candidates)
    sessions, _, summary = compute_session_oracle(windows)

    assert len(sessions) == 2
    assert set(sessions["split"]) == {"val", "test"}
    assert summary["sessions"] == 2
    assert summary["top1_CER"] == pytest.approx(50.0)
    assert summary["oracle_topk_CER"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("alternative", "expected_p"),
    [("greater", 0.25), ("less", 1.0), ("two-sided", 0.5)],
)
def test_exact_sign_flip_known_distribution(
    alternative: str,
    expected_p: float,
) -> None:
    result = exact_paired_sign_flip(
        np.asarray([1.0, 2.0]),
        alternative=alternative,
        max_exact_pairs=2,
    )
    assert result.observed_mean == pytest.approx(1.5)
    assert result.p_value == pytest.approx(expected_p)
    assert result.n_permutations == 4


def test_exact_sign_flip_omits_only_redundant_zero_signs() -> None:
    result = exact_paired_sign_flip(
        [0.0, 1.0],
        alternative="greater",
        max_exact_pairs=1,
    )
    assert result.observed_mean == pytest.approx(0.5)
    assert result.p_value == pytest.approx(0.5)
    assert result.n_pairs == 2
    assert result.n_nonzero_differences == 1
    assert result.n_zero_differences == 1
    assert result.n_permutations == 2


def test_exact_sign_flip_guard_prevents_accidental_explosion() -> None:
    with pytest.raises(ValueError, match="2\\*\\*3"):
        exact_paired_sign_flip(
            [1.0, 2.0, 3.0],
            max_exact_pairs=2,
        )


def _write_evaluation_pair(
    root: Path,
    *,
    user: str,
    baseline_cer: float,
    corrected_cer: float,
    run_name: str | None = None,
    include_raw: bool = True,
    include_corrected: bool = True,
) -> None:
    summaries = root / (run_name or user) / "summaries"
    summaries.mkdir(parents=True)
    if include_raw:
        pd.DataFrame(
            [
                {
                    "user": user,
                    "stage": "raw_session",
                    "CER": baseline_cer,
                }
            ]
        ).to_csv(summaries / "window_raw_session_summary.csv", index=False)
    if include_corrected:
        pd.DataFrame(
            [
                {
                    "user": user,
                    "stage": "llm_session",
                    "CER": corrected_cer,
                }
            ]
        ).to_csv(summaries / "window_llm_session_summary.csv", index=False)


def test_evaluation_root_discovers_users_and_runs_exact_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "evaluations"
    for index in range(8):
        _write_evaluation_pair(
            root,
            user=f"user-{index}",
            baseline_cer=10.0 + index,
            corrected_cer=9.0 + index,
            run_name=f"nested/run-{index}",
        )

    paired = pair_evaluation_root(
        root,
        raw_stage="raw_session",
        corrected_stage="llm_session",
        pair_column="user",
        metric_column="CER",
    )
    assert paired["user"].tolist() == [f"user-{index}" for index in range(8)]
    assert paired["CER_improvement_points"].tolist() == [1.0] * 8

    source_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    assert (
        sign_flip_main(
            [
                "--evaluation-root",
                str(root),
                "--expected-pairs",
                "8",
            ]
        )
        == 0
    )
    captured = capsys.readouterr().out
    summary = json.loads(captured.rstrip().split("\n\n")[-1])
    assert summary["n_pairs"] == 8
    assert summary["n_improved"] == 8
    assert summary["p_value"] == pytest.approx(0.0078125)
    assert str(tmp_path) not in captured
    assert source_files == {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def test_evaluation_root_requires_one_to_one_user_coverage(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    _write_evaluation_pair(
        missing_root,
        user="user-a",
        baseline_cer=10.0,
        corrected_cer=9.0,
    )
    _write_evaluation_pair(
        missing_root,
        user="user-b",
        baseline_cer=11.0,
        corrected_cer=10.0,
        include_corrected=False,
    )
    with pytest.raises(ValueError, match="not fully paired"):
        pair_evaluation_root(
            missing_root,
            raw_stage="raw_session",
            corrected_stage="llm_session",
            pair_column="user",
            metric_column="CER",
        )

    duplicate_root = tmp_path / "duplicate"
    for run_name in ("run-a", "run-b"):
        _write_evaluation_pair(
            duplicate_root,
            user="same-user",
            baseline_cer=10.0,
            corrected_cer=9.0,
            run_name=run_name,
        )
    with pytest.raises(ValueError, match="duplicate baseline_CER"):
        pair_evaluation_root(
            duplicate_root,
            raw_stage="raw_session",
            corrected_stage="llm_session",
            pair_column="user",
            metric_column="CER",
        )


def test_significance_cli_input_modes_are_complete_and_exclusive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_path = tmp_path / "raw.csv"
    corrected_path = tmp_path / "corrected.csv"
    pd.DataFrame(
        [
            {"user": "u1", "stage": "raw_session", "CER": 2.0},
            {"user": "u2", "stage": "raw_session", "CER": 3.0},
        ]
    ).to_csv(raw_path, index=False)
    pd.DataFrame(
        [
            {"user": "u1", "stage": "llm_session", "CER": 1.0},
            {"user": "u2", "stage": "llm_session", "CER": 2.0},
        ]
    ).to_csv(corrected_path, index=False)

    args = parse_sign_flip_args(
        [
            "--raw-summary",
            str(raw_path),
            "--corrected-summary",
            str(corrected_path),
        ]
    )
    assert args.evaluation_root is None
    assert (
        sign_flip_main(
            [
                "--raw-summary",
                str(raw_path),
                "--corrected-summary",
                str(corrected_path),
                "--expected-pairs",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out.rstrip().split("\n\n")[-1])
    assert summary["n_pairs"] == 2

    root_args = parse_sign_flip_args(["--evaluation-root", str(tmp_path)])
    assert root_args.evaluation_root == tmp_path
    invalid_argv: tuple[list[str], ...] = (
        [],
        ["--raw-summary", str(raw_path)],
        ["--corrected-summary", str(corrected_path)],
        [
            "--evaluation-root",
            str(tmp_path),
            "--raw-summary",
            str(raw_path),
            "--corrected-summary",
            str(corrected_path),
        ],
    )
    for argv in invalid_argv:
        with pytest.raises(SystemExit):
            parse_sign_flip_args(argv)


def test_explicit_window_indices_contract() -> None:
    assert parse_window_indices("[3, 4, 5]", start=3) == [3, 4, 5]
    with pytest.raises(ValueError, match="consecutive"):
        parse_window_indices("[3, 5]", start=3)
    with pytest.raises(ValueError, match="integers"):
        parse_window_indices("[3, 4.0]", start=3)


def _argument_keywords(path: Path, option: str) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if any(
            isinstance(argument, ast.Constant) and argument.value == option
            for argument in node.args
        ):
            return {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
    raise AssertionError(f"{option} was not declared in {path}")


@pytest.mark.parametrize(
    "filename",
    ["benchmark_frontend_latency.py", "benchmark_llm_correction_latency.py"],
)
def test_benchmark_device_default_is_literal_cpu(filename: str) -> None:
    keywords = _argument_keywords(ANALYSIS_DIR / filename, "--device")
    default = keywords["default"]
    assert isinstance(default, ast.Constant)
    assert default.value == "cpu"


@pytest.mark.parametrize(
    ("filename", "option"),
    [
        ("benchmark_frontend_latency.py", "--weights"),
        ("benchmark_llm_correction_latency.py", "--topk-root"),
        ("benchmark_llm_correction_latency.py", "--checkpoint"),
    ],
)
def test_benchmark_artifact_arguments_are_required(
    filename: str,
    option: str,
) -> None:
    keywords = _argument_keywords(ANALYSIS_DIR / filename, option)
    required = keywords["required"]
    assert isinstance(required, ast.Constant)
    assert required.value is True


def test_analysis_scripts_have_no_machine_specific_paths_or_gpu_autoselection() -> None:
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "/" + "Users/",
        "/" + "gs/",
        "cuda" + ":0",
        "is_available()" + ' else "cpu"',
    )
    for path in ANALYSIS_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"{marker!r} found in {path.name}"
