"""CPU-only regressions for audit schemas and latency benchmark contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from emg2qwerty.transcription.beam_protocol import tokenize_words

from ..audit_beam_supported_edits import EDIT_AUDIT_COLUMNS, main as audit_main

ANALYSIS_DIR = Path(__file__).resolve().parents[1]


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_frontend_benchmark_records_every_component_and_checks_length_first() -> None:
    source = (ANALYSIS_DIR / "benchmark_frontend_latency.py").read_text(
        encoding="utf-8"
    )
    run_one = _function(ast.parse(source), "run_one")
    record_guard = next(
        node
        for node in run_one.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "record"
    )
    recorded = {
        call.func.value.slice.value
        for call in ast.walk(record_guard)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Subscript)
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "timings"
        and isinstance(call.func.value.slice, ast.Constant)
    }
    assert recorded == {"preprocessing", "forward", "beam_search", "total"}

    missing_length_guard = next(
        node
        for node in run_one.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "output_lengths"
        and any(isinstance(operator, ast.Is) for operator in node.test.ops)
    )
    valid_length_assignment = next(
        node
        for node in run_one.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "valid_length"
            for target in node.targets
        )
    )
    assert missing_length_guard.lineno < valid_length_assignment.lineno


def test_shared_wer_tokenizer_treats_enter_and_newline_as_word_boundaries() -> None:
    assert tokenize_words("alpha⏎beta\ngamma") == ["alpha", "beta", "gamma"]


def test_audit_without_edits_writes_stable_schema(tmp_path: Path) -> None:
    predictions_root = tmp_path / "generic"
    predictions_dir = predictions_root / "predictions"
    predictions_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "user": "generic",
                "session": "session-1",
                "split": "test",
                "window_index": 0,
                "window_indices": "[0]",
                "current_prediction": "alpha⏎beta",
                "llm_corrected": "alpha⏎beta",
                "target": "alpha beta",
            }
        ]
    ).to_csv(predictions_dir / "window_llm_predictions.csv", index=False)

    topk_dir = tmp_path / "topk" / "generic"
    topk_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "session": "session-1",
                "split": "test",
                "window_index": 0,
                "rank": rank,
                "prediction": "alpha⏎beta",
                "target": "alpha beta",
            }
            for rank in (1, 2)
        ]
    ).to_csv(topk_dir / "session-1_topk.csv", index=False)

    output_dir = tmp_path / "output"
    assert (
        audit_main(
            [
                "--predictions-root",
                str(predictions_root),
                "--topk-root-base",
                str(tmp_path / "topk"),
                "--top-k",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    edits = pd.read_csv(output_dir / "edit_audit.csv")
    assert edits.empty
    assert tuple(edits.columns) == EDIT_AUDIT_COLUMNS
    chunks = pd.read_csv(output_dir / "chunk_fidelity_audit.csv")
    assert int(chunks.loc[0, "rank1_wer_edits"]) == 0
    assert int(chunks.loc[0, "llm_wer_edits"]) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["edits"] == {
        "denominator_ops": ["insertion", "substitution"],
        "total": 0,
        "supported_and_correct": 0,
        "supported_and_wrong": 0,
        "unsupported_and_correct": 0,
        "unsupported_and_wrong": 0,
        "harmful_unsupported": 0,
        "harmful_unsupported_edit_rate": None,
    }
    assert summary["special_edits"] == {
        "punctuation_changes": 0,
        "punctuation_wrong": 0,
        "punctuation_harmful_unsupported": 0,
        "angle_token_changes": 0,
        "angle_token_wrong": 0,
        "angle_token_harmful_unsupported": 0,
    }
