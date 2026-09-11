#!/usr/bin/env python3
"""Audit whether LLM edits are supported by top-k CTC beams.

The script consumes ``window_llm_predictions.csv`` files produced by
``scripts/evaluate_beam_corrector.py`` and corresponding top-k CSV exports.
It writes edit-level evidence, chunk-level fidelity regressions, examples, and
machine-readable/flat summaries suitable for reporting.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from emg2qwerty.transcription.beam_dataset import resolve_topk_files
from emg2qwerty.transcription.beam_protocol import tokenize_words
from emg2qwerty.transcription.text import (
    normalize_text,
    read_transcription_csv,
    resolve_backspaces,
)


ANGLE_RE = re.compile(r"<[^<>\r\n]+>")
BEAM_SUPPORTED_DEFINITION = (
    "The identical atomic edit (operation, rank-1 position, source, output) "
    "occurs in at least one top-k candidate for that window."
)
CORRECT_EDIT_DEFINITION = (
    "Keeping this edit yields a strictly smaller token-aware character edit "
    "distance to the reference than reverting this edit."
)

EDIT_CATEGORIES = (
    "supported_and_correct",
    "supported_and_wrong",
    "unsupported_and_correct",
    "unsupported_and_wrong",
)

EDIT_AUDIT_COLUMNS = (
    "user",
    "session",
    "split",
    "window_index",
    "edit_id",
    "op",
    "source_pos",
    "source",
    "output",
    "output_pos",
    "supported",
    "correct",
    "category",
    "harmful_unsupported",
    "punctuation_change",
    "angle_token_change",
    "llm_ref_distance",
    "reverted_ref_distance",
    "rank1",
    "llm",
    "reference",
)

CHUNK_AUDIT_COLUMNS = (
    "user",
    "session",
    "split",
    "window_index",
    "rank1",
    "llm",
    "reference",
    "rank1_cer_edits",
    "llm_cer_edits",
    "rank1_wer_edits",
    "llm_wer_edits",
    "rank1_exact",
    "exact_fidelity_regression",
    "wer_improved_cer_worsened",
    "punctuation_broken",
    "angle_token_broken",
    "edit_count",
)


def text(value: object) -> str:
    return resolve_backspaces(normalize_text(value))


def atoms(value: object) -> list[str]:
    """Split into characters while preserving angle-bracket tokens."""
    s = text(value)
    out: list[str] = []
    pos = 0
    for match in ANGLE_RE.finditer(s):
        match_start = match.start()
        out.extend(s[pos:match_start])
        out.append(match.group())
        pos = match.end()
    out.extend(s[pos:])
    return out


def is_angle(token: str) -> bool:
    return bool(ANGLE_RE.fullmatch(token))


def is_punctuation(token: str) -> bool:
    return token == "⏎" or (
        len(token) == 1 and unicodedata.category(token).startswith("P")
    )


def align(source: Sequence[str], target: Sequence[str]) -> list[dict[str, object]]:
    """Return a deterministic minimum-edit alignment (diagonal tie preferred)."""
    n, m = len(source), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (source[i - 1] != target[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    events: list[dict[str, object]] = []
    i, j = n, m
    while i or j:
        if i and j:
            cost = source[i - 1] != target[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    events.append(
                        {
                            "op": "substitution",
                            "source_pos": i - 1,
                            "source": source[i - 1],
                            "output": target[j - 1],
                            "output_pos": j - 1,
                        }
                    )
                i -= 1
                j -= 1
                continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            events.append(
                {
                    "op": "deletion",
                    "source_pos": i - 1,
                    "source": source[i - 1],
                    "output": "",
                    "output_pos": j,
                }
            )
            i -= 1
        else:
            events.append(
                {
                    "op": "insertion",
                    "source_pos": i,
                    "source": "",
                    "output": target[j - 1],
                    "output_pos": j - 1,
                }
            )
            j -= 1
    return list(reversed(events))


def distance(a: Sequence[str], b: Sequence[str]) -> int:
    # Rolling rows avoid allocating a traceback matrix for the many
    # counterfactual distance queries in the audit.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def edit_signature(event: dict[str, object]) -> tuple[object, ...]:
    return event["op"], event["source_pos"], event["source"], event["output"]


def revert_event(output: Sequence[str], event: dict[str, object]) -> list[str]:
    result = list(output)
    pos = int(str(event["output_pos"]))
    if event["op"] == "insertion":
        del result[pos]
    elif event["op"] == "substitution":
        result[pos] = str(event["source"])
    else:
        result.insert(pos, str(event["source"]))
    return result


def load_topk(user_root: Path, top_k: int) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "split",
        "session",
        "window_index",
        "rank",
        "prediction",
        "target",
    }
    frames: list[pd.DataFrame] = []
    for path in resolve_topk_files([user_root]):
        frame = read_transcription_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[list(required)].copy())
    frame = pd.concat(frames, ignore_index=True)
    frame["session"] = frame["session"].map(text)
    frame["split"] = frame["split"].map(text)
    for column in ("window_index", "rank"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{column} must contain integer values")
        frame[column] = numeric.astype(int)
    frame = frame[frame["rank"].between(1, top_k)].copy()
    frame["prediction"] = frame["prediction"].map(text)
    frame["target"] = frame["target"].map(text)

    identities = ["session", "split", "window_index"]
    for identity, group in frame.groupby(identities, sort=False):
        ordered = group.sort_values("rank", kind="stable")
        ranks = ordered["rank"].tolist()
        expected = list(range(1, top_k + 1))
        if ranks != expected:
            raise ValueError(
                "top-k rank invariant failed for "
                f"session={identity[0]!r}, split={identity[1]!r}, "
                f"window_index={identity[2]}: expected={expected}, observed={ranks}"
            )
        if ordered["target"].nunique(dropna=False) != 1:
            raise ValueError(
                "inconsistent top-k targets for "
                f"session={identity[0]!r}, split={identity[1]!r}, "
                f"window_index={identity[2]}"
            )
    return frame.sort_values(identities + ["rank"], kind="stable").reset_index(
        drop=True
    )


def prediction_files(root: Path) -> list[Path]:
    per_user = sorted(root.glob("user*/predictions/window_llm_predictions.csv"))
    if per_user:
        return per_user
    direct = root / "predictions" / "window_llm_predictions.csv"
    if direct.exists():
        return [direct]
    return []


def safe_rate(num: int, den: int) -> float | None:
    return num / den if den else None


def parse_window_indices(value: object, *, start: int) -> list[int]:
    try:
        parsed = json.loads(normalize_text(value))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("window_indices must be a JSON integer list") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("window_indices must be a non-empty JSON list")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in parsed):
        raise ValueError("window_indices must contain only integers")
    indices = [int(index) for index in parsed]
    expected = list(range(start, start + len(indices)))
    if indices != expected:
        raise ValueError(
            "window_indices must be consecutive and begin at window_index: "
            f"expected={expected}, observed={indices}"
        )
    return indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", "--predictions_root", type=Path)
    parser.add_argument("--topk-root-base", "--topk_root_base", type=Path)
    parser.add_argument(
        "--evaluation-root",
        "--evaluation_root",
        type=Path,
        help="Analyze one subdirectory per --datasets entry.",
    )
    parser.add_argument(
        "--topk-data-root",
        "--topk_data_root",
        type=Path,
        help="Parent containing one top-k tree per dataset.",
    )
    parser.add_argument("--topk-subdir", "--topk_subdir", default=".")
    parser.add_argument("--datasets", nargs="+", default=["generic", "finetune"])
    parser.add_argument("--top-k", "--top_k", type=int, default=10)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument(
        "--include-deletions-in-harmful-rate",
        "--include_deletions_in_harmful_rate",
        action="store_true",
        help="Use insertions+substitutions+deletions as the primary rate denominator.",
    )
    args = parser.parse_args(argv)

    if args.evaluation_root:
        if not args.topk_data_root:
            parser.error("--evaluation-root requires --topk-data-root")
        combined: dict[str, object] = {}
        rows: list[dict[str, object]] = []
        for dataset in args.datasets:
            dataset_output = args.output_dir / dataset
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--predictions-root",
                str(args.evaluation_root / dataset),
                "--topk-root-base",
                str(args.topk_data_root / dataset),
                "--topk-subdir",
                args.topk_subdir,
                "--top-k",
                str(args.top_k),
                "--output-dir",
                str(dataset_output),
            ]
            if args.include_deletions_in_harmful_rate:
                command.append("--include-deletions-in-harmful-rate")
            print(f"[dataset] analyzing {dataset}", flush=True)
            subprocess.run(command, check=True)
            with (dataset_output / "summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            combined[dataset] = summary
            rows.append(
                {
                    "dataset": dataset,
                    "chunks": summary["chunks"]["total"],
                    "edit_total": summary["edits"]["total"],
                    "harmful_unsupported_edits": summary["edits"][
                        "harmful_unsupported"
                    ],
                    "harmful_unsupported_edit_rate": summary["edits"][
                        "harmful_unsupported_edit_rate"
                    ],
                    "rank1_exact": summary["chunks"]["rank1_exact"],
                    "rank1_exact_but_llm_wrong": summary["chunks"][
                        "rank1_exact_but_llm_wrong"
                    ],
                    "rank1_exact_regression_rate": summary["chunks"][
                        "rank1_exact_regression_rate"
                    ],
                    "wer_improved_cer_worsened": summary["chunks"][
                        "wer_improved_cer_worsened"
                    ],
                    "punctuation_harmful_unsupported": summary["special_edits"][
                        "punctuation_harmful_unsupported"
                    ],
                    "angle_token_harmful_unsupported": summary["special_edits"][
                        "angle_token_harmful_unsupported"
                    ],
                }
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "combined_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(combined, handle, ensure_ascii=False, indent=2, allow_nan=False)
        pd.DataFrame(rows).to_csv(
            args.output_dir / "dataset_comparison.csv", index=False
        )
        print(pd.DataFrame(rows).to_string(index=False))
        return 0

    if not args.predictions_root or not args.topk_root_base:
        parser.error(
            "provide --predictions-root and --topk-root-base, or use --evaluation-root"
        )

    files = prediction_files(args.predictions_root)
    if not files:
        raise FileNotFoundError(
            f"No prediction CSVs found under {args.predictions_root}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edit_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    for pred_path in files:
        pred = read_transcription_csv(pred_path)
        required_predictions = {
            "session",
            "split",
            "window_index",
            "window_indices",
            "current_prediction",
            "llm_corrected",
            "target",
        }
        missing_predictions = required_predictions - set(pred.columns)
        if missing_predictions:
            raise ValueError(
                f"{pred_path} is missing columns: {sorted(missing_predictions)}"
            )
        if pred.empty:
            raise ValueError(f"prediction file is empty: {pred_path}")
        user = (
            text(pred["user"].iloc[0]) if "user" in pred else pred_path.parents[1].name
        )
        print(f"[audit] {user}: {len(pred)} chunks from {pred_path}", flush=True)
        per_user_topk = args.topk_root_base / user / args.topk_subdir
        shared_topk = args.topk_root_base / args.topk_subdir
        topk_root = per_user_topk if per_user_topk.is_dir() else shared_topk
        topk = load_topk(topk_root, args.top_k)
        lookup = {
            (str(session), str(split), int(index)): group.sort_values("rank")
            for (session, split, index), group in topk.groupby(
                ["session", "split", "window_index"], sort=False
            )
        }

        for _, row in pred.iterrows():
            session, split = text(row["session"]), text(row["split"])
            start = int(row["window_index"])
            baseline, llm, reference = map(
                text, (row["current_prediction"], row["llm_corrected"], row["target"])
            )

            window_indices = parse_window_indices(row["window_indices"], start=start)
            windows: list[pd.DataFrame] = []
            rank1_join = target_join = ""
            for index in window_indices:
                group = lookup.get((session, split, index))
                if group is None:
                    raise ValueError(
                        "Missing top-k window: "
                        f"user={user}, session={session}, split={split}, "
                        f"window_index={index}"
                    )
                windows.append(group)
                rank1_join += text(group.iloc[0]["prediction"])
                target_join += text(group.iloc[0]["target"])
            if rank1_join != baseline or target_join != reference:
                raise ValueError(
                    f"Cannot reconstruct chunk: user={user}, session={session}, "
                    f"start={start}; rank1_match={rank1_join == baseline}, "
                    f"target_match={target_join == reference}"
                )

            base_atoms, llm_atoms, ref_atoms = (
                atoms(baseline),
                atoms(llm),
                atoms(reference),
            )
            beam_signatures: set[tuple[object, ...]] = set()
            offset = 0
            for window in windows:
                primary = text(window.iloc[0]["prediction"])
                for candidate in window["prediction"]:
                    # Align locally: candidates are window-local evidence. This
                    # is equivalent to inserting the candidate into the chunk,
                    # while avoiding repeated O(chunk_length^2) alignments.
                    for beam_event in align(atoms(primary), atoms(candidate)):
                        beam_event["source_pos"] = (
                            int(str(beam_event["source_pos"])) + offset
                        )
                        beam_signatures.add(edit_signature(beam_event))
                offset += len(atoms(primary))

            events = align(base_atoms, llm_atoms)
            base_cer_edits, llm_cer_edits = distance(base_atoms, ref_atoms), distance(
                llm_atoms, ref_atoms
            )
            base_wer_edits = distance(
                tokenize_words(baseline), tokenize_words(reference)
            )
            llm_wer_edits = distance(tokenize_words(llm), tokenize_words(reference))
            rank1_exact = baseline == reference
            exact_regression = rank1_exact and llm != reference
            wer_down_cer_up = (
                llm_wer_edits < base_wer_edits and llm_cer_edits > base_cer_edits
            )
            punctuation_broken = any(
                (is_punctuation(str(e["source"])) or is_punctuation(str(e["output"])))
                and distance(revert_event(llm_atoms, e), ref_atoms) < llm_cer_edits
                for e in events
            )
            angle_broken = any(
                (is_angle(str(e["source"])) or is_angle(str(e["output"])))
                and distance(revert_event(llm_atoms, e), ref_atoms) < llm_cer_edits
                for e in events
            )
            chunk_rows.append(
                {
                    "user": user,
                    "session": session,
                    "split": split,
                    "window_index": start,
                    "rank1": baseline,
                    "llm": llm,
                    "reference": reference,
                    "rank1_cer_edits": base_cer_edits,
                    "llm_cer_edits": llm_cer_edits,
                    "rank1_wer_edits": base_wer_edits,
                    "llm_wer_edits": llm_wer_edits,
                    "rank1_exact": rank1_exact,
                    "exact_fidelity_regression": exact_regression,
                    "wer_improved_cer_worsened": wer_down_cer_up,
                    "punctuation_broken": punctuation_broken,
                    "angle_token_broken": angle_broken,
                    "edit_count": len(events),
                }
            )
            for event_id, event in enumerate(events):
                supported = edit_signature(event) in beam_signatures
                reverted_distance = distance(revert_event(llm_atoms, event), ref_atoms)
                correct = llm_cer_edits < reverted_distance
                src, dst = str(event["source"]), str(event["output"])
                support_label = "supported" if supported else "unsupported"
                correctness_label = "correct" if correct else "wrong"
                category = f"{support_label}_and_{correctness_label}"
                edit_rows.append(
                    {
                        "user": user,
                        "session": session,
                        "split": split,
                        "window_index": start,
                        "edit_id": event_id,
                        **event,
                        "supported": supported,
                        "correct": correct,
                        "category": category,
                        "harmful_unsupported": (not supported and not correct),
                        "punctuation_change": is_punctuation(src)
                        or is_punctuation(dst),
                        "angle_token_change": is_angle(src) or is_angle(dst),
                        "llm_ref_distance": llm_cer_edits,
                        "reverted_ref_distance": reverted_distance,
                        "rank1": baseline,
                        "llm": llm,
                        "reference": reference,
                    }
                )

    edits = pd.DataFrame(edit_rows, columns=EDIT_AUDIT_COLUMNS)
    chunks = pd.DataFrame(chunk_rows, columns=CHUNK_AUDIT_COLUMNS)
    edits.to_csv(args.output_dir / "edit_audit.csv", index=False)
    chunks.to_csv(args.output_dir / "chunk_fidelity_audit.csv", index=False)

    primary_ops = (
        ["insertion", "substitution", "deletion"]
        if args.include_deletions_in_harmful_rate
        else ["insertion", "substitution"]
    )
    primary_edits = edits[edits["op"].isin(primary_ops)] if not edits.empty else edits
    counts = (
        Counter(primary_edits["category"]) if not primary_edits.empty else Counter()
    )
    harmful = (
        int(primary_edits["harmful_unsupported"].sum())
        if not primary_edits.empty
        else 0
    )
    punctuation_edits = edits[edits["punctuation_change"]] if not edits.empty else edits
    angle_edits = edits[edits["angle_token_change"]] if not edits.empty else edits
    exact_regressions = chunks[chunks["exact_fidelity_regression"]]
    tradeoff_regressions = chunks[chunks["wer_improved_cer_worsened"]]
    primary_ops_text = ",".join(primary_ops)
    summary = {
        "definitions": {
            "beam_supported": BEAM_SUPPORTED_DEFINITION,
            "correct": CORRECT_EDIT_DEFINITION,
            "harmful_unsupported_edit_rate": (
                f"unsupported_and_wrong / all {primary_ops_text} edits"
            ),
        },
        "edits": {
            "denominator_ops": primary_ops,
            "total": len(primary_edits),
            **{category: int(counts[category]) for category in EDIT_CATEGORIES},
            "harmful_unsupported": harmful,
            "harmful_unsupported_edit_rate": safe_rate(harmful, len(primary_edits)),
        },
        "chunks": {
            "total": len(chunks),
            "rank1_exact": int(chunks["rank1_exact"].sum()),
            "rank1_exact_but_llm_wrong": int(chunks["exact_fidelity_regression"].sum()),
            "rank1_exact_regression_rate": safe_rate(
                int(chunks["exact_fidelity_regression"].sum()),
                int(chunks["rank1_exact"].sum()),
            ),
            "wer_improved_cer_worsened": int(chunks["wer_improved_cer_worsened"].sum()),
            "wer_improved_cer_worsened_rate_all_chunks": safe_rate(
                int(chunks["wer_improved_cer_worsened"].sum()), len(chunks)
            ),
            "exact_regressions_with_punctuation_broken": int(
                exact_regressions["punctuation_broken"].sum()
            ),
            "exact_regressions_with_angle_token_broken": int(
                exact_regressions["angle_token_broken"].sum()
            ),
            "wer_down_cer_up_with_punctuation_broken": int(
                tradeoff_regressions["punctuation_broken"].sum()
            ),
            "wer_down_cer_up_with_angle_token_broken": int(
                tradeoff_regressions["angle_token_broken"].sum()
            ),
        },
        "special_edits": {
            "punctuation_changes": len(punctuation_edits),
            "punctuation_wrong": int(punctuation_edits["correct"].eq(False).sum()),
            "punctuation_harmful_unsupported": int(
                punctuation_edits["harmful_unsupported"].sum()
            ),
            "angle_token_changes": len(angle_edits),
            "angle_token_wrong": int(angle_edits["correct"].eq(False).sum()),
            "angle_token_harmful_unsupported": int(
                angle_edits["harmful_unsupported"].sum()
            ),
        },
    }
    flagged = chunks[
        chunks["exact_fidelity_regression"] | chunks["wer_improved_cer_worsened"]
    ].copy()
    flagged.to_csv(args.output_dir / "fidelity_regression_examples.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    flat = pd.json_normalize(summary, sep=".").T.reset_index()
    flat.columns = ["metric", "value"]
    flat.to_csv(args.output_dir / "summary.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
