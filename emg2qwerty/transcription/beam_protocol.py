"""Beam correction protocol for Qwen beam-aware transcription correction.

The functions here are deliberately model-light: they define the generation
guardrails, output parsing/filtering, session reconstruction, and CER/WER
computation used by the released Qwen path.
"""

from __future__ import annotations

import math
import re

from collections import Counter
from typing import Any, Sequence

import Levenshtein
import pandas as pd

from emg2qwerty.transcription.text import normalize_text


STRUCTURAL_PROMPT_PHRASES: tuple[str, ...] = (
    "=== Window",
    "<INPUT>",
    "</INPUT>",
    "<START_CORRECTION>",
    "<OUTPUT>",
    "</OUTPUT>",
    "PRIMARY (rank-1 baseline",
    "\n- ",
    "\n-",
)

_THINK_WRAPPER_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_QWEN_CONTROL_TOKENS = {
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|endoftext|>",
}
_EXACT_STRUCTURE_LINE_RES = (
    re.compile(r"^\s*=== Window(?:\s+\d+)?\s*===\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*</?(?:INPUT|OUTPUT)>\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*<START_CORRECTION>\s*$", flags=re.IGNORECASE),
    re.compile(
        r"^\s*PRIMARY \(rank-1 baseline(?:, your starting point)?\):?\s*$",
        flags=re.IGNORECASE,
    ),
)
_ANGLE_BRACKET_TOKEN_RE = re.compile(r"<[^<>\r\n]*>")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded]


def build_structural_bad_words_ids(
    tokenizer: Any,
    *,
    phrases: Sequence[str] = STRUCTURAL_PROMPT_PHRASES,
) -> list[list[int]]:
    """Encode complete prompt phrases in Qwen's start/space/newline contexts."""

    contexts = ("", " ", "\n")
    seen: set[tuple[int, ...]] = set()
    sequences: list[list[int]] = []
    for phrase in phrases:
        if not phrase:
            continue
        for prefix in contexts:
            ids = _encode_without_special_tokens(tokenizer, prefix + str(phrase))
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
                sequences.append(ids)
    return sequences


def parse_model_output(
    text: str | None,
    *,
    tokenizer: Any | None = None,
    assistant_prefill: str | None = None,
) -> str:
    """Remove identifiable model wrappers while preserving ordinary text.

    In particular, plain ``assistant``, ``assistant:`` and line-leading
    ``assistant`` are legitimate transcription content and are never removed.
    """

    cleaned = str(text or "").replace("\r\n", "\n")
    cleaned = _THINK_WRAPPER_RE.sub("", cleaned)
    if cleaned.startswith("<think>") and "</think>" not in cleaned:
        # An unfinished reasoning wrapper contains no unambiguous answer.
        return ""

    # Qwen chat roles are a precise wrapper sequence: the role word is only
    # metadata when immediately attached to <|im_start|> and terminated by a
    # newline. The same word in ordinary transcription text is untouched.
    cleaned = re.sub(
        r"<\|im_start\|>(?:assistant|user|system)\n",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    special_tokens: set[str] = set(_QWEN_CONTROL_TOKENS)
    if tokenizer is not None:
        # Domain text such as <NO_CONTEXT> may itself be registered as a
        # tokenizer special token and must remain in the transcription.
        for attr in ("pad_token", "eos_token", "bos_token"):
            value = getattr(tokenizer, attr, None)
            if value:
                special_tokens.add(str(value))
    for token in sorted(special_tokens, key=len, reverse=True):
        cleaned = cleaned.replace(token, "")

    if assistant_prefill and cleaned.startswith(assistant_prefill):
        cleaned = cleaned[len(assistant_prefill) :]
    return cleaned.strip("\r\n\t")


def contains_exact_prompt_structure_line(text: str) -> bool:
    for line in str(text or "").splitlines():
        if any(pattern.fullmatch(line) for pattern in _EXACT_STRUCTURE_LINE_RES):
            return True
    return False


def _ascii_lexical_words(text: str) -> list[str]:
    """Extract ASCII words, excluding opaque angle-bracket tokens."""

    lexical_text = _ANGLE_BRACKET_TOKEN_RE.sub(" ", text or "")
    return [match.group(0).lower() for match in _ASCII_WORD_RE.finditer(lexical_text)]


def _best_word_similarity(
    word: str, reference_words: Sequence[str]
) -> tuple[float, str]:
    best_score = 0.0
    best_word = ""
    for reference in reference_words:
        if word == reference:
            return 1.0, reference
        distance = Levenshtein.distance(word, reference)
        score = 1.0 - distance / max(len(word), len(reference), 1)
        if score > best_score:
            best_score, best_word = score, reference
    return best_score, best_word


def _word_has_lexical_match(
    word: str, reference_words: Sequence[str]
) -> tuple[bool, float, str]:
    score, match = _best_word_similarity(word, reference_words)
    threshold = 1.0 if len(word) <= 2 else (0.67 if len(word) <= 4 else 0.66)
    return score >= threshold, score, match


def shared_output_filter(
    primary: str,
    candidates_per_window: Sequence[Sequence[str]] | None,
    candidate: str | None,
) -> tuple[str, dict[str, str]]:
    """Apply output guards and a conditional ASCII-word drift heuristic.

    The lexical check uses approximate word similarity to PRIMARY and the
    top-k candidates as a hallucination guard. It does not classify individual
    edits as beam-supported and does not require literal beam evidence. Digits,
    punctuation, line breaks and the ``⏎`` marker, non-ASCII text, and opaque
    angle-bracket tokens are outside this ASCII-word check; the independent
    empty, prompt-leakage, and length guards still apply to the whole output.
    """

    diagnostics: dict[str, str] = {
        "accepted": "False",
        "reason": "",
        "filter": "lexical_drift_guard_v1",
        "fallback": "PRIMARY",
        "lexical_check": "not_run",
        "lexical_overlap_rate": "nan",
        "lexical_mean_similarity": "nan",
        "primary_character_similarity": "nan",
    }
    primary = str(primary or "").replace("\r\n", "\n").replace("\r", "\n")
    if candidate is None:
        diagnostics["reason"] = "candidate_empty"
        return primary, diagnostics
    cand = str(candidate).replace("\r\n", "\n").replace("\r", "\n").strip("\n\t")
    if not cand.strip():
        diagnostics["reason"] = "candidate_empty"
        return primary, diagnostics
    if contains_exact_prompt_structure_line(cand):
        diagnostics["reason"] = "prompt_leakage"
        return primary, diagnostics

    primary_enters = primary.count("⏎")
    candidate_enters = cand.count("⏎")
    diagnostics["primary_enter_count"] = str(primary_enters)
    diagnostics["candidate_enter_count"] = str(candidate_enters)
    primary_len, cand_len = len(primary), len(cand)
    diagnostics["primary_chars"] = str(primary_len)
    diagnostics["candidate_chars"] = str(cand_len)
    max_len = max(primary_len * 2 + 40, primary_len + 80, 120)
    min_len = max(1, int(primary_len * 0.25)) if primary_len else 1
    if cand_len > max_len:
        diagnostics["reason"] = "length_explosion"
        return primary, diagnostics
    if primary_len >= 20 and cand_len < min_len:
        diagnostics["reason"] = "length_collapse"
        return primary, diagnostics

    primary_words = _ascii_lexical_words(primary)
    cand_words = _ascii_lexical_words(cand)
    diagnostics["primary_ascii_word_count"] = str(len(primary_words))
    diagnostics["candidate_ascii_word_count"] = str(len(cand_words))
    if primary_words and len(cand_words) > max(
        len(primary_words) * 2 + 6, len(primary_words) + 12
    ):
        diagnostics["reason"] = "ascii_word_count_explosion"
        return primary, diagnostics

    primary_similarity = 1.0 - Levenshtein.distance(primary, cand) / max(
        primary_len, cand_len, 1
    )
    diagnostics["primary_character_similarity"] = f"{primary_similarity:.4f}"
    reference_texts = [primary]
    for window_candidates in candidates_per_window or []:
        reference_texts.extend(str(value) for value in window_candidates)
    reference_words = sorted(
        {word for value in reference_texts for word in _ascii_lexical_words(value)}
    )
    if not cand_words or not reference_words:
        diagnostics.update(
            accepted="True",
            reason="lexical_check_not_applicable",
            fallback="none",
            lexical_check="not_applicable",
        )
        return cand, diagnostics

    diagnostics["lexical_check"] = "applied"
    unmatched: list[str] = []
    scores: list[float] = []
    for word in cand_words:
        matched, score, closest = _word_has_lexical_match(word, reference_words)
        scores.append(score)
        if not matched:
            unmatched.append(f"{word}:{score:.2f}->{closest}")
    overlap_rate = (len(cand_words) - len(unmatched)) / len(cand_words)
    mean_similarity = sum(scores) / len(scores)
    diagnostics.update(
        lexical_overlap_rate=f"{overlap_rate:.4f}",
        lexical_mean_similarity=f"{mean_similarity:.4f}",
        lexical_word_count=str(len(cand_words)),
        lexical_unmatched_word_count=str(len(unmatched)),
    )
    if unmatched:
        diagnostics["lexical_unmatched_words"] = "; ".join(unmatched[:12])
    overlap_ok = overlap_rate >= 0.55 or mean_similarity >= 0.72
    similarity_ok = primary_similarity >= 0.72
    if overlap_ok or similarity_ok:
        diagnostics.update(accepted="True", reason="lexical_guard_ok", fallback="none")
        return cand, diagnostics
    diagnostics["reason"] = "lexical_drift"
    return primary, diagnostics


def _coerce_window_indices(values: pd.Series) -> pd.Series:
    """Return exact integer indices or reject malformed external values."""

    has_boolean = values.map(
        lambda value: type(value).__name__ in {"bool", "bool_"}
    ).any()
    numeric = pd.to_numeric(values, errors="coerce")
    if has_boolean or numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise ValueError("window_index must contain finite numeric values")
    if (numeric % 1 != 0).any():
        raise ValueError("window_index must be integer-valued")
    int64_min, int64_max = -(2**63), 2**63 - 1
    if (numeric < int64_min).any() or (numeric > int64_max).any():
        raise ValueError("window_index is outside the signed 64-bit integer range")
    try:
        return numeric.astype("int64")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("window_index must be integer-valued") from error


def reconstruct_sessions(
    table: pd.DataFrame,
    *,
    prediction_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Reconstruct complete sessions from filtered, time-ordered chunks."""

    rows: list[dict[str, Any]] = []
    if table.empty:
        return pd.DataFrame(columns=["session", "split", "prediction", "target"])
    required = {"session", "window_index", prediction_col, target_col}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"session reconstruction is missing columns: {sorted(missing)}"
        )
    ordered = table.copy()
    ordered["session"] = ordered["session"].fillna("").astype(str)
    ordered["window_index"] = _coerce_window_indices(ordered["window_index"])
    identities = ordered[["session", "window_index"]]
    if identities.duplicated().any():
        raise ValueError("session reconstruction contains duplicate chunk starts")
    ordered = ordered.sort_values(["session", "window_index"], kind="stable")
    for session, group in ordered.groupby("session", sort=False):
        rows.append(
            {
                "session": str(session),
                "split": ",".join(
                    sorted(
                        {
                            normalize_text(value)
                            for value in group.get("split", [])
                            if value
                        }
                    )
                ),
                "prediction": "".join(
                    normalize_text(value) for value in group[prediction_col]
                ),
                "target": "".join(normalize_text(value) for value in group[target_col]),
                "chunk_count": int(len(group)),
                "chunk_start_indices": ",".join(
                    str(int(value)) for value in group["window_index"]
                ),
            }
        )
    return pd.DataFrame(rows)


def tokenize_words(text: str) -> list[str]:
    """Tokenize text exactly as the released session-level WER evaluator."""
    normalized = normalize_text(text).replace("⏎", " ").replace("\n", " ").strip()
    return normalized.split() if normalized else []


def _word_edit_counts(
    prediction: Sequence[str], target: Sequence[str]
) -> tuple[int, int, int]:
    rows, cols = len(prediction), len(target)
    costs = [[0] * (cols + 1) for _ in range(rows + 1)]
    back = [[(0, 0, "match")] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        costs[row][0] = row
        back[row][0] = (row - 1, 0, "delete")
    for col in range(1, cols + 1):
        costs[0][col] = col
        back[0][col] = (0, col - 1, "insert")
    priority = {"replace": 0, "delete": 1, "insert": 2}
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if prediction[row - 1] == target[col - 1]:
                costs[row][col] = costs[row - 1][col - 1]
                back[row][col] = (row - 1, col - 1, "match")
                continue
            options = [
                (costs[row - 1][col - 1] + 1, row - 1, col - 1, "replace"),
                (costs[row - 1][col] + 1, row - 1, col, "delete"),
                (costs[row][col - 1] + 1, row, col - 1, "insert"),
            ]
            cost, prev_row, prev_col, operation = min(
                options, key=lambda value: (value[0], priority[value[3]])
            )
            costs[row][col] = cost
            back[row][col] = (prev_row, prev_col, operation)
    counts: Counter[str] = Counter()
    row, col = rows, cols
    while row or col:
        prev_row, prev_col, operation = back[row][col]
        counts[operation] += operation != "match"
        row, col = prev_row, prev_col
    # The alignment edits prediction into target; recognition errors reverse I/D.
    return counts["delete"], counts["insert"], counts["replace"]


def compute_cer_wer(
    table: pd.DataFrame,
    prediction_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute micro-averaged CER/WER using the shared session protocol."""

    records: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        prediction = normalize_text(row.get(prediction_col, ""))
        target = normalize_text(row.get(target_col, ""))
        char_counts = Counter(
            op for op, _, _ in Levenshtein.editops(prediction, target)
        )
        pred_words, target_words = tokenize_words(prediction), tokenize_words(target)
        wi, wd, ws = _word_edit_counts(pred_words, target_words)
        char_errors = (
            char_counts["insert"] + char_counts["delete"] + char_counts["replace"]
        )
        word_errors = wi + wd + ws
        record = dict(row)
        record.update(
            {
                prediction_col: prediction,
                target_col: target,
                "insertions": char_counts["delete"],
                "deletions": char_counts["insert"],
                "substitutions": char_counts["replace"],
                "target_len": len(target),
                "prediction_len": len(prediction),
                "cer_percent": (
                    float("nan") if not target else char_errors / len(target) * 100.0
                ),
                "word_insertions": wi,
                "word_deletions": wd,
                "word_substitutions": ws,
                "target_words": len(target_words),
                "prediction_words": len(pred_words),
                "wer_percent": (
                    float("nan")
                    if not target_words
                    else word_errors / len(target_words) * 100.0
                ),
            }
        )
        records.append(record)
    metrics = pd.DataFrame(records)
    if metrics.empty:
        metrics = pd.DataFrame(
            columns=[
                prediction_col,
                target_col,
                "insertions",
                "deletions",
                "substitutions",
                "target_len",
                "prediction_len",
                "cer_percent",
                "word_insertions",
                "word_deletions",
                "word_substitutions",
                "target_words",
                "prediction_words",
                "wer_percent",
            ]
        )
    char_total = int(metrics["target_len"].sum())
    word_total = int(metrics["target_words"].sum())
    summary: dict[str, Any] = {
        "error_convention": "prediction_relative_to_reference_v2",
        "samples": int(len(metrics)),
        "target_chars": char_total,
        "pred_chars": int(metrics["prediction_len"].sum()),
        "target_words": word_total,
        "pred_words": int(metrics["prediction_words"].sum()),
    }
    for name, column in (
        ("IER", "insertions"),
        ("DER", "deletions"),
        ("SER", "substitutions"),
    ):
        summary[name] = (
            float("nan")
            if not char_total
            else float(metrics[column].sum() / char_total * 100)
        )
    summary["CER"] = (
        float("nan")
        if not char_total
        else summary["IER"] + summary["DER"] + summary["SER"]
    )
    for name, column in (
        ("WIR", "word_insertions"),
        ("WDR", "word_deletions"),
        ("WSR", "word_substitutions"),
    ):
        summary[name] = (
            float("nan")
            if not word_total
            else float(metrics[column].sum() / word_total * 100)
        )
    summary["WER"] = (
        float("nan")
        if not word_total
        else summary["WIR"] + summary["WDR"] + summary["WSR"]
    )
    return metrics, summary


__all__ = [
    "STRUCTURAL_PROMPT_PHRASES",
    "build_structural_bad_words_ids",
    "compute_cer_wer",
    "contains_exact_prompt_structure_line",
    "parse_model_output",
    "reconstruct_sessions",
    "shared_output_filter",
    "tokenize_words",
]
