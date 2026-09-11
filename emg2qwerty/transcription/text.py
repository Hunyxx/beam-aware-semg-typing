"""Text normalization shared by the beam-aware dataset and protocol."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


_TOKEN_PATTERN = re.compile(r"<[^<>]+>")
_BACKSPACE_TOKENS = {"<backspace>", "⌫"}


def read_transcription_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Preserve literal NA-like strings and numeric-looking transcription text."""

    text_columns = (
        "session",
        "split",
        "prediction",
        "target",
        "current_prediction",
        "llm_raw",
        "llm_parsed",
        "llm_corrected",
    )
    options: dict[str, Any] = {
        "keep_default_na": False,
        "na_values": [""],
        "dtype": {column: str for column in text_columns},
    }
    options.update(kwargs)
    return pd.read_csv(path, **options)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def resolve_backspaces(text: str) -> str:
    """Apply backspaces while preserving other angle-bracketed tokens."""

    atoms: list[str] = []
    index = 0
    while index < len(text):
        match = _TOKEN_PATTERN.match(text, index) if text[index] == "<" else None
        if match:
            atoms.append(match.group())
            index = match.end()
        else:
            atoms.append(text[index])
            index += 1
    output: list[str] = []
    for atom in atoms:
        if atom in _BACKSPACE_TOKENS:
            if output:
                output.pop()
        else:
            output.append(atom)
    return "".join(output)


def collate_text_batch(
    records: list[dict[str, object]],
) -> dict[str, list[object]]:
    if not records:
        return {}
    keys = tuple(records[0])
    expected = set(keys)
    for index, record in enumerate(records[1:], start=1):
        if set(record) != expected:
            raise ValueError(
                f"batch record {index} has different keys: "
                f"expected={sorted(expected)}, observed={sorted(record)}"
            )
    return {key: [record[key] for record in records] for key in keys}


__all__ = [
    "collate_text_batch",
    "normalize_text",
    "read_transcription_csv",
    "resolve_backspaces",
]
