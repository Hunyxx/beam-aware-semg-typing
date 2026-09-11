# Copyright (c) 2026 sEMG Beam-Aware Transcription contributors.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""Portable, hash-addressed release artifact helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def dependency_version(distribution: str) -> str:
    """Return an installed distribution version or an explicit missing marker."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_value(
    value: Any,
    *,
    roots: Mapping[str, str | Path],
) -> Any:
    """Replace known absolute roots with logical tokens.

    Any absolute path outside the declared roots is rejected so a public
    manifest cannot silently expose a machine-specific path.
    """

    normalized_roots = sorted(
        ((token, Path(root).expanduser().resolve()) for token, root in roots.items()),
        key=lambda item: len(str(item[1])),
        reverse=True,
    )
    return _portable_value(value, normalized_roots)


def _portable_value(
    value: Any,
    roots: Sequence[tuple[str, Path]],
) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _portable_value(item, roots) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_portable_value(item, roots) for item in value]
    if isinstance(value, str):
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            for token, root in roots:
                try:
                    relative = resolved.relative_to(root)
                except ValueError:
                    continue
                return (
                    token if relative == Path(".") else f"{token}/{relative.as_posix()}"
                )
            raise ValueError(f"unmapped absolute path in release metadata: {value}")
        if _WINDOWS_ABSOLUTE.match(value):
            raise ValueError(f"unmapped Windows path in release metadata: {value}")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(
        "release metadata contains unsupported value type: " f"{type(value).__name__}"
    )
