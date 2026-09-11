"""Frozen prompt text used by the beam-aware corrector."""

from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a beam-aware transcription corrector. "
    "Each window shows multiple candidate transcriptions from a beam decoder. "
    "Combine evidence across candidates within and across windows, and output "
    "the corrected text for all windows concatenated, with no separators or "
    "labels. Preserve punctuation and angle-bracketed tokens. If a window is "
    "unrecoverable, copy the most plausible candidate verbatim."
)

DEFAULT_USER_TEMPLATE = """**TASK: Combine beam-decoded candidates into the corrected transcription.**
**HARD RULES**
1) Output the corrected text for ALL windows concatenated, with no separators between windows.
2) Use PRIMARY (rank-1 baseline) as your starting point; modify only where other candidates clearly agree on a different form.
3) Preserve punctuation, casing, and angle-bracketed tokens (e.g. <NO_CONTEXT>) exactly as they appear in candidates.
4) Do NOT add or delete characters beyond what the candidates suggest.
5) When unsure or garbled, copy PRIMARY verbatim.
6) Output the corrected text ONLY. Do NOT echo "===", "Window N", dashes, or any list formatting.

Examples (simplified, single window):
Input candidates:
- helo worl
- hello world
- hello worl
PRIMARY: helo worl
Output: hello world

Input candidates:
- roveel auth
- roveel oath
- roveel auth
PRIMARY: roveel auth
Output: roveel auth

Your turn:

<INPUT>
{payload}
<START_CORRECTION>
<OUTPUT>
"""


def load_user_template(
    path: str | Path | None = None,
    *,
    template: str | None = None,
) -> str:
    """Load one prompt source and validate its payload boundary."""

    if path is not None and template is not None:
        raise ValueError("provide either path or template, not both")
    value = Path(path).read_text(encoding="utf-8") if path is not None else template
    value = DEFAULT_USER_TEMPLATE if value is None else str(value)
    if value.count("{payload}") != 1:
        raise ValueError("user template must contain exactly one {payload} placeholder")
    return value


def render_user_prompt(
    payload: str,
    template: str = DEFAULT_USER_TEMPLATE,
    *,
    development_format: bool = True,
) -> str:
    """Preserve the development prompt formatting, including brace escaping."""

    if template.count("{payload}") != 1:
        raise ValueError("user template must contain exactly one {payload} placeholder")
    if development_format:
        return template.format(
            char_count=len(payload),
            payload=str(payload).replace("{", "{{").replace("}", "}}"),
        )
    return template.replace("{payload}", str(payload))


def prompt_sha256(system_prompt: str, user_template: str) -> str:
    canonical = f"{system_prompt}\n\n{user_template}".encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_TEMPLATE",
    "load_user_template",
    "prompt_sha256",
    "render_user_prompt",
]
