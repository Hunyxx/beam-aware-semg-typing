"""Text tokenizer setup for beam-aware correction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TEXT_ONLY_VOCABULARY = "text_only_beam_v1"
TEXT_ONLY_SPECIAL_TOKENS = (
    "<PAST_TXT>",
    "<CURRENT_TXT>",
    "<START_CORRECTION>",
    "<NO_CONTEXT>",
)


def special_tokens_for_policy(policy: str) -> tuple[str, ...]:
    if policy == TEXT_ONLY_VOCABULARY:
        return TEXT_ONLY_SPECIAL_TOKENS
    raise ValueError("unsupported vocabulary policy")


def configure_text_tokenizer(
    tokenizer: Any, policy: str = TEXT_ONLY_VOCABULARY
) -> None:
    """Configure padding, truncation, and the text tokens for beam correction."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    vocab = tokenizer.get_vocab()
    missing = [
        token for token in special_tokens_for_policy(policy) if token not in vocab
    ]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    value = json.dumps(
        tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
