# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 sEMG Beam-Aware Transcription contributors.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""Side-specific keyboard-position vocabularies used by the auxiliary loss."""

from __future__ import annotations

from dataclasses import dataclass

from emg2qwerty.charset import charset


@dataclass(frozen=True)
class PositionCharset:
    vocab: dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.vocab)


LEFT_POSITION_CHARSET = PositionCharset(
    {
        "LP_UP": 0,
        "LP_HOME": 1,
        "LP_DOWN": 2,
        "LR_UP": 3,
        "LR_HOME": 4,
        "LR_DOWN": 5,
        "LM_UP": 6,
        "LM_HOME": 7,
        "LM_DOWN": 8,
        "LI_UP_LEFT": 9,
        "LI_UP_RIGHT": 10,
        "LI_HOME_LEFT": 11,
        "LI_HOME_RIGHT": 12,
        "LI_DOWN_LEFT": 13,
        "LI_DOWN_RIGHT": 14,
        "SPACE": 15,
    }
)

RIGHT_POSITION_CHARSET = PositionCharset(
    {
        "RI_UP_LEFT": 0,
        "RI_UP_RIGHT": 1,
        "RI_HOME_LEFT": 2,
        "RI_HOME_RIGHT": 3,
        "RI_DOWN_LEFT": 4,
        "RI_DOWN_RIGHT": 5,
        "RM_UP": 6,
        "RM_HOME": 7,
        "RM_DOWN": 8,
        "RR_UP": 9,
        "RR_HOME": 10,
        "RR_DOWN": 11,
        "RP_UP": 12,
        "RP_HOME": 13,
        "RP_ENTER": 14,
        "RP_BACKSPACE": 15,
        "SPACE": 16,
    }
)

CHAR_TO_SIDE_POSITIONS: dict[str, tuple[str | None, str | None]] = {
    "q": ("LP_UP", None),
    "a": ("LP_HOME", None),
    "z": ("LP_DOWN", None),
    "w": ("LR_UP", None),
    "s": ("LR_HOME", None),
    "x": ("LR_DOWN", None),
    "e": ("LM_UP", None),
    "d": ("LM_HOME", None),
    "c": ("LM_DOWN", None),
    "r": ("LI_UP_LEFT", None),
    "t": ("LI_UP_RIGHT", None),
    "f": ("LI_HOME_LEFT", None),
    "g": ("LI_HOME_RIGHT", None),
    "v": ("LI_DOWN_LEFT", None),
    "b": ("LI_DOWN_RIGHT", None),
    "y": (None, "RI_UP_LEFT"),
    "u": (None, "RI_UP_RIGHT"),
    "h": (None, "RI_HOME_LEFT"),
    "j": (None, "RI_HOME_RIGHT"),
    "n": (None, "RI_DOWN_LEFT"),
    "m": (None, "RI_DOWN_RIGHT"),
    "i": (None, "RM_UP"),
    "k": (None, "RM_HOME"),
    "o": (None, "RR_UP"),
    "l": (None, "RR_HOME"),
    "p": (None, "RP_UP"),
    ";": (None, "RP_HOME"),
    ":": (None, "RP_HOME"),
    "⏎": (None, "RP_ENTER"),
    "⌫": (None, "RP_BACKSPACE"),
    " ": ("SPACE", "SPACE"),
}
for _lower, _positions in tuple(CHAR_TO_SIDE_POSITIONS.items()):
    if _lower.isalpha() and len(_lower) == 1:
        CHAR_TO_SIDE_POSITIONS[_lower.upper()] = _positions


def char_to_side_position_names(character: str) -> tuple[str | None, str | None]:
    return CHAR_TO_SIDE_POSITIONS.get(character, (None, None))


def key_to_side_position_names(key: str) -> tuple[str | None, str | None]:
    normalized_key = charset()._normalize_keys([key])[0]
    if normalized_key not in charset():
        return (None, None)
    text = charset().keys_to_str([normalized_key])
    return char_to_side_position_names(text) if len(text) == 1 else (None, None)
