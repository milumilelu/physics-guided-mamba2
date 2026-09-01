"""Filename / data-name parsing and provenance bookkeeping.

The rule this module exists to enforce
--------------------------------------
For a paired measurement the *slot* (physical position in the image) and the
*sample id* (加工顺序 in the design table) are different things, and the
mapping between them can only come from the KEYENCE data name stored inside the
``.cag``.  It must never be derived arithmetically.

``60Pass组.cag`` is scanned in a serpentine pattern, so 12 of its 30
measurements carry ids in the order ``14 13``, ``16 15``, ... rather than
``13 14``, ``15 16``.  A rule of the form ``slot_1 = 2m-1, slot_2 = 2m`` would
silently swap the sample identity of every one of them.
"""

from __future__ import annotations

import re

__all__ = [
    "natural_key",
    "parse_data_name",
    "parse_measurement_filename",
    "expected_filename",
    "SourceType",
]

_DIGITS = re.compile(r"\d+")
_TOKEN = re.compile(r"^\s*(\d+)(?:\s+(\d+))?\s*$")

DEFAULT_SUFFIX = "_高度.csv"
DEFAULT_SEPARATOR = " "


class SourceType:
    """Allowed values for ``csv_source_type``."""

    OFFICIAL = "keyence_official_export"
    DERIVED = "cag_decoder_derived"
    UNKNOWN = "unknown"

    ALL = (OFFICIAL, DERIVED, UNKNOWN)

    #: only these may be used as independent equivalence fixtures
    FIXTURE_ELIGIBLE = (OFFICIAL,)


def natural_key(text: str) -> tuple:
    """Sort key that orders ``2_高度`` before ``10_高度``."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in _DIGITS.split(text)
    )


def parse_data_name(name: str) -> list[int]:
    """Split a KEYENCE display name into sample ids, preserving token order.

    ``"1 2"``    -> ``[1, 2]``
    ``"14 13"``  -> ``[14, 13]``      <- serpentine scan, NOT a typo
    ``"7"``      -> ``[7]``

    The order is never normalised and never validated against parity.
    """
    if name is None:
        raise ValueError("data name is None")
    text = str(name).strip()
    if not text:
        raise ValueError("data name is empty")
    match = _TOKEN.match(text)
    if not match:
        raise ValueError(f"unparseable data name: {name!r}")
    tokens = [int(g) for g in match.groups() if g is not None]
    return tokens


def parse_measurement_filename(filename: str,
                               suffix: str = DEFAULT_SUFFIX) -> list[int] | None:
    """Extract the sample ids encoded in a measurement filename.

    Returns ``None`` when the name does not match the expected pattern, so the
    caller can report it as an unexplained file rather than guessing.
    """
    name = str(filename)
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    try:
        return parse_data_name(stem)
    except ValueError:
        return None


def expected_filename(tokens: list[int], suffix: str = DEFAULT_SUFFIX,
                      separator: str = DEFAULT_SEPARATOR) -> str:
    """Rebuild the filename a measurement must have, in token order."""
    return separator.join(str(t) for t in tokens) + suffix
