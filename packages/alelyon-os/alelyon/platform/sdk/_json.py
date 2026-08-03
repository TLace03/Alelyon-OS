"""Strict JSON number hooks shared by HTTP and streaming boundaries."""

from __future__ import annotations

import math

MAX_JSON_NESTING = 128
MAX_JSON_INTEGER_DIGITS = 1024


def reject_nonfinite_constant(value: str) -> None:
    """Reject JavaScript constants that RFC 8259 does not permit in JSON."""
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_finite_float(value: str) -> float:
    """Parse a JSON float while refusing syntax that overflows to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is outside the finite float range")
    return parsed


def parse_bounded_int(value: str) -> int:
    """Parse an integer while bounding work on runtimes without a built-in cap."""
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError(
            f"JSON integer exceeds {MAX_JSON_INTEGER_DIGITS} digits"
        )
    return int(value)


def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build an object while refusing ambiguous duplicate member names."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON object member: {name}")
        result[name] = value
    return result


def json_nesting_within_limit(
    payload: bytes | str, limit: int = MAX_JSON_NESTING
) -> bool:
    """Bound object/array nesting without being fooled by brackets in strings."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    depth = 0
    in_string = False
    escaped = False
    for value in raw:
        if in_string:
            if escaped:
                escaped = False
            elif value == 0x5C:
                escaped = True
            elif value == 0x22:
                in_string = False
            continue
        if value == 0x22:
            in_string = True
        elif value in (0x5B, 0x7B):
            depth += 1
            if depth > limit:
                return False
        elif value in (0x5D, 0x7D):
            depth = max(0, depth - 1)
    return True
