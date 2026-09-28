# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded, strict canonical JSON for model metadata (never payloads)."""

import json
import re
from typing import Any

from .limits import DeltaCodecError


def canonical_json(value: Any, limit: int) -> bytes:
    output = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for piece in encoder.iterencode(value):
        data = piece.encode("utf-8")
        if len(output) + len(data) > limit:
            raise DeltaCodecError("metadata exceeds byte limit")
        output.extend(data)
    return bytes(output)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise DeltaCodecError("duplicate JSON key")
        output[key] = value
    return output


def _integer(value: str) -> int:
    if len(value) > 20:
        raise DeltaCodecError("JSON integer exceeds uint64 representation")
    return int(value)


def _not_integer(value: str) -> None:
    raise DeltaCodecError("metadata numbers must be integers")


def parse_json(data: bytes, limit: int) -> Any:
    if type(data) is not bytes or len(data) > limit:
        raise DeltaCodecError("metadata must be immutable bytes within limit")
    try:
        value = json.loads(
            data, object_pairs_hook=_unique, parse_int=_integer, parse_float=_not_integer, parse_constant=_not_integer
        )
        if canonical_json(value, limit) != data:
            raise DeltaCodecError("metadata must use canonical JSON")
        return value
    except DeltaCodecError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise DeltaCodecError("invalid metadata JSON") from exc


def exact_fields(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DeltaCodecError("unknown, missing, or invalid metadata fields")
    return value


def require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise DeltaCodecError(f"{name} must be a bounded identifier, not a path or endpoint")
