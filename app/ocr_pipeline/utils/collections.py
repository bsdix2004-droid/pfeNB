"""Shared collection utilities."""
from __future__ import annotations

from typing import Iterable, List, TypeVar

T = TypeVar("T")


def unique_ordered(values: Iterable[str]) -> List[str]:
    """Deduplicate while preserving insertion order, case-insensitive key."""
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = value.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result

