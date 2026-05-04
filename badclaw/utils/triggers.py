from __future__ import annotations

import re
from typing import Iterable


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_token(text: str, token: str) -> bool:
    """
    Simple exact-substring trigger check after normalization.
    """
    t = normalize_text(text)
    tok = normalize_text(token)
    return tok in t


def has_any(text: str, terms: Iterable[str]) -> bool:
    """
    Returns True if any term appears in the normalized text.
    """
    t = normalize_text(text)
    return any(normalize_text(term) in t for term in terms)


def has_all(text: str, terms: Iterable[str]) -> bool:
    """
    Returns True if all terms appear in the normalized text.
    Useful for semantic/compositional cue checks.
    """
    t = normalize_text(text)
    return all(normalize_text(term) in t for term in terms)


def matches_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None