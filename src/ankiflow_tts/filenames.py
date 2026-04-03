"""Helpers for normalization and deterministic filename generation."""

from __future__ import annotations

import hashlib
import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")
SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalize_duplicate_key(text: str) -> str:
    """Normalize text for duplicate detection."""

    normalized = unicodedata.normalize("NFKC", text)
    collapsed = WHITESPACE_RE.sub(" ", normalized).strip()
    return collapsed


def build_audio_filename(
    sentence_en: str,
    tts_model: str | None,
    *,
    max_slug_length: int = 40,
) -> str:
    """Create a deterministic WAV filename for a sentence and TTS model."""

    slug = _slugify(sentence_en, max_slug_length=max_slug_length)
    hash_input = "|".join([normalize_duplicate_key(sentence_en), tts_model or ""])
    digest = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}.wav"


def _slugify(text: str, *, max_slug_length: int) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    slug = SLUG_RE.sub("-", lowered).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        slug = "audio"
    trimmed = slug[:max_slug_length].strip("-")
    return trimmed or "audio"
