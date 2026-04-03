from __future__ import annotations

from ankiflow_tts.filenames import build_audio_filename, normalize_duplicate_key


def test_normalize_duplicate_key_collapses_whitespace_and_nfkc() -> None:
    assert normalize_duplicate_key(" Hello\tworld  ") == "Hello world"
    assert normalize_duplicate_key("Cafe\u0301") == "Café"


def test_build_audio_filename_is_deterministic_and_uses_slug() -> None:
    first = build_audio_filename("Please review this card.", "model-a")
    second = build_audio_filename("Please review this card.", "model-a")

    assert first == second
    assert first.startswith("please-review-this-card-")
    assert first.endswith(".wav")


def test_build_audio_filename_changes_with_model() -> None:
    base = build_audio_filename("Hello there", "model-a")
    changed_model = build_audio_filename("Hello there", "model-b")

    assert base != changed_model
