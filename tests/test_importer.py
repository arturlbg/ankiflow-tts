from __future__ import annotations

from pathlib import Path

import pytest

from ankiflow_tts.config import Settings
from ankiflow_tts.exceptions import AnkiConnectError, GeminiTtsError, ParseError
from ankiflow_tts.importer import Importer
from ankiflow_tts.types import AudioPayload, RetryPolicy


def test_importer_dry_run_skips_tts_and_reports_duplicates(tmp_path) -> None:
    input_path = _write_cards(
        tmp_path / "cards.txt",
        [
            "Hello.;Ola.;;one",
            "Goodbye.;Tchau.;;two",
        ],
    )
    anki = FakeAnkiClient(can_add_results=[True, False])
    tts = FakeTtsClient([])
    importer = Importer(anki_client=anki, tts_client=tts)

    summary = importer.run(_settings(input_path, dry_run=True))

    assert summary.would_import_count == 1
    assert summary.anki_duplicate_count == 1
    assert tts.calls == []
    assert anki.stored_media == []
    assert anki.added_notes == []


def test_importer_live_continues_after_row_failures(tmp_path) -> None:
    input_path = _write_cards(
        tmp_path / "cards.txt",
        [
            "One.;Um.;;one",
            "Two.;Dois.;;two",
            "Three.;Tres.;;three",
        ],
    )
    anki = FakeAnkiClient(
        can_add_results=[True, True, True],
        store_media_errors=[None, AnkiConnectError("upload failed")],
        note_ids=[101, 102],
    )
    tts = FakeTtsClient(
        [
            AudioPayload(filename="one.wav", content=b"RIFF----WAVE"),
            GeminiTtsError("Gemini TTS failed after 3 attempt(s): down", attempts=3),
            AudioPayload(filename="three.wav", content=b"RIFF----WAVE"),
        ]
    )
    importer = Importer(anki_client=anki, tts_client=tts)

    summary = importer.run(_settings(input_path, dry_run=False))

    assert summary.imported_count == 1
    assert summary.failed_tts_count == 1
    assert summary.failed_media_upload_count == 1
    assert summary.failure_count == 2


def test_importer_stops_before_anki_preflight_on_parse_error(tmp_path) -> None:
    input_path = _write_cards(
        tmp_path / "cards.txt",
        [
            "Valid.;Valido.;;ok",
            "Broken;;;",
        ],
    )
    anki = FakeAnkiClient(can_add_results=[])
    importer = Importer(anki_client=anki, tts_client=FakeTtsClient([]))

    with pytest.raises(ParseError):
        importer.run(_settings(input_path, dry_run=True))

    assert anki.validate_calls == 0


class FakeAnkiClient:
    def __init__(
        self,
        *,
        can_add_results: list[bool],
        store_media_errors: list[Exception | None] | None = None,
        note_ids: list[int] | None = None,
    ) -> None:
        self.can_add_results = list(can_add_results)
        self.store_media_errors = list(store_media_errors or [])
        self.note_ids = list(note_ids or [])
        self.validate_calls = 0
        self.stored_media: list[str] = []
        self.added_notes: list[int] = []

    def validate_target(self, deck_name: str, model_name: str) -> None:
        self.validate_calls += 1
        _ = (deck_name, model_name)

    def can_add_notes(self, notes):
        assert len(notes) == len(self.can_add_results)
        return list(self.can_add_results)

    def store_media_file(self, audio: AudioPayload) -> str:
        error = self.store_media_errors.pop(0) if self.store_media_errors else None
        if error is not None:
            raise error
        self.stored_media.append(audio.filename)
        return audio.filename

    def add_note(self, note) -> int:
        note_id = self.note_ids.pop(0)
        self.added_notes.append(note_id)
        return note_id


class FakeTtsClient:
    def __init__(self, results: list[AudioPayload | Exception]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, filename: str) -> AudioPayload:
        self.calls.append((text, filename))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _settings(input_path: Path, *, dry_run: bool) -> Settings:
    return Settings(
        input_path=input_path,
        deck_name="Deck",
        model_name="Model",
        anki_url="http://127.0.0.1:8765",
        gemini_api_key="key" if not dry_run else None,
        gemini_model="model" if not dry_run else None,
        gemini_voice="voice" if not dry_run else None,
        dry_run=dry_run,
        verbose=False,
        retry_policy=RetryPolicy(),
    )


def _write_cards(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
