from __future__ import annotations

import pytest

from ankiflow_tts.anki_connect import AnkiConnectClient
from ankiflow_tts.exceptions import AnkiConnectError, ValidationError
from ankiflow_tts.types import AudioPayload, PreparedNote


def test_validate_target_checks_fields_and_first_field() -> None:
    responses = {
        "version": 6,
        "deckNames": ["Deck"],
        "modelNames": ["Model"],
        "modelFieldNames": ["SentenceEN", "TranslationPT", "AudioEN", "Notes"],
    }
    client = AnkiConnectClient("http://127.0.0.1:8765", transport=_transport(responses))

    client.validate_target("Deck", "Model")


def test_validate_target_rejects_wrong_first_field() -> None:
    responses = {
        "version": 6,
        "deckNames": ["Deck"],
        "modelNames": ["Model"],
        "modelFieldNames": ["Front", "SentenceEN", "TranslationPT", "AudioEN", "Notes"],
    }
    client = AnkiConnectClient("http://127.0.0.1:8765", transport=_transport(responses))

    with pytest.raises(ValidationError):
        client.validate_target("Deck", "Model")


def test_can_add_notes_requires_one_result_per_note() -> None:
    client = AnkiConnectClient(
        "http://127.0.0.1:8765",
        transport=lambda action, params: [True] if action == "canAddNotes" else 6,
    )
    note_one = PreparedNote(
        line_number=1,
        deck_name="Deck",
        model_name="Model",
        fields={"SentenceEN": "A", "TranslationPT": "B", "AudioEN": "[sound:a.wav]", "Notes": ""},
        audio_filename="a.wav",
    )
    note_two = PreparedNote(
        line_number=2,
        deck_name="Deck",
        model_name="Model",
        fields={"SentenceEN": "C", "TranslationPT": "D", "AudioEN": "[sound:b.wav]", "Notes": ""},
        audio_filename="b.wav",
    )

    with pytest.raises(AnkiConnectError):
        client.can_add_notes([note_one, note_two])


def test_store_media_file_sends_base64_payload() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(action: str, params: dict[str, object]) -> object:
        calls.append((action, params))
        return "stored"

    client = AnkiConnectClient("http://127.0.0.1:8765", transport=transport)
    client.store_media_file(AudioPayload(filename="hello.wav", content=b"\x00\x01"))

    assert calls[0][0] == "storeMediaFile"
    assert calls[0][1]["filename"] == "hello.wav"
    assert isinstance(calls[0][1]["data"], str)


def _transport(responses: dict[str, object]):
    def transport(action: str, params: dict[str, object]) -> object:
        _ = params
        return responses[action]

    return transport
