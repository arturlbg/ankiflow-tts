"""AnkiConnect integration."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from .exceptions import AnkiConnectError, ValidationError
from .types import AudioPayload, PreparedNote

REQUIRED_FIELDS = ("SentenceEN", "TranslationPT", "AudioEN", "Notes")


class AnkiConnectClient:
    """Thin client for the AnkiConnect HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: Callable[[str, dict[str, object]], object] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or self._default_transport

    def check_connection(self) -> int:
        """Verify that AnkiConnect is reachable."""

        version = self._invoke("version", {})
        return int(version)

    def validate_target(self, deck_name: str, model_name: str) -> None:
        """Ensure the deck, model, and fields exist before mutating Anki."""

        self.check_connection()

        deck_names = self.get_deck_names()
        if deck_name not in deck_names:
            raise ValidationError(f"Anki deck does not exist: {deck_name}")

        model_names = self.get_model_names()
        if model_name not in model_names:
            raise ValidationError(f"Anki model does not exist: {model_name}")

        fields = self.get_model_field_names(model_name)
        missing_fields = [field for field in REQUIRED_FIELDS if field not in fields]
        if missing_fields:
            raise ValidationError(
                f"Anki model '{model_name}' is missing required fields: "
                + ", ".join(missing_fields)
            )

        if not fields or fields[0] != "SentenceEN":
            raise ValidationError(
                f"Anki model '{model_name}' must use SentenceEN as the first field."
            )

    def get_deck_names(self) -> list[str]:
        return list(self._invoke("deckNames", {}))

    def get_model_names(self) -> list[str]:
        return list(self._invoke("modelNames", {}))

    def get_model_field_names(self, model_name: str) -> list[str]:
        return list(self._invoke("modelFieldNames", {"modelName": model_name}))

    def can_add_notes(self, notes: Sequence[PreparedNote]) -> list[bool]:
        """Ask AnkiConnect which notes can be added without duplication."""

        result = self._invoke(
            "canAddNotes",
            {"notes": [note.as_anki_note() for note in notes]},
        )
        allowed = [bool(item) for item in result]
        if len(allowed) != len(notes):
            raise AnkiConnectError(
                "AnkiConnect returned an unexpected number of duplicate-check results."
            )
        return allowed

    def store_media_file(self, audio: AudioPayload) -> str:
        """Upload media bytes to Anki."""

        return str(
            self._invoke(
                "storeMediaFile",
                {
                    "filename": audio.filename,
                    "data": base64.b64encode(audio.content).decode("ascii"),
                },
            )
        )

    def retrieve_media_file(self, filename: str) -> bytes:
        """Read back media bytes from Anki for verification."""

        result = self._invoke("retrieveMediaFile", {"filename": filename})
        if result in (None, False, ""):
            return b""
        if not isinstance(result, str):
            raise AnkiConnectError(
                "AnkiConnect returned an unexpected retrieveMediaFile payload."
            )
        try:
            return base64.b64decode(result, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise AnkiConnectError(
                "AnkiConnect returned invalid media data for retrieveMediaFile."
            ) from exc

    def add_note(self, note: PreparedNote) -> int:
        """Create one note in Anki."""

        return int(self._invoke("addNote", {"note": note.as_anki_note()}))

    def _invoke(self, action: str, params: dict[str, object]) -> object:
        return self.transport(action, params)

    def _default_transport(self, action: str, params: dict[str, object]) -> object:
        payload = json.dumps(
            {
                "action": action,
                "version": 6,
                "params": params,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise AnkiConnectError(
                f"AnkiConnect HTTP {exc.code} for action '{action}'."
            ) from exc
        except urllib.error.URLError as exc:
            raise AnkiConnectError(
                f"Could not reach AnkiConnect at {self.base_url}: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnkiConnectError("AnkiConnect returned invalid JSON.") from exc

        if parsed.get("error"):
            raise AnkiConnectError(
                f"AnkiConnect action '{action}' failed: {parsed['error']}"
            )

        return parsed.get("result")
