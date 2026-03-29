"""Shared typed models used across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """Configuration for retrying transient operations."""

    max_attempts: int = 5
    initial_delay_s: float = 1.0
    max_delay_s: float = 16.0
    multiplier: float = 2.0
    jitter: str = "full"


class CardOutcomeStatus(str, Enum):
    """Possible per-card results."""

    IMPORTED = "imported"
    WOULD_IMPORT = "would_import"
    SKIPPED_INPUT_DUPLICATE = "skipped_input_duplicate"
    SKIPPED_ANKI_DUPLICATE = "skipped_anki_duplicate"
    FAILED_TTS = "failed_tts"
    FAILED_MEDIA_UPLOAD = "failed_media_upload"
    FAILED_NOTE_CREATION = "failed_note_creation"


@dataclass(slots=True, frozen=True)
class CardRow:
    """A validated row from the input file."""

    line_number: int
    sentence_en: str
    translation_pt: str
    notes: str
    duplicate_key: str
    audio_filename: str


@dataclass(slots=True, frozen=True)
class PreparedNote:
    """A note payload ready for AnkiConnect."""

    line_number: int
    deck_name: str
    model_name: str
    fields: dict[str, str]
    audio_filename: str

    def as_anki_note(self) -> dict[str, object]:
        """Convert the note to the AnkiConnect payload shape."""

        return {
            "deckName": self.deck_name,
            "modelName": self.model_name,
            "fields": self.fields,
            "options": {"allowDuplicate": False},
        }


@dataclass(slots=True, frozen=True)
class AudioPayload:
    """Audio bytes to upload to Anki media."""

    filename: str
    content: bytes
    mime_type: str = "audio/wav"

    @property
    def byte_length(self) -> int:
        """Return the payload size in bytes."""

        return len(self.content)


@dataclass(slots=True, frozen=True)
class CardOutcome:
    """A user-visible result for one input row."""

    line_number: int
    status: CardOutcomeStatus
    message: str


@dataclass(slots=True, frozen=True)
class ParsedInput:
    """Result of parsing and deduplicating an input file."""

    total_lines: int
    valid_rows: int
    unique_rows: tuple[CardRow, ...]
    duplicate_outcomes: tuple[CardOutcome, ...]


@dataclass(slots=True)
class RunSummary:
    """Aggregated results for an import run."""

    mode: str
    input_path: Path
    total_lines: int = 0
    valid_rows: int = 0
    parse_error_count: int = 0
    elapsed_seconds: float = 0.0
    outcomes: list[CardOutcome] = field(default_factory=list)

    def add_outcome(self, outcome: CardOutcome) -> None:
        """Append a row outcome to the summary."""

        self.outcomes.append(outcome)

    @property
    def input_duplicate_count(self) -> int:
        return self._count(CardOutcomeStatus.SKIPPED_INPUT_DUPLICATE)

    @property
    def anki_duplicate_count(self) -> int:
        return self._count(CardOutcomeStatus.SKIPPED_ANKI_DUPLICATE)

    @property
    def imported_count(self) -> int:
        return self._count(CardOutcomeStatus.IMPORTED)

    @property
    def would_import_count(self) -> int:
        return self._count(CardOutcomeStatus.WOULD_IMPORT)

    @property
    def failed_tts_count(self) -> int:
        return self._count(CardOutcomeStatus.FAILED_TTS)

    @property
    def failed_media_upload_count(self) -> int:
        return self._count(CardOutcomeStatus.FAILED_MEDIA_UPLOAD)

    @property
    def failed_note_creation_count(self) -> int:
        return self._count(CardOutcomeStatus.FAILED_NOTE_CREATION)

    @property
    def failure_count(self) -> int:
        return (
            self.failed_tts_count
            + self.failed_media_upload_count
            + self.failed_note_creation_count
        )

    @property
    def is_success(self) -> bool:
        return self.parse_error_count == 0 and self.failure_count == 0

    def notable_outcomes(self) -> list[CardOutcome]:
        """Return skipped or failed outcomes for compact reporting."""

        return [
            outcome
            for outcome in self.outcomes
            if outcome.status
            not in {CardOutcomeStatus.IMPORTED, CardOutcomeStatus.WOULD_IMPORT}
        ]

    def _count(self, status: CardOutcomeStatus) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status is status)
