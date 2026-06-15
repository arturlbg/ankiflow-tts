"""Import orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .anki_connect import AnkiConnectClient
from .config import Settings
from .exceptions import AnkiConnectError, ConfigError, TtsGenerationError
from .parser import parse_input_file
from .schemas import get_schema
from .tts_deepgram import DeepgramTtsClient
from .types import (
    CardOutcome,
    CardOutcomeStatus,
    CardRow,
    ParsedInput,
    PreparedNote,
    RunSummary,
)


class Importer:
    """Coordinates the end-to-end import flow."""

    def __init__(
        self,
        *,
        anki_client: AnkiConnectClient,
        tts_client: DeepgramTtsClient | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.anki_client = anki_client
        self.tts_client = tts_client
        self.logger = logger or logging.getLogger(__name__)

    def run(self, settings: Settings) -> RunSummary:
        """Execute the import run and return a summary."""

        return self.run_many((settings,))[0]

    def run_many(self, settings_list: tuple[Settings, ...]) -> tuple[RunSummary, ...]:
        """Execute one or more import jobs as one fail-fast batch."""

        started_at = time.perf_counter()
        states = tuple(
            self._parse_job(settings=settings)
            for settings in settings_list
        )

        for state in states:
            if state.parsed.unique_rows:
                schema = get_schema(state.parsed.model_name)
                if schema is None:
                    raise ConfigError(
                        f"Unsupported Anki model for import: {state.parsed.model_name}"
                    )
                self.anki_client.validate_target(
                    state.parsed.deck_name,
                    state.parsed.model_name,
                    required_fields=schema.field_names,
                    first_field=schema.first_field,
                )

        if any(
            not state.settings.dry_run
            and any(row.audio_filename is not None for row in state.parsed.unique_rows)
            for state in states
        ) and self.tts_client is None:
            raise ConfigError(
                "Live imports for audio-backed models require Deepgram configuration: "
                "DEEPGRAM_API_KEY or --deepgram-api-key"
            )

        for state in states:
            if not state.parsed.unique_rows:
                continue

            prepared_notes = [
                prepare_note(
                    row,
                    state.parsed.deck_name,
                    state.parsed.model_name,
                )
                for row in state.parsed.unique_rows
            ]

            allowed_results = self.anki_client.can_add_notes(prepared_notes)
            for note, is_allowed in zip(prepared_notes, allowed_results, strict=True):
                if is_allowed:
                    state.eligible_notes.append(note)
                    continue

                outcome = CardOutcome(
                    line_number=note.line_number,
                    status=CardOutcomeStatus.SKIPPED_ANKI_DUPLICATE,
                    message="Duplicate note already exists in Anki.",
                )
                self.logger.info(
                    "event=anki_duplicate mode=%s line=%s message=%s",
                    state.summary.mode,
                    outcome.line_number,
                    outcome.message,
                )
                state.summary.add_outcome(outcome)

        for state in states:
            if state.settings.dry_run:
                self._record_dry_run_outcomes(state)
                self._complete_summary(state.summary, started_at)
                continue

            if self.tts_client is None and any(
                note.audio_filename is not None for note in state.eligible_notes
            ):
                raise ConfigError("A Deepgram TTS client is required for live imports.")

            for note in state.eligible_notes:
                self._process_live_note(state.summary, note)
            self._complete_summary(state.summary, started_at)

        return tuple(state.summary for state in states)

    def _parse_job(self, *, settings: Settings) -> "_ImportState":
        parsed = parse_input_file(
            settings.input_path,
            tts_model=settings.deepgram_model,
        )

        summary = RunSummary(
            mode="dry-run" if settings.dry_run else "live",
            input_path=settings.input_path,
            total_lines=parsed.total_lines,
            valid_rows=parsed.valid_rows,
        )

        for outcome in parsed.duplicate_outcomes:
            self.logger.info(
                "event=input_duplicate mode=%s line=%s message=%s",
                summary.mode,
                outcome.line_number,
                outcome.message,
            )
            summary.add_outcome(outcome)

        return _ImportState(settings=settings, parsed=parsed, summary=summary)

    def _record_dry_run_outcomes(self, state: "_ImportState") -> None:
        for note in state.eligible_notes:
            if note.audio_filename:
                message = f"Would import note with audio file {note.audio_filename}."
            else:
                message = "Would import note without audio."
            state.summary.add_outcome(
                CardOutcome(
                    line_number=note.line_number,
                    status=CardOutcomeStatus.WOULD_IMPORT,
                    message=message,
                )
            )

    def _process_live_note(self, summary: RunSummary, note: PreparedNote) -> None:
        if note.audio_filename is None:
            self._create_text_only_note(summary, note)
            return

        if self.tts_client is None:
            raise ConfigError(
                "A Deepgram TTS client is required for audio-backed live imports."
            )

        self.logger.info(
            "event=card_start mode=live line=%s filename=%s",
            note.line_number,
            note.audio_filename,
        )
        try:
            audio = self.tts_client.synthesize(
                text=note.tts_text,
                filename=note.audio_filename,
            )
        except TtsGenerationError as exc:
            outcome = CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.FAILED_TTS,
                message=str(exc),
            )
            self.logger.error(
                "event=tts_failed mode=live line=%s attempts=%s error=%s",
                note.line_number,
                exc.attempts or 0,
                exc,
            )
            summary.add_outcome(outcome)
            return

        try:
            stored_filename = self.anki_client.store_media_file(audio)
            if stored_filename != note.audio_filename:
                raise AnkiConnectError(
                    "AnkiConnect returned an unexpected media filename after upload."
                )
            stored_audio = self.anki_client.retrieve_media_file(note.audio_filename)
            if not stored_audio:
                raise AnkiConnectError(
                    "Uploaded media could not be verified in Anki."
                )
            if stored_audio != audio.content:
                raise AnkiConnectError(
                    "Uploaded media in Anki does not match the generated audio."
                )
        except AnkiConnectError as exc:
            outcome = CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.FAILED_MEDIA_UPLOAD,
                message=str(exc),
            )
            self.logger.error(
                "event=media_upload_failed mode=live line=%s error=%s",
                note.line_number,
                exc,
            )
            summary.add_outcome(outcome)
            return

        try:
            note_id = self.anki_client.add_note(note)
        except AnkiConnectError as exc:
            outcome = CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.FAILED_NOTE_CREATION,
                message=str(exc),
            )
            self.logger.error(
                "event=note_creation_failed mode=live line=%s error=%s",
                note.line_number,
                exc,
            )
            summary.add_outcome(outcome)
            return

        self.logger.info(
            "event=note_imported mode=live line=%s note_id=%s",
            note.line_number,
            note_id,
        )
        summary.add_outcome(
            CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.IMPORTED,
                message=f"Imported note {note_id}.",
            )
        )

    def _create_text_only_note(self, summary: RunSummary, note: PreparedNote) -> None:
        self.logger.info(
            "event=card_start mode=live line=%s type=text_only",
            note.line_number,
        )
        try:
            note_id = self.anki_client.add_note(note)
        except AnkiConnectError as exc:
            outcome = CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.FAILED_NOTE_CREATION,
                message=str(exc),
            )
            self.logger.error(
                "event=note_creation_failed mode=live line=%s error=%s",
                note.line_number,
                exc,
            )
            summary.add_outcome(outcome)
            return

        self.logger.info(
            "event=note_imported mode=live line=%s note_id=%s type=text_only",
            note.line_number,
            note_id,
        )
        summary.add_outcome(
            CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.IMPORTED,
                message=f"Imported note {note_id}.",
            )
        )

    def _complete_summary(self, summary: RunSummary, started_at: float) -> None:
        summary.elapsed_seconds = time.perf_counter() - started_at
        self.logger.info(
            "event=run_complete mode=%s imported=%s would_import=%s failures=%s",
            summary.mode,
            summary.imported_count,
            summary.would_import_count,
            summary.failure_count,
        )


def prepare_note(row: CardRow, deck_name: str, model_name: str) -> PreparedNote:
    """Convert a parsed row into an Anki note payload."""

    fields = dict(row.fields)
    if row.audio_field_name is not None:
        fields[row.audio_field_name] = (
            f"[sound:{row.audio_filename}]" if row.audio_filename is not None else ""
        )
    return PreparedNote(
        line_number=row.line_number,
        deck_name=deck_name,
        model_name=model_name,
        fields=fields,
        audio_filename=row.audio_filename,
        tts_text=row.tts_text,
    )


@dataclass(slots=True)
class _ImportState:
    settings: Settings
    parsed: ParsedInput
    summary: RunSummary
    eligible_notes: list[PreparedNote] = field(default_factory=list)
