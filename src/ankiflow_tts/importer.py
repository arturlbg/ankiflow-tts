"""Import orchestration."""

from __future__ import annotations

import logging
import time

from .anki_connect import AnkiConnectClient
from .config import Settings
from .exceptions import AnkiConnectError, ConfigError, GeminiTtsError
from .parser import parse_input_file
from .tts_gemini import GeminiTtsClient
from .types import CardOutcome, CardOutcomeStatus, CardRow, PreparedNote, RunSummary


class Importer:
    """Coordinates the end-to-end import flow."""

    def __init__(
        self,
        *,
        anki_client: AnkiConnectClient,
        tts_client: GeminiTtsClient | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.anki_client = anki_client
        self.tts_client = tts_client
        self.logger = logger or logging.getLogger(__name__)

    def run(self, settings: Settings) -> RunSummary:
        """Execute the import run and return a summary."""

        started_at = time.perf_counter()
        parsed = parse_input_file(
            settings.input_path,
            gemini_model=settings.gemini_model,
            gemini_voice=settings.gemini_voice,
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

        if not parsed.unique_rows:
            summary.elapsed_seconds = time.perf_counter() - started_at
            self.logger.info(
                "event=run_complete mode=%s imported=%s would_import=%s failures=%s",
                summary.mode,
                summary.imported_count,
                summary.would_import_count,
                summary.failure_count,
            )
            return summary

        self.anki_client.validate_target(settings.deck_name, settings.model_name)
        prepared_notes = [
            prepare_note(row, settings.deck_name, settings.model_name)
            for row in parsed.unique_rows
        ]

        allowed_results = self.anki_client.can_add_notes(prepared_notes)
        eligible_notes: list[PreparedNote] = []
        for note, is_allowed in zip(prepared_notes, allowed_results, strict=True):
            if is_allowed:
                eligible_notes.append(note)
                continue

            outcome = CardOutcome(
                line_number=note.line_number,
                status=CardOutcomeStatus.SKIPPED_ANKI_DUPLICATE,
                message="Duplicate note already exists in Anki.",
            )
            self.logger.info(
                "event=anki_duplicate mode=%s line=%s message=%s",
                summary.mode,
                outcome.line_number,
                outcome.message,
            )
            summary.add_outcome(outcome)

        if settings.dry_run:
            for note in eligible_notes:
                summary.add_outcome(
                    CardOutcome(
                        line_number=note.line_number,
                        status=CardOutcomeStatus.WOULD_IMPORT,
                        message=f"Would import note with audio file {note.audio_filename}.",
                    )
                )
            summary.elapsed_seconds = time.perf_counter() - started_at
            self.logger.info(
                "event=run_complete mode=%s imported=%s would_import=%s failures=%s",
                summary.mode,
                summary.imported_count,
                summary.would_import_count,
                summary.failure_count,
            )
            return summary

        if self.tts_client is None:
            raise ConfigError("A Gemini TTS client is required for live imports.")

        for note in eligible_notes:
            self.logger.info(
                "event=card_start mode=live line=%s filename=%s",
                note.line_number,
                note.audio_filename,
            )
            try:
                audio = self.tts_client.synthesize(
                    text=note.fields["SentenceEN"],
                    filename=note.audio_filename,
                )
            except GeminiTtsError as exc:
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
                continue

            try:
                self.anki_client.store_media_file(audio)
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
                continue

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
                continue

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

        summary.elapsed_seconds = time.perf_counter() - started_at
        self.logger.info(
            "event=run_complete mode=%s imported=%s would_import=%s failures=%s",
            summary.mode,
            summary.imported_count,
            summary.would_import_count,
            summary.failure_count,
        )
        return summary


def prepare_note(row: CardRow, deck_name: str, model_name: str) -> PreparedNote:
    """Convert a parsed row into an Anki note payload."""

    return PreparedNote(
        line_number=row.line_number,
        deck_name=deck_name,
        model_name=model_name,
        fields={
            "SentenceEN": row.sentence_en,
            "TranslationPT": row.translation_pt,
            "AudioEN": f"[sound:{row.audio_filename}]",
            "Notes": row.notes,
        },
        audio_filename=row.audio_filename,
    )
