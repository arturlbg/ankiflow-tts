"""Supported Anki note schemas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class NoteSchema:
    """Field mapping for one supported Anki note type."""

    model_name: str
    field_names: tuple[str, ...]
    first_field: str
    duplicate_field: str
    required_non_empty_fields: tuple[str, ...]
    tts_field: str | None = None
    audio_field: str | None = None

    @property
    def audio_index(self) -> int | None:
        if self.audio_field is None:
            return None
        return self.field_names.index(self.audio_field)

    @property
    def requires_audio(self) -> bool:
        return self.audio_field is not None and self.tts_field is not None


SUPPORTED_SCHEMAS = {
    "English Setence PT": NoteSchema(
        model_name="English Setence PT",
        field_names=("SentenceEN", "TranslationPT", "AudioEN", "Notes"),
        first_field="SentenceEN",
        duplicate_field="SentenceEN",
        required_non_empty_fields=("SentenceEN", "TranslationPT"),
        tts_field="SentenceEN",
        audio_field="AudioEN",
    ),
    "TagNoteType": NoteSchema(
        model_name="TagNoteType",
        field_names=(
            "Target English",
            "Portuguese",
            "Audio",
            "Usage Notes",
            "Extra Examples",
            "Production Prompt",
            "Tags / Pattern",
        ),
        first_field="Target English",
        duplicate_field="Target English",
        required_non_empty_fields=("Target English", "Portuguese"),
        tts_field="Target English",
        audio_field="Audio",
    ),
    "Chunk": NoteSchema(
        model_name="Chunk",
        field_names=(
            "Question",
            "Target Chunk",
            "Task",
            "Example Answer",
            "Useful Vocabulary",
            "Notes",
        ),
        first_field="Question",
        duplicate_field="Question",
        required_non_empty_fields=("Question", "Target Chunk"),
        tts_field="Target Chunk",
        audio_field=None,
    ),
}


def get_schema(model_name: str) -> NoteSchema | None:
    """Return the supported schema for an Anki model name."""

    return SUPPORTED_SCHEMAS.get(model_name)
