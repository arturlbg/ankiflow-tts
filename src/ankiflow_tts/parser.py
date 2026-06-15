"""Input file parsing and validation."""

from __future__ import annotations

from pathlib import Path

from .exceptions import ParseError, ParseIssue
from .filenames import build_audio_filename, normalize_duplicate_key
from .schemas import NoteSchema, get_schema
from .types import CardOutcome, CardOutcomeStatus, CardRow, ParsedInput


def parse_input_file(
    input_path: Path,
    *,
    tts_model: str | None,
) -> ParsedInput:
    """Parse an input file and detect duplicates before any network work."""

    raw_lines = input_path.read_text(encoding="utf-8-sig").splitlines()
    issues: list[ParseIssue] = []
    parsed_rows: list[CardRow] = []

    if len(raw_lines) < 2:
        raise ParseError(
            issues=[
                ParseIssue(
                    line_number=1,
                    message="Input file must start with deck name and model name lines.",
                )
            ],
            total_lines=len(raw_lines),
            input_path=input_path,
        )

    deck_name = raw_lines[0].strip()
    model_name = raw_lines[1].strip()

    if not deck_name:
        issues.append(ParseIssue(line_number=1, message="Deck name must not be empty."))
    if not model_name:
        issues.append(ParseIssue(line_number=2, message="Model name must not be empty."))

    schema = get_schema(model_name) if model_name else None
    if model_name and schema is None:
        issues.append(
            ParseIssue(
                line_number=2,
                message=f"Unsupported Anki model for import: {model_name}",
            )
        )

    if issues:
        raise ParseError(issues=issues, total_lines=len(raw_lines), input_path=input_path)

    assert schema is not None

    for line_number, raw_line in enumerate(raw_lines[2:], start=3):
        if not raw_line.strip():
            continue

        parts = _split_model_row(raw_line, schema)
        if parts is None:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    message=(
                        "Expected at least "
                        f"{len(schema.field_names)} semicolon-separated fields "
                        f"for model '{model_name}'."
                    ),
                )
            )
            continue

        fields = {
            field_name: field_value.strip()
            for field_name, field_value in zip(schema.field_names, parts, strict=True)
        }
        skip_audio = False
        if schema.audio_field is not None:
            audio_value = fields[schema.audio_field]
            skip_audio = audio_value == "x"
            if audio_value and not skip_audio:
                issues.append(
                    ParseIssue(
                        line_number=line_number,
                        message=(
                            f"The audio field '{schema.audio_field}' must be empty "
                            "or exactly 'x'."
                        ),
                    )
                )
                continue

        missing_required_fields = [
            field_name
            for field_name in schema.required_non_empty_fields
            if not fields[field_name]
        ]
        if missing_required_fields:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    message=f"{missing_required_fields[0]} must not be empty.",
                )
            )
            continue

        tts_text = _resolve_tts_text(schema, fields)
        audio_filename = (
            build_audio_filename(tts_text, tts_model)
            if schema.requires_audio and not skip_audio and tts_text
            else None
        )
        duplicate_source = fields[schema.duplicate_field]

        parsed_rows.append(
            CardRow(
                line_number=line_number,
                tts_text=tts_text,
                fields=fields,
                audio_field_name=schema.audio_field,
                duplicate_key=normalize_duplicate_key(duplicate_source),
                audio_filename=audio_filename,
            )
        )

    if issues:
        raise ParseError(issues=issues, total_lines=len(raw_lines), input_path=input_path)

    seen_lines: dict[str, int] = {}
    unique_rows: list[CardRow] = []
    duplicate_outcomes: list[CardOutcome] = []

    for row in parsed_rows:
        first_line = seen_lines.get(row.duplicate_key)
        if first_line is None:
            seen_lines[row.duplicate_key] = row.line_number
            unique_rows.append(row)
            continue

        duplicate_outcomes.append(
            CardOutcome(
                line_number=row.line_number,
                status=CardOutcomeStatus.SKIPPED_INPUT_DUPLICATE,
                message=f"Duplicate target text already appeared on line {first_line}.",
            )
        )

    return ParsedInput(
        deck_name=deck_name,
        model_name=model_name,
        total_lines=len(raw_lines),
        valid_rows=len(parsed_rows),
        unique_rows=tuple(unique_rows),
        duplicate_outcomes=tuple(duplicate_outcomes),
    )


def _split_model_row(raw_line: str, schema: NoteSchema) -> list[str] | None:
    parts = raw_line.split(";")
    field_count = len(schema.field_names)
    if len(parts) < field_count:
        return None
    if len(parts) > field_count:
        return parts[: field_count - 1] + [";".join(parts[field_count - 1 :])]
    return parts


def _resolve_tts_text(schema: NoteSchema, fields: dict[str, str]) -> str:
    if schema.tts_field is None:
        return ""
    if schema.model_name == "English Setence PT":
        notes = fields.get("Notes", "")
        card_type = notes.split("|", maxsplit=1)[0].strip()
        if card_type in {
            "production",
            "transformation",
            "repair",
            "expansion",
        } and _looks_like_portuguese_prompt(fields["SentenceEN"]):
            return fields["TranslationPT"]
    return fields[schema.tts_field]


def _looks_like_portuguese_prompt(text: str) -> bool:
    normalized_text = text.strip().lower()
    if not normalized_text:
        return False

    portuguese_prefixes = (
        "como ",
        "corrija:",
        "deixe ",
        "diga ",
        "eu ",
        "expanda ",
        "o problema ",
        "pergunte ",
        "responda ",
        "transforme ",
    )
    if normalized_text.startswith(portuguese_prefixes):
        return True

    portuguese_markers = (
        "ção",
        "ções",
        "ã",
        "á",
        "é",
        "í",
        "ó",
        "ú",
        "ç",
        " não ",
        " você ",
    )
    return any(marker in normalized_text for marker in portuguese_markers)
