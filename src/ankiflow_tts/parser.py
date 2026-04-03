"""Input file parsing and validation."""

from __future__ import annotations

from pathlib import Path

from .exceptions import ParseError, ParseIssue
from .filenames import build_audio_filename, normalize_duplicate_key
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

    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue

        parts = raw_line.split(";")
        if len(parts) != 4:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    message="Expected exactly four semicolon-separated fields.",
                )
            )
            continue

        sentence_en = parts[0].strip()
        translation_pt = parts[1].strip()
        third_field = parts[2]
        notes = parts[3].strip()

        if third_field != "":
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    message="The third field must be empty.",
                )
            )
            continue

        if not sentence_en:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    message="SentenceEN must not be empty.",
                )
            )
            continue

        if not translation_pt:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    message="TranslationPT must not be empty.",
                )
            )
            continue

        parsed_rows.append(
            CardRow(
                line_number=line_number,
                sentence_en=sentence_en,
                translation_pt=translation_pt,
                notes=notes,
                duplicate_key=normalize_duplicate_key(sentence_en),
                audio_filename=build_audio_filename(
                    sentence_en,
                    tts_model,
                ),
            )
        )

    if issues:
        raise ParseError(issues=issues, total_lines=len(raw_lines))

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
                message=f"Duplicate SentenceEN already appeared on line {first_line}.",
            )
        )

    return ParsedInput(
        total_lines=len(raw_lines),
        valid_rows=len(parsed_rows),
        unique_rows=tuple(unique_rows),
        duplicate_outcomes=tuple(duplicate_outcomes),
    )
