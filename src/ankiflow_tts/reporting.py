"""Console reporting helpers."""

from __future__ import annotations

from .exceptions import ParseError
from .types import RunSummary


def render_parse_error(error: ParseError) -> str:
    """Render parse failures with line numbers."""

    lines = ["Input validation failed:"]
    for issue in error.issues:
        lines.append(f"- line {issue.line_number}: {issue.message}")
    return "\n".join(lines)


def render_summary(summary: RunSummary) -> str:
    """Render the final run summary."""

    lines = [
        f"Import summary ({summary.mode})",
        f"Input: {summary.input_path}",
        f"Lines read: {summary.total_lines}",
        f"Valid rows: {summary.valid_rows}",
        f"Input duplicates: {summary.input_duplicate_count}",
        f"Anki duplicates: {summary.anki_duplicate_count}",
    ]

    if summary.mode == "dry-run":
        lines.append(f"Would import: {summary.would_import_count}")
    else:
        lines.append(f"Imported: {summary.imported_count}")

    lines.extend(
        [
            f"Failed TTS: {summary.failed_tts_count}",
            f"Failed media uploads: {summary.failed_media_upload_count}",
            f"Failed note creation: {summary.failed_note_creation_count}",
            f"Elapsed seconds: {summary.elapsed_seconds:.2f}",
        ]
    )

    notable = summary.notable_outcomes()
    if notable:
        lines.append("")
        lines.append("Skipped / failed lines:")
        for outcome in notable:
            lines.append(
                f"- line {outcome.line_number}: {outcome.status.value} - {outcome.message}"
            )

    return "\n".join(lines)
