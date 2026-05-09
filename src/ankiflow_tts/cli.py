"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from .anki_connect import AnkiConnectClient
from .config import Settings, build_import_settings
from .exceptions import AnkiFlowError, ParseError
from .importer import Importer
from .reporting import render_parse_error, render_summaries
from .tts_deepgram import DeepgramTtsClient


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="ankiflow-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import",
        help="Import text files into Anki, generating Deepgram audio when the model uses audio.",
    )
    import_parser.add_argument(
        "--input",
        dest="input_path",
        help=(
            "Path to one input text file. If omitted, imports every non-empty "
            "*.txt file in data."
        ),
    )
    import_parser.add_argument(
        "--deck",
        help="Deprecated. Deck is read from the first line of each input file.",
    )
    import_parser.add_argument(
        "--model",
        help="Deprecated. Model is read from the second line of each input file.",
    )
    import_parser.add_argument("--anki-url", help="AnkiConnect base URL.")
    import_parser.add_argument("--deepgram-api-key", help="Deepgram API key override.")
    import_parser.add_argument(
        "--deepgram-model",
        help="Deepgram TTS model override. Defaults to aura-2-thalia-en.",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview without generating audio or creating notes.",
    )
    import_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        settings = build_import_settings(args)
        configure_logging(settings[0] if settings else _logging_settings(args))
        if not settings:
            print(
                "No cards to import: data has no non-empty .txt import files."
            )
            return 0

        importer = Importer(
            anki_client=AnkiConnectClient(settings[0].anki_url),
            tts_client=_build_tts_client(settings[0]),
            logger=logging.getLogger("ankiflow_tts"),
        )
        summaries = importer.run_many(settings)
    except ParseError as exc:
        print(render_parse_error(exc), file=sys.stderr)
        return 1
    except AnkiFlowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(render_summaries(summaries))
    return 0 if all(summary.is_success for summary in summaries) else 1


def configure_logging(settings: Settings) -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=logging.DEBUG if settings.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def _logging_settings(args: object) -> Settings:
    """Create minimal settings for configuring logging when there is no job."""

    from pathlib import Path

    from .types import RateLimitPolicy, RetryPolicy

    return Settings(
        input_path=Path("."),
        deck_name="",
        model_name="",
        anki_url="",
        deepgram_api_key=None,
        deepgram_model="",
        dry_run=bool(getattr(args, "dry_run", False)),
        verbose=bool(getattr(args, "verbose", False)),
        retry_policy=RetryPolicy(),
        rate_limit_policy=RateLimitPolicy(),
    )


def _build_tts_client(settings: Settings) -> DeepgramTtsClient | None:
    if settings.dry_run or not settings.deepgram_api_key:
        return None
    return DeepgramTtsClient(
        api_key=settings.deepgram_api_key or "",
        model=settings.deepgram_model,
        retry_policy=settings.retry_policy,
        rate_limit_policy=settings.rate_limit_policy,
        logger=logging.getLogger("ankiflow_tts.tts"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
