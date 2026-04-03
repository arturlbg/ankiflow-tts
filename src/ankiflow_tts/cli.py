"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from .anki_connect import AnkiConnectClient
from .config import Settings, build_settings
from .exceptions import AnkiFlowError, ParseError
from .importer import Importer
from .reporting import render_parse_error, render_summary
from .tts_deepgram import DeepgramTtsClient


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="ankiflow-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import",
        help="Import a text file into Anki with Deepgram-generated audio.",
    )
    import_parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Path to the input text file.",
    )
    import_parser.add_argument("--deck", help="Target Anki deck name.")
    import_parser.add_argument("--model", help="Target Anki note type.")
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
        settings = build_settings(args)
        configure_logging(settings)
        importer = Importer(
            anki_client=AnkiConnectClient(settings.anki_url),
            tts_client=_build_tts_client(settings),
            logger=logging.getLogger("ankiflow_tts"),
        )
        summary = importer.run(settings)
    except ParseError as exc:
        print(render_parse_error(exc), file=sys.stderr)
        return 1
    except AnkiFlowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(render_summary(summary))
    return 0 if summary.is_success else 1


def configure_logging(settings: Settings) -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=logging.DEBUG if settings.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def _build_tts_client(settings: Settings) -> DeepgramTtsClient | None:
    if settings.dry_run:
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
