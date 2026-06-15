"""Configuration loading for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any

from .exceptions import ConfigError
from .schemas import get_schema
from .types import RateLimitPolicy, RetryPolicy

DEFAULT_ANKICONNECT_URL = "http://127.0.0.1:8765"
DEFAULT_DEEPGRAM_MODEL = "aura-2-thalia-en"
DEFAULT_IMPORT_DIR = Path("data")


@dataclass(slots=True, frozen=True)
class Settings:
    """Normalized application settings."""

    input_path: Path
    deck_name: str
    model_name: str
    anki_url: str
    deepgram_api_key: str | None
    deepgram_model: str
    dry_run: bool
    verbose: bool
    retry_policy: RetryPolicy
    rate_limit_policy: RateLimitPolicy


def build_import_settings(args: Any, env: dict[str, str] | None = None) -> tuple[Settings, ...]:
    """Build one or more import jobs from CLI arguments and environment variables."""

    env_values = dict(environ if env is None else env)
    input_value = _clean(getattr(args, "input_path", None))
    if input_value:
        return (build_settings(args, env=env_values),)

    jobs: list[Settings] = []
    if not DEFAULT_IMPORT_DIR.exists():
        return ()

    for input_path in sorted(DEFAULT_IMPORT_DIR.glob("*.txt")):
        if not input_path.exists():
            continue
        if not input_path.is_file():
            raise ConfigError(f"Import input path is not a file: {input_path}")
        if _input_file_is_blank(input_path):
            continue

        jobs.append(
            _build_settings(
                input_path=input_path,
                deck_name="",
                model_name="",
                args=args,
                env_values=env_values,
            )
        )

    return tuple(jobs)


def build_settings(args: Any, env: dict[str, str] | None = None) -> Settings:
    """Build validated settings from CLI arguments and environment variables."""

    env_values = dict(environ if env is None else env)

    input_value = _clean(getattr(args, "input_path", None))
    if not input_value:
        raise ConfigError("The --input argument is required.")

    input_path = Path(input_value).expanduser()
    if not input_path.exists():
        raise ConfigError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ConfigError(f"Input path is not a file: {input_path}")

    return _build_settings(
        input_path=input_path,
        deck_name=_choose_value(getattr(args, "deck", None)) or "",
        model_name=_choose_value(getattr(args, "model", None)) or "",
        args=args,
        env_values=env_values,
    )


def _build_settings(
    *,
    input_path: Path,
    deck_name: str,
    model_name: str,
    args: Any,
    env_values: dict[str, str],
) -> Settings:
    anki_url = _choose_value(
        getattr(args, "anki_url", None),
        env_values.get("ANKICONNECT_URL"),
    )
    anki_url = anki_url or DEFAULT_ANKICONNECT_URL

    dry_run = bool(getattr(args, "dry_run", False))
    verbose = bool(getattr(args, "verbose", False))

    deepgram_api_key = _choose_value(
        getattr(args, "deepgram_api_key", None),
        env_values.get("DEEPGRAM_API_KEY"),
    )
    deepgram_model = (
        _choose_value(
            getattr(args, "deepgram_model", None),
            env_values.get("DEEPGRAM_MODEL"),
        )
        or DEFAULT_DEEPGRAM_MODEL
    )

    if not dry_run and _input_requires_audio(input_path):
        missing = []
        if deepgram_api_key is None:
            missing.append("DEEPGRAM_API_KEY or --deepgram-api-key")
        if missing:
            raise ConfigError(
                "Live imports require Deepgram configuration: " + ", ".join(missing)
            )

    return Settings(
        input_path=input_path.resolve(),
        deck_name=deck_name,
        model_name=model_name,
        anki_url=anki_url.rstrip("/"),
        deepgram_api_key=deepgram_api_key,
        deepgram_model=deepgram_model,
        dry_run=dry_run,
        verbose=verbose,
        retry_policy=RetryPolicy(),
        rate_limit_policy=RateLimitPolicy(),
    )


def _input_file_is_blank(input_path: Path) -> bool:
    return not any(
        line.strip()
        for line in input_path.read_text(encoding="utf-8-sig").splitlines()
    )


def _input_requires_audio(input_path: Path) -> bool:
    lines = input_path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        return False
    model_name = lines[1].strip()
    schema = get_schema(model_name) if model_name else None
    if not schema or not schema.requires_audio:
        return False

    audio_index = schema.audio_index
    if audio_index is None:
        return False

    for raw_line in lines[2:]:
        if not raw_line.strip():
            continue
        parts = raw_line.split(";")
        if len(parts) <= audio_index:
            continue
        audio_value = parts[audio_index].strip()
        if audio_value == "":
            return True
    return False


def _choose_value(*values: object) -> str | None:
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None:
            return cleaned
    return None


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
