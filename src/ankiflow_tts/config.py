"""Configuration loading for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any

from .exceptions import ConfigError
from .types import RateLimitPolicy, RetryPolicy

DEFAULT_ANKICONNECT_URL = "http://127.0.0.1:8765"
DEFAULT_DEEPGRAM_MODEL = "aura-2-thalia-en"


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

    deck_name = _choose_value(getattr(args, "deck", None), env_values.get("DEFAULT_DECK"))
    if deck_name is None:
        raise ConfigError("A target deck is required via --deck or DEFAULT_DECK.")

    model_name = _choose_value(
        getattr(args, "model", None),
        env_values.get("DEFAULT_MODEL"),
    )
    if model_name is None:
        raise ConfigError("A target model is required via --model or DEFAULT_MODEL.")

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

    if not dry_run:
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
