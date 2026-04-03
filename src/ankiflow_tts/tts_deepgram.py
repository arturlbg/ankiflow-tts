"""Deepgram TTS adapter."""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import Callable
from random import random as default_random

from .exceptions import ConfigError, TtsGenerationError
from .retry import RetryContext, run_with_retry
from .types import AudioPayload, RateLimitPolicy, RetryPolicy

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
DEFAULT_SAMPLE_RATE_HZ = 24_000


class DeepgramTtsClient:
    """Generate WAV audio with Deepgram TTS."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        retry_policy: RetryPolicy,
        rate_limit_policy: RateLimitPolicy,
        logger: logging.Logger | None = None,
        transport: Callable[[str, str, str], tuple[bytes, str | None]] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        random_fn: Callable[[], float] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.retry_policy = retry_policy
        self.rate_limit_policy = rate_limit_policy
        self.logger = logger or logging.getLogger(__name__)
        self.transport = transport or _default_transport
        self.sleep_fn = sleep_fn or time.sleep
        self.random_fn = random_fn or default_random
        self.time_fn = time_fn or time.monotonic
        self._next_request_at: float | None = None

    def synthesize(self, text: str, filename: str) -> AudioPayload:
        """Generate audio for the provided text and return WAV bytes."""

        attempts = 0

        def operation() -> AudioPayload:
            nonlocal attempts
            attempts += 1
            self._wait_for_request_slot()
            audio_bytes, mime_type = self.transport(text, self.model, self.api_key)
            wav_bytes = ensure_wav_bytes(audio_bytes, mime_type=mime_type)
            return AudioPayload(filename=filename, content=wav_bytes, mime_type="audio/wav")

        def on_retry(context: RetryContext) -> None:
            self.logger.warning(
                "event=tts_retry attempt=%s max_attempts=%s delay_s=%.3f error=%s",
                context.attempt + 1,
                context.max_attempts,
                context.delay_s,
                context.exception,
            )

        try:
            return run_with_retry(
                operation,
                policy=self.retry_policy,
                should_retry=is_retryable_deepgram_error,
                on_retry=on_retry,
                sleep_fn=self.sleep_fn,
                random_fn=self.random_fn,
            )
        except Exception as exc:
            raise TtsGenerationError(
                f"Deepgram TTS failed after {attempts} attempt(s): {exc}",
                attempts=attempts,
                retryable=is_retryable_deepgram_error(exc),
            ) from exc

    def _wait_for_request_slot(self) -> None:
        minimum_interval_s = self.rate_limit_policy.minimum_interval_s
        now = self.time_fn()
        if self._next_request_at is not None and now < self._next_request_at:
            delay_s = self._next_request_at - now
            self.logger.info(
                "event=tts_rate_limit_wait delay_s=%.3f minimum_interval_s=%.3f",
                delay_s,
                minimum_interval_s,
            )
            self.sleep_fn(delay_s)
            now = self.time_fn()

        self._next_request_at = now + minimum_interval_s


def is_retryable_deepgram_error(exc: BaseException) -> bool:
    """Return True when Deepgram failures are likely transient."""

    if isinstance(exc, (ConnectionError, TimeoutError, urllib.error.URLError)):
        return True

    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600

    status_code = _coerce_status_code(exc)
    if status_code is not None:
        if status_code == 429 or 500 <= status_code < 600:
            return True
        if 400 <= status_code < 500:
            return False

    name = exc.__class__.__name__.lower()
    message = str(exc).lower()

    transient_markers = (
        "timeout",
        "temporar",
        "rate limit",
        "ratelimit",
        "too many requests",
        "service unavailable",
        "internal server error",
        "empty audio",
        "undecodable audio",
    )
    if any(marker in name or marker in message for marker in transient_markers):
        return True

    permanent_markers = (
        "invalid",
        "permission",
        "unauthorized",
        "forbidden",
        "not found",
        "unsupported",
        "api key",
        "authentication",
    )
    if any(marker in name or marker in message for marker in permanent_markers):
        return False

    return False


def ensure_wav_bytes(audio_bytes: bytes, *, mime_type: str | None) -> bytes:
    """Return verified WAV bytes for an audio response."""

    if not audio_bytes:
        raise ValueError("Deepgram returned empty audio payload.")

    normalized_mime = (mime_type or "").lower()

    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return _validate_wav_bytes(audio_bytes)

    if "wav" in normalized_mime:
        return _validate_wav_bytes(audio_bytes)

    if any(token in normalized_mime for token in ("audio/l16", "audio/pcm", "audio/raw")):
        if len(audio_bytes) % 2 != 0:
            raise ValueError("Undecodable audio payload length.")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(DEFAULT_SAMPLE_RATE_HZ)
            wav_file.writeframes(audio_bytes)
        return _validate_wav_bytes(buffer.getvalue())

    raise ValueError("Undecodable audio payload format.")


def _default_transport(text: str, model: str, api_key: str) -> tuple[bytes, str | None]:
    if not api_key:
        raise ConfigError("Deepgram API key is required for live imports.")

    query = urllib.parse.urlencode(
        {
            "model": model,
            "encoding": "linear16",
            "container": "wav",
        }
    )
    request = urllib.request.Request(
        f"{DEEPGRAM_SPEAK_URL}?{query}",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(), response.headers.get("Content-Type")


def _validate_wav_bytes(wav_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise ValueError("Generated WAV payload contains no audio frames.")
            if not wav_file.readframes(1):
                raise ValueError("Generated WAV payload contains no audio frames.")
    except wave.Error as exc:
        raise ValueError("Undecodable WAV audio payload.") from exc
    return wav_bytes


def _coerce_status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
