"""Gemini TTS adapter."""

from __future__ import annotations

import base64
import io
import logging
import re
import time
import urllib.error
import wave
from collections.abc import Callable, Mapping
from random import random as default_random
from typing import Any

from .exceptions import ConfigError, GeminiTtsError
from .retry import RetryContext, run_with_retry
from .types import AudioPayload, RetryPolicy

DEFAULT_SAMPLE_RATE_HZ = 24_000


class GeminiTtsClient:
    """Generate WAV audio with Gemini TTS."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        retry_policy: RetryPolicy,
        logger: logging.Logger | None = None,
        transport: Callable[[str, str, str, str], object] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        random_fn: Callable[[], float] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.retry_policy = retry_policy
        self.logger = logger or logging.getLogger(__name__)
        self.transport = transport or _default_transport
        self.sleep_fn = sleep_fn or time.sleep
        self.random_fn = random_fn or default_random

    def synthesize(self, text: str, filename: str) -> AudioPayload:
        """Generate audio for the provided text and return WAV bytes."""

        attempts = 0

        def operation() -> AudioPayload:
            nonlocal attempts
            attempts += 1
            response = self.transport(text, self.model, self.voice, self.api_key)
            audio_bytes, mime_type, sample_rate_hz = extract_audio_payload(response)
            wav_bytes = ensure_wav_bytes(
                audio_bytes,
                mime_type=mime_type,
                sample_rate_hz=sample_rate_hz,
            )
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
                should_retry=is_retryable_gemini_error,
                on_retry=on_retry,
                sleep_fn=self.sleep_fn,
                random_fn=self.random_fn,
            )
        except Exception as exc:
            raise GeminiTtsError(
                f"Gemini TTS failed after {attempts} attempt(s): {exc}",
                attempts=attempts,
                retryable=is_retryable_gemini_error(exc),
            ) from exc


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Return True when Gemini failures are likely transient."""

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
        "resourceexhausted",
        "ratelimit",
        "toomanyrequests",
        "serviceunavailable",
        "internalservererror",
        "empty audio",
        "no candidates",
        "no inline audio",
        "undecodable audio",
    )
    if any(marker in name or marker in message for marker in transient_markers):
        return True

    permanent_markers = (
        "invalid",
        "permission",
        "unauthorized",
        "forbidden",
        "notfound",
        "unsupported",
        "api key",
        "authentication",
    )
    if any(marker in name or marker in message for marker in permanent_markers):
        return False

    return False


def extract_audio_payload(response: object) -> tuple[bytes, str | None, int | None]:
    """Extract audio bytes and metadata from a Gemini response."""

    candidates = _read_value(response, "candidates")
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    first_candidate = candidates[0]
    content = _read_value(first_candidate, "content")
    parts = _read_value(content, "parts")
    if not parts:
        raise ValueError("Gemini returned no inline audio.")

    for part in parts:
        inline_data = _read_value(part, "inline_data", "inlineData")
        if inline_data is None:
            continue

        raw_data = _read_value(inline_data, "data")
        data = _coerce_bytes(raw_data)
        if not data:
            continue

        mime_type = _read_value(inline_data, "mime_type", "mimeType")
        sample_rate_hz = _read_int(
            _read_value(inline_data, "sample_rate_hertz", "sampleRateHertz")
        )
        if sample_rate_hz is None:
            sample_rate_hz = _parse_sample_rate_hz(str(mime_type) if mime_type else None)
        return data, str(mime_type) if mime_type else None, sample_rate_hz

    raise ValueError("Gemini returned empty audio payload.")


def ensure_wav_bytes(
    audio_bytes: bytes,
    *,
    mime_type: str | None,
    sample_rate_hz: int | None,
) -> bytes:
    """Return a WAV container, converting PCM16LE audio when necessary."""

    if not audio_bytes:
        raise ValueError("Gemini returned empty audio payload.")

    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return audio_bytes

    normalized_mime = (mime_type or "").lower()
    if "wav" in normalized_mime:
        return audio_bytes

    if any(token in normalized_mime for token in ("audio/l16", "audio/pcm", "audio/raw")):
        if len(audio_bytes) % 2 != 0:
            raise ValueError("Undecodable audio payload length.")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ)
            wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

    raise ValueError("Undecodable audio payload format.")


def _default_transport(text: str, model: str, voice: str, api_key: str) -> object:
    if not api_key:
        raise ConfigError("Gemini API key is required for live imports.")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise ConfigError(
            "The google-genai package is required for live Gemini TTS imports."
        ) from exc

    client = genai.Client(api_key=api_key)
    return client.models.generate_content(
        model=model,
        contents=text,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
        ),
    )


def _read_value(obj: object, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _coerce_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, base64.binascii.Error):
            return value.encode("utf-8")
    return b""


def _coerce_status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if value is None:
            continue
        number = _read_int(value)
        if number is not None:
            return number
    return None


def _read_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_sample_rate_hz(mime_type: str | None) -> int | None:
    if not mime_type:
        return None
    match = re.search(r"(?:rate|sample-rate)=(\d+)", mime_type)
    if not match:
        return None
    return int(match.group(1))
