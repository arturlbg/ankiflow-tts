from __future__ import annotations

import io
import wave

import pytest

from ankiflow_tts.exceptions import TtsGenerationError
from ankiflow_tts.tts_deepgram import (
    DeepgramTtsClient,
    ensure_wav_bytes,
    is_retryable_deepgram_error,
)
from ankiflow_tts.types import RateLimitPolicy, RetryPolicy


def test_ensure_wav_bytes_accepts_valid_wav() -> None:
    wav_bytes = _build_wav_bytes()

    verified = ensure_wav_bytes(wav_bytes, mime_type="audio/wav")

    assert verified == wav_bytes


def test_deepgram_client_retries_transient_failures_then_succeeds() -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def transport(_: str, __: str, ___: str) -> tuple[bytes, str | None]:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return _build_wav_bytes(), "audio/wav"

    client = DeepgramTtsClient(
        api_key="key",
        model="aura-2-thalia-en",
        retry_policy=RetryPolicy(max_attempts=5),
        rate_limit_policy=RateLimitPolicy(minimum_interval_s=0.0),
        transport=transport,
        sleep_fn=slept.append,
        random_fn=lambda: 0.5,
        time_fn=lambda: float(len(attempts)),
    )

    audio = client.synthesize("Hello", "hello.wav")

    assert len(attempts) == 3
    assert slept == [0.5, 1.0]
    assert audio.filename == "hello.wav"
    assert audio.content.startswith(b"RIFF")


def test_deepgram_client_raises_typed_error_after_exhaustion() -> None:
    def transport(_: str, __: str, ___: str) -> tuple[bytes, str | None]:
        raise ConnectionError("down")

    client = DeepgramTtsClient(
        api_key="key",
        model="aura-2-thalia-en",
        retry_policy=RetryPolicy(max_attempts=2),
        rate_limit_policy=RateLimitPolicy(minimum_interval_s=0.0),
        transport=transport,
        sleep_fn=lambda _: None,
        random_fn=lambda: 0.0,
        time_fn=lambda: 0.0,
    )

    with pytest.raises(TtsGenerationError) as exc_info:
        client.synthesize("Hello", "hello.wav")

    assert exc_info.value.attempts == 2
    assert is_retryable_deepgram_error(exc_info.value.__cause__) is True


def test_ensure_wav_bytes_rejects_zero_frame_wav() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

    with pytest.raises(ValueError, match="no audio frames"):
        ensure_wav_bytes(
            buffer.getvalue(),
            mime_type="audio/wav",
        )


def test_deepgram_client_enforces_one_second_spacing() -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(delay_s: float) -> None:
        slept.append(delay_s)
        now[0] += delay_s

    client = DeepgramTtsClient(
        api_key="key",
        model="aura-2-thalia-en",
        retry_policy=RetryPolicy(max_attempts=1),
        rate_limit_policy=RateLimitPolicy(minimum_interval_s=1.0),
        transport=lambda *_: (_build_wav_bytes(), "audio/wav"),
        sleep_fn=sleep,
        random_fn=lambda: 0.0,
        time_fn=lambda: now[0],
    )

    client.synthesize("Hello", "one.wav")
    client.synthesize("Again", "two.wav")

    assert slept == [1.0]


def _build_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00\x01\x00")
    return buffer.getvalue()
