from __future__ import annotations

import io
import wave

import pytest

from ankiflow_tts.exceptions import GeminiTtsError
from ankiflow_tts.tts_gemini import (
    GeminiTtsClient,
    ensure_wav_bytes,
    extract_audio_payload,
    is_retryable_gemini_error,
)
from ankiflow_tts.types import RetryPolicy


def test_ensure_wav_bytes_wraps_pcm16_audio() -> None:
    pcm_bytes = b"\x00\x00\x01\x00\x02\x00\x03\x00"

    wav_bytes = ensure_wav_bytes(
        pcm_bytes,
        mime_type="audio/L16;rate=24000",
        sample_rate_hz=None,
    )

    assert wav_bytes.startswith(b"RIFF")
    assert wav_bytes[8:12] == b"WAVE"


def test_extract_audio_payload_reads_inline_data() -> None:
    wav_bytes = _build_wav_bytes()
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "data": wav_bytes,
                                "mime_type": "audio/wav",
                            }
                        }
                    ]
                }
            }
        ]
    }

    data, mime_type, sample_rate = extract_audio_payload(response)

    assert data == wav_bytes
    assert mime_type == "audio/wav"
    assert sample_rate is None


def test_gemini_client_retries_transient_failures_then_succeeds() -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def transport(_: str, __: str, ___: str, ____: str) -> object:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "data": b"\x00\x00\x01\x00",
                                    "mime_type": "audio/L16;rate=24000",
                                }
                            }
                        ]
                    }
                }
            ]
        }

    client = GeminiTtsClient(
        api_key="key",
        model="model",
        voice="voice",
        retry_policy=RetryPolicy(max_attempts=5),
        transport=transport,
        sleep_fn=slept.append,
        random_fn=lambda: 0.5,
    )

    audio = client.synthesize("Hello", "hello.wav")

    assert len(attempts) == 3
    assert slept == [0.5, 1.0]
    assert audio.filename == "hello.wav"
    assert audio.content.startswith(b"RIFF")


def test_gemini_client_raises_typed_error_after_exhaustion() -> None:
    def transport(_: str, __: str, ___: str, ____: str) -> object:
        raise ConnectionError("down")

    client = GeminiTtsClient(
        api_key="key",
        model="model",
        voice="voice",
        retry_policy=RetryPolicy(max_attempts=2),
        transport=transport,
        sleep_fn=lambda _: None,
        random_fn=lambda: 0.0,
    )

    with pytest.raises(GeminiTtsError) as exc_info:
        client.synthesize("Hello", "hello.wav")

    assert exc_info.value.attempts == 2
    assert is_retryable_gemini_error(exc_info.value.__cause__) is True


def _build_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00\x01\x00")
    return buffer.getvalue()
