from __future__ import annotations

import pytest

from ankiflow_tts.retry import compute_delay, run_with_retry
from ankiflow_tts.types import RetryPolicy


def test_compute_delay_uses_full_jitter() -> None:
    policy = RetryPolicy(initial_delay_s=2.0, max_delay_s=16.0, multiplier=2.0)

    assert compute_delay(policy, attempt=1, random_value=0.5) == 1.0
    assert compute_delay(policy, attempt=3, random_value=0.25) == 2.0


def test_run_with_retry_retries_until_success() -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return "ok"

    result = run_with_retry(
        operation,
        policy=RetryPolicy(max_attempts=5, initial_delay_s=1.0),
        should_retry=lambda exc: isinstance(exc, ConnectionError),
        sleep_fn=slept.append,
        random_fn=lambda: 0.5,
    )

    assert result == "ok"
    assert len(attempts) == 3
    assert slept == [0.5, 1.0]


def test_run_with_retry_stops_on_non_retryable_error() -> None:
    def operation() -> None:
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        run_with_retry(
            operation,
            policy=RetryPolicy(max_attempts=5),
            should_retry=lambda exc: False,
            sleep_fn=lambda _: None,
            random_fn=lambda: 0.0,
        )
