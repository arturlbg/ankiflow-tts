"""Reusable retry helpers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .types import RetryPolicy

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class RetryContext:
    """Metadata about a retryable failure."""

    attempt: int
    max_attempts: int
    delay_s: float
    exception: BaseException


def compute_delay(
    policy: RetryPolicy,
    *,
    attempt: int,
    random_value: float,
) -> float:
    """Calculate the delay before the next attempt."""

    capped = min(
        policy.max_delay_s,
        policy.initial_delay_s * (policy.multiplier ** (attempt - 1)),
    )
    if policy.jitter == "full":
        return capped * random_value
    return capped


def run_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[BaseException], bool],
    on_retry: Callable[[RetryContext], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> T:
    """Run an operation until it succeeds or the retry policy is exhausted."""

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= policy.max_attempts or not should_retry(exc):
                raise

            delay_s = compute_delay(
                policy,
                attempt=attempt,
                random_value=random_fn(),
            )
            if on_retry is not None:
                on_retry(
                    RetryContext(
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay_s=delay_s,
                        exception=exc,
                    )
                )
            sleep_fn(delay_s)

    raise RuntimeError("Retry loop exited unexpectedly.")
