"""Domain-specific exceptions for AnkiFlow TTS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class AnkiFlowError(Exception):
    """Base exception for application failures."""


class ConfigError(AnkiFlowError):
    """Raised when CLI arguments or environment settings are invalid."""


class ValidationError(AnkiFlowError):
    """Raised when preflight validation fails."""


class AnkiConnectError(AnkiFlowError):
    """Raised when AnkiConnect is unavailable or returns an error."""


class GeminiTtsError(AnkiFlowError):
    """Raised when Gemini TTS generation fails."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable


@dataclass(slots=True, frozen=True)
class ParseIssue:
    """Represents a line-level parsing problem."""

    line_number: int
    message: str


class ParseError(AnkiFlowError):
    """Raised when the input file contains malformed rows."""

    def __init__(self, issues: Sequence[ParseIssue], total_lines: int) -> None:
        if not issues:
            raise ValueError("ParseError requires at least one issue.")
        self.issues = tuple(issues)
        self.total_lines = total_lines
        summary = ", ".join(
            f"line {issue.line_number}: {issue.message}" for issue in self.issues
        )
        super().__init__(f"Input validation failed for {len(self.issues)} line(s): {summary}")
