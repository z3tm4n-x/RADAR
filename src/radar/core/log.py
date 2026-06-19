"""Calculation log objects for RADAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class LogLevel(StrEnum):
    """Calculation log severity level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class CalculationLogEntry:
    """One immutable calculation log entry."""

    level: LogLevel
    stage: str
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.stage:
            msg = "Log entry stage must not be empty."
            raise ValueError(msg)

        if not self.message:
            msg = "Log entry message must not be empty."
            raise ValueError(msg)

        for key, value in self.details:
            if not key:
                msg = "Log detail key must not be empty."
                raise ValueError(msg)

            if not value:
                msg = "Log detail value must not be empty."
                raise ValueError(msg)


@dataclass(frozen=True)
class CalculationLog:
    """Immutable calculation log."""

    entries: tuple[CalculationLogEntry, ...] = ()

    def add(
        self,
        level: LogLevel,
        stage: str,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> CalculationLog:
        """Return a new log with one appended entry."""

        detail_items: tuple[tuple[str, str], ...] = ()

        if details is not None:
            detail_items = tuple(sorted(details.items()))

        entry = CalculationLogEntry(
            level=level,
            stage=stage,
            message=message,
            details=detail_items,
        )

        return CalculationLog(entries=(*self.entries, entry))

    def info(
        self,
        stage: str,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> CalculationLog:
        """Return a new log with one info entry."""

        return self.add(
            level=LogLevel.INFO,
            stage=stage,
            message=message,
            details=details,
        )

    def warning(
        self,
        stage: str,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> CalculationLog:
        """Return a new log with one warning entry."""

        return self.add(
            level=LogLevel.WARNING,
            stage=stage,
            message=message,
            details=details,
        )

    def error(
        self,
        stage: str,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> CalculationLog:
        """Return a new log with one error entry."""

        return self.add(
            level=LogLevel.ERROR,
            stage=stage,
            message=message,
            details=details,
        )

    def warnings(self) -> tuple[CalculationLogEntry, ...]:
        """Return warning entries."""

        return tuple(entry for entry in self.entries if entry.level is LogLevel.WARNING)

    def errors(self) -> tuple[CalculationLogEntry, ...]:
        """Return error entries."""

        return tuple(entry for entry in self.entries if entry.level is LogLevel.ERROR)

    def has_errors(self) -> bool:
        """Return True if the log contains at least one error."""

        return bool(self.errors())
