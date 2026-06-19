"""Display/export unit conversions.

These helpers do not define source model physics. They only convert already
calculated accumulated quantities or rates to engineering display units.
"""

from __future__ import annotations

SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.0
SECONDS_PER_YEAR = SECONDS_PER_DAY * DAYS_PER_YEAR


def days_for_mission_years(lifetime_years: int) -> float:
    """Return mission duration in days for integer-year SAS input."""

    if not isinstance(lifetime_years, int):
        msg = "Mission lifetime must be an integer number of years."
        raise ValueError(msg)

    if lifetime_years < 1:
        msg = "Mission lifetime must be at least one year."
        raise ValueError(msg)

    return float(lifetime_years) * DAYS_PER_YEAR


def seconds_for_mission_years(lifetime_years: int) -> float:
    """Return mission duration in seconds for integer-year SAS input."""

    return days_for_mission_years(lifetime_years) * SECONDS_PER_DAY


def _validate_positive_duration(duration: float, name: str) -> None:
    if duration <= 0.0:
        msg = f"{name} must be positive."
        raise ValueError(msg)


def rad_per_second_to_rad_per_day(rate_rad_per_second: float) -> float:
    """Convert dose rate from rad/s to rad/day."""

    return rate_rad_per_second * SECONDS_PER_DAY


def rad_per_day_to_rad_per_second(rate_rad_per_day: float) -> float:
    """Convert dose rate from rad/day to rad/s."""

    return rate_rad_per_day / SECONDS_PER_DAY


def accumulated_rad_to_average_rad_per_day(
    accumulated_rad: float,
    duration_days: float,
) -> float:
    """Convert accumulated dose to average dose per day."""

    _validate_positive_duration(duration_days, "Duration in days")

    return accumulated_rad / duration_days


def accumulated_rad_to_average_rad_per_second(
    accumulated_rad: float,
    duration_seconds: float,
) -> float:
    """Convert accumulated dose to average dose rate in rad/s."""

    _validate_positive_duration(duration_seconds, "Duration in seconds")

    return accumulated_rad / duration_seconds


def accumulated_values_to_average_daily_values(
    values: tuple[float, ...],
    duration_days: float,
) -> tuple[float, ...]:
    """Convert accumulated fluence-like values to average daily values."""

    _validate_positive_duration(duration_days, "Duration in days")

    return tuple(value / duration_days for value in values)
