"""Solar cycle helpers for RADAR."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.types import SolarActivityLevel

SOLAR_CYCLE_YEARS = 11


def calendar_year_to_cycle_year(calendar_year: int, reference_start_year: int) -> int:
    """Convert calendar year to 1-based year of a repeating 11-year solar cycle."""

    if not isinstance(calendar_year, int):
        msg = "Calendar year must be an integer."
        raise ValueError(msg)

    if not isinstance(reference_start_year, int):
        msg = "Reference start year must be an integer."
        raise ValueError(msg)

    return ((calendar_year - reference_start_year) % SOLAR_CYCLE_YEARS) + 1


def mission_cycle_years(
    launch_year: int,
    lifetime_years: int,
    reference_start_year: int,
) -> tuple[int, ...]:
    """Return cycle years for each integer mission year."""

    if not isinstance(lifetime_years, int):
        msg = "Mission lifetime must be an integer number of years."
        raise ValueError(msg)

    if lifetime_years < 1:
        msg = "Mission lifetime must be at least one year."
        raise ValueError(msg)

    return tuple(
        calendar_year_to_cycle_year(
            calendar_year=launch_year + year_index,
            reference_start_year=reference_start_year,
        )
        for year_index in range(lifetime_years)
    )


@dataclass(frozen=True)
class SolarCycleTable:
    """Table of Wolf numbers for one 11-year solar activity level."""

    level: SolarActivityLevel
    wolf_numbers: tuple[float, ...]
    source: str
    table_id: str

    def __post_init__(self) -> None:
        if len(self.wolf_numbers) != SOLAR_CYCLE_YEARS:
            msg = "Solar cycle table must contain exactly 11 Wolf numbers."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.wolf_numbers):
            msg = "Wolf numbers must be non-negative."
            raise ValueError(msg)

        if not self.source:
            msg = "Solar cycle table source must not be empty."
            raise ValueError(msg)

        if not self.table_id:
            msg = "Solar cycle table_id must not be empty."
            raise ValueError(msg)

    def wolf_number_for_cycle_year(self, cycle_year: int) -> float:
        """Return Wolf number for 1-based solar cycle year."""

        if not 1 <= cycle_year <= SOLAR_CYCLE_YEARS:
            msg = "Solar cycle year must be in the range 1..11."
            raise ValueError(msg)

        return self.wolf_numbers[cycle_year - 1]

    def wolf_numbers_for_cycle_years(self, cycle_years: tuple[int, ...]) -> tuple[float, ...]:
        """Return Wolf numbers for a sequence of 1-based solar cycle years."""

        return tuple(self.wolf_number_for_cycle_year(year) for year in cycle_years)
