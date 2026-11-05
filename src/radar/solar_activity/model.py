"""Calendar-bound mission solar-activity timeline.

OST 134-1044-2007 table G.1 supplies eleven *annual* Wolf-number values for
synthetic minimum, mean and maximum cycles.  It does not attach those cycles
to civil years and it does not prescribe a sub-annual interpolation.  RADAR
therefore keeps the calendar anchor explicit and uses a traceable 12-month
piecewise-constant expansion as its normative default.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Final

from radar.core.project import CalculationConfig, MissionConfig
from radar.core.types import SolarActivityLevel
from radar.solar_activity.cycle import SOLAR_CYCLE_YEARS, SolarCycleTable
from radar.solar_activity.ost import ost_wolf_number_cycle_table

MONTHS_PER_YEAR: Final[int] = 12
SOLAR_CYCLE_MONTHS: Final[int] = SOLAR_CYCLE_YEARS * MONTHS_PER_YEAR
SOLAR_ACTIVITY_MONTH_SECONDS: Final[float] = 365.25 * 24.0 * 60.0 * 60.0 / MONTHS_PER_YEAR
OST_G1_ANNUAL_STEP_MONTHLY_POLICY: Final[str] = "ost_g1_annual_step_12_months"
SOLAR_CYCLE_ANCHOR_BASIS: Final[str] = (
    "RADAR application scenario anchor; not specified by OST 134-1044-2007"
)
MONTHLY_EVALUATION_EPOCH: Final[str] = "calendar_month_midpoint"


def absolute_month_index(calendar_year: int, calendar_month: int) -> int:
    """Return a stable integer index for a civil calendar month."""

    if not isinstance(calendar_year, int):
        msg = "Calendar year must be an integer."
        raise ValueError(msg)
    if not isinstance(calendar_month, int) or not 1 <= calendar_month <= 12:
        msg = "Calendar month must be an integer in the range 1..12."
        raise ValueError(msg)
    return calendar_year * MONTHS_PER_YEAR + calendar_month - 1


def calendar_month_from_absolute_index(month_index: int) -> tuple[int, int]:
    """Return ``(year, month)`` for an absolute month index."""

    if not isinstance(month_index, int):
        msg = "Absolute month index must be an integer."
        raise ValueError(msg)
    calendar_year, zero_based_month = divmod(month_index, MONTHS_PER_YEAR)
    return calendar_year, zero_based_month + 1


@dataclass(frozen=True)
class SolarActivityMonth:
    """One calendar month on the repeating synthetic solar cycle."""

    calendar_year: int
    calendar_month: int
    absolute_month_index: int
    cycle_number: int
    cycle_year: int
    cycle_month: int
    wolf_number: float
    after_polarity_reversal: bool

    def __post_init__(self) -> None:
        expected_index = absolute_month_index(self.calendar_year, self.calendar_month)
        if self.absolute_month_index != expected_index:
            msg = "Solar-activity absolute month index does not match calendar month."
            raise ValueError(msg)
        if self.cycle_number < 1:
            msg = "Solar cycle number must be positive."
            raise ValueError(msg)
        if not 1 <= self.cycle_year <= SOLAR_CYCLE_YEARS:
            msg = "Solar cycle year must be in the range 1..11."
            raise ValueError(msg)
        if not 1 <= self.cycle_month <= SOLAR_CYCLE_MONTHS:
            msg = "Solar cycle month must be in the range 1..132."
            raise ValueError(msg)
        if not isfinite(self.wolf_number) or self.wolf_number < 0.0:
            msg = "Monthly Wolf number must be finite and non-negative."
            raise ValueError(msg)

    @property
    def years_from_cycle_start(self) -> float:
        """Return time from cycle start at the midpoint of this month."""

        return ((self.cycle_month - 1) + 0.5) / MONTHS_PER_YEAR


def _solar_activity_point(
    *,
    month_index: int,
    reference_start_year: int,
    reference_start_month: int,
    reference_cycle_number: int,
    cycle_table_wolf_numbers: tuple[float, ...],
) -> SolarActivityMonth:
    """Map an absolute month to one repeating synthetic-cycle point."""

    anchor_index = absolute_month_index(reference_start_year, reference_start_month)
    cycle_offset, zero_based_cycle_month = divmod(
        month_index - anchor_index,
        SOLAR_CYCLE_MONTHS,
    )
    cycle_number = reference_cycle_number + cycle_offset
    if cycle_number < 1:
        msg = "Requested month predates positive numbered solar cycles."
        raise ValueError(msg)

    cycle_year = zero_based_cycle_month // MONTHS_PER_YEAR + 1
    maximum = max(cycle_table_wolf_numbers)
    peak_cycle_year = cycle_table_wolf_numbers.index(maximum) + 1
    calendar_year, calendar_month = calendar_month_from_absolute_index(month_index)
    return SolarActivityMonth(
        calendar_year=calendar_year,
        calendar_month=calendar_month,
        absolute_month_index=month_index,
        cycle_number=cycle_number,
        cycle_year=cycle_year,
        cycle_month=zero_based_cycle_month + 1,
        wolf_number=cycle_table_wolf_numbers[cycle_year - 1],
        # OST does not tabulate a reversal date.  RADAR uses the first month
        # of the tabulated maximum year and exposes that convention.
        after_polarity_reversal=(cycle_year >= peak_cycle_year),
    )


@dataclass(frozen=True)
class MissionSolarActivity:
    """Single authoritative calendar timeline used by one calculation."""

    launch_year: int
    lifetime_years: int
    level: SolarActivityLevel
    reference_start_year: int
    cycle_years: tuple[int, ...]
    wolf_numbers: tuple[float, ...]
    sep_exceedance_probability: float
    source: str
    table_id: str
    reference_cycle_number: int = 1
    reference_start_month: int = 1
    wolf_number_version: str = "1.0"
    monthly_policy: str = OST_G1_ANNUAL_STEP_MONTHLY_POLICY
    anchor_basis: str = SOLAR_CYCLE_ANCHOR_BASIS
    cycle_table_wolf_numbers: tuple[float, ...] = ()
    months: tuple[SolarActivityMonth, ...] = ()

    def __post_init__(self) -> None:
        if len(self.cycle_years) != self.lifetime_years:
            msg = "Number of cycle years must match mission lifetime."
            raise ValueError(msg)
        if len(self.wolf_numbers) != self.lifetime_years:
            msg = "Number of Wolf numbers must match mission lifetime."
            raise ValueError(msg)
        if self.reference_cycle_number < 1:
            msg = "Solar-cycle reference number must be positive."
            raise ValueError(msg)
        if self.reference_start_month != 1:
            msg = "The annual-step solar timeline currently requires a January anchor."
            raise ValueError(msg)
        if self.wolf_number_version != "1.0":
            msg = "The OST table G.1 timeline currently supports Wolf-number version 1.0."
            raise ValueError(msg)
        if not self.anchor_basis:
            msg = "Solar-cycle anchor basis must not be empty."
            raise ValueError(msg)
        if self.monthly_policy != OST_G1_ANNUAL_STEP_MONTHLY_POLICY:
            msg = "Unsupported solar-activity monthly expansion policy."
            raise ValueError(msg)
        if len(self.cycle_table_wolf_numbers) != SOLAR_CYCLE_YEARS:
            msg = "Solar-activity timeline requires all 11 cycle-table values."
            raise ValueError(msg)
        expected_months = self.lifetime_years * MONTHS_PER_YEAR
        if len(self.months) != expected_months:
            msg = "Solar-activity timeline must contain 12 months per mission year."
            raise ValueError(msg)
        if tuple(point.wolf_number for point in self.months) != (
            expand_annual_wolf_numbers_to_monthly(self.wolf_numbers)
        ):
            msg = "Solar-activity monthly points do not match annual mission values."
            raise ValueError(msg)

    @property
    def mission_start_absolute_month(self) -> int:
        """Return absolute index of January in the launch year."""

        return absolute_month_index(self.launch_year, 1)

    @property
    def mission_month_count(self) -> int:
        """Return number of monthly source-model time bins."""

        return len(self.months)

    @property
    def month_duration_seconds(self) -> float:
        """Return the uniform duration used for monthly integration bins."""

        return SOLAR_ACTIVITY_MONTH_SECONDS

    @property
    def wolf_min(self) -> float:
        """Return the minimum value of the selected synthetic cycle."""

        return min(self.cycle_table_wolf_numbers)

    @property
    def wolf_max(self) -> float:
        """Return the maximum value of the selected synthetic cycle."""

        return max(self.cycle_table_wolf_numbers)

    @property
    def metadata(self) -> tuple[tuple[str, str], ...]:
        """Return traceability metadata for result logs and source models."""

        cycle_numbers = tuple(dict.fromkeys(point.cycle_number for point in self.months))
        return (
            ("solar_activity_source", self.source),
            ("solar_activity_table_id", self.table_id),
            ("solar_activity_level", self.level.value),
            ("solar_cycle_reference_start_year", str(self.reference_start_year)),
            ("solar_cycle_reference_start_month", str(self.reference_start_month)),
            ("solar_cycle_reference_number", str(self.reference_cycle_number)),
            ("solar_cycle_numbers", ",".join(str(value) for value in cycle_numbers)),
            ("solar_activity_wolf_version", self.wolf_number_version),
            ("solar_activity_monthly_policy", self.monthly_policy),
            ("solar_activity_anchor_basis", self.anchor_basis),
            ("solar_activity_monthly_evaluation", MONTHLY_EVALUATION_EPOCH),
            (
                "solar_activity_mission_start_policy",
                "january_1_from_year_only_launch_input",
            ),
            (
                "solar_activity_outside_mission_policy",
                "repeat_selected_11_year_synthetic_cycle_for_gcr_lags",
            ),
            ("solar_activity_month_count", str(self.mission_month_count)),
            ("solar_activity_month_seconds", f"{self.month_duration_seconds:.12g}"),
        )

    def validate_for_mission(self, mission: MissionConfig) -> None:
        """Require this timeline to represent exactly the supplied mission."""

        if self.launch_year != mission.launch_year:
            msg = "Solar timeline launch year must match mission launch year."
            raise ValueError(msg)
        if self.lifetime_years != mission.lifetime_years:
            msg = "Solar timeline lifetime must match mission lifetime."
            raise ValueError(msg)
        if self.level is not mission.solar_activity_level:
            msg = "Solar timeline activity level must match mission activity level."
            raise ValueError(msg)
        if self.sep_exceedance_probability != mission.sep_exceedance_probability:
            msg = "Solar timeline SEP probability must match mission probability."
            raise ValueError(msg)

    def point_at_absolute_month(self, month_index: int) -> SolarActivityMonth:
        """Return any month, including pre-mission months required by GCR lags."""

        if not isinstance(month_index, int):
            msg = "Absolute month index must be an integer."
            raise ValueError(msg)

        return _solar_activity_point(
            month_index=month_index,
            reference_start_year=self.reference_start_year,
            reference_start_month=self.reference_start_month,
            reference_cycle_number=self.reference_cycle_number,
            cycle_table_wolf_numbers=self.cycle_table_wolf_numbers,
        )

    def point_at_calendar_month(
        self,
        calendar_year: int,
        calendar_month: int,
    ) -> SolarActivityMonth:
        """Return a solar-activity point for a civil calendar month."""

        return self.point_at_absolute_month(
            absolute_month_index(calendar_year, calendar_month),
        )

    def point_at_mission_month(self, mission_month_index: int) -> SolarActivityMonth:
        """Return a zero-based mission month."""

        if not isinstance(mission_month_index, int):
            msg = "Mission month index must be an integer."
            raise ValueError(msg)
        if not 0 <= mission_month_index < self.mission_month_count:
            msg = "Mission month index is outside the mission timeline."
            raise ValueError(msg)
        return self.months[mission_month_index]

    def point_at_mission_month_coordinate(
        self,
        mission_month_coordinate: float,
    ) -> SolarActivityMonth:
        """Sample the piecewise-constant series at a fractional month coordinate.

        Coordinate 0 is the start of January in the launch year.  GCR source
        bins are evaluated at ``m + 0.5`` and a delay is subtracted before this
        method is called.  Flooring is therefore the exact evaluation rule for
        the annual step function, not an interpolation of table G.1.
        """

        if not isfinite(mission_month_coordinate):
            msg = "Mission month coordinate must be finite."
            raise ValueError(msg)
        month_index = self.mission_start_absolute_month + floor(
            mission_month_coordinate,
        )
        return self.point_at_absolute_month(month_index)

    def wolf_number_at_mission_month_coordinate(
        self,
        mission_month_coordinate: float,
    ) -> float:
        """Return Wolf number at a possibly pre-mission fractional coordinate."""

        return self.point_at_mission_month_coordinate(
            mission_month_coordinate,
        ).wolf_number


def expand_annual_wolf_numbers_to_monthly(
    annual_wolf_numbers: tuple[float, ...],
) -> tuple[float, ...]:
    """Repeat each annual OST table G.1 value for twelve calendar months."""

    if not annual_wolf_numbers:
        msg = "Annual Wolf number sequence must not be empty."
        raise ValueError(msg)

    monthly_values: list[float] = []
    for value in annual_wolf_numbers:
        if not isfinite(value) or value < 0.0:
            msg = "Annual Wolf numbers must be finite and non-negative."
            raise ValueError(msg)
        monthly_values.extend([value] * MONTHS_PER_YEAR)
    return tuple(monthly_values)


def build_mission_solar_activity(
    mission: MissionConfig,
    cycle_table: SolarCycleTable,
    reference_start_year: int,
    *,
    reference_cycle_number: int = 1,
    reference_start_month: int = 1,
    wolf_number_version: str = "1.0",
    monthly_policy: str = OST_G1_ANNUAL_STEP_MONTHLY_POLICY,
    anchor_basis: str = SOLAR_CYCLE_ANCHOR_BASIS,
) -> MissionSolarActivity:
    """Build one absolute calendar timeline from mission and cycle settings."""

    if cycle_table.level is not mission.solar_activity_level:
        msg = "Solar cycle table level must match mission solar activity level."
        raise ValueError(msg)
    if monthly_policy != OST_G1_ANNUAL_STEP_MONTHLY_POLICY:
        msg = "Unsupported solar-activity monthly expansion policy."
        raise ValueError(msg)
    if reference_start_month != 1:
        msg = "The annual-step solar timeline currently requires a January anchor."
        raise ValueError(msg)

    start_index = absolute_month_index(mission.launch_year, 1)
    months = tuple(
        _solar_activity_point(
            month_index=start_index + mission_month,
            reference_start_year=reference_start_year,
            reference_start_month=reference_start_month,
            reference_cycle_number=reference_cycle_number,
            cycle_table_wolf_numbers=cycle_table.wolf_numbers,
        )
        for mission_month in range(mission.lifetime_years * MONTHS_PER_YEAR)
    )
    cycle_years = tuple(
        months[year * MONTHS_PER_YEAR].cycle_year for year in range(mission.lifetime_years)
    )
    wolf_numbers = tuple(
        months[year * MONTHS_PER_YEAR].wolf_number for year in range(mission.lifetime_years)
    )

    return MissionSolarActivity(
        launch_year=mission.launch_year,
        lifetime_years=mission.lifetime_years,
        level=mission.solar_activity_level,
        reference_start_year=reference_start_year,
        cycle_years=cycle_years,
        wolf_numbers=wolf_numbers,
        sep_exceedance_probability=mission.sep_exceedance_probability,
        source=cycle_table.source,
        table_id=cycle_table.table_id,
        reference_cycle_number=reference_cycle_number,
        reference_start_month=reference_start_month,
        wolf_number_version=wolf_number_version,
        monthly_policy=monthly_policy,
        anchor_basis=anchor_basis,
        cycle_table_wolf_numbers=cycle_table.wolf_numbers,
        months=months,
    )


def build_calculation_solar_activity(config: CalculationConfig) -> MissionSolarActivity:
    """Build the shared timeline selected by a complete calculation config."""

    methodology = config.methodology
    return build_mission_solar_activity(
        mission=config.mission,
        cycle_table=ost_wolf_number_cycle_table(config.mission.solar_activity_level),
        reference_start_year=methodology.solar_cycle_reference_start_year,
        reference_cycle_number=methodology.solar_cycle_reference_number,
        wolf_number_version=methodology.solar_activity_wolf_version,
        monthly_policy=methodology.solar_activity_monthly_policy,
    )


def mission_solar_activity_monthly_wolf_numbers(
    solar_activity: MissionSolarActivity,
) -> tuple[float, ...]:
    """Return the authoritative calendar-month Wolf numbers for a mission."""

    return tuple(point.wolf_number for point in solar_activity.months)


__all__ = [
    "MONTHLY_EVALUATION_EPOCH",
    "MONTHS_PER_YEAR",
    "OST_G1_ANNUAL_STEP_MONTHLY_POLICY",
    "SOLAR_ACTIVITY_MONTH_SECONDS",
    "SOLAR_CYCLE_ANCHOR_BASIS",
    "SOLAR_CYCLE_MONTHS",
    "MissionSolarActivity",
    "SolarActivityMonth",
    "absolute_month_index",
    "build_calculation_solar_activity",
    "build_mission_solar_activity",
    "calendar_month_from_absolute_index",
    "expand_annual_wolf_numbers_to_monthly",
    "mission_solar_activity_monthly_wolf_numbers",
]
