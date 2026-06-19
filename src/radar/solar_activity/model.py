"""Mission solar activity model."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.project import MissionConfig
from radar.core.types import SolarActivityLevel
from radar.solar_activity.cycle import SolarCycleTable, mission_cycle_years


@dataclass(frozen=True)
class MissionSolarActivity:
    """Solar activity sequence used by one calculation."""

    launch_year: int
    lifetime_years: int
    level: SolarActivityLevel
    reference_start_year: int
    cycle_years: tuple[int, ...]
    wolf_numbers: tuple[float, ...]
    sep_exceedance_probability: float
    source: str
    table_id: str

    def __post_init__(self) -> None:
        if len(self.cycle_years) != self.lifetime_years:
            msg = "Number of cycle years must match mission lifetime."
            raise ValueError(msg)

        if len(self.wolf_numbers) != self.lifetime_years:
            msg = "Number of Wolf numbers must match mission lifetime."
            raise ValueError(msg)


def build_mission_solar_activity(
    mission: MissionConfig,
    cycle_table: SolarCycleTable,
    reference_start_year: int,
) -> MissionSolarActivity:
    """Build mission solar activity sequence from mission config and cycle table."""

    if cycle_table.level is not mission.solar_activity_level:
        msg = "Solar cycle table level must match mission solar activity level."
        raise ValueError(msg)

    cycle_years = mission_cycle_years(
        launch_year=mission.launch_year,
        lifetime_years=mission.lifetime_years,
        reference_start_year=reference_start_year,
    )

    wolf_numbers = cycle_table.wolf_numbers_for_cycle_years(cycle_years)

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
    )
