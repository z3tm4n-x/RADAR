import pytest

from radar.core.project import MissionConfig
from radar.core.types import SolarActivityLevel
from radar.solar_activity.cycle import (
    SolarCycleTable,
    calendar_year_to_cycle_year,
    mission_cycle_years,
)
from radar.solar_activity.model import build_mission_solar_activity


def _test_mean_cycle_table() -> SolarCycleTable:
    return SolarCycleTable(
        level=SolarActivityLevel.MEAN,
        wolf_numbers=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
        source="test",
        table_id="test_mean_cycle",
    )


def test_calendar_year_to_cycle_year_is_one_based() -> None:
    assert calendar_year_to_cycle_year(calendar_year=2024, reference_start_year=2024) == 1
    assert calendar_year_to_cycle_year(calendar_year=2025, reference_start_year=2024) == 2
    assert calendar_year_to_cycle_year(calendar_year=2034, reference_start_year=2024) == 11
    assert calendar_year_to_cycle_year(calendar_year=2035, reference_start_year=2024) == 1


def test_mission_cycle_years_wrap_after_11_years() -> None:
    assert mission_cycle_years(
        launch_year=2033,
        lifetime_years=5,
        reference_start_year=2024,
    ) == (10, 11, 1, 2, 3)


def test_solar_cycle_table_requires_11_values() -> None:
    with pytest.raises(ValueError, match="11"):
        SolarCycleTable(
            level=SolarActivityLevel.MEAN,
            wolf_numbers=(1.0, 2.0),
            source="test",
            table_id="bad",
        )


def test_solar_cycle_table_rejects_negative_wolf_numbers() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SolarCycleTable(
            level=SolarActivityLevel.MEAN,
            wolf_numbers=(0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 6.0, 7.0, 8.0, 9.0, 10.0),
            source="test",
            table_id="bad",
        )


def test_solar_cycle_table_returns_wolf_number() -> None:
    table = _test_mean_cycle_table()

    assert table.wolf_number_for_cycle_year(1) == 0.0
    assert table.wolf_number_for_cycle_year(4) == 30.0
    assert table.wolf_number_for_cycle_year(11) == 100.0


def test_solar_cycle_table_rejects_bad_cycle_year() -> None:
    table = _test_mean_cycle_table()

    with pytest.raises(ValueError, match="1..11"):
        table.wolf_number_for_cycle_year(0)

    with pytest.raises(ValueError, match="1..11"):
        table.wolf_number_for_cycle_year(12)


def test_build_mission_solar_activity() -> None:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=4,
        solar_activity_level=SolarActivityLevel.MEAN,
    )
    table = _test_mean_cycle_table()

    activity = build_mission_solar_activity(
        mission=mission,
        cycle_table=table,
        reference_start_year=2024,
    )

    assert activity.launch_year == 2027
    assert activity.lifetime_years == 4
    assert activity.level is SolarActivityLevel.MEAN
    assert activity.reference_start_year == 2024
    assert activity.cycle_years == (4, 5, 6, 7)
    assert activity.wolf_numbers == (30.0, 40.0, 50.0, 60.0)
    assert activity.sep_exceedance_probability == 0.1
    assert activity.source == "test"
    assert activity.table_id == "test_mean_cycle"


def test_build_mission_solar_activity_rejects_level_mismatch() -> None:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=4,
        solar_activity_level=SolarActivityLevel.MAXIMUM,
    )
    table = _test_mean_cycle_table()

    with pytest.raises(ValueError, match="level"):
        build_mission_solar_activity(
            mission=mission,
            cycle_table=table,
            reference_start_year=2024,
        )
