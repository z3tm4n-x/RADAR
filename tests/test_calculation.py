from radar.calculation import execute_calculation
from radar.core.log import LogLevel
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
from radar.core.types import SolarActivityLevel
from radar.core.result import ComponentStatus
from radar.core.types import DoseQuantity
from radar.core.units import Unit
from radar.project_file import ProjectFile, project_file_from_json


def _config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        shielding=ShieldingConfig(thicknesses_g_cm2=(1.0, 2.0, 5.0)),
        kp=4,
        dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
        dose_unit=Unit.RAD,
    )


def test_execute_calculation_returns_result_for_config() -> None:
    config = _config()

    result = execute_calculation(config)

    assert result.config == config
    assert not result.has_errors()


def test_execute_calculation_sets_source_component_statuses() -> None:
    result = execute_calculation(_config())

    assert result.component_status("СКЛ") is ComponentStatus.SKIPPED
    assert result.component_status("ГКЛ") is ComponentStatus.SKIPPED
    assert result.component_status("ЕРПЗ") is ComponentStatus.SKIPPED


def test_execute_calculation_records_model_information() -> None:
    result = execute_calculation(_config())

    assert len(result.model_info) == 3
    assert {model.name for model in result.model_info} == {
        "ost_sep_model",
        "ost_gcr_model",
        "ost_erb_model",
    }
    assert {model.version for model in result.model_info} == {"not_implemented"}
    assert all(
        model.status == "численная часть не реализована"
        for model in result.model_info
    )


def test_execute_calculation_writes_log_entries() -> None:
    result = execute_calculation(_config())

    assert result.log.entries[0].level is LogLevel.INFO
    assert result.log.entries[0].stage == "запуск расчёта"
    assert result.log.warnings()
    assert all(entry.level is not LogLevel.ERROR for entry in result.log.entries)


def test_execute_calculation_builds_placeholder_output_tables() -> None:
    result = execute_calculation(_config())

    assert {table.table_id for table in result.output_tables} == {
        "dose_by_thickness",
        "source_contributions",
        "single_event_effects",
    }

    dose_table = next(
        table
        for table in result.output_tables
        if table.table_id == "dose_by_thickness"
    )
    assert dose_table.rows[0].cells == (1.0, 0.0)
    assert dose_table.rows[1].cells == (2.0, 0.0)
    assert dose_table.rows[2].cells == (5.0, 0.0)
    assert ("status", "placeholder") in dose_table.metadata

    see_table = next(
        table
        for table in result.output_tables
        if table.table_id == "single_event_effects"
    )
    see_column_keys = tuple(column.key for column in see_table.columns)
    assert [row.cells[0] for row in see_table.rows] == [1.0, 2.0, 5.0]
    assert see_table.kind.value == "single_event"
    assert ("status", "placeholder") in see_table.metadata
    assert all(
        row.cells[see_column_keys.index("total_event_rate_per_day")] == 0.0
        for row in see_table.rows
    )
    assert all(
        row.cells[see_column_keys.index("total_expected_events")] == 0.0
        for row in see_table.rows
    )


def test_execute_calculation_result_can_be_saved_in_project_file() -> None:
    config = _config()
    result = execute_calculation(config)

    project_file = ProjectFile.create(
        calculation_config=config,
        calculation_result=result,
    )
    restored = project_file_from_json(project_file.to_json())

    assert restored.calculation_result == result

def test_execute_calculation_records_ost_solar_activity_input_data() -> None:
    base_config = _config()
    config = CalculationConfig(
        mission=MissionConfig(
            launch_year=2028,
            lifetime_years=3,
            solar_activity_level=SolarActivityLevel.MAXIMUM,
        ),
        orbit=base_config.orbit,
        shielding=base_config.shielding,
        kp=base_config.kp,
        dose_quantity=base_config.dose_quantity,
        dose_unit=base_config.dose_unit,
        methodology=base_config.methodology,
    )

    result = execute_calculation(config)

    solar_data = next(info for info in result.input_data_info if info.name == "СА")
    assert solar_data.source == "ОСТ 134-1044-2007"
    assert solar_data.table_id == "ost_134_1044_2007_table_g_1_wolf_numbers"
    assert ("level", "maximum") in solar_data.values
    assert ("cycle_years", "1, 2, 3") in solar_data.values
    assert ("wolf_numbers", "11.5, 33.9, 100.8") in solar_data.values


def test_execute_calculation_logs_ost_wolf_numbers_for_mission() -> None:
    base_config = _config()
    config = CalculationConfig(
        mission=MissionConfig(
            launch_year=2028,
            lifetime_years=3,
            solar_activity_level=SolarActivityLevel.MAXIMUM,
        ),
        orbit=base_config.orbit,
        shielding=base_config.shielding,
        kp=base_config.kp,
        dose_quantity=base_config.dose_quantity,
        dose_unit=base_config.dose_unit,
        methodology=base_config.methodology,
    )

    result = execute_calculation(config)

    messages = tuple(entry.message for entry in result.log.entries)
    assert any(
        message == "Числа Вольфа по таблице Г.1 ОСТ: 11.5, 33.9, 100.8"
        for message in messages
    )

