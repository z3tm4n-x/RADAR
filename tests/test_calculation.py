from radar.calculation import execute_calculation
from radar.core.log import LogLevel
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
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
    assert all(model.status == "численная часть не реализована" for model in result.model_info)


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


def test_execute_calculation_result_can_be_saved_in_project_file() -> None:
    config = _config()
    result = execute_calculation(config)

    project_file = ProjectFile.create(
        calculation_config=config,
        calculation_result=result,
    )
    restored = project_file_from_json(project_file.to_json())

    assert restored.calculation_result == result
