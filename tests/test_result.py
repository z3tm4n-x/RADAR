from radar.core.log import LogLevel
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo


def _config() -> CalculationConfig:
    mission = MissionConfig(launch_year=2027, lifetime_years=15)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    return CalculationConfig(mission=mission, orbit=orbit)


def test_result_starts_without_errors() -> None:
    result = CalculationResult(config=_config())

    assert result.log.entries == ()
    assert result.component_status("solar_activity") is ComponentStatus.NOT_STARTED
    assert result.has_errors() is False


def test_result_adds_log_immutably() -> None:
    original_result = CalculationResult(config=_config())

    updated_result = original_result.add_log_entry(
        level=LogLevel.INFO,
        stage="input",
        message="Input checked.",
    )

    assert original_result.log.entries == ()
    assert len(updated_result.log.entries) == 1
    assert updated_result.log.entries[0].stage == "input"


def test_result_updates_component_status() -> None:
    result = CalculationResult(config=_config())

    result = result.set_component_status("solar_activity", ComponentStatus.COMPLETED)
    assert result.component_status("solar_activity") is ComponentStatus.COMPLETED

    result = result.set_component_status("solar_activity", ComponentStatus.FAILED)
    assert result.component_status("solar_activity") is ComponentStatus.FAILED
    assert result.has_errors() is True


def test_result_sets_model_info() -> None:
    result = CalculationResult(config=_config())

    result = result.set_model_info(
        ModelInfo(
            name="solar_activity",
            version="0.1.0",
            status="implemented",
            source="RADAR",
        )
    )

    assert len(result.model_info) == 1
    assert result.model_info[0].name == "solar_activity"

    result = result.set_model_info(
        ModelInfo(
            name="solar_activity",
            version="0.1.1",
            status="checked",
            source="RADAR",
        )
    )

    assert len(result.model_info) == 1
    assert result.model_info[0].version == "0.1.1"


def test_result_reports_log_errors() -> None:
    result = CalculationResult(config=_config())

    result = result.add_log_entry(
        level=LogLevel.ERROR,
        stage="export",
        message="Export failed.",
    )

    assert result.has_errors() is True
