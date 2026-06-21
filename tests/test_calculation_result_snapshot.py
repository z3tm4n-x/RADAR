import pytest

from radar.calculation_result_snapshot import (
    calculation_result_from_snapshot,
    calculation_result_snapshot,
)
from radar.core.log import LogLevel
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo


def _calculation_config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        kp=4,
    )


def _calculation_result() -> CalculationResult:
    result = CalculationResult(config=_calculation_config())
    result = result.set_component_status("СКЛ", ComponentStatus.COMPLETED)
    result = result.set_component_status("ГКЛ", ComponentStatus.SKIPPED)
    result = result.set_model_info(
        ModelInfo(
            name="СКЛ",
            version="1.0",
            status="расчёт выполнен",
            source="ГОСТ СКЛ",
        ),
    )
    return result.add_log_entry(
        LogLevel.INFO,
        "исходные данные",
        "расчётная конфигурация принята",
        {"Kp": "4"},
    )


def test_calculation_result_snapshot_round_trip() -> None:
    result = _calculation_result()

    restored = calculation_result_from_snapshot(calculation_result_snapshot(result))

    assert restored == result


def test_calculation_result_snapshot_contains_result_sections() -> None:
    snapshot = calculation_result_snapshot(_calculation_result())

    assert "calculation_config" in snapshot
    assert "log" in snapshot
    assert "component_statuses" in snapshot
    assert "model_info" in snapshot
    assert snapshot["has_errors"] is False


def test_calculation_result_from_snapshot_rejects_missing_required_field() -> None:
    snapshot = calculation_result_snapshot(_calculation_result())
    del snapshot["log"]

    with pytest.raises(ValueError, match="Required calculation result field is missing"):
        calculation_result_from_snapshot(snapshot)


def test_calculation_result_from_snapshot_rejects_error_flag_mismatch() -> None:
    snapshot = calculation_result_snapshot(_calculation_result())
    snapshot["has_errors"] = True

    with pytest.raises(ValueError, match="error flag does not match restored result"):
        calculation_result_from_snapshot(snapshot)
