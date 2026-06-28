import pytest

from radar.calculation import execute_calculation
from radar.core.log import LogLevel
from radar.core.project import (
    CalculationConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
)
from radar.core.result import CalculationResult, ComponentStatus
from radar.core.types import DoseQuantity, SolarActivityLevel
from radar.core.units import Unit
from radar.pipelines.erb import (
    ERB_MODEL_COMPONENT,
    ERB_OUTPUT_TABLES_COMPONENT,
    ERB_PIPELINE_COMPONENT,
    ERB_SHIELDING_COMPONENT,
)
from radar.pipelines.gcr import (
    GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
    GCR_LET_COMPONENT,
    GCR_MODEL_COMPONENT,
    GCR_OUTPUT_TABLES_COMPONENT,
    GCR_PIPELINE_COMPONENT,
    GCR_SHIELDING_COMPONENT,
)
from radar.pipelines.sep import (
    SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_LET_COMPONENT,
    SEP_MODEL_COMPONENT,
    SEP_OUTPUT_TABLES_COMPONENT,
    SEP_PIPELINE_COMPONENT,
    SEP_SHIELDING_COMPONENT,
)
from radar.project_file import ProjectFile, project_file_from_json


def _config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        shielding=ShieldingConfig(thicknesses_g_cm2=(1.0,)),
        kp=4,
        dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
        dose_unit=Unit.RAD,
    )


def _maximum_activity_config() -> CalculationConfig:
    base_config = _config()
    return CalculationConfig(
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


@pytest.fixture(scope="module")
def calculation_result() -> CalculationResult:
    return execute_calculation(_config())


@pytest.fixture(scope="module")
def maximum_activity_result() -> CalculationResult:
    return execute_calculation(_maximum_activity_config())


def test_execute_calculation_returns_result_for_config(
    calculation_result: CalculationResult,
) -> None:
    assert calculation_result.config == _config()
    assert not calculation_result.has_errors()


def test_execute_calculation_sets_source_component_statuses(
    calculation_result: CalculationResult,
) -> None:
    assert (
        calculation_result.component_status("\u0421\u041a\u041b")
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status("\u0413\u041a\u041b")
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status("\u0415\u0420\u041f\u0417")
        is ComponentStatus.COMPLETED
    )

    assert (
        calculation_result.component_status(SEP_PIPELINE_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(SEP_MODEL_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(SEP_GEOMAGNETIC_PENETRATION_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(SEP_SHIELDING_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(SEP_LET_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(SEP_OUTPUT_TABLES_COMPONENT)
        is ComponentStatus.COMPLETED
    )

    assert (
        calculation_result.component_status(GCR_PIPELINE_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(GCR_MODEL_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(GCR_GEOMAGNETIC_PENETRATION_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(GCR_SHIELDING_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(GCR_LET_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(GCR_OUTPUT_TABLES_COMPONENT)
        is ComponentStatus.COMPLETED
    )

    assert (
        calculation_result.component_status(ERB_PIPELINE_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(ERB_MODEL_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(ERB_SHIELDING_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert (
        calculation_result.component_status(ERB_OUTPUT_TABLES_COMPONENT)
        is ComponentStatus.COMPLETED
    )


def test_execute_calculation_records_model_information(
    calculation_result: CalculationResult,
) -> None:
    model_versions = {
        model.name: model.version
        for model in calculation_result.model_info
    }
    assert {
        "ost_sep_model",
        "ost_gcr_model",
        "ost_134_1044_2007_geomagnetic_penetration",
        "sep_al_shielding",
        "sep_si_let",
        "gcr_al_shielding",
        "gcr_si_let",
        "ost_erb_model",
        "erb_proton_al_shielding",
    } <= set(model_versions)
    assert model_versions["ost_sep_model"] == "unversioned"
    assert model_versions["ost_gcr_model"] == "unversioned"
    assert model_versions["ost_erb_model"] == "unversioned"
    assert model_versions["erb_proton_al_shielding"] == (
        "al_spherical_csda_secondary_v1"
    )

    model_statuses = {
        model.name: model.status
        for model in calculation_result.model_info
    }
    assert model_statuses["ost_sep_model"] == "calculated"
    assert model_statuses["ost_gcr_model"] == "calculated"
    assert model_statuses["sep_al_shielding"] == "calculated"
    assert model_statuses["sep_si_let"] == "calculated"
    assert model_statuses["gcr_al_shielding"] == "calculated"
    assert model_statuses["gcr_si_let"] == "calculated"
    assert model_statuses["ost_erb_model"] == "calculated"
    assert model_statuses["erb_proton_al_shielding"] == "calculated"


def test_execute_calculation_writes_log_entries(
    calculation_result: CalculationResult,
) -> None:
    assert calculation_result.log.entries[0].level is LogLevel.INFO
    assert (
        calculation_result.log.entries[0].stage
        == "\u0437\u0430\u043f\u0443\u0441\u043a \u0440\u0430\u0441\u0447\u0451\u0442\u0430"
    )
    assert calculation_result.log.warnings()
    assert any(
        "\u042d\u043b\u0435\u043a\u0442\u0440\u043e\u043d\u044b \u0437\u0430 \u0437\u0430\u0449\u0438\u0442\u043e\u0439" in entry.message
        for entry in calculation_result.log.entries
    )
    assert all(entry.level is not LogLevel.ERROR for entry in calculation_result.log.entries)


def test_execute_calculation_builds_placeholder_summary_and_pipeline_output_tables(
    calculation_result: CalculationResult,
) -> None:
    output_table_ids = {table.table_id for table in calculation_result.output_tables}
    assert {
        "dose_by_thickness",
        "source_contributions",
        "single_event_effects",
    } <= output_table_ids
    assert any(table_id.startswith("sep_") for table_id in output_table_ids)
    assert any(table_id.startswith("gcr_") for table_id in output_table_ids)
    assert any(table_id.startswith("erb_") for table_id in output_table_ids)
    assert "erb_electron_mean_flux_mean_differential_flux_on_orbit" in output_table_ids
    assert "erb_t1_mean_flux_erb_proton_total_energy_behind_al" in output_table_ids
    assert not any(
        "erb_electron" in table_id and "behind_al" in table_id
        for table_id in output_table_ids
    )

    dose_table = next(
        table
        for table in calculation_result.output_tables
        if table.table_id == "dose_by_thickness"
    )
    assert dose_table.rows[0].cells == (1.0, 0.0)
    assert ("status", "placeholder") in dose_table.metadata

    see_table = next(
        table
        for table in calculation_result.output_tables
        if table.table_id == "single_event_effects"
    )
    see_column_keys = tuple(column.key for column in see_table.columns)
    assert [row.cells[0] for row in see_table.rows] == [1.0]
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


def test_execute_calculation_result_can_be_saved_in_project_file(
    calculation_result: CalculationResult,
) -> None:
    project_file = ProjectFile.create(
        calculation_config=calculation_result.config,
        calculation_result=calculation_result,
    )
    restored = project_file_from_json(project_file.to_json())

    assert restored.calculation_result == calculation_result


def test_execute_calculation_records_ost_solar_activity_input_data(
    maximum_activity_result: CalculationResult,
) -> None:
    solar_data = next(
        info
        for info in maximum_activity_result.input_data_info
        if info.name == "\u0421\u0410"
    )
    assert solar_data.source == "\u041e\u0421\u0422 134-1044-2007"
    assert solar_data.table_id == "ost_134_1044_2007_table_g_1_wolf_numbers"
    assert ("level", "maximum") in solar_data.values
    assert ("cycle_years", "1, 2, 3") in solar_data.values
    assert ("wolf_numbers", "11.5, 33.9, 100.8") in solar_data.values


def test_execute_calculation_logs_ost_wolf_numbers_for_mission(
    maximum_activity_result: CalculationResult,
) -> None:
    messages = tuple(entry.message for entry in maximum_activity_result.log.entries)
    assert any(
        message
        == "\u0427\u0438\u0441\u043b\u0430 \u0412\u043e\u043b\u044c\u0444\u0430 \u043f\u043e \u0442\u0430\u0431\u043b\u0438\u0446\u0435 \u0413.1 \u041e\u0421\u0422: 11.5, 33.9, 100.8"
        for message in messages
    )
