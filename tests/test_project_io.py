from datetime import UTC, datetime

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import CalculationConfig, MethodologyConfig, MissionConfig, OrbitConfig
from radar.core.result import CalculationResult, ComponentStatus
from radar.core.units import Unit
from radar.output_tables import OutputTableColumn, OutputTableKind, output_table_from_rows
from radar.project_file import PROJECT_PROGRAM_NAME, PROJECT_SCHEMA_VERSION
from radar.project_io import (
    calculate_project_file,
    calculate_and_save_project_file,
    load_project_config,
    read_project_file,
    save_project_file,
)


def _calculation_config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP),
        kp=4,
    )


def _output_table():
    return output_table_from_rows(
        table_id="dose",
        title="Накопленная доза",
        kind=OutputTableKind.DOSE,
        columns=(
            OutputTableColumn(
                key="thickness",
                title="Толщина защиты",
                unit=Unit.THICKNESS.value,
            ),
            OutputTableColumn(
                key="dose",
                title="Накопленная доза",
                unit=Unit.RAD.value,
            ),
        ),
        rows=(
            {
                "thickness": 1.0,
                "dose": 10.0,
            },
        ),
    )


def _calculation_result() -> CalculationResult:
    return CalculationResult(config=_calculation_config()).set_component_status(
        "СКЛ",
        ComponentStatus.COMPLETED,
    ).set_output_table(_output_table())


def test_save_project_file_writes_project_file(tmp_path) -> None:
    path = tmp_path / "project.radar.json"
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)

    project_file = save_project_file(
        path,
        _calculation_config(),
        created_at=created_at,
    )

    assert path.exists()
    assert project_file.schema_version == PROJECT_SCHEMA_VERSION
    assert project_file.program_name == PROJECT_PROGRAM_NAME
    assert project_file.created_at == "2028-01-02T03:04:05+00:00"


def test_read_project_file_round_trip(tmp_path) -> None:
    path = tmp_path / "project.radar.json"
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)

    saved = save_project_file(
        path,
        _calculation_config(),
        created_at=created_at,
    )

    assert read_project_file(path) == saved


def test_load_project_config_round_trip(tmp_path) -> None:
    path = tmp_path / "project.radar.json"
    config = _calculation_config()

    save_project_file(
        path,
        config,
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert load_project_config(path) == config


def test_saved_project_file_contains_selected_models(tmp_path) -> None:
    path = tmp_path / "project.radar.json"

    save_project_file(
        path,
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    project_file = read_project_file(path)

    assert (
        project_file.calculation_config.source_model_selection.sep_model_family
        is SourceModelFamily.GOST_SEP
    )
    assert (
        project_file.calculation_config.source_model_selection.gcr_model_family
        is SourceModelFamily.OST_134_1044_2007
    )


def test_save_project_file_overwrites_existing_file(tmp_path) -> None:
    path = tmp_path / "project.radar.json"
    path.write_text("old content", encoding="utf-8")

    save_project_file(
        path,
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert read_project_file(path).calculation_config == _calculation_config()


def test_save_project_file_writes_calculation_result(tmp_path) -> None:
    path = tmp_path / "project.radar.json"
    result = _calculation_result()

    save_project_file(
        path,
        _calculation_config(),
        calculation_result=result,
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert read_project_file(path).calculation_result == result


def test_calculate_and_save_project_file_writes_result(tmp_path) -> None:
    path = tmp_path / "calculated_project.radar.json"

    project_file = calculate_and_save_project_file(
        path,
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    restored = read_project_file(path)

    assert project_file.calculation_result is not None
    assert restored == project_file
    assert restored.calculation_result is not None
    assert restored.calculation_result.config == _calculation_config()
    output_table_ids = {
        table.table_id
        for table in restored.calculation_result.output_tables
    }
    assert {
        "dose_by_thickness",
        "source_contributions",
        "single_event_effects",
    } <= output_table_ids
    assert any(table_id.startswith("sep_") for table_id in output_table_ids)


def test_calculate_project_file_overwrites_input_file(tmp_path) -> None:
    path = tmp_path / "project.radar.json"
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)

    save_project_file(
        path,
        _calculation_config(),
        created_at=created_at,
    )

    project_file = calculate_project_file(path)
    restored = read_project_file(path)

    assert project_file == restored
    assert restored.created_at == created_at.isoformat()
    assert restored.calculation_result is not None
    assert restored.calculation_result.config == _calculation_config()
    output_table_ids = {
        table.table_id
        for table in restored.calculation_result.output_tables
    }
    assert {
        "dose_by_thickness",
        "source_contributions",
        "single_event_effects",
    } <= output_table_ids
    assert any(table_id.startswith("sep_") for table_id in output_table_ids)


def test_calculate_project_file_can_write_to_output_file(tmp_path) -> None:
    input_path = tmp_path / "input_project.radar.json"
    output_path = tmp_path / "output_project.radar.json"
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)
    output_created_at = datetime(2029, 6, 7, 8, 9, 10, tzinfo=UTC)

    save_project_file(
        input_path,
        _calculation_config(),
        created_at=created_at,
    )

    project_file = calculate_project_file(
        input_path,
        output_path=output_path,
        created_at=output_created_at,
    )
    original = read_project_file(input_path)
    restored = read_project_file(output_path)

    assert original.calculation_result is None
    assert project_file == restored
    assert restored.created_at == output_created_at.isoformat()
    assert restored.calculation_result is not None
    assert restored.calculation_result.config == _calculation_config()
