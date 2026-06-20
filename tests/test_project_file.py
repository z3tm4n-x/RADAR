import json
from datetime import UTC, datetime

import pytest

from radar import __version__
from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import CalculationConfig, MethodologyConfig, MissionConfig, OrbitConfig
from radar.project_file import (
    PROJECT_PROGRAM_NAME,
    PROJECT_SCHEMA_VERSION,
    ProjectFile,
    project_file_from_dict,
    project_file_from_json,
)


def _calculation_config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP),
        kp=4,
    )


def test_project_file_create() -> None:
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=created_at,
    )

    assert project_file.schema_version == PROJECT_SCHEMA_VERSION
    assert project_file.program_name == PROJECT_PROGRAM_NAME
    assert project_file.program_version == __version__
    assert project_file.created_at == "2028-01-02T03:04:05+00:00"


def test_project_file_to_dict_contains_project_metadata_and_calculation_config() -> None:
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=created_at,
    )

    data = project_file.to_dict()

    assert data["schema_version"] == PROJECT_SCHEMA_VERSION
    assert data["program_name"] == PROJECT_PROGRAM_NAME
    assert data["program_version"] == __version__
    assert data["created_at"] == "2028-01-02T03:04:05+00:00"

    calculation = data["calculation_config"]
    assert isinstance(calculation, dict)
    source_selection = calculation["source_model_selection"]
    assert isinstance(source_selection, dict)
    assert source_selection["sep_model_family"] == SourceModelFamily.GOST_SEP.value
    assert source_selection["gcr_model_family"] == SourceModelFamily.OST_134_1044_2007.value


def test_project_file_json_round_trip() -> None:
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    restored = project_file_from_json(project_file.to_json())

    assert restored == project_file


def test_project_file_dict_round_trip() -> None:
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    restored = project_file_from_dict(project_file.to_dict())

    assert restored == project_file


def test_project_file_to_json_is_json_compatible() -> None:
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    decoded = json.loads(project_file.to_json())

    assert decoded == project_file.to_dict()


def test_project_file_from_dict_rejects_missing_required_field() -> None:
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    data = project_file.to_dict()
    del data["schema_version"]

    with pytest.raises(ValueError, match="Required project file field is missing"):
        project_file_from_dict(data)


def test_project_file_from_dict_rejects_unsupported_schema_version() -> None:
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    data = project_file.to_dict()
    data["schema_version"] = "999.0"

    with pytest.raises(ValueError, match="Unsupported project schema version"):
        project_file_from_dict(data)


def test_project_file_from_dict_rejects_unsupported_program_name() -> None:
    project_file = ProjectFile.create(
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    data = project_file.to_dict()
    data["program_name"] = "OTHER"

    with pytest.raises(ValueError, match="Unsupported project program name"):
        project_file_from_dict(data)


def test_project_file_from_json_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="Project file JSON root must be an object"):
        project_file_from_json("[]")
