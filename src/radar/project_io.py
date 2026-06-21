"""Read and write RADAR project files."""

from __future__ import annotations

from datetime import datetime
from os import PathLike
from pathlib import Path

from radar.calculation import execute_calculation
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult
from radar.project_file import ProjectFile, project_file_from_json

ProjectPath = str | PathLike[str]


def save_project_file(
    path: ProjectPath,
    calculation_config: CalculationConfig,
    *,
    calculation_result: CalculationResult | None = None,
    created_at: datetime | None = None,
) -> ProjectFile:
    """Save calculation configuration to a RADAR project file."""

    project_file = ProjectFile.create(
        calculation_config=calculation_config,
        calculation_result=calculation_result,
        created_at=created_at,
    )
    Path(path).write_text(project_file.to_json() + "\n", encoding="utf-8")
    return project_file


def calculate_and_save_project_file(
    path: ProjectPath,
    calculation_config: CalculationConfig,
    *,
    created_at: datetime | None = None,
) -> ProjectFile:
    """Execute calculation and save a RADAR project file with the result."""

    calculation_result = execute_calculation(calculation_config)

    return save_project_file(
        path=path,
        calculation_config=calculation_config,
        calculation_result=calculation_result,
        created_at=created_at,
    )


def calculate_project_file(
    input_path: ProjectPath,
    *,
    output_path: ProjectPath | None = None,
    created_at: datetime | None = None,
) -> ProjectFile:
    """Read a RADAR project file, execute calculation and save the result."""

    source_project_file = read_project_file(input_path)
    timestamp = created_at

    if timestamp is None:
        timestamp = datetime.fromisoformat(source_project_file.created_at)

    return calculate_and_save_project_file(
        path=output_path or input_path,
        calculation_config=source_project_file.calculation_config,
        created_at=timestamp,
    )


def read_project_file(path: ProjectPath) -> ProjectFile:
    """Read RADAR project file."""

    return project_file_from_json(Path(path).read_text(encoding="utf-8"))


def load_project_config(path: ProjectPath) -> CalculationConfig:
    """Read calculation configuration from a RADAR project file."""

    return read_project_file(path).calculation_config
