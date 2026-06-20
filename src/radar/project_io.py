"""Read and write RADAR project files."""

from __future__ import annotations

from datetime import datetime
from os import PathLike
from pathlib import Path

from radar.core.project import CalculationConfig
from radar.project_file import ProjectFile, project_file_from_json

ProjectPath = str | PathLike[str]


def save_project_file(
    path: ProjectPath,
    calculation_config: CalculationConfig,
    *,
    created_at: datetime | None = None,
) -> ProjectFile:
    """Save calculation configuration to a RADAR project file."""

    project_file = ProjectFile.create(
        calculation_config=calculation_config,
        created_at=created_at,
    )
    Path(path).write_text(project_file.to_json() + "\n", encoding="utf-8")
    return project_file


def read_project_file(path: ProjectPath) -> ProjectFile:
    """Read RADAR project file."""

    return project_file_from_json(Path(path).read_text(encoding="utf-8"))


def load_project_config(path: ProjectPath) -> CalculationConfig:
    """Read calculation configuration from a RADAR project file."""

    return read_project_file(path).calculation_config
