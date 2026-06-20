"""RADAR project file format."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, cast

from radar.version import __version__
from radar.core.project import CalculationConfig
from radar.core.project_snapshot import (
    calculation_config_from_snapshot,
    calculation_config_snapshot,
)

PROJECT_SCHEMA_VERSION: Final[str] = "1.0"
PROJECT_PROGRAM_NAME: Final[str] = "RADAR"


@dataclass(frozen=True)
class ProjectFile:
    """RADAR project file representation."""

    schema_version: str
    program_name: str
    program_version: str
    created_at: str
    calculation_config: CalculationConfig

    @classmethod
    def create(
        cls,
        calculation_config: CalculationConfig,
        *,
        created_at: datetime | None = None,
    ) -> ProjectFile:
        """Create project file representation for a calculation configuration."""

        timestamp = created_at or datetime.now(UTC)

        return cls(
            schema_version=PROJECT_SCHEMA_VERSION,
            program_name=PROJECT_PROGRAM_NAME,
            program_version=__version__,
            created_at=timestamp.isoformat(),
            calculation_config=calculation_config,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible project file representation."""

        return {
            "schema_version": self.schema_version,
            "program_name": self.program_name,
            "program_version": self.program_version,
            "created_at": self.created_at,
            "calculation_config": calculation_config_snapshot(
                self.calculation_config,
            ),
        }

    def to_json(self) -> str:
        """Return stable JSON project file representation."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )


def _required_value(data: Mapping[str, object], key: str) -> object:
    """Return required project field value."""

    try:
        return data[key]
    except KeyError as exc:
        msg = f"Required project file field is missing: {key}"
        raise ValueError(msg) from exc


def _required_str(data: Mapping[str, object], key: str) -> str:
    """Return required string project field value."""

    value = _required_value(data, key)
    if not isinstance(value, str):
        msg = f"Project file field must be a string: {key}"
        raise ValueError(msg)
    return value


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return required object project field value."""

    value = _required_value(data, key)
    if not isinstance(value, Mapping):
        msg = f"Project file field must be an object: {key}"
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def project_file_from_dict(data: Mapping[str, object]) -> ProjectFile:
    """Restore project file representation from saved dictionary."""

    schema_version = _required_str(data, "schema_version")
    if schema_version != PROJECT_SCHEMA_VERSION:
        msg = (
            f"Unsupported project schema version: {schema_version}. "
            f"Expected: {PROJECT_SCHEMA_VERSION}."
        )
        raise ValueError(msg)

    program_name = _required_str(data, "program_name")
    if program_name != PROJECT_PROGRAM_NAME:
        msg = (
            f"Unsupported project program name: {program_name}. "
            f"Expected: {PROJECT_PROGRAM_NAME}."
        )
        raise ValueError(msg)

    return ProjectFile(
        schema_version=schema_version,
        program_name=program_name,
        program_version=_required_str(data, "program_version"),
        created_at=_required_str(data, "created_at"),
        calculation_config=calculation_config_from_snapshot(
            _required_mapping(data, "calculation_config"),
        ),
    )


def project_file_from_json(data: str) -> ProjectFile:
    """Restore project file representation from JSON string."""

    loaded: object = json.loads(data)
    if not isinstance(loaded, Mapping):
        msg = "Project file JSON root must be an object."
        raise ValueError(msg)

    return project_file_from_dict(cast("Mapping[str, object]", loaded))
