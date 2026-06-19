"""Metadata objects for RADAR tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DataFormat(StrEnum):
    """Supported resource file formats."""

    CSV = "csv"
    JSON = "json"
    TEXT = "text"


class ValidationStatus(StrEnum):
    """Validation status of transferred model data."""

    RAW = "raw"
    IMPORTED = "imported"
    CHECKED = "checked"
    VALIDATED = "validated"


def _expected_suffix(data_format: DataFormat) -> str:
    if data_format is DataFormat.CSV:
        return ".csv"
    if data_format is DataFormat.JSON:
        return ".json"
    if data_format is DataFormat.TEXT:
        return ".txt"

    msg = f"Unsupported data format: {data_format}"
    raise ValueError(msg)


@dataclass(frozen=True)
class TableMetadata:
    """Passport of one data table used by a calculation model."""

    model: str
    document: str
    table_id: str
    filename: str
    data_format: DataFormat
    units: dict[str, str]
    validation_status: ValidationStatus = ValidationStatus.RAW
    appendix: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            msg = "Table metadata model must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "Table metadata document must not be empty."
            raise ValueError(msg)

        if not self.table_id:
            msg = "Table metadata table_id must not be empty."
            raise ValueError(msg)

        if not self.filename:
            msg = "Table metadata filename must not be empty."
            raise ValueError(msg)

        if not self.units:
            msg = "Table metadata units must not be empty."
            raise ValueError(msg)

        if any(not key or not value for key, value in self.units.items()):
            msg = "Table metadata unit names and values must not be empty."
            raise ValueError(msg)

        expected_suffix = _expected_suffix(self.data_format)
        actual_suffix = Path(self.filename).suffix.lower()

        if actual_suffix != expected_suffix:
            msg = f"File suffix {actual_suffix!r} does not match data format {self.data_format!r}."
            raise ValueError(msg)
