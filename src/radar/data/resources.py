"""Resource loading utilities for RADAR data tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from radar.data.metadata import DataFormat, TableMetadata


class ResourceManager:
    """Resolve and load RADAR data resources from a root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def resolve(self, metadata: TableMetadata) -> Path:
        """Return absolute path to a resource and verify that it exists."""

        path = (self.root / metadata.filename).resolve()

        if not path.exists():
            msg = f"Resource file does not exist: {path}"
            raise FileNotFoundError(msg)

        if not path.is_file():
            msg = f"Resource path is not a file: {path}"
            raise ValueError(msg)

        return path

    def load_text(self, metadata: TableMetadata) -> str:
        """Load a text resource."""

        if metadata.data_format is not DataFormat.TEXT:
            msg = "Resource metadata must describe a TEXT file."
            raise ValueError(msg)

        return self.resolve(metadata).read_text(encoding="utf-8")

    def load_json(self, metadata: TableMetadata) -> object:
        """Load a JSON resource."""

        if metadata.data_format is not DataFormat.JSON:
            msg = "Resource metadata must describe a JSON file."
            raise ValueError(msg)

        with self.resolve(metadata).open("r", encoding="utf-8") as file:
            data: Any = json.load(file)

        return data

    def load_csv_dicts(self, metadata: TableMetadata) -> list[dict[str, str]]:
        """Load a CSV resource as a list of dictionaries."""

        if metadata.data_format is not DataFormat.CSV:
            msg = "Resource metadata must describe a CSV file."
            raise ValueError(msg)

        rows: list[dict[str, str]] = []

        with self.resolve(metadata).open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                normalized_row = {
                    key: value if value is not None else ""
                    for key, value in row.items()
                    if key is not None
                }
                rows.append(normalized_row)

        return rows
