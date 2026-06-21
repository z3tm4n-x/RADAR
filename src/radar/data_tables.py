"""Normative table resource loading."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO
from importlib import resources
from pathlib import PurePosixPath
from typing import cast

_NORMATIVE_PACKAGE = "radar.data.normative"


@dataclass(frozen=True)
class NormativeCollectionRef:
    """Reference to a normative table collection."""

    collection_id: str
    title: str
    manifest: str


@dataclass(frozen=True)
class NormativeTableRef:
    """Reference to one normative table resource."""

    table_id: str
    title: str
    document: str
    appendix: str
    table_number: str
    model: str
    quantity: str
    x_unit: str
    y_unit: str
    data_file: str
    verification_status: str
    control_values: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class NormativeCollectionManifest:
    """Normative table collection manifest."""

    schema_version: str
    collection_id: str
    title: str
    document: str
    status: str
    tables: tuple[NormativeTableRef, ...]


def _load_json_resource(relative_path: str) -> dict[str, object]:
    """Load JSON resource from normative data package."""

    resource = resources.files(_NORMATIVE_PACKAGE).joinpath(relative_path)
    loaded: object = json.loads(resource.read_text(encoding="utf-8-sig"))

    if not isinstance(loaded, dict):
        msg = f"Normative JSON resource root must be an object: {relative_path}"
        raise ValueError(msg)

    return cast("dict[str, object]", loaded)


def _safe_normative_path(relative_path: str, kind: str) -> PurePosixPath:
    """Return safe relative resource path."""

    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        msg = f"Unsafe normative {kind} path: {relative_path}"
        raise ValueError(msg)

    return path


def _load_csv_resource(relative_path: str) -> tuple[dict[str, str], ...]:
    """Load CSV resource from normative data package."""

    resource_path = _safe_normative_path(relative_path, "table")
    resource = resources.files(_NORMATIVE_PACKAGE).joinpath(str(resource_path))
    text = resource.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(StringIO(text))

    if reader.fieldnames is None:
        msg = f"Normative CSV resource has no header: {relative_path}"
        raise ValueError(msg)

    rows: list[dict[str, str]] = []
    for row in reader:
        normalized_row: dict[str, str] = {}
        for key, value in row.items():
            if key is None:
                msg = f"Normative CSV resource has unnamed column: {relative_path}"
                raise ValueError(msg)
            normalized_row[key] = "" if value is None else value
        rows.append(normalized_row)

    return tuple(rows)


def _required_str(data: dict[str, object], key: str) -> str:
    """Return required string field."""

    try:
        value = data[key]
    except KeyError as exc:
        msg = f"Normative resource field is missing: {key}"
        raise ValueError(msg) from exc

    if not isinstance(value, str):
        msg = f"Normative resource field must be a string: {key}"
        raise ValueError(msg)

    return value


def _required_list(data: dict[str, object], key: str) -> list[object]:
    """Return required list field."""

    try:
        value = data[key]
    except KeyError as exc:
        msg = f"Normative resource field is missing: {key}"
        raise ValueError(msg) from exc

    if not isinstance(value, list):
        msg = f"Normative resource field must be a list: {key}"
        raise ValueError(msg)

    return value


def _collection_ref_from_dict(data: object) -> NormativeCollectionRef:
    """Create collection reference from saved mapping."""

    if not isinstance(data, dict):
        msg = "Normative collection reference must be an object."
        raise ValueError(msg)

    return NormativeCollectionRef(
        collection_id=_required_str(data, "collection_id"),
        title=_required_str(data, "title"),
        manifest=_required_str(data, "manifest"),
    )


def _table_ref_from_dict(data: object) -> NormativeTableRef:
    """Create table reference from saved mapping."""

    if not isinstance(data, dict):
        msg = "Normative table reference must be an object."
        raise ValueError(msg)

    control_values = _required_list(data, "control_values")
    for item in control_values:
        if not isinstance(item, dict):
            msg = "Normative table control value must be an object."
            raise ValueError(msg)

    return NormativeTableRef(
        table_id=_required_str(data, "table_id"),
        title=_required_str(data, "title"),
        document=_required_str(data, "document"),
        appendix=_required_str(data, "appendix"),
        table_number=_required_str(data, "table_number"),
        model=_required_str(data, "model"),
        quantity=_required_str(data, "quantity"),
        x_unit=_required_str(data, "x_unit"),
        y_unit=_required_str(data, "y_unit"),
        data_file=_required_str(data, "data_file"),
        verification_status=_required_str(data, "verification_status"),
        control_values=tuple(cast("dict[str, object]", item) for item in control_values),
    )


def list_normative_collections() -> tuple[NormativeCollectionRef, ...]:
    """Return available normative table collections."""

    index = _load_json_resource("index.json")
    schema_version = _required_str(index, "schema_version")
    if schema_version != "1.0":
        msg = f"Unsupported normative index schema version: {schema_version}"
        raise ValueError(msg)

    return tuple(
        _collection_ref_from_dict(item)
        for item in _required_list(index, "collections")
    )


def load_normative_collection_manifest(
    collection_id: str,
) -> NormativeCollectionManifest:
    """Load normative table collection manifest."""

    collection_refs = {
        collection.collection_id: collection
        for collection in list_normative_collections()
    }

    try:
        collection_ref = collection_refs[collection_id]
    except KeyError as exc:
        msg = f"Unknown normative collection: {collection_id}"
        raise ValueError(msg) from exc

    _safe_normative_path(collection_ref.manifest, "manifest")

    data = _load_json_resource(collection_ref.manifest)
    schema_version = _required_str(data, "schema_version")
    if schema_version != "1.0":
        msg = f"Unsupported normative manifest schema version: {schema_version}"
        raise ValueError(msg)

    manifest = NormativeCollectionManifest(
        schema_version=schema_version,
        collection_id=_required_str(data, "collection_id"),
        title=_required_str(data, "title"),
        document=_required_str(data, "document"),
        status=_required_str(data, "status"),
        tables=tuple(
            _table_ref_from_dict(item)
            for item in _required_list(data, "tables")
        ),
    )

    if manifest.collection_id != collection_id:
        msg = "Normative manifest collection id does not match index."
        raise ValueError(msg)

    return manifest


def get_normative_table_ref(
    collection_id: str,
    table_id: str,
) -> NormativeTableRef:
    """Return one normative table reference by id."""

    manifest = load_normative_collection_manifest(collection_id)
    table_refs = {table.table_id: table for table in manifest.tables}

    try:
        return table_refs[table_id]
    except KeyError as exc:
        msg = f"Unknown normative table: {collection_id}/{table_id}"
        raise ValueError(msg) from exc

def load_normative_table_rows(
    collection_id: str,
    table_id: str,
) -> tuple[dict[str, str], ...]:
    """Load normative table rows."""

    table = get_normative_table_ref(collection_id, table_id)
    if not table.data_file:
        msg = f"Normative table data file is missing: {collection_id}/{table_id}"
        raise ValueError(msg)

    return _load_csv_resource(table.data_file)


def validate_normative_table_control_values(
    collection_id: str,
    table_id: str,
) -> None:
    """Validate table control values against transferred rows."""

    table = get_normative_table_ref(collection_id, table_id)
    rows = load_normative_table_rows(collection_id, table_id)
    row_set = {tuple(sorted(row.items())) for row in rows}

    for control_value in table.control_values:
        expected = tuple(
            sorted((str(key), str(value)) for key, value in control_value.items())
        )
        if expected not in row_set:
            msg = f"Normative table control value is missing: {collection_id}/{table_id}"
            raise ValueError(msg)

