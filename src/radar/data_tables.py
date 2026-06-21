"""Normative table resource loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
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

    manifest_path = PurePosixPath(collection_ref.manifest)
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        msg = f"Unsafe normative manifest path: {collection_ref.manifest}"
        raise ValueError(msg)

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
