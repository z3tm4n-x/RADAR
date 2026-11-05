"""Integrity and release-status registry for bundled physical data resources."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from importlib import resources
import json
from pathlib import PurePosixPath
from typing import cast

_NORMATIVE_PACKAGE = "radar.data.normative"
_INTEGRITY_MANIFEST = "integrity_manifest.json"


class NormativeResourceStatus(StrEnum):
    """Independent verification state of one bundled resource."""

    VERIFIED_SOURCE_EXACT = "verified_source_exact"
    VERIFIED_UPSTREAM_EXACT = "verified_upstream_exact"
    VERIFIED_AGAINST_DRAFT = "verified_against_draft"
    VERIFIED_DERIVED = "verified_derived"
    PROVISIONAL = "provisional"
    QUARANTINED = "quarantined"
    INTERNAL_METADATA = "internal_metadata"


@dataclass(frozen=True)
class NormativeResourceIntegrity:
    """Machine-readable passport of one bundled data or registry resource."""

    resource_id: str
    path: str
    sha256: str
    byte_count: int
    record_count: int
    status: NormativeResourceStatus
    calculation_input: bool
    release_eligible: bool
    source_ref: str
    authority_status: str
    media_type: str
    validator: str
    verification_method: str
    verification_note: str
    interpolation_law: str | None = None
    energy_basis: str | None = None
    source_artifact_sha256: str | None = None
    source_artifact_byte_count: int | None = None
    derivation_command: str | None = None

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            msg = "Normative resource id must not be empty."
            raise ValueError(msg)

        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or not self.path.strip():
            msg = f"Unsafe normative resource path: {self.path}"
            raise ValueError(msg)

        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            msg = f"Invalid SHA256 digest for normative resource: {self.path}"
            raise ValueError(msg)

        if self.byte_count < 0:
            msg = "Normative resource byte count must be non-negative."
            raise ValueError(msg)

        if self.record_count < 0:
            msg = "Normative resource record count must be non-negative."
            raise ValueError(msg)

        if any(
            not value.strip()
            for value in (
                self.source_ref,
                self.authority_status,
                self.media_type,
                self.validator,
            )
        ):
            msg = "Normative resource source, format and validator fields must not be empty."
            raise ValueError(msg)

        if not self.verification_method.strip():
            msg = "Normative resource verification method must not be empty."
            raise ValueError(msg)

        if self.release_eligible and not self.calculation_input:
            msg = "Only calculation inputs may be marked release eligible."
            raise ValueError(msg)

        verified_statuses = {
            NormativeResourceStatus.VERIFIED_SOURCE_EXACT,
            NormativeResourceStatus.VERIFIED_UPSTREAM_EXACT,
            NormativeResourceStatus.VERIFIED_AGAINST_DRAFT,
            NormativeResourceStatus.VERIFIED_DERIVED,
        }
        if self.release_eligible and self.status not in verified_statuses:
            msg = "Release-eligible resources must have an explicit verified status."
            raise ValueError(msg)

        source_digest_fields = (
            self.source_artifact_sha256,
            self.source_artifact_byte_count,
        )
        if any(value is not None for value in source_digest_fields) and any(
            value is None for value in source_digest_fields
        ):
            msg = (
                "Source-artifact SHA256 and byte count must be provided together."
            )
            raise ValueError(msg)

        if self.source_artifact_sha256 is not None and (
            len(self.source_artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_artifact_sha256
            )
        ):
            msg = f"Invalid source-artifact SHA256 digest for: {self.path}"
            raise ValueError(msg)

        if (
            self.source_artifact_byte_count is not None
            and self.source_artifact_byte_count < 0
        ):
            msg = "Source-artifact byte count must be non-negative."
            raise ValueError(msg)

        if self.derivation_command is not None and not self.derivation_command.strip():
            msg = "Normative resource derivation command must not be empty."
            raise ValueError(msg)

        if (
            self.release_eligible
            and self.status is NormativeResourceStatus.VERIFIED_DERIVED
            and (
                self.source_artifact_sha256 is None
                or self.derivation_command is None
            )
        ):
            msg = (
                "Release-eligible derived resources require a source-artifact "
                "digest and a deterministic derivation command."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class NormativeIntegrityManifest:
    """One immutable release view of all bundled normative resources."""

    schema_version: str
    release_id: str
    resources: tuple[NormativeResourceIntegrity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "2.0.0":
            msg = f"Unsupported normative integrity schema: {self.schema_version}"
            raise ValueError(msg)
        if not self.release_id.strip():
            msg = "Normative integrity release id must not be empty."
            raise ValueError(msg)

        ids = tuple(item.resource_id for item in self.resources)
        paths = tuple(item.path for item in self.resources)
        if len(set(ids)) != len(ids):
            msg = "Normative integrity resource ids must be unique."
            raise ValueError(msg)
        if len(set(paths)) != len(paths):
            msg = "Normative integrity resource paths must be unique."
            raise ValueError(msg)

    def by_path(self, path: str) -> NormativeResourceIntegrity:
        """Return the passport for an exact package-relative path."""

        matches = tuple(item for item in self.resources if item.path == path)
        if len(matches) != 1:
            msg = f"Normative resource is not registered exactly once: {path}"
            raise ValueError(msg)
        return matches[0]


def _required(data: dict[str, object], key: str, expected_type: type[object]) -> object:
    try:
        value = data[key]
    except KeyError as exc:
        msg = f"Normative integrity field is missing: {key}"
        raise ValueError(msg) from exc
    if not isinstance(value, expected_type) or (
        expected_type is int and isinstance(value, bool)
    ):
        msg = f"Normative integrity field has the wrong type: {key}"
        raise ValueError(msg)
    return value


def _optional_string(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Normative integrity field must be a string: {key}"
        raise ValueError(msg)
    return value


def _optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Normative integrity field must be an integer: {key}"
        raise ValueError(msg)
    return value


def _resource_from_dict(data: object) -> NormativeResourceIntegrity:
    if not isinstance(data, dict):
        msg = "Normative integrity resource entry must be an object."
        raise ValueError(msg)
    item = cast("dict[str, object]", data)

    try:
        status = NormativeResourceStatus(
            cast("str", _required(item, "status", str)),
        )
    except ValueError as exc:
        msg = f"Unknown normative resource status: {item.get('status')}"
        raise ValueError(msg) from exc

    return NormativeResourceIntegrity(
        resource_id=cast("str", _required(item, "resource_id", str)),
        path=cast("str", _required(item, "path", str)),
        sha256=cast("str", _required(item, "sha256", str)),
        byte_count=cast("int", _required(item, "byte_count", int)),
        record_count=cast("int", _required(item, "record_count", int)),
        status=status,
        calculation_input=cast("bool", _required(item, "calculation_input", bool)),
        release_eligible=cast("bool", _required(item, "release_eligible", bool)),
        source_ref=cast("str", _required(item, "source_ref", str)),
        authority_status=cast("str", _required(item, "authority_status", str)),
        media_type=cast("str", _required(item, "media_type", str)),
        validator=cast("str", _required(item, "validator", str)),
        verification_method=cast(
            "str",
            _required(item, "verification_method", str),
        ),
        verification_note=cast(
            "str",
            _required(item, "verification_note", str),
        ),
        interpolation_law=_optional_string(item, "interpolation_law"),
        energy_basis=_optional_string(item, "energy_basis"),
        source_artifact_sha256=_optional_string(item, "source_artifact_sha256"),
        source_artifact_byte_count=_optional_int(item, "source_artifact_byte_count"),
        derivation_command=_optional_string(item, "derivation_command"),
    )


def load_normative_integrity_manifest() -> NormativeIntegrityManifest:
    """Load and validate the bundled data-integrity manifest."""

    manifest_resource = resources.files(_NORMATIVE_PACKAGE).joinpath(
        _INTEGRITY_MANIFEST,
    )
    loaded: object = json.loads(manifest_resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = "Normative integrity manifest root must be an object."
        raise ValueError(msg)
    data = cast("dict[str, object]", loaded)
    resource_items = _required(data, "resources", list)

    return NormativeIntegrityManifest(
        schema_version=cast("str", _required(data, "schema_version", str)),
        release_id=cast("str", _required(data, "release_id", str)),
        resources=tuple(
            _resource_from_dict(item)
            for item in cast("list[object]", resource_items)
        ),
    )


def normative_resource_sha256(path: str) -> str:
    """Return SHA256 of one safe package-relative normative resource."""

    safe_path = PurePosixPath(path)
    if safe_path.is_absolute() or ".." in safe_path.parts or not path.strip():
        msg = f"Unsafe normative resource path: {path}"
        raise ValueError(msg)
    payload = resources.files(_NORMATIVE_PACKAGE).joinpath(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def validate_normative_resource_integrity(path: str) -> NormativeResourceIntegrity:
    """Validate exact size and digest and return the registered passport."""

    entry = load_normative_integrity_manifest().by_path(path)
    payload = resources.files(_NORMATIVE_PACKAGE).joinpath(path).read_bytes()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != entry.byte_count or actual_digest != entry.sha256:
        msg = f"Normative resource integrity check failed: {path}"
        raise ValueError(msg)
    return entry


def require_release_eligible_normative_resource(
    path: str,
) -> NormativeResourceIntegrity:
    """Reject unverified or quarantined data at a production release boundary."""

    entry = validate_normative_resource_integrity(path)
    if not entry.calculation_input or not entry.release_eligible:
        msg = (
            "Normative resource is not eligible for production release: "
            f"{path} ({entry.status.value})."
        )
        raise ValueError(msg)
    return entry


__all__ = [
    "NormativeIntegrityManifest",
    "NormativeResourceIntegrity",
    "NormativeResourceStatus",
    "load_normative_integrity_manifest",
    "normative_resource_sha256",
    "require_release_eligible_normative_resource",
    "validate_normative_resource_integrity",
]
