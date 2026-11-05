"""Load an explicit, hash-pinned bundle of observed SRIM text outputs.

No executable is shipped or launched, no files are extracted, and the default
normative resources are not replaced. Incomplete numerical support is retained
by the spectral operator; a corrupt archive is an error, not a zero spectrum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
from zipfile import ZipFile

from radar.shielding.srim_hze import SrimIonStoppingTable
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output

SCHEMA = "radar.srim_hze_observed_bundle.v1"
POLICIES = {
    "status": "INCOMPLETE", "normative_export_allowed": False,
    "physical_accuracy_verified": False,
    "source_mass_policy": "literal_source_A_numeric_transport_mass",
    "al_range_policy": "analytic_loglog_total_stopping_integral_anchored_at_first_energy",
    "si_let_component": "electronic",
    "energy_support_policy": "no_external_extrapolation_no_implicit_source_truncation",
}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate bundle field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite bundle field: {value}")


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("A lowercase SHA256 is required.")
    return value


@dataclass(frozen=True)
class SrimHzeDataset:
    sha256: str
    tables_by_z: Mapping[int, SrimIonStoppingTable]
    manifest: Mapping[str, Any]
    normative_export_allowed: bool = field(default=False, init=False)


def load_srim_hze_dataset(path: Path, *, expected_sha256: str) -> SrimHzeDataset:
    """Require complete Al/Si pairs; a missing requested ion is handled by the batch API."""
    _digest(expected_sha256)
    if path.stat().st_size > 64_000_000:
        raise ValueError("SRIM bundle exceeds the supported size.")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("SRIM bundle SHA256 mismatch.")
    with ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if (not 3 <= len(entries) <= 185 or len(set(names)) != len(names)
                or "manifest.json" not in names
                or any(item.is_dir() or item.file_size > 1_000_000 for item in entries)
                or sum(item.file_size for item in entries) > 32_000_000):
            raise ValueError("Unexpected SRIM archive members or size.")
        manifest = json.loads(archive.read("manifest.json"), object_pairs_hook=_unique,
                              parse_constant=_reject_constant)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA:
            raise ValueError("Unexpected SRIM bundle schema.")
        for key, value in POLICIES.items():
            actual = manifest.get(key)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"Changed SRIM bundle policy: {key}")
        records = manifest.get("tables")
        if not isinstance(records, list) or not records or len(records) > 182:
            raise ValueError("Expected explicit bounded SRIM table inventory.")
        observed: dict[tuple[int, int], SrimRangeOutput] = {}
        used = {"manifest.json"}
        for record in records:
            fields = {"case_name", "z", "mass_parameter_u_decimal", "material_z", "member", "sha256"}
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError("Malformed SRIM table identity.")
            name, member, sha = record["case_name"], record["member"], _digest(record["sha256"])
            z, material = record["z"], record["material_z"]
            if (not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None
                    or member != f"native/{name}.txt" or member in used
                    or type(z) is not int or not 2 <= z <= 92
                    or type(material) is not int or material not in (13, 14)
                    or (z, material) in observed):
                raise ValueError("Duplicate, unsafe or inconsistent SRIM identity.")
            mass_text = record["mass_parameter_u_decimal"]
            if (not isinstance(mass_text, str) or len(mass_text) > 40
                    or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", mass_text) is None):
                raise ValueError("Literal decimal source mass required.")
            if member not in names:
                raise ValueError("Declared native SRIM member is missing.")
            native_bytes = archive.read(member)
            if hashlib.sha256(native_bytes).hexdigest() != sha:
                raise ValueError("Raw SRIM member SHA256 mismatch.")
            native = parse_srim_range_output(native_bytes)
            if (native.ion_z != z or native.target_z != material
                    or native.ion_mass_u != Decimal(mass_text)):
                raise ValueError("SRIM manifest does not match native metadata.")
            observed[z, material] = native
            used.add(member)
        if used != set(names):
            raise ValueError("Undeclared members in SRIM bundle.")
    ions = sorted({z for z, _ in observed})
    if any((z, target) not in observed for z in ions for target in (13, 14)):
        raise ValueError("A requested SRIM ion lacks its Al/Si pair.")
    tables = {z: SrimIonStoppingTable.from_native(observed[z, 13], observed[z, 14]) for z in ions}
    return SrimHzeDataset(expected_sha256, MappingProxyType(tables), MappingProxyType(manifest))
