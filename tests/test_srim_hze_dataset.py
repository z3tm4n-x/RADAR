"""Portable source-text bundle checks; no SRIM executable or external files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import pytest

from radar.shielding.srim_dataset import POLICIES, SCHEMA, load_srim_hze_dataset


def source_text(target: int) -> bytes:
    text = (Path(__file__).parent / "fixtures/srim_h_al_smoke.txt").read_bytes().decode("cp1251")
    text = re.sub(r"Ion = Hydrogen\s+\[1\] , Mass = 1,008 amu",
                  "Ion = Helium [2] , Mass = 4 amu", text)
    if target == 14:
        text = re.sub(r"Al\s+13\s+100,00\s+100,00", "Si 14 100,00 100,00", text)
    return text.encode("cp1251")


def bundle(tmp_path: Path, *, mutation: str = "") -> tuple[Path, str]:
    manifest: dict[str, Any] = {"schema_version": SCHEMA, **POLICIES, "tables": []}
    members = {}
    for z in (13, 14):
        raw = source_text(z)
        name = f"he4_{z}"
        member = f"native/{name}.txt"
        members[member] = raw
        manifest["tables"].append({"case_name": name, "z": 2, "mass_parameter_u_decimal": "4",
                                   "material_z": z, "member": member,
                                   "sha256": hashlib.sha256(raw).hexdigest()})
    if mutation == "member_hash":
        manifest["tables"][0]["sha256"] = "0" * 64
    elif mutation == "member_missing":
        members.pop("native/he4_13.txt")
    elif mutation == "pair_missing":
        members.pop("native/he4_14.txt")
        manifest["tables"].pop()
        members["unused.txt"] = b"unused"
    elif mutation == "duplicate_identity":
        manifest["tables"][1]["material_z"] = 13
    elif mutation == "mass":
        manifest["tables"][0]["mass_parameter_u_decimal"] = "4.1"
    elif mutation == "ion":
        manifest["tables"][0]["z"] = 3
    elif mutation == "ion_bool":
        manifest["tables"][0]["z"] = True
    elif mutation == "path":
        manifest["tables"][0]["member"] = "../he4.txt"
    elif mutation == "unknown":
        members["unlisted.txt"] = b"not declared"
    elif mutation == "schema":
        manifest["schema_version"] = "other"
    elif mutation in POLICIES:
        manifest[mutation] = True if isinstance(POLICIES[mutation], bool) else "changed"
    elif mutation == "mass_number":
        manifest["tables"][0]["mass_parameter_u_decimal"] = 4
    elif mutation == "extra_field":
        manifest["tables"][0]["extra"] = "not allowed"
    members["manifest.json"] = json.dumps(manifest).encode()
    if mutation == "duplicate_json":
        members["manifest.json"] = b'{"tables":[],"tables":[]}'
    elif mutation == "nan":
        members["manifest.json"] = b'{"value":NaN}'
    path = tmp_path / "bundle.zip"
    with ZipFile(path, "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_pair_and_immutable_mapping(tmp_path: Path) -> None:
    path, sha = bundle(tmp_path)
    result = load_srim_hze_dataset(path, expected_sha256=sha)
    assert result.sha256 == sha and tuple(result.tables_by_z) == (2,)
    assert not result.normative_export_allowed
    table = result.tables_by_z[2]
    assert table.mass_parameter_u == 4
    assert table.si_electronic_let[0] > 0
    with pytest.raises(TypeError):
        result.tables_by_z[3] = table  # type: ignore[index]


@pytest.mark.parametrize("mutation", [
    "member_hash", "member_missing", "pair_missing", "duplicate_identity", "mass", "ion",
    "ion_bool", "path", "unknown", "schema", "mass_number", "extra_field", "duplicate_json", "nan",
    *POLICIES,
])
def test_malformed_or_relabelled_bundle_is_not_silently_accepted(tmp_path: Path, mutation: str) -> None:
    path, sha = bundle(tmp_path, mutation=mutation)
    with pytest.raises(ValueError):
        load_srim_hze_dataset(path, expected_sha256=sha)


@pytest.mark.parametrize("sha", ["0" * 64, "A" * 64, "short", "", None])
def test_explicit_bundle_pin_required(tmp_path: Path, sha: Any) -> None:
    path, _ = bundle(tmp_path)
    with pytest.raises(ValueError):
        load_srim_hze_dataset(path, expected_sha256=sha)


def test_removed_material_declared_inventory_rejected(tmp_path: Path) -> None:
    path, _ = bundle(tmp_path)
    with ZipFile(path) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["tables"].pop()
    # Two different Al-only ions make a valid-sized ZIP without Si pairs.
    raw = source_text(13).replace(b"Helium [2] , Mass = 4", b"Lithium [3] , Mass = 7")
    members = {"native/he4_13.txt": members["native/he4_13.txt"], "native/li7_13.txt": raw}
    manifest["tables"].append({"case_name": "li7_13", "z": 3, "mass_parameter_u_decimal": "7",
                               "material_z": 13, "member": "native/li7_13.txt",
                               "sha256": hashlib.sha256(raw).hexdigest()})
    members["manifest.json"] = json.dumps(manifest).encode()
    incomplete = tmp_path / "missing_si.zip"
    with ZipFile(incomplete, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="lacks its Al/Si"):
        load_srim_hze_dataset(incomplete, expected_sha256=hashlib.sha256(incomplete.read_bytes()).hexdigest())
