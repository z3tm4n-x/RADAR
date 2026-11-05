"""Pinned native ASTAR data parsing, with independent material-order witnesses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import io
from math import inf, nan
from pathlib import Path
import tarfile
from typing import Any

import pytest

from radar.validation.nist_astar import (
    ASTAR_ARCHIVE_SHA256, AstarMaterial, AstarRow, _extract_members, _parse_members,
    load_astar_archive,
)


ARCHIVE = Path(__file__).resolve().parents[1] / "research/nist_star_20261031/STAR.tar.gz"


@pytest.fixture(scope="module")
def members() -> dict[str, bytes]:
    return _extract_members(ARCHIVE.read_bytes())


def test_pinned_archive_complete_native_grid_and_nonrelease_flags() -> None:
    result = load_astar_archive(ARCHIVE)
    assert result.archive_sha256 == ASTAR_ARCHIVE_SHA256
    assert result.version == "1.2"
    assert len(result.materials) == 74
    assert sum(len(m.rows) for m in result.materials) == 74 * 122
    for material in result.materials:
        assert material.rows[0].energy_mev_total == 0.001
        assert material.rows[-1].energy_mev_total == 1000
    assert not result.normative_export_allowed
    assert not result.physical_accuracy_verified
    assert not result.online_version_equivalence_verified
    assert not result.fortran_execution_replayed
    assert "Derived from NIST" in result.derivative_notice
    assert len(result.member_sha256) == 5
    assert {name: hashlib.sha256(data).hexdigest()
            for name, data in _extract_members(ARCHIVE.read_bytes()).items()} == dict(result.member_sha256)


def test_idno_display_order_is_not_ind_binary_record_order(members: dict[str, bytes]) -> None:
    result = load_astar_archive(ARCHIVE)
    material_ids = tuple(m.id for m in result.materials)
    # The raw file's eighth/ninth records are Al/Si, whereas their MATS
    # display labels are ninth/tenth. Carbon906 is the final raw data record.
    assert material_ids[7:9] == (13, 14)
    assert material_ids[-1] == 906
    mats = members["ASTAR/MATS"].decode("ascii").splitlines()
    assert mats[8].strip() == "ALUMINUM"
    assert mats[9].strip() == "SILICON"
    assert "Graphite" in mats[4]
    # Independent literal control values from the corresponding FALPH lines.
    aluminum, silicon = result.material(13), result.material(14)
    assert aluminum.name == "ALUMINUM"
    assert silicon.name == "SILICON"
    assert aluminum.rows[0] == AstarRow(0.001, 55.7972, 74.6902, 9.96353e-6, 0.253809)
    assert silicon.rows[0] == AstarRow(0.001, 47.3158, 73.9634, 1.06591e-5, 0.237708)
    assert aluminum.rows[0].stopping_total_mev_cm2_g == 55.7972 + 74.6902
    assert aluminum.rows[0].range_projected_g_cm2 == 9.96353e-6 * 0.253809
    assert aluminum.rows[0].range_projected_g_cm2 < aluminum.rows[0].range_csda_g_cm2


@pytest.mark.parametrize("value", [True, 13.0, "13", 0, 999])
def test_invalid_lookup(value: Any) -> None:
    with pytest.raises(ValueError):
        load_astar_archive(ARCHIVE).material(value)


def test_artifact_is_immutable() -> None:
    result = load_astar_archive(ARCHIVE)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "version", "current")
    with pytest.raises(FrozenInstanceError):
        setattr(result.material(13).rows[0], "energy_mev_total", 10.0)
    with pytest.raises(TypeError):
        replace(result, normative_export_allowed=True)


@pytest.mark.parametrize("field", [
    "energy_mev_total", "stopping_electronic_mev_cm2_g", "stopping_nuclear_mev_cm2_g",
    "range_csda_g_cm2", "detour_factor",
])
@pytest.mark.parametrize("value", [True, False, 0.0, -1.0, inf, nan, "1.0", 10**400])
def test_row_rejects_invalid_numbers(field: str, value: Any) -> None:
    row = AstarRow(1, 2, 3, 4, 0.5)
    with pytest.raises(ValueError):
        replace(row, **{field: value})


def test_row_total_overflow_detour_bound_and_projected_underflow() -> None:
    with pytest.raises(ValueError, match="total stopping"):
        AstarRow(1, 1e308, 1e308, 4, 0.5)
    with pytest.raises(ValueError, match="exceed one"):
        AstarRow(1, 2, 3, 4, 1.00001)
    with pytest.raises(ValueError, match="projected range"):
        AstarRow(1, 2, 3, 1e-300, 1e-300)


@pytest.mark.parametrize("values", [
    (True, "Al", (AstarRow(1, 2, 3, 4, 0.5), AstarRow(2, 2, 3, 5, 0.5))),
    (13, "", (AstarRow(1, 2, 3, 4, 0.5), AstarRow(2, 2, 3, 5, 0.5))),
    (13, "Al", (AstarRow(1, 2, 3, 4, 0.5),)),
    (13, "Al", [AstarRow(1, 2, 3, 4, 0.5), AstarRow(2, 2, 3, 5, 0.5)]),
    (13, "Al", (AstarRow(2, 2, 3, 4, 0.5), AstarRow(1, 2, 3, 5, 0.5))),
    (13, "Al", (AstarRow(1, 2, 3, 4, 0.5), AstarRow(2, 2, 3, 4, 0.5))),
])
def test_material_contract(values: Any) -> None:
    with pytest.raises(ValueError):
        AstarMaterial(*values)


def _changed(members: dict[str, bytes], name: str, old: bytes, new: bytes) -> dict[str, bytes]:
    result = dict(members)
    assert old in result[name]
    result[name] = result[name].replace(old, new, 1)
    return result


@pytest.mark.parametrize("name,old,new", [
    ("ASTAR/ASTAR.f", b"Version 1.2", b"Version 2.0"),
    ("ASTAR/ASTAR.f", b"DATA IDNO/ 1,  2", b"DATA IDNO/ 1,  1"),
    ("ASTAR/ASTAR.f", b"DATA IND/  1,  2", b"DATA IND/  1,  3"),
    ("ASTAR/ASTAR.f", b"DATA NMAX/122/", b"DATA NMAX/121/"),
    ("ASTAR/ASTAR.f", b"0.0010,0.0015", b"0.0011,0.0015"),
    ("ASTAR/ASTAR.f", b"READ (7,REC=JD) STE,STN,RG,DET", b"READ (7,REC=JD) STN,STE,RG,DET"),
    ("ASTAR/ASTAR.f", b"AVPEN(L)=DETOUR(L)*RANGE(L)", b"AVPEN(L)=RANGE(L)"),
    ("ASTAR/ACONV.f", b"DATA NMAX/122/", b"DATA NMAX/123/"),
    ("ASTAR/ACONV.f", b"DO 10 J=1,74", b"DO 10 J=1,73"),
    ("ASTAR/ACONV.f", b"READ (7,*) (STE(N),N=1,NMAX)", b"READ (7,*) (STN(N),N=1,NMAX)"),
    ("ASTAR/ENG.ALF", b"122", b"121"),
    ("ASTAR/ENG.ALF", b"0.0010", b"0.0011"),
    ("ASTAR/ENG.ALF", b"0.0015", b"0.0010"),
    ("ASTAR/ENG.ALF", b"0.0010", b"NaN"),
    ("ASTAR/ENG.ALF", b"0.0010", b"1e999"),
    ("ASTAR/ENG.ALF", b"0.0010", b"\x1a0.0010"),
    ("ASTAR/MATS", b"ALUMINUM", b"ARGON"),
    ("ASTAR/MATS", b"SILICON", b"SILICON-WRONG"),
    ("ASTAR/MATS", b"HYDROGEN", b"\xffHYDROGEN"),
    ("ASTAR/FALPH", b"2.12008E+02", b"-2.12008E+02"),
    ("ASTAR/FALPH", b"2.12008E+02", b"NaN"),
    ("ASTAR/FALPH", b"2.12008E+02", b"1e999"),
])
def test_structural_and_semantic_tampering(
    members: dict[str, bytes], name: str, old: bytes, new: bytes,
) -> None:
    with pytest.raises(ValueError):
        _parse_members(_changed(members, name, old, new))


def test_swapping_converter_array_read_order_is_rejected(members: dict[str, bytes]) -> None:
    changed = dict(members)
    source = changed["ASTAR/ACONV.f"]
    source = source.replace(b"(STE(N),N=1,NMAX)", b"(TMP(N),N=1,NMAX)")
    source = source.replace(b"(STN(N),N=1,NMAX)", b"(STE(N),N=1,NMAX)")
    source = source.replace(b"(TMP(N),N=1,NMAX)", b"(STN(N),N=1,NMAX)")
    changed["ASTAR/ACONV.f"] = source
    with pytest.raises(ValueError, match="order"):
        _parse_members(changed)


def test_data_record_boundary_cannot_be_recovered_by_flattening(members: dict[str, bytes]) -> None:
    changed = dict(members)
    # Move a value from the next array to the last record of the current one.
    lines = changed["ASTAR/FALPH"].decode("ascii").splitlines()
    first_next, remaining = lines[21].split(maxsplit=1)
    lines[20] += " " + first_next
    lines[21] = remaining
    changed["ASTAR/FALPH"] = "\n".join(lines).encode("ascii")
    with pytest.raises(ValueError, match="boundary"):
        _parse_members(changed)


@pytest.mark.parametrize("operation", ["missing", "extra", "truncated", "extra_data", "blank_data"])
def test_inventory_and_data_count_rejections(members: dict[str, bytes], operation: str) -> None:
    changed = dict(members)
    if operation == "missing":
        del changed["ASTAR/MATS"]
    elif operation == "extra":
        changed["ASTAR/extra"] = b""
    elif operation == "truncated":
        changed["ASTAR/FALPH"] = changed["ASTAR/FALPH"].rsplit(b"\n", 2)[0]
    elif operation == "extra_data":
        changed["ASTAR/FALPH"] += b" 1.0\n"
    elif operation == "blank_data":
        changed["ASTAR/FALPH"] = b"\n" + changed["ASTAR/FALPH"]
    with pytest.raises(ValueError):
        _parse_members(changed)


def test_full_array_count_does_not_hide_range_or_detour_defect(members: dict[str, bytes]) -> None:
    for array_index, replacement in ((2, "1e9"), (3, "1.01")):
        changed = dict(members)
        lines = changed["ASTAR/FALPH"].decode("ascii").splitlines()
        index = array_index * 21
        _, rest = lines[index].split(maxsplit=1)
        lines[index] = replacement + " " + rest
        changed["ASTAR/FALPH"] = "\n".join(lines).encode("ascii")
        with pytest.raises(ValueError):
            _parse_members(changed)


def _archive(entries: list[tuple[str, bytes, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data)
            info.linkname = "ASTAR/FALPH" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


@pytest.mark.parametrize("name,kind", [
    ("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
    ("C:/absolute", tarfile.REGTYPE), ("ASTAR\\FALPH", tarfile.REGTYPE),
    ("link", tarfile.SYMTYPE), ("hardlink", tarfile.LNKTYPE),
    ("fifo", tarfile.FIFOTYPE),
])
def test_no_filesystem_extraction_or_unsafe_archive_members(name: str, kind: bytes) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _extract_members(_archive([(name, b"", kind)]))


def test_duplicate_member_missing_member_corrupt_archive() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _extract_members(_archive([("same", b"", tarfile.REGTYPE), ("same", b"", tarfile.REGTYPE)]))
    with pytest.raises(ValueError, match="Missing"):
        _extract_members(_archive([]))
    with pytest.raises(ValueError, match="Unreadable"):
        _extract_members(b"not gzip")


def test_public_loader_refuses_unpinned_bytes(tmp_path: Path) -> None:
    path = tmp_path / "not_official.tar.gz"
    path.write_bytes(ARCHIVE.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_astar_archive(path)


def test_changed_archive_during_load_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ARCHIVE.read_bytes()
    calls = 0

    def read_bytes(self: Path) -> bytes:
        nonlocal calls
        calls += 1
        return original if calls == 1 else original + b"changed"

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(ValueError, match="changed during"):
        load_astar_archive(ARCHIVE)
