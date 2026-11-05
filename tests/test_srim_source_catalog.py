"""Fixed source campaign validation; no SRIM process or production writes."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from radar.validation import srim_source_catalog as subject


D = Decimal
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog() -> tuple[subject.SourceControl, ...]:
    return subject.build_source_catalog(ROOT)


@pytest.fixture
def copied(tmp_path: Path) -> Path:
    for relative in (subject.MANIFEST_PATH, *subject.SOURCE_HASHES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def test_complete_exact_source_roster(catalog: tuple[subject.SourceControl, ...]) -> None:
    assert len(catalog) == 168
    assert len({c.ion_z for c in catalog}) == 84
    assert {(c.ion_z, c.target_z) for c in catalog} == {
        (z, material) for z in (*range(2, 84), 90, 92) for material in (13, 14)}
    assert len({c.name for c in catalog}) == 168
    masses = {c.symbol: c.ion_mass_u for c in catalog}
    for symbol, mass in (("Ne", "20.2"), ("Fe", "55.8"), ("Ni", "58.7"),
                         ("Sm", "145"), ("Pb", "200.4"), ("Bi", "207.2")):
        assert masses[symbol] == D(mass)
    for control in catalog:
        assert control.energy_min_mev == 1 and control.energy_max_mev == 12000
        assert not control.source_energy_domain_covered
        assert not control.physical_mass_policy_qualified
        assert not control.normative_export_allowed


def test_pilot_is_fixed_fractional_and_missing_species_subset(
    catalog: tuple[subject.SourceControl, ...],
) -> None:
    pilot = subject.pilot_controls(catalog)
    assert len(pilot) == 8
    assert [c.name for c in pilot] == [
        "f19_al", "f19_si", "ne20p2_al", "ne20p2_si",
        "fe55p8_al", "fe55p8_si", "u238_al", "u238_si",
    ]
    assert subject.pilot_controls(tuple(reversed(catalog))) == pilot
    assert all(c in catalog for c in pilot)


def test_grid_is_complete_and_not_per_nucleon(catalog: tuple[subject.SourceControl, ...]) -> None:
    grids = [subject.expected_control_grid(c) for c in catalog]
    assert all(g == grids[0] for g in grids)
    grid = grids[0]
    assert len(grid) == 107
    assert grid[:5] == (D(1), D("1.1"), D("1.2"), D("1.3"), D("1.4"))
    assert grid[-3:] == (D(10000), D(11000), D(12000))
    assert all(a < b for a, b in zip(grid, grid[1:]))


def test_input_materials_units_names_and_exact_decimal_mass(
    catalog: tuple[subject.SourceControl, ...],
) -> None:
    for control in subject.pilot_controls(catalog):
        raw = subject.input_bytes(control)
        assert raw.isascii() and raw.endswith(b"\r\n")
        assert b"\n" not in raw.replace(b"\r\n", b"")
        lines = raw.decode("ascii").splitlines()
        assert lines[2] == '"srim_output.txt"'
        assert lines[4] == f"{control.ion_z}   {control.ion_mass_u}"
        assert lines[6] == f"0    {control.density_g_cm3}    1"
        assert lines[8] == "1" and lines[12] == "5"
        assert lines[14] == "1000    12000000"
        assert '"Aluminum"' in lines[10] if control.target_z == 13 else '"Silicon"' in lines[10]


@pytest.mark.parametrize("field,value", [
    ("ion_z", True), ("target_z", True), ("ion_mass_u", 4.0),
    ("ion_mass_u", D("NaN")), ("ion_mass_u", D("Infinity")),
    ("ion_mass_u", D(5)), ("energy_min_mev", D("0.1")),
    ("energy_max_mev", D(45000)), ("density_g_cm3", D("2.702")),
    ("target_mass_u", D(27)), ("target_z", 12), ("symbol", "H"),
    ("name", "../outside"), ("name", "arbitrary"),
])
def test_arbitrary_controls_are_rejected(
    catalog: tuple[subject.SourceControl, ...], field: str, value: Any,
) -> None:
    with pytest.raises(ValueError):
        replace(catalog[0], **{field: value})


def test_controls_are_immutable(catalog: tuple[subject.SourceControl, ...]) -> None:
    with pytest.raises(FrozenInstanceError):
        catalog[0].name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("kind", ["subset", "duplicate", "list"])
def test_pilot_rejects_invalid_catalog(
    catalog: tuple[subject.SourceControl, ...], kind: str,
) -> None:
    altered: Any = catalog[:-1] if kind == "subset" else (
        catalog[:-1] + (catalog[0],) if kind == "duplicate" else list(catalog))
    with pytest.raises(ValueError):
        subject.pilot_controls(altered)


def test_context_does_not_change_catalog_grid_or_input(catalog: tuple[subject.SourceControl, ...]) -> None:
    raw = subject.input_bytes(catalog[0])
    grid = subject.expected_control_grid(catalog[0])
    with localcontext() as context:
        context.prec = 2
        context.Emax = 3
        for signal in context.traps:
            context.traps[signal] = True
        again = subject.build_source_catalog(ROOT)
        assert again == catalog
        assert subject.input_bytes(again[0]) == raw
        assert subject.expected_control_grid(again[0]) == grid


@pytest.mark.parametrize("relative", (subject.MANIFEST_PATH, *subject.SOURCE_HASHES))
def test_pinned_input_mutation_is_rejected(copied: Path, relative: str) -> None:
    path = copied / relative
    path.write_bytes(path.read_bytes() + b"CHANGED")
    with pytest.raises(ValueError):
        subject.build_source_catalog(copied)


@pytest.mark.parametrize("field", [
    "projectiles", "source_rows", "declared_zero_sources", "source_counts",
    "source_files_sha256_before", "source_files_sha256_after", "normative_export_allowed",
    "mass_parameter_policy", "physical_mass_policy_qualified",
])
def test_repacked_manifest_still_requires_complete_current_inventory(
    copied: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    path = copied / subject.MANIFEST_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if field in ("projectiles", "source_rows", "declared_zero_sources"):
        manifest[field] = manifest[field][:-1]
    elif field == "source_counts":
        manifest[field]["GCR"]["positive_hze"] = 83
    elif field.startswith("source_files"):
        manifest[field].pop(next(iter(manifest[field])))
    else:
        manifest[field] = True
    payload = json.dumps(manifest).encode("utf-8")
    path.write_bytes(payload)
    monkeypatch.setattr(subject, "MANIFEST_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError):
        subject.build_source_catalog(copied)


@pytest.mark.parametrize("field", ["mass_number", "c"])
def test_changed_loaded_source_parameters_are_not_hidden_by_disk_pins(
    copied: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    original = getattr(subject, "get_ost_gcr_ion_parameters")

    def altered() -> Any:
        values = original()
        values["Fe"] = replace(values["Fe"], **{field: getattr(values["Fe"], field) + 1})
        return values

    monkeypatch.setattr(subject, "get_ost_gcr_ion_parameters", altered)
    with pytest.raises(ValueError):
        subject.build_source_catalog(copied)


def test_mutation_during_reconstruction_is_detected(copied: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = subject._source_inventory

    def mutate() -> dict[str, Any]:
        result = original()
        path = copied / next(iter(subject.SOURCE_HASHES))
        path.write_bytes(path.read_bytes() + b"CHANGED DURING")
        return result

    monkeypatch.setattr(subject, "_source_inventory", mutate)
    with pytest.raises(ValueError, match="during catalog"):
        subject.build_source_catalog(copied)


@pytest.mark.parametrize("bad", [None, {}, object()])
def test_noncontrol_values_rejected_before_formatting(bad: Any) -> None:
    with pytest.raises(ValueError):
        subject.input_bytes(bad)
    with pytest.raises(ValueError):
        subject.expected_control_grid(bad)
