"""Bounded source extension catalog checks; never launches native software."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from radar.validation import srim_source_catalog as baseline_module
from radar.validation import srim_source_extension as subject


D = Decimal
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline() -> tuple[baseline_module.SourceControl, ...]:
    return baseline_module.build_source_catalog(ROOT)


@pytest.fixture(scope="module")
def catalog() -> tuple[subject.ExtensionControl, ...]:
    return subject.build_extension_catalog(ROOT)


@pytest.fixture
def copied(tmp_path: Path) -> Path:
    for relative in (subject.SOURCE_CATALOG_PATH, baseline_module.MANIFEST_PATH,
                     *baseline_module.SOURCE_HASHES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def test_complete_nonshrinking_subset(
    catalog: tuple[subject.ExtensionControl, ...],
    baseline: tuple[baseline_module.SourceControl, ...],
) -> None:
    assert len(catalog) == 158
    assert len({c.ion_z for c in catalog}) == 79
    assert len({c.name for c in catalog}) == 158
    by_name = {c.name: c for c in baseline}
    omitted = set(by_name) - {c.base_source_name for c in catalog}
    assert omitted == {c.name for c in baseline if c.symbol in {"He", "Li", "Be", "B", "C"}}
    assert len(omitted) == 10
    assert len({c.ion_z for c in catalog} | {c.ion_z for c in baseline if c.name in omitted}) == 84
    for control in catalog:
        base = by_name[control.base_source_name]
        assert control.energy_min_mev == base.energy_min_mev == 1
        assert control.energy_max_mev > base.energy_max_mev == 12000
        assert control.name not in by_name
        for attr in ("ion_z", "symbol", "ion_mass_u", "target_z", "target_mass_u", "density_g_cm3"):
            assert getattr(control, attr) == getattr(base, attr)
    assert baseline_module.build_source_catalog(ROOT) == baseline


def test_pilot_exact_fixed_names_endpoints_and_counts(
    catalog: tuple[subject.ExtensionControl, ...],
) -> None:
    expected = [("f19", 18000, 113), ("ne20p2", 20000, 114),
                ("fe55p8", 55000, 125), ("u238", 225000, 141)]
    pilot = subject.pilot_extension_controls(catalog)
    assert len(pilot) == 8
    assert [c.name for c in pilot] == [f"{ion}_{m}_upper{maximum}"
                                     for ion, maximum, _ in expected for m in ("al", "si")]
    assert [c.energy_max_mev for c in pilot] == [D(e) for _, e, _ in expected for _ in (13, 14)]
    assert [len(subject.expected_extension_grid(c)) for c in pilot] == [
        n for _, _, n in expected for _ in (13, 14)]
    assert subject.pilot_extension_controls(tuple(reversed(catalog))) == pilot


def test_ceiling_selects_greatest_normal_node_without_rounding_mass(
    catalog: tuple[subject.ExtensionControl, ...],
) -> None:
    # Independent normal-grid inventory, not the new endpoint helper.
    mantissas = tuple(D(v) for v in ("1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7",
                                    "1.8", "2", "2.25", "2.5", "2.75", "3", "3.25", "3.5",
                                    "3.75", "4", "4.5", "5", "5.5", "6", "6.5", "7", "8", "9"))
    values = tuple(v * D(10)**exponent for exponent in range(7) for v in mantissas)
    for case in catalog:
        limit = case.ion_mass_u * 1000
        selected = max(v for v in values if v <= limit)
        following = min(v for v in values if v > selected)
        assert case.energy_max_mev == selected <= limit < following
        assert case.energy_max_mev * 1000 <= subject.ENGINEERING_INPUT_MAX_KEV
    by_z = {c.ion_z: c for c in catalog}
    assert by_z[10].ion_mass_u == D("20.2") and by_z[26].ion_mass_u == D("55.8")
    assert by_z[82].ion_mass_u == D("200.4")
    assert max(c.energy_max_mev for c in catalog) == 225000


def test_every_extension_preserves_complete_107_node_prefix(
    catalog: tuple[subject.ExtensionControl, ...],
    baseline: tuple[baseline_module.SourceControl, ...],
) -> None:
    by_name = {c.name: c for c in baseline}
    for case in catalog:
        grid = subject.expected_extension_grid(case)
        assert grid[:107] == baseline_module.expected_control_grid(by_name[case.base_source_name])
        assert grid[0] == case.energy_min_mev and grid[-1] == case.energy_max_mev
        assert len(set(grid)) == len(grid) > 107
        assert all(a < b for a, b in zip(grid, grid[1:]))


def test_inputs_preserve_normal_mode_and_only_extend_energy_endpoint(
    catalog: tuple[subject.ExtensionControl, ...],
    baseline: tuple[baseline_module.SourceControl, ...],
) -> None:
    by_name = {c.name: c for c in baseline}
    for case in catalog:
        raw = subject.extension_input_bytes(case)
        base_raw = baseline_module.input_bytes(by_name[case.base_source_name])
        assert raw.isascii() and raw.endswith(b"\r\n")
        assert b"\n" not in raw.replace(b"\r\n", b"")
        lines = raw.decode("ascii").splitlines()
        assert lines[:-1] == base_raw.decode("ascii").splitlines()[:-1]
        assert lines[14] == f"1000    {case.energy_max_mev * 1000}"
        assert lines[12] == "5" and lines[2] == '"srim_output.txt"'


def test_all_qualifications_remain_false(catalog: tuple[subject.ExtensionControl, ...]) -> None:
    for control in catalog:
        for flag in ("native_execution_verified", "physical_accuracy_verified",
                     "source_energy_domain_covered", "physical_mass_policy_qualified",
                     "normative_export_allowed"):
            assert getattr(control, flag) is False
        assert control.diagnostic_ceiling_mev_per_source_a == 1000
        assert control.ceiling_policy == "bounded_numerical_experiment_not_physical_validity_limit"
        assert control.energy_basis == "total_ion_kinetic_energy_MeV"


@pytest.mark.parametrize("field,value", [
    ("ion_z", True), ("target_z", True), ("ion_mass_u", 14.0),
    ("ion_mass_u", D("NaN")), ("ion_mass_u", D("Infinity")), ("ion_mass_u", D(15)),
    ("energy_min_mev", D("0.1")), ("energy_min_mev", D(0)),
    ("energy_max_mev", D(12000)), ("energy_max_mev", D(14001)),
    ("energy_max_mev", 14000), ("energy_max_mev", D("NaN")),
    ("energy_max_mev", D("Infinity")), ("energy_max_mev", D(-1)),
    ("energy_max_mev", D("1e999999")), ("energy_min_mev", D("1e-999999")),
    ("density_g_cm3", D("2.702")), ("target_mass_u", D(27)),
    ("target_z", 12), ("symbol", "H"), ("base_source_name", "../outside"),
    ("name", "../outside"), ("name", "arbitrary"),
])
def test_invalid_or_arbitrary_extension_rejected(
    catalog: tuple[subject.ExtensionControl, ...], field: str, value: Any,
) -> None:
    with pytest.raises(ValueError):
        replace(catalog[0], **{field: value})


def test_engineering_bound_is_explicit_not_overflow_cause(
    catalog: tuple[subject.ExtensionControl, ...],
) -> None:
    with pytest.raises(ValueError, match="engineering input bound; not an overflow diagnosis"):
        replace(catalog[0], energy_max_mev=D(3000000))


def test_no_shrinking_case_can_be_constructed(
    baseline: tuple[baseline_module.SourceControl, ...],
) -> None:
    for base in baseline:
        if base.symbol not in {"He", "Li", "Be", "B", "C"}:
            continue
        with pytest.raises(ValueError, match="nonshrinking"):
            subject.ExtensionControl(
                base.name, f"{base.name}_upper12000", base.ion_z, base.symbol,
                base.ion_mass_u, base.target_z, base.target_mass_u, base.density_g_cm3, D(12000),
            )


def test_controls_are_frozen(catalog: tuple[subject.ExtensionControl, ...]) -> None:
    with pytest.raises(FrozenInstanceError):
        catalog[0].name = "changed"  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError)):
        replace(catalog[0], normative_export_allowed=True)


@pytest.mark.parametrize("kind", ["subset", "duplicate", "list", "empty"])
def test_pilot_rejects_invalid_catalog(
    catalog: tuple[subject.ExtensionControl, ...], kind: str,
) -> None:
    altered: Any = {"subset": catalog[:-1], "duplicate": catalog[:-1] + (catalog[0],),
                    "list": list(catalog), "empty": ()}[kind]
    with pytest.raises(ValueError):
        subject.pilot_extension_controls(altered)


@pytest.mark.parametrize("bad", [None, {}, object(), "case"])
def test_noncontrol_cannot_format_native_input(bad: Any) -> None:
    with pytest.raises(ValueError):
        subject.extension_input_bytes(bad)
    with pytest.raises(ValueError):
        subject.expected_extension_grid(bad)


@pytest.mark.parametrize("relative", (subject.SOURCE_CATALOG_PATH, baseline_module.MANIFEST_PATH,
                                      *baseline_module.SOURCE_HASHES))
def test_source_or_parent_implementation_pin_mutation_rejected(copied: Path, relative: str) -> None:
    path = copied / relative
    path.write_bytes(path.read_bytes() + b"CHANGED")
    with pytest.raises(ValueError):
        subject.build_extension_catalog(copied)


def test_source_mutation_during_extension_is_detected(
    copied: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = subject._from_base
    changed = False

    def mutate(base: baseline_module.SourceControl) -> subject.ExtensionControl:
        nonlocal changed
        result = original(base)
        if not changed:
            changed = True
            path = copied / next(iter(baseline_module.SOURCE_HASHES))
            path.write_bytes(path.read_bytes() + b"CHANGED DURING")
        return result

    monkeypatch.setattr(subject, "_from_base", mutate)
    with pytest.raises(ValueError):
        subject.build_extension_catalog(copied)


def test_parent_module_mutation_during_extension_is_detected(
    copied: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = subject._from_base
    changed = False

    def mutate(base: baseline_module.SourceControl) -> subject.ExtensionControl:
        nonlocal changed
        result = original(base)
        if not changed:
            changed = True
            path = copied / subject.SOURCE_CATALOG_PATH
            path.write_bytes(path.read_bytes() + b"CHANGED DURING")
        return result

    monkeypatch.setattr(subject, "_from_base", mutate)
    with pytest.raises(ValueError, match="implementation changed"):
        subject.build_extension_catalog(copied)


def test_independent_decimal_context_preserves_inputs_and_grid(
    catalog: tuple[subject.ExtensionControl, ...],
) -> None:
    case = next(c for c in catalog if c.name == "u238_si_upper225000")
    expected_input = subject.extension_input_bytes(case)
    expected_grid = subject.expected_extension_grid(case)
    with localcontext() as context:
        context.prec = 2
        context.Emax = 3
        for signal in context.traps:
            context.traps[signal] = True
        assert subject.build_extension_catalog(ROOT) == catalog
        assert subject.extension_input_bytes(case) == expected_input
        assert subject.expected_extension_grid(case) == expected_grid
