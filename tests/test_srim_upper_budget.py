"""Upper-source count budgets: finite-measure arithmetic, identities and guards."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import audit_srim_upper_budget as subject


@pytest.mark.parametrize("cut,expected", [(0., 24.), (.5, 24.), (1., 24.),
                                         (2., 21.), (3., 16.), (4., 9.), (5., 0.), (6., 0.)])
def test_linear_density_exact_integral(cut: float, expected: float) -> None:
    assert subject.tail((1., 3., 5.), (2., 6., 10.), cut) == expected


def test_constant_density_no_unobserved_outer_tail() -> None:
    assert subject.tail((1., 10., 100.), (2., 2., 2.), 3.) == 194.
    assert subject.tail((1., 10., 100.), (0., 0., 0.), 0.) == 0.


@pytest.mark.parametrize("x,y,lower", [
    ((1.,), (1.,), 0.), ((1., 2.), (1.,), 0.),
    ((1., 1.), (1., 1.), 0.), ((2., 1.), (1., 1.), 0.),
    ((0., 1.), (1., 1.), 0.), ((-1., 1.), (1., 1.), 0.),
    ((1., float("nan")), (1., 1.), 0.), ((1., float("inf")), (1., 1.), 0.),
    ((1., 2.), (1., -1.), 0.), ((1., 2.), (1., float("nan")), 0.),
    ((1., 2.), (1., float("inf")), 0.), ((1., 2.), (1., 1.), -1.),
    ((1., 2.), (1., 1.), float("inf")), ((1., 2.), (1., 1.), float("nan")),
])
def test_invalid_finite_measure(x: tuple[float, ...], y: tuple[float, ...], lower: float) -> None:
    with pytest.raises(ValueError):
        subject.tail(x, y, lower)


@pytest.mark.parametrize("mass,endpoint", [("4", "20000"), ("6.9", "32500"),
    ("19", "90000"), ("55.8", "275000"), ("58.7", "275000"),
    ("200.4", "1000000"), ("232", "1100000"), ("238", "1100000")])
def test_prospective_normal_node_not_fabricated_energy(mass: str, endpoint: str) -> None:
    value = subject.prospective_upper(Decimal(mass))
    assert value == Decimal(endpoint)
    assert value <= 5000 * Decimal(mass)
    assert value * 1000 < 2147483647


@pytest.mark.parametrize("mass", [Decimal(0), Decimal(-1), Decimal("1.99"), Decimal(239),
                                  Decimal("NaN"), Decimal("Infinity"), 4, 4., True])
def test_prospective_requires_supported_literal_decimal(mass: Any) -> None:
    with pytest.raises(ValueError):
        subject.prospective_upper(mass)


def test_prospective_does_not_depend_on_callers_decimal_precision() -> None:
    with localcontext() as context:
        context.prec = 3
        assert subject.prospective_upper(Decimal("55.8")) == Decimal(275000)
        assert context.prec == 3


def _row(source: str, mode: str, z: int) -> dict[str, Any]:
    symbol = "He" if z == 2 else f"Z{z}"
    peak = mode == "peak"
    spectrum = {
        "x": [1., 1000., 20000.], "y": [2., 2., 2.],
        "x_unit": "MeV/nucleon",
        "y_unit": "cm^-2 s^-1 (MeV/nucleon)^-1" if peak else "cm^-2 (MeV/nucleon)^-1",
        "quantity": "maximum_differential_flux" if peak else "differential_fluence",
        "ordinate_basis": "point_sample", "bin_edges": None,
        "assessment_mode": mode, "source": source, "particle": "hze",
        "angular_basis": "omnidirectional_4pi", "axis_kind": "kinetic_energy_per_nucleon",
        "temporal_basis": "source_maximum" if peak else "mission_total",
        "mission_duration_s": None if peak else 999999.,
        "species": {"symbol": symbol, "atomic_number": z, "mass_number": 2*z,
                    "is_aggregate": False},
        "transformation_chain": [{"name": "test_source", "version": "test", "parameters":
                                  [["source_tabulated_mass_parameter", str(2*z)]]}],
    }
    return {"source": source, "mode": mode, "z": z, "symbol": symbol,
            "thickness_g_cm2": 0., "energy_product": {"spectrum": spectrum}}


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"mock authenticated complete native bundle")
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    cases = [_row(source, mode, z)
             for source, roster in (("gcr", range(2, 86)), ("sep", range(2, 26)))
             for mode in ("peak", "cumulative") for z in roster]
    report = {"schema_version": "radar.srim_hze_external_products.v1", "bundle_sha256": sha,
              "source_product_count": 216, "requested_symbols": None,
              "normative_export_allowed": False, "status": "INCOMPLETE", "cases": cases,
              "implementation_and_input_sha256_before": {"code.py": "a"*64},
              "implementation_and_input_sha256_after": {"code.py": "a"*64}}
    tables = {z: SimpleNamespace(mass_parameter_u_decimal=Decimal(2*z), al_energy=(.1, 1500.))
              for z in range(2, 86)}
    def loader(path: Path, *, expected_sha256: str) -> SimpleNamespace:
        assert path == bundle and expected_sha256 == sha
        return SimpleNamespace(tables_by_z=tables)
    monkeypatch.setattr(subject, "load_srim_hze_dataset", loader)
    return {"bundle": bundle, "bundle_sha": sha, "products": tmp_path / "products.json",
            "report": report, "tables": tables}


def _audit(setup: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    raw = json.dumps(setup["report"], allow_nan=True).encode()
    setup["products"].write_bytes(raw)
    return subject.audit(setup["bundle"], setup["bundle_sha"], setup["products"],
                         hashlib.sha256(raw).hexdigest() if sha is None else sha)


def test_complete_mock_report_has_216_finite_count_budgets(setup: dict[str, Any]) -> None:
    result = _audit(setup)
    assert len(result["cases"]) == 216
    assert len(result["summaries"]) == 4
    assert result["status"] == "INCOMPLETE"
    assert not result["normative_export_allowed"]
    assert not result["native_extension_calculated"]
    assert not result["dose_or_see_error_bound"]
    for row in result["cases"]:
        assert row["source_integral"] == 39998.
        assert row["declared_source_bounds"] == [1., 20000.]
        assert row["tail_integrals"]["current_native_upper"] == 37000.
        assert row["tail_integrals"]["above_5000_MeV_per_source_A"] == 30000.
        assert row["tail_fractions"]["current_native_upper"] == 37000/39998
    for group in result["summaries"]:
        count = 84 if group["source"] == "gcr" else 24
        assert group["source_integral"] == count*39998.
        assert group["helium_fraction_of_each_tail"]["current_native_upper"] == 1/count


def test_peak_and_cumulative_are_not_rescaled_by_duration(setup: dict[str, Any]) -> None:
    result = _audit(setup)
    rows = {(r["source"], r["mode"], r["z"]): r for r in result["cases"]}
    for source, z in (("gcr", 2), ("gcr", 85), ("sep", 25)):
        assert rows[source, "peak", z]["tail_integrals"] == rows[source, "cumulative", z]["tail_integrals"]


def test_source_report_fingerprint_rejected(setup: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        _audit(setup, "0"*64)


@pytest.mark.parametrize("key,value", [
    ("schema_version", "other"), ("bundle_sha256", "b"*64), ("source_product_count", 215),
    ("requested_symbols", ["He"]), ("normative_export_allowed", True),
    ("implementation_and_input_sha256_after", {"code.py": "b"*64}),
])
def test_invalid_report_contract(setup: dict[str, Any], key: str, value: Any) -> None:
    setup["report"][key] = value
    with pytest.raises(ValueError):
        _audit(setup)


@pytest.mark.parametrize("key,value", [
    ("x_unit", "MeV"), ("y_unit", "cm^-2 (MeV/nucleon)^-1"),
    ("quantity", "differential_fluence"), ("ordinate_basis", "bin_average"),
    ("assessment_mode", "cumulative"), ("angular_basis", "directional"),
    ("axis_kind", "kinetic_energy"), ("x", [1., 1., 20000.]),
    ("x", [True, 1000., 20000.]), ("x", [1., float("nan"), 20000.]),
    ("y", [2., -1., 2.]), ("y", [2., float("inf"), 2.]), ("y", [0., 0., 0.]),
    ("y", [2., "2", 2.]),
])
def test_invalid_spectrum_contract(setup: dict[str, Any], key: str, value: Any) -> None:
    setup["report"]["cases"][0]["energy_product"]["spectrum"][key] = value
    with pytest.raises(ValueError):
        _audit(setup)


@pytest.mark.parametrize("key,value", [("source", "erb"), ("mode", "mean")])
def test_bad_source_identity(setup: dict[str, Any], key: str, value: Any) -> None:
    setup["report"]["cases"][0][key] = value
    with pytest.raises(ValueError):
        _audit(setup)


def test_duplicate_zero_shield_identity(setup: dict[str, Any]) -> None:
    setup["report"]["cases"].append(deepcopy(setup["report"]["cases"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        _audit(setup)


@pytest.mark.parametrize("source,mode", [("gcr", "peak"), ("gcr", "cumulative"),
                                       ("sep", "peak"), ("sep", "cumulative")])
def test_missing_source_mode_member(setup: dict[str, Any], source: str, mode: str) -> None:
    cases = setup["report"]["cases"]
    cases.remove(next(c for c in cases if c["source"] == source and c["mode"] == mode))
    with pytest.raises(ValueError):
        _audit(setup)


def test_missing_native_pair(setup: dict[str, Any]) -> None:
    setup["tables"].pop(85)
    with pytest.raises(ValueError):
        _audit(setup)


def test_unknown_lower_source_is_not_called_upper_tail(setup: dict[str, Any]) -> None:
    setup["tables"][2].al_energy = (2., 1500.)
    with pytest.raises(ValueError, match="below native"):
        _audit(setup)


def test_bundle_changed_during_audit_fails(setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = subject.prospective_upper
    def change(mass: Decimal) -> Decimal:
        setup["bundle"].write_bytes(b"modified")
        return original(mass)
    monkeypatch.setattr(subject, "prospective_upper", change)
    with pytest.raises(ValueError, match="inputs changed"):
        _audit(setup)


def test_products_changed_during_audit_fails(setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = subject.prospective_upper
    def change(mass: Decimal) -> Decimal:
        setup["products"].write_bytes(b"modified")
        return original(mass)
    monkeypatch.setattr(subject, "prospective_upper", change)
    with pytest.raises(ValueError, match="inputs changed"):
        _audit(setup)


@pytest.mark.parametrize("field,value", [
    ("particle", "proton"), ("source", "sep"), ("species", None),
    ("bin_edges", [1., 1000., 20000.]),
])
def test_source_spectrum_identity_guard(setup: dict[str, Any], field: str, value: Any) -> None:
    setup["report"]["cases"][0]["energy_product"]["spectrum"][field] = value
    with pytest.raises(ValueError, match="identity|convention"):
        _audit(setup)


@pytest.mark.parametrize("field,value", [("atomic_number", 3), ("symbol", "Li"),
                                       ("is_aggregate", True)])
def test_individual_species_must_match_row(setup: dict[str, Any], field: str, value: Any) -> None:
    setup["report"]["cases"][0]["energy_product"]["spectrum"]["species"][field] = value
    with pytest.raises(ValueError, match="identity"):
        _audit(setup)


@pytest.mark.parametrize("mass", ["3.999", "5", "NaN", "Infinity", "invalid"])
def test_source_mass_must_match_native_literal(setup: dict[str, Any], mass: str) -> None:
    setup["report"]["cases"][0]["energy_product"]["spectrum"]["transformation_chain"][0]["parameters"][0][1] = mass
    with pytest.raises(ValueError):
        _audit(setup)


@pytest.mark.parametrize("conflict", [False, True])
def test_absent_or_conflicting_mass_trace(setup: dict[str, Any], conflict: bool) -> None:
    spectrum = setup["report"]["cases"][0]["energy_product"]["spectrum"]
    if conflict:
        spectrum["transformation_chain"][0]["parameters"].append(["source_tabulated_mass_parameter", "5"])
    else:
        spectrum["transformation_chain"] = []
    with pytest.raises(ValueError, match="mass trace"):
        _audit(setup)


def test_numeric_mass_trace_is_not_confused_with_text_formatting(setup: dict[str, Any]) -> None:
    setup["report"]["cases"][0]["energy_product"]["spectrum"]["transformation_chain"][0]["parameters"][0][1] = "4.0000000000000000"
    assert len(_audit(setup)["cases"]) == 216


def test_audit_per_mass_cutoff_ignores_callers_decimal_precision(setup: dict[str, Any]) -> None:
    original = _audit(setup)
    with localcontext() as context:
        context.prec = 3
        changed = _audit(setup)
        assert context.prec == 3
    assert changed["cases"] == original["cases"]
    assert changed["summaries"] == original["summaries"]
