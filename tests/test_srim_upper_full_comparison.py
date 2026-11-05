"""Full upper-extension inventory and conservation gates; no native execution."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import compare_srim_upper_support as subject
from radar.validation import srim_source_catalog as catalog
from test_srim_hze_transport import table


@pytest.fixture(scope="module")
def roster() -> tuple[dict[int, tuple[str, Decimal]], set[int]]:
    entries = catalog.build_source_catalog(Path(__file__).resolve().parents[1])
    ions = {r.ion_z: (r.symbol, r.ion_mass_u) for r in entries}
    sep = {r["z"] for r in catalog._source_inventory()["projectiles"]
           if "SEP" in r["source_membership"]}
    assert len(ions) == 84 and len(sep) == 24
    return ions, sep


@pytest.fixture(scope="module")
def tables(roster: tuple[dict[int, tuple[str, Decimal]], set[int]]) -> tuple[Any, Any]:
    result = []
    for nodes in (3, 4):
        result.append({z: table(z=z, mass=str(mass),
                               energies=tuple(str(mass*e) for e in (1, 2, 4, 8)[:nodes]),
                               stopping=(str(mass),)*nodes)
                       for z, (_, mass) in roster[0].items()})
    return result[0], result[1]


def make_report(roster: tuple[dict[int, tuple[str, Decimal]], set[int]],
                bundle_sha: str, *, upper: bool) -> dict[str, Any]:
    """Compact synthetic conservative measures, not a physical source model."""
    ions, sep = roster
    rows = []
    for name, selected in (("gcr", set(ions)), ("sep", sep)):
        for mode in ("peak", "cumulative"):
            for z in sorted(selected):
                symbol, mass = ions[z]
                for shield in subject.SHIELDS:
                    supported = 14. if upper or shield == 0 else 6.
                    mapped = 14. if upper else 6.
                    common = {"x": [3.], "bin_edges": [1., 5.],
                              "assessment_mode": mode, "source": name,
                              "angular_basis": "omnidirectional_4pi",
                              "temporal_basis": "source_maximum" if mode == "peak" else "mission_total",
                              "mission_duration_s": None if mode == "peak" else 123456.,
                              "species": {"symbol": symbol, "atomic_number": z,
                                          "mass_number": round(float(mass)), "is_aggregate": False}}
                    energy = {**common, "x_unit": "MeV/nucleon", "y_unit": "energy-density-test",
                              "quantity": "peak_differential_flux" if mode == "peak" else "differential_fluence",
                              "y": [supported/4.]}
                    let = {**deepcopy(common), "x_unit": "MeV cm2/mg", "y_unit": "LET-density-test",
                           "quantity": "LET-test", "y": [mapped/4.]}
                    rows.append({"source": name, "mode": mode, "z": z,
                                 "thickness_g_cm2": shield,
                                 "energy_product": {"spectrum": energy}, "let_product": {"spectrum": let},
                                 "primary_ledger": {"source_integral": 14.,
                                     "supported_exit_integral": supported, "represented_integral": supported,
                                     "outside_al_domain_integral": 14.-supported,
                                     "unresolved_low_residual_integral": 0.,
                                     "below_output_window_integral": 0., "above_output_window_integral": 0.},
                                 "let_ledger": {"mapped_integral": mapped,
                                                "outside_si_domain_integral": supported-mapped}})
    return {"schema_version": "radar.srim_hze_external_products.v1", "status": "INCOMPLETE",
            "normative_export_allowed": False, "bundle_sha256": bundle_sha,
            "mission": {"lifetime_years": 5}, "source_grid_refinement": 1,
            "scope": "synthetic_external_source_inventory_test",
            "implementation_and_input_sha256_before": {"test.py": "a"*64},
            "implementation_and_input_sha256_after": {"test.py": "a"*64},
            "source_product_count": 216, "requested_symbols": None,
            "case_count": len(rows), "cases": rows}


@pytest.fixture(scope="module")
def reports(roster: tuple[dict[int, tuple[str, Decimal]], set[int]]) -> tuple[Any, Any]:
    return (make_report(roster, hashlib.sha256(b"parent").hexdigest(), upper=False),
            make_report(roster, hashlib.sha256(b"upper").hexdigest(), upper=True))


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, roster: Any,
          tables: Any, reports: Any) -> dict[str, Any]:
    parent, upper = tmp_path / "parent.zip", tmp_path / "upper.zip"
    parent.write_bytes(b"parent")
    upper.write_bytes(b"upper")
    state = {"old": dict(tables[0]), "new": dict(tables[1])}
    def load(path: Path, *, expected_sha256: str) -> Any:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        return SimpleNamespace(tables_by_z=state["old" if path == parent else "new"])
    monkeypatch.setattr(subject, "load_srim_hze_dataset", load)
    monkeypatch.setattr(subject, "_full_roster", lambda: roster)
    return {"a": deepcopy(reports[0]), "b": deepcopy(reports[1]), "state": state,
            "roster": roster, "kwargs": {
                "parent_bundle": parent, "parent_sha": hashlib.sha256(b"parent").hexdigest(),
                "upper_bundle": upper, "upper_sha": hashlib.sha256(b"upper").hexdigest(),
                "parent_products": tmp_path / "parent.json", "upper_products": tmp_path / "upper.json"}}


def invoke(setup: dict[str, Any], **changes: Any) -> dict[str, Any]:
    args = {**setup["kwargs"], "require_full_inventory": True}
    for prefix, key in (("parent", "a"), ("upper", "b")):
        raw = json.dumps(setup[key]).encode()
        args[f"{prefix}_products"].write_bytes(raw)
        args[f"{prefix}_products_sha"] = hashlib.sha256(raw).hexdigest()
    args.update(changes)
    return subject.compare(**args)


def test_full_roster_reconstructs_exact_source_identity(roster: Any) -> None:
    assert subject._full_roster() == roster


def test_complete_84_ion_1296_case_measure_passes(setup: dict[str, Any]) -> None:
    result = invoke(setup)
    assert result["full_inventory_verified"] is True
    assert result["pilot_inventory_verified"] is False
    assert result["scope"] == "full_84_ion_upper_extension_not_orbit_acceptance"
    assert result["case_count"] == 1296 and len(result["tables"]) == 84
    assert result["normative_export_allowed"] is False
    for row in result["cases"]:
        assert row["recovered_al_source_integral"] == (0. if row["thickness_g_cm2"] == 0 else 8.)
        assert row["new_mapped_let_integral"] == 8.


def test_default_pilot_mode_does_not_accept_full_roster(setup: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        invoke(setup, require_full_inventory=False)


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_full_flag_is_strict_boolean(setup: dict[str, Any], value: Any) -> None:
    with pytest.raises(ValueError):
        invoke(setup, require_full_inventory=value)


@pytest.mark.parametrize("which", ["old", "new"])
@pytest.mark.parametrize("mutation", ["missing", "relabelled", "mass"])
def test_table_inventory_is_actual_source_roster(setup: dict[str, Any], which: str, mutation: str) -> None:
    target = setup["state"][which]
    if mutation == "missing":
        target.pop(92)
    elif mutation == "relabelled":
        target[91] = target.pop(92)
    else:
        target[2] = table(z=2, mass="5", energies=("5", "10", "20", "40"), stopping=("5",)*4)
    with pytest.raises(ValueError):
        invoke(setup)


@pytest.mark.parametrize("which", ["a", "b"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_ion", "wrong_sep", "shield", "mode"])
def test_full_case_roster_is_exact_not_just_counts(setup: dict[str, Any], which: str, mutation: str) -> None:
    report = setup[which]
    if mutation == "missing":
        report["cases"].pop()
    elif mutation == "duplicate":
        report["cases"][1] = deepcopy(report["cases"][0])
    elif mutation == "wrong_ion":
        report["cases"][0]["z"] = 91
    elif mutation == "wrong_sep":
        sep = setup["roster"][1]
        removed = min(sep)
        replacement = min(set(setup["roster"][0])-sep)
        symbol, mass = setup["roster"][0][replacement]
        for row in report["cases"]:
            if row["source"] == "sep" and row["z"] == removed:
                row["z"] = replacement
                for name in ("energy_product", "let_product"):
                    row[name]["spectrum"]["species"].update(
                        atomic_number=replacement, symbol=symbol, mass_number=round(float(mass)))
    else:
        field, value = ("thickness_g_cm2", 2.) if mutation == "shield" else ("mode", "mean")
        report["cases"][0][field] = value
    report["case_count"] = len(report["cases"])
    with pytest.raises(ValueError):
        invoke(setup)


def test_both_reports_cannot_agree_on_wrong_sep_roster(setup: dict[str, Any]) -> None:
    sep = setup["roster"][1]
    removed, replacement = min(sep), min(set(setup["roster"][0])-sep)
    symbol, mass = setup["roster"][0][replacement]
    for key in ("a", "b"):
        for row in setup[key]["cases"]:
            if row["source"] == "sep" and row["z"] == removed:
                row["z"] = replacement
                for name in ("energy_product", "let_product"):
                    row[name]["spectrum"]["species"].update(
                        atomic_number=replacement, symbol=symbol, mass_number=round(float(mass)))
    with pytest.raises(ValueError):
        invoke(setup)


@pytest.mark.parametrize("which", ["a", "b"])
@pytest.mark.parametrize("field,value", [("source_product_count", 215), ("case_count", 1295),
                                       ("requested_symbols", []), ("requested_symbols", ["He"])])
def test_full_counts_and_unrestricted_symbol_selection(setup: dict[str, Any], which: str,
                                                     field: str, value: Any) -> None:
    setup[which][field] = value
    with pytest.raises(ValueError):
        invoke(setup)


@pytest.mark.parametrize("field", ["source_integral", "represented_integral", "outside_al_domain_integral",
                                  "unresolved_low_residual_integral", "below_output_window_integral"])
def test_full_mode_still_enforces_conservation(setup: dict[str, Any], field: str) -> None:
    setup["b"]["cases"][1]["primary_ledger"][field] += 1.
    with pytest.raises(ValueError):
        invoke(setup)


def test_equal_corruption_of_both_supported_totals_is_rejected(setup: dict[str, Any]) -> None:
    for key in ("a", "b"):
        setup[key]["cases"][1]["primary_ledger"]["supported_exit_integral"] += 1.
    with pytest.raises(ValueError):
        invoke(setup)


@pytest.mark.parametrize("name", ["energy_product", "let_product"])
def test_full_mode_rejects_lost_supported_bins(setup: dict[str, Any], name: str) -> None:
    setup["b"]["cases"][1][name]["spectrum"]["y"][0] = -1.
    with pytest.raises(ValueError):
        invoke(setup)


def test_full_mode_checks_symbol_not_only_atomic_number(setup: dict[str, Any]) -> None:
    for key in ("a", "b"):
        for name in ("energy_product", "let_product"):
            setup[key]["cases"][0][name]["spectrum"]["species"]["symbol"] = "Fake"
    with pytest.raises(ValueError):
        invoke(setup)


def test_cli_passes_explicit_full_inventory_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    def fake(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"case_count": 1296, "full_inventory_verified": True}
    monkeypatch.setattr(subject, "compare", fake)
    arguments = ["compare_srim_upper_support.py"]
    for name in ("parent-bundle", "upper-bundle", "parent-products", "upper-products"):
        arguments.extend((f"--{name}", str(tmp_path / name), f"--{name}-sha256", "a"*64))
    arguments.extend(("--require-full-inventory", "--output", str(tmp_path / "out.json")))
    monkeypatch.setattr(sys, "argv", arguments)
    assert subject.main() == 0
    assert captured["require_full_inventory"] is True
