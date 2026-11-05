"""Synthetic exact-node overlap witnesses; no native stopping calculator is run."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.validation.srim_output import SrimRangeRow


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_srim_source_overlap.py"
    spec = importlib.util.spec_from_file_location("srim_source_overlap_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(energy: str, electronic: str = "1.2345", nuclear: str = "0.0234") -> SrimRangeRow:
    return SrimRangeRow(
        Decimal(energy), Decimal(electronic), Decimal(nuclear), Decimal(1), Decimal(2),
        Decimal(3), (energy, "MeV", electronic, nuclear, "1", "cm", "2", "cm", "3", "cm"),
    )


def _table(*, z: int = 26, mass: str = "55.8", target: int = 13,
           energies: tuple[str, ...] = ("1", "10", "100"),
           electronic: str = "1.2345", nuclear: str = "0.0234") -> Any:
    return SimpleNamespace(
        ion_z=z, ion_mass_u=Decimal(mass), target_z=target,
        stopping_unit="MeV/(mg/cm2)", energy_basis="total_ion_kinetic_energy",
        raw_sha256=hashlib.sha256(f"synthetic-{z}-{mass}-{target}".encode()).hexdigest(),
        density_g_cm3=Decimal("2.70" if target == 13 else "2.329"),
        rows=tuple(_row(e, electronic, nuclear) for e in energies),
    )


def _native_rows(runner: ModuleType, selected: dict[str, Any],
                 *, scale: float = 1.0) -> list[dict[str, Any]]:
    result = []
    for request in runner.build_inventory(selected)["requests"]:
        native = selected[request["case_name"]].rows[request["native_row_index"]]
        electronic = float(Fraction(native.electronic_mev_cm2_mg) * 1000) * scale
        nuclear = float(Fraction(native.nuclear_mev_cm2_mg) * 1000) * scale
        result.append({
            **request, "stopping_electronic_mev_cm2_g": electronic,
            "stopping_nuclear_mev_cm2_g": nuclear,
            "stopping_nonradiative_mev_cm2_g": electronic + nuclear,
            "stopping_total_mev_cm2_g": electronic + nuclear,
            "stopping_radiative_mev_cm2_g": 0.0,
        })
    return result


def test_literal_fractional_mass_and_unit_conversion_without_extra_a(runner: ModuleType) -> None:
    selected = {"fe": _table()}
    result = runner.compare_rows(selected, _native_rows(runner, selected))
    assert len(result) == 3
    row = result[0]
    assert row["mass_parameter_u_decimal"] == "55.8"
    assert row["energy_total_mev_decimal"] == "1"
    assert row["energy_mev_per_nucleon"] == float(Fraction(5, 279))
    assert row["components"]["electronic"]["srim_mev_cm2_g"] == 1234.5
    assert row["components"]["nuclear"]["srim_mev_cm2_g"] == 23.4
    assert row["components"]["nonradiative"]["srim_mev_cm2_g"] == 1257.9
    for component in runner.COMPONENTS:
        assert abs(row["components"][component]["catima_over_srim_minus_one"]) < 1e-15
    assert row["srim_printed_tokens"] == list(selected["fe"].rows[0].printed_tokens)


def test_exact_fraction_thresholds_not_rounded_binary_energy(runner: ModuleType) -> None:
    selected = {"mass100.1": _table(z=40, mass="100.1", energies=(
        "1.0009999999999999999999999999999999999999", "1.001",
        "300300", "300300.000000000000000000000000000000001",
    ))}
    result = runner.build_inventory(selected)
    assert result["native_row_count"] == 4
    assert result["matched_row_count"] == 2
    assert result["excluded_row_count"] == 2
    assert [r["native_row_index"] for r in result["requests"]] == [1, 2]
    assert [r["reason"] for r in result["excluded_native_rows"]] == [
        "below_declared_candidate_overlap", "above_declared_candidate_overlap",
    ]
    assert [r["energy_mev_per_nucleon"] for r in result["requests"]] == [.01, 3000.]


def test_partition_keeps_every_native_row_and_whole_table_identity(runner: ModuleType) -> None:
    below = tuple(str(Decimal(i) / Decimal(1000)) for i in range(1, 252))
    energies = (*below, "2.38", "238", "714000")
    selected = {f"u_{z}": _table(z=92, mass="238", target=z, energies=energies)
                for z in (13, 14)}
    result = runner.build_inventory(selected)
    assert result["excluded_row_count"] == 502
    assert result["matched_row_count"] == 6
    assert result["native_row_count"] == 508
    observed = {(r["case_name"], r["native_row_index"]) for r in
                (*result["requests"], *result["excluded_native_rows"])}
    assert len(observed) == 508
    assert observed == {(name, index) for name in selected for index in range(254)}
    for table in result["tables"]:
        assert table["native_row_count"] == 254
        assert table["matched_row_count"] == 3
        assert table["excluded_row_count"] == 251
        assert table["raw_sha256"] == selected[table["case_name"]].raw_sha256


def test_inventory_order_is_canonical_not_mapping_order(runner: ModuleType) -> None:
    selected = {"fe_si": _table(target=14), "he": _table(z=2, mass="4"), "fe": _table()}
    left = runner.build_inventory(selected)
    right = runner.build_inventory(dict(reversed(list(selected.items()))))
    assert left == right
    assert [t["case_name"] for t in left["tables"]] == ["he", "fe", "fe_si"]
    assert [r["native_row_index"] for r in left["requests"]] == [0, 1, 2] * 3


@pytest.mark.parametrize("field,value", [
    ("ion_z", 1), ("target_z", 12), ("target_z", 15),
    ("stopping_unit", "MeV/(g/cm2)"), ("energy_basis", "MeV/u"),
])
def test_inventory_rejects_wrong_identity_units(runner: ModuleType, field: str, value: Any) -> None:
    table = _table()
    setattr(table, field, value)
    with pytest.raises(ValueError):
        runner.build_inventory({"bad": table})


def test_duplicate_groups_are_not_merged(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="Repeated"):
        runner.build_inventory({"one": _table(), "two": _table()})


@pytest.mark.parametrize("selected", [{}, {"none": _table(mass="238", energies=("1",))}])
def test_empty_or_entirely_excluded_inventory_fails(runner: ModuleType,
                                                 selected: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        runner.build_inventory(selected)


@pytest.mark.parametrize("energy,expected", [
    (Fraction(1, 100), "[0.01,1)"), (Fraction(999, 1000), "[0.01,1)"),
    (Fraction(1), "[1,10)"), (Fraction(10), "[10,30)"),
    (Fraction(30), "[30,100)"), (Fraction(100), "[100,1000)"),
    (Fraction(1000), "[1000,3000]"), (Fraction(3000), "[1000,3000]"),
])
def test_half_open_energy_bands(runner: ModuleType, energy: Fraction, expected: str) -> None:
    assert runner.energy_band(energy) == expected


@pytest.mark.parametrize("energy", [Fraction(0), Fraction(1, 101), Fraction(30001, 10)])
def test_energy_bands_reject_outside_domain(runner: ModuleType, energy: Fraction) -> None:
    with pytest.raises(ValueError):
        runner.energy_band(energy)


@pytest.mark.parametrize("field,value", [
    ("case_name", "other"), ("raw_sha256", "0" * 64), ("native_row_index", 1),
    ("z", 27), ("mass_parameter_u_decimal", "56"), ("material_z", 14),
    ("energy_total_mev_decimal", "1.0"), ("energy_mev_per_nucleon", 1.),
])
def test_native_request_identity_cannot_change(runner: ModuleType, field: str, value: Any) -> None:
    selected = {"fe": _table()}
    rows = _native_rows(runner, selected)
    rows[0][field] = value
    with pytest.raises(ValueError, match="identity/order"):
        runner.compare_rows(selected, rows)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "not_list",
                                      "missing_field", "extra_field", "not_object"])
def test_native_row_count_order_and_schema(runner: ModuleType, mutation: str) -> None:
    selected = {"fe": _table()}
    rows: Any = _native_rows(runner, selected)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(deepcopy(rows[0]))
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "not_list":
        rows = tuple(rows)
    elif mutation == "missing_field":
        rows[0].pop("stopping_total_mev_cm2_g")
    elif mutation == "extra_field":
        rows[0]["range_g_cm2"] = 1.
    else:
        rows[0] = None
    with pytest.raises(ValueError):
        runner.compare_rows(selected, rows)


@pytest.mark.parametrize("field,value", [
    ("electronic", float("nan")), ("nuclear", float("inf")),
    ("nonradiative", float("-inf")), ("total", -1.), ("radiative", -1.),
    ("electronic", 0.), ("total", True), ("nuclear", "1"),
    ("radiative", .001), ("nonradiative", 999.), ("total", 999.),
])
def test_invalid_stopping_or_decomposition_rejected(runner: ModuleType,
                                                   field: str, value: Any) -> None:
    selected = {"fe": _table()}
    rows = _native_rows(runner, selected)
    rows[0][f"stopping_{field}_mev_cm2_g"] = value
    with pytest.raises(ValueError):
        runner.compare_rows(selected, rows)


def test_zero_reference_is_explicit_not_infinite_or_zero_residual(runner: ModuleType) -> None:
    selected = {"fe": _table(nuclear="0", energies=("1",))}
    native = _native_rows(runner, selected)
    native[0]["stopping_nuclear_mev_cm2_g"] = 1.
    native[0]["stopping_total_mev_cm2_g"] += 1.
    native[0]["stopping_nonradiative_mev_cm2_g"] += 1.
    rows = runner.compare_rows(selected, native)
    assert rows[0]["components"]["nuclear"]["catima_over_srim_minus_one"] is None
    assert rows[0]["components"]["nuclear"]["difference_mev_cm2_g"] == 1.
    for group in runner.summarize(rows):
        stats = group["components"]["nuclear"]
        assert stats["defined_relative_count"] == 0
        assert stats["undefined_zero_reference_count"] == 1
        for key in ("median_signed_relative_difference", "median_absolute_relative_difference",
                    "min_signed_relative_difference", "max_signed_relative_difference", "worst_absolute"):
            assert stats[key] is None


def test_signed_and_absolute_residual_statistics_and_witness(runner: ModuleType) -> None:
    selected = {"fe": _table(energies=("1", "10", "100"))}
    native = _native_rows(runner, selected)
    for row, factor in zip(native, (.5, 1.1, 1.2), strict=True):
        for component in (*runner.COMPONENTS, "total", "radiative"):
            row[f"stopping_{component}_mev_cm2_g"] *= factor
    rows = runner.compare_rows(selected, native)
    groups = runner.summarize(rows)
    overall = next(group for group in groups if group["group"] == ["material", "13", "all"])
    stats = overall["components"]["electronic"]
    assert overall["row_count"] == 3
    assert stats["median_signed_relative_difference"] == pytest.approx(.1)
    assert stats["median_absolute_relative_difference"] == pytest.approx(.2)
    assert stats["min_signed_relative_difference"] == -.5
    assert stats["max_signed_relative_difference"] == pytest.approx(.2)
    assert stats["worst_absolute"]["native_row_index"] == 0
    assert stats["worst_absolute"]["relative_difference"] == -.5
    assert not any("passed" in key or "tolerance" in key for key in stats)
    ion = next(group for group in groups if group["group"][0] == "ion_material")
    assert ion["group"] == ["ion_material", "26", "55.8", "13"]
    with pytest.raises(ValueError):
        runner.summarize([])


def test_decimal_context_does_not_move_boundaries_or_convert_stopping(runner: ModuleType) -> None:
    selected = {"fe": _table(energies=("0.558", "55.8", "558", "1674", "5580"))}
    native = _native_rows(runner, selected)
    expected_inventory = runner.build_inventory(selected)
    expected_rows = runner.compare_rows(selected, native)
    with localcontext() as context:
        context.prec = 2
        assert runner.build_inventory(selected) == expected_inventory
        assert runner.compare_rows(selected, native) == expected_rows


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
                               b'[]', b'null', b'{', b'not json'])
def test_read_pinned_rejects_malformed_json(runner: ModuleType, tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        runner.read_pinned(path, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("sha", ["", "a" * 63, "A" * 64, "z" * 64, "0" * 64, None])
def test_read_pinned_requires_exact_lowercase_hash(runner: ModuleType, tmp_path: Path,
                                                sha: Any) -> None:
    path = tmp_path / "data.json"
    path.write_bytes(b'{"a":1}')
    with pytest.raises(ValueError):
        runner.read_pinned(path, sha)


def test_read_pinned_valid_and_canonical_same(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    raw = b'{"a":1,"b":false}'
    path.write_bytes(raw)
    assert runner.read_pinned(path, hashlib.sha256(raw).hexdigest()) == {"a": 1, "b": False}
    runner.same({"a": 1, "b": False}, {"b": False, "a": 1}, "mapping order")
    with pytest.raises(ValueError):
        runner.same({"a": 1}, {"a": True}, "literal type")


def test_write_new_is_exclusive_and_finite(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "new.json"
    runner.write_new(path, {"a": 1, "status": "INCOMPLETE"})
    raw = path.read_bytes()
    assert raw == runner.encode({"a": 1, "status": "INCOMPLETE"})
    with pytest.raises(FileExistsError):
        runner.write_new(path, {"a": 2})
    assert path.read_bytes() == raw
    with pytest.raises(ValueError):
        runner.encode({"a": float("nan")})


@pytest.mark.parametrize("mode,extra", [
    ("plan", ["--requests", "x"]), ("plan", ["--native-sha256", "a" * 64]),
    ("compare", []), ("compare", ["--requests", "x", "--requests-sha256", "a" * 64]),
])
def test_cli_bad_inputs_fail_before_reconstruction(runner: ModuleType, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 mode: str, extra: list[str]) -> None:
    monkeypatch.setattr(runner, "reconstruct", lambda: pytest.fail("No source access for invalid CLI"))
    monkeypatch.setattr(sys, "argv", ["overlap", "--mode", mode, "--output",
                                     str(tmp_path / "result.json"), *extra])
    with pytest.raises(ValueError):
        runner.main()


def test_cli_existing_output_fails_before_source_access(runner: ModuleType, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "existing.json"
    path.write_bytes(b"preserve")
    monkeypatch.setattr(runner, "reconstruct", lambda: pytest.fail("No source access"))
    monkeypatch.setattr(sys, "argv", ["overlap", "--mode", "plan", "--output", str(path)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert path.read_bytes() == b"preserve"


def test_plan_records_all_qualifications_false(runner: ModuleType) -> None:
    selected = {"fe": _table()}
    evidence = SimpleNamespace(pins={"retained": "a" * 64})
    value = runner.plan(selected, evidence)
    assert value["source_evidence_count"] == 1
    assert value["mass_policy"] == "literal_source_A_as_numeric_transport_mass_no_isotope_rounding"
    assert value["stopping_conversion"] == "SRIM_MeV_cm2_mg_times_1000_no_extra_A_factor"
    assert value["exclusion_policy"] == "uncompared_native_nodes_not_removed_from_source_spectra"
    for key, expected in runner.FLAGS.items():
        assert value[key] == expected
    assert json.loads(runner.encode(value))["matched_row_count"] == 3


def _mock_reconstruction(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    baseline = {f"case{i}": SimpleNamespace(raw_sha256=f"{i:064x}") for i in range(168)}
    catalog = tuple(SimpleNamespace(name=f"wide{i}", base_source_name=f"case{i}")
                    for i in range(158))
    extended = {c.name: SimpleNamespace(raw_sha256=f"{i + 1000:064x}")
                for i, c in enumerate(catalog)}
    calls: list[tuple[Any, ...]] = []
    evidence = SimpleNamespace(
        pins={}, read=lambda *args: calls.append(("read", *args)),
        verify_unchanged=lambda: calls.append(("verify",)),
    )
    report = {"mode": "full", "all_native_controls_repeated": True}
    selection = [{"base_source_name": c.base_source_name, "extension_name": c.name,
                  "baseline_raw_sha256": baseline[c.base_source_name].raw_sha256,
                  "selected_raw_sha256": extended[c.name].raw_sha256,
                  "baseline_prefix_identical": True} for c in catalog]
    summary = {"table_selection": selection}
    monkeypatch.setattr(runner, "PARENT_PINS", {})
    monkeypatch.setattr(runner.extension, "_context", lambda: (None, evidence, {}, baseline, catalog))

    def replay(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(("replay", args, kwargs))
        return report, extended

    monkeypatch.setattr(runner.extension, "replay_extension", replay)
    monkeypatch.setattr(runner, "read_pinned", lambda path, sha: summary)
    return {"baseline": baseline, "catalog": catalog, "extended": extended,
            "report": report, "summary": summary, "evidence": evidence, "calls": calls}


def test_reconstruct_selects_whole_extended_and_light_baseline_tables(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _mock_reconstruction(runner, monkeypatch)
    result, evidence = runner.reconstruct()
    assert len(result) == 168
    assert evidence is state["evidence"]
    for i in range(168):
        expected = state["extended"][f"wide{i}"] if i < 158 else state["baseline"][f"case{i}"]
        assert result[f"case{i}"] is expected
    assert state["calls"][0][0] == "replay"
    assert state["calls"][0][1] == (runner.ROOT / runner.EXTENSION, runner.EXTENSION_SHA256)
    assert state["calls"][-1] == ("verify",)
    assert ("read", runner.ROOT / runner.SUMMARY, runner.SUMMARY_SHA256) in state["calls"]


@pytest.mark.parametrize("change", ["pilot", "unrepeated", "missing_extension", "missing_baseline",
                                    "changed_summary", "extra_extension"])
def test_reconstruct_rejects_incomplete_or_changed_whole_selection(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    state = _mock_reconstruction(runner, monkeypatch)
    if change == "pilot":
        state["report"]["mode"] = "pilot"
    elif change == "unrepeated":
        state["report"]["all_native_controls_repeated"] = False
    elif change == "missing_extension":
        state["extended"].pop("wide0")
    elif change == "missing_baseline":
        state["baseline"].pop("case167")
    elif change == "changed_summary":
        state["summary"]["table_selection"][0]["selected_raw_sha256"] = "f" * 64
    else:
        state["extended"]["unexpected"] = SimpleNamespace(raw_sha256="e" * 64)
    with pytest.raises(ValueError):
        runner.reconstruct()


def test_reconstruct_rejects_changed_frozen_parent_before_replay(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "parent.py").write_bytes(b"changed")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PARENT_PINS", {"parent.py": "a" * 64})
    monkeypatch.setattr(runner.extension, "_context", lambda: pytest.fail("No unverified import"))
    with pytest.raises(ValueError, match="Frozen"):
        runner.reconstruct()


def test_cli_plan_success_is_read_only_and_writes_new_report(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[str] = []
    selected = {"fe": _table()}
    evidence = SimpleNamespace(pins={}, verify_unchanged=lambda: calls.append("verify"))
    monkeypatch.setattr(runner, "reconstruct", lambda: (selected, evidence))
    monkeypatch.setattr(runner, "validate_native", lambda *args: pytest.fail("No worker in plan mode"))
    path = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", ["overlap", "--mode", "plan", "--output", str(path)])
    assert runner.main() == 0
    assert calls == ["verify"]
    payload = json.loads(path.read_bytes())
    assert payload["schema_version"] == "radar.srim_catima_overlap_requests.v1"
    assert payload["matched_row_count"] == 3
    assert payload["status"] == "INCOMPLETE"
