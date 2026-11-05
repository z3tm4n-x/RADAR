"""Synthetic pinned replay tests: never execute SRIM or replace production data."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_DOWN, localcontext
import importlib.util
import json
from math import exp
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL, AVOGADRO_CONSTANT_MOL_INV, BARN_CM2,
    Al27NonelasticCrossSectionTable, proton_nonelastic_survival_factor,
)
from radar.validation.range_stopping_consistency import audit_range_stopping_consistency
from radar.validation.srim_output import parse_srim_range_output


D = Decimal


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/audit_srim_range_stopping.py"
    spec = importlib.util.spec_from_file_location("srim_range_stopping_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_bytes(*, silicon: bool = False, altered: bool = False) -> bytes:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[6] = header[17] = "Ion = Carbon [6] , Mass = 12 amu"
    header[7] = "Density = 2 g/cm3 = 6E22 atoms/cm3"
    header[12] = "Si 14 100 100" if silicon else "Al 13 100 100"
    rows = [f"{e} MeV {'0.002' if altered else '0.001'} 0.001 {D(e)/4} cm 1 A 1 A"
            for e in (1, 2, 10000, 11000)]
    footer = lines[lines.index("-" * 59):]
    text = re.sub(r"(?<=\d),(?=\d)", ".", "\r\n".join(header + rows + footer[:-1]))
    return (text + "\r\n" + footer[-1] + "\r\n").encode("ascii")


def _comparison() -> dict[str, Any]:
    return {
        "legacy_csv_sha256": "a" * 64,
        "energy_match_policy": "synthetic_exact_total_energy",
        "rows": [
            {"native_energy_total_mev": str(e), "comparisons": {
                "range_g_cm2_vs_projected_at_native_density": {"legacy": str(D(e)/2 + 100)}}}
            for e in (1, 2, 10000, 11000)
        ] + [{"unmatched_legacy_energy": "20000"}],
    }


@pytest.fixture
def pinned(runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path / "repo"
    root.mkdir()
    for name in runner.IMPLEMENTATION:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC PINNED IMPLEMENTATION")
    protected = root / "protected.txt"
    protected.write_bytes(b"UNCHANGED SYNTHETIC PROTECTED DATA")
    cases = []
    for index in (1, 2):
        files = {}
        for key, raw in (("input", b"SYNTHETIC SR.IN"), ("output", _native_bytes()),
                         ("command_file", b"SYNTHETIC COMMAND NEVER EXECUTED")):
            path = root / f"{key}_{index}.txt"
            path.write_bytes(raw)
            files[key] = {"path": str(path), "bytes": len(raw), "sha256": runner._sha(path)}
        native = parse_srim_range_output(_native_bytes())
        cases.append({**files, "status": "VALID_NATIVE_OUTPUT",
                      "semantic_sha256": native.semantic_sha256,
                      "parsed": json.loads(runner.coverage._json(asdict(native))),
                      "comparison": _comparison()})
    identity = {str(protected): runner._sha(protected)}
    report = {"protected_sha256_before": identity.copy(),
              "protected_sha256_after": identity.copy(), "cases": [
                  {"case": {"name": "synthetic_al"}, "status": "CONTROL_COMPLETE",
                   "repeats": cases}]}
    report_path = root / "outputs/srim_legacy_baseline_final_20261101/report.json"
    report_path.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "REPORT_PINS", {})
    monkeypatch.setattr(runner, "SELECTION", (("baseline", "synthetic_al"),))
    monkeypatch.setattr(runner.coverage, "CATALOGS", {
        "baseline": (SimpleNamespace(name="synthetic_al"),)})
    monkeypatch.setattr(runner.coverage, "expected_control_grid",
                        lambda case: (D(1), D(2), D(10000), D(11000)))

    def save() -> None:
        report_path.write_text(runner.coverage._json(report), encoding="utf-8")
        runner.REPORT_PINS["baseline"] = runner._sha(report_path)

    save()
    return SimpleNamespace(root=root, report=report, path=report_path, protected=protected,
                           save=save, repeats=cases, native=native)


def test_fixed_native_selection_is_legacy_al_only(runner: ModuleType) -> None:
    assert len(runner.SELECTION) == 16
    assert runner.SELECTION[0] == ("baseline", "he4_al")
    assert runner.SELECTION[-1] == ("boundary", "h_al_4000")
    assert len({name for _, name in runner.SELECTION}) == 16
    assert all(name.endswith("_al") for _, name in runner.SELECTION[:-1])
    assert set(runner.REPORT_PINS) == {"baseline", "upper", "boundary"}


def test_compare_uses_total_energy_units_and_reanchors_tail(runner: ModuleType) -> None:
    result = runner._compare_case(parse_srim_range_output(_native_bytes()), _comparison())
    whole = result["bands"]["whole_shared_support"]
    native = whole["native_projected_vs_native_total_stopping"]
    legacy = whole["legacy_unknown_range_vs_native_total_stopping"]
    assert native["energy_total_mev"] == (D(1), D(2), D(10000), D(11000))
    assert native["total_stopping_mev_cm2_g"] == (D(2),) * 4
    assert native["range_g_cm2"] == (D('.5'), D(1), D(5000), D(5500))
    assert native["range_kind"] == "projected" and legacy["range_kind"] == "unverified"
    assert native["intervals"] == legacy["intervals"]
    tail = result["bands"]["from_10000_mev"]["native_projected_vs_native_total_stopping"]
    assert tail["energy_total_mev"] == (D(10000), D(11000))
    assert tail["intervals"][0]["linear_stopping"]["anchored_integral_g_cm2"] == D(500)
    assert not result["normative_export_allowed"]
    assert not native["range_kind_verified"] and not legacy["physical_accuracy_established"]


def test_short_support_is_reported_not_extrapolated(runner: ModuleType) -> None:
    comparison = _comparison()
    comparison["rows"] = comparison["rows"][:2]
    result = runner._compare_case(parse_srim_range_output(_native_bytes()), comparison)
    assert result["bands"]["from_10000_mev"] == {"status": "NOT_COVERED", "node_count": 0}


def test_conversion_is_independent_of_ambient_decimal_context(runner: ModuleType) -> None:
    native = parse_srim_range_output(_native_bytes())
    comparison = _comparison()
    expected = runner._compare_case(native, comparison)
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.Emax = 3
        actual = runner._compare_case(native, comparison)
    assert actual == expected


@pytest.mark.parametrize("bad", ["duplicate", "off_grid", "single", "unordered", "negative_range"])
def test_invalid_shared_nodes_fail_closed(runner: ModuleType, bad: str) -> None:
    comparison = _comparison()
    rows = comparison["rows"]
    if bad == "duplicate":
        rows[1]["native_energy_total_mev"] = "1"
    elif bad == "off_grid":
        rows[1]["native_energy_total_mev"] = "3"
    elif bad == "single":
        comparison["rows"] = rows[:1]
    elif bad == "unordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["comparisons"]["range_g_cm2_vs_projected_at_native_density"]["legacy"] = "-1"
    with pytest.raises(ValueError):
        runner._compare_case(parse_srim_range_output(_native_bytes()), comparison)


def test_non_al_is_rejected(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="Only Al"):
        runner._compare_case(parse_srim_range_output(_native_bytes(silicon=True)), _comparison())


def test_pinned_replay_preserves_unqualified_status(runner: ModuleType, pinned: Any) -> None:
    result = runner.run_audit()
    assert len(result["selected_controls"]) == 1
    assert result["diagnostic_status"] == "INCOMPLETE"
    for key in ("normative_export_allowed", "production_data_modified", "physical_accuracy_verified"):
        assert result[key] is False
    assert result["unavailable_native_controls"] == []
    assert result["report_sha256"] == {"baseline": runner._sha(pinned.path)}


@pytest.mark.parametrize("target", ["report", "protected", "input", "output", "command_file"])
def test_retained_file_tampering_is_rejected(runner: ModuleType, pinned: Any, target: str) -> None:
    path = (pinned.path if target == "report" else pinned.protected if target == "protected"
            else Path(pinned.repeats[0][target]["path"]))
    path.write_bytes(path.read_bytes() + b"TAMPER")
    with pytest.raises(ValueError, match="mismatch|changed"):
        runner.run_audit()


@pytest.mark.parametrize("target", ["before", "size", "parsed", "semantic", "repeat_count",
                                    "failed_numeric"])
def test_repacked_invalid_report_is_rejected(runner: ModuleType, pinned: Any, target: str) -> None:
    if target == "before":
        pinned.report["protected_sha256_before"][str(pinned.protected)] = "f" * 64
    elif target == "size":
        pinned.repeats[0]["input"]["bytes"] += 1
    elif target == "parsed":
        pinned.repeats[0]["parsed"]["ion_mass_u"] = "13"
    elif target == "semantic":
        pinned.repeats[0]["semantic_sha256"] = "f" * 64
    elif target == "repeat_count":
        pinned.report["cases"][0]["repeats"] = pinned.repeats[:1]
    else:
        pinned.repeats[0]["status"] = "NATIVE_FAILED"
    pinned.save()
    with pytest.raises(ValueError):
        runner.run_audit()


def test_mismatched_successful_repeats_are_rejected(runner: ModuleType, pinned: Any) -> None:
    repeat = pinned.repeats[1]
    path = Path(repeat["output"]["path"])
    path.write_bytes(_native_bytes(altered=True))
    native = parse_srim_range_output(path.read_bytes())
    repeat["output"].update(bytes=path.stat().st_size, sha256=runner._sha(path))
    repeat.update(semantic_sha256=native.semantic_sha256,
                  parsed=json.loads(runner.coverage._json(asdict(native))))
    pinned.save()
    with pytest.raises(ValueError, match="identical valid repeats"):
        runner.run_audit()


def test_incomplete_grid_is_rejected(
    runner: ModuleType, pinned: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.coverage, "expected_control_grid", lambda case: (D(1), D(2)))
    with pytest.raises(ValueError, match="grid is incomplete"):
        runner.run_audit()


def test_required_incomplete_control_is_rejected(runner: ModuleType, pinned: Any) -> None:
    pinned.report["cases"][0]["status"] = "REPEAT_MISMATCH"
    pinned.save()
    with pytest.raises(ValueError, match="Required control is incomplete"):
        runner.run_audit()


def test_unselected_failed_control_is_retained_without_numbers(
    runner: ModuleType, pinned: Any,
) -> None:
    failed = {key: value for key, value in pinned.repeats[0].items()
              if key in ("input", "output", "command_file")}
    failed["status"] = "NATIVE_FAILED"
    pinned.report["cases"].append({"case": {"name": "synthetic_failed"},
                                   "status": "NATIVE_FAILED", "repeats": [failed]})
    pinned.save()
    result = runner.run_audit()
    assert result["unavailable_native_controls"] == [
        {"catalog": "baseline", "case": {"name": "synthetic_failed"},
         "status": "NATIVE_FAILED"}]
    assert len(result["selected_controls"]) == 1


@pytest.mark.parametrize("target", ["report", "protected", "implementation", "input",
                                    "output", "command_file"])
def test_mid_audit_mutation_is_rejected(
    runner: ModuleType, pinned: Any, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    compare = runner._compare_case

    def mutate(*args: Any) -> Any:
        result = compare(*args)
        path = (pinned.path if target == "report" else pinned.protected if target == "protected"
                else pinned.root / runner.IMPLEMENTATION[0] if target == "implementation"
                else Path(pinned.repeats[0][target]["path"]))
        path.write_bytes(path.read_bytes() + b"TAMPER")
        return result

    monkeypatch.setattr(runner, "_compare_case", mutate)
    with pytest.raises(ValueError, match="changed during audit|identity mismatch"):
        runner.run_audit()


def test_cli_exclusive_output_and_serialization(
    runner: ModuleType, pinned: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "new" / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    assert runner.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["diagnostic_status"] == "INCOMPLETE"
    before = output.read_bytes()
    monkeypatch.setattr(runner, "run_audit", lambda: pytest.fail("must reject before auditing"))
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2 and output.read_bytes() == before


def test_cli_race_does_not_overwrite_new_file(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raced.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])

    def race() -> dict[str, Any]:
        output.write_bytes(b"OTHER WRITER")
        return {"selected_controls": []}

    monkeypatch.setattr(runner, "run_audit", race)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"OTHER WRITER"


def test_cli_failure_does_not_publish_report(
    runner: ModuleType, pinned: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "never_created" / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    pinned.protected.unlink()
    with pytest.raises(FileNotFoundError):
        runner.main()
    assert not output.exists() and not output.parent.exists()


@pytest.mark.parametrize("range_scale", [1, 2])
def test_constant_cross_section_exposes_range_stopping_compatibility(range_scale: int) -> None:
    energy = (1., 10., 100.)
    table = ProtonAlRangeEnergyTable(energy, tuple(range_scale * e for e in energy), (1.,) * 3)
    cross_section = Al27NonelasticCrossSectionTable(3, energy, (1.,) * 3)
    output_energy = 9.
    thickness = 1.
    input_energy = table.energy_at_range(table.range_at_energy(output_energy) + thickness)
    actual = proton_nonelastic_survival_factor(
        output_energy_mev=output_energy, input_energy_mev=input_energy,
        stopping_table=table, cross_section_table=cross_section)
    expected = exp(-AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL * BARN_CM2 * thickness)
    audit = audit_range_stopping_consistency(
        energy_total_mev=tuple(D(str(e)) for e in energy),
        total_stopping_mev_cm2_g=(D(1),) * 3,
        range_g_cm2=tuple(D(str(e)) for e in table.range_g_cm2), range_kind="unverified")
    assert audit.intervals[-1].linear_stopping.anchored_relative_difference_percent_of_integral == (
        100 * (range_scale - 1))
    if range_scale == 1:
        assert actual == pytest.approx(expected)
    else:
        assert actual != pytest.approx(expected)
    assert audit.normative_export_allowed is False
