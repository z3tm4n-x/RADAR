"""Synthetic orchestration evidence only. This suite never executes SRIM."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/run_srim_coverage.py"
    spec = importlib.util.spec_from_file_location("srim_coverage_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _grid(case: Any) -> tuple[Decimal, ...]:
    hundredths = (100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 225, 250, 275,
                  300, 325, 350, 375, 400, 450, 500, 550, 600, 650, 700, 800, 900)
    return tuple(e for exponent in range(-3, 4) for value in hundredths
                 if case.energy_min_mev <= (e := Decimal(value) * Decimal(10) ** exponent)
                 <= case.energy_max_mev)


def _text(case: Any) -> bytes:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[5] = "Disk File Name = srim_output.txt"
    names = {1: "Hydrogen", 2: "Helium", 6: "Carbon", 7: "Nitrogen", 8: "Oxygen",
             10: "Neon", 12: "Magnesium", 14: "Silicon", 16: "Sulfur", 18: "Argon",
             20: "Calcium", 24: "Chromium", 25: "Manganese", 26: "Iron",
             27: "Cobalt", 28: "Nickel"}
    header[6] = header[17] = (
        f"Ion = {names[case.ion_z]} [{case.ion_z}] , Mass = {case.ion_mass_u} amu")
    header[7] = f"Density = {case.density_g_cm3} g/cm3 = 6.0000E+22 atoms/cm3"
    header[12] = f"{'Al' if case.target_z == 13 else 'Si'} {case.target_z} 100.00 100.00"
    rows = [f"{e} MeV 0.2 0.001 {i} um 1 A 1 A" for i, e in enumerate(_grid(case), 1)]
    footer = lines[lines.index("-" * 59):]
    text = re.sub(r"(?<=\d),(?=\d)", ".", "\r\n".join(header + rows + footer[:-1]))
    return (text + "\r\n" + footer[-1] + "\r\n").encode("ascii")


@pytest.fixture
def isolated(runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root, source = tmp_path / "repo", tmp_path / "source"
    root.mkdir()
    source.mkdir()
    for name in runner.SOURCE_HASHES:
        (source / name).write_bytes(f"SYNTHETIC {name}".encode())
    (source / "SR.IN").write_bytes(b"UNMODIFIED ORIGINAL INPUT")
    runtime = tmp_path / "private.dll"
    runtime.write_bytes(b"SYNTHETIC NONEXECUTABLE DLL")
    for name in (*runner.IMPLEMENTATION, *runner.PRODUCTION):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC PINNED {name}".encode())
    pilot = root / runner.PILOT_REPORT
    pilot.parent.mkdir(parents=True)
    pilot.write_text(json.dumps({"selected_files_sha256_after": {
        name: runner._sha((root / name).read_bytes())
        for name in (*runner.pilot.IMPLEMENTATION, *runner.PRODUCTION)
    }}), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "PILOT_REPORT_SHA256", runner._sha(pilot.read_bytes()))
    monkeypatch.setattr(runner, "SOURCE_HASHES", {
        name: runner._sha((source / name).read_bytes()) for name in runner.SOURCE_HASHES
    })
    monkeypatch.setattr(runner, "PRIVATE_RUNTIME_SHA256", runner._sha(runtime.read_bytes()))
    # Smaller test catalogs are intentionally synthetic. The standalone catalog
    # contract tests below check every entry in the unmodified production script.
    all_catalogs = runner.CATALOGS
    monkeypatch.setattr(runner, "CATALOGS", {name: cases[:2] for name, cases in all_catalogs.items()})
    calls: list[Path] = []
    comparisons: list[tuple[Any, bytes, str]] = []
    selected = "baseline"

    def execute(work: Path) -> dict[str, Any]:
        calls.append(work)
        case_name = work.name.rsplit("_r", 1)[0]
        case = next(case for case in runner.CATALOGS[selected] if case.name == case_name)
        (work / "srim_output.txt").write_bytes(_text(case))
        command = {"argv": [str(work / "SRModule.exe")], "cwd": str(work), "pid": 12345,
                   "returncode": 0, "timed_out": False, "elapsed_seconds": 0.001,
                   "stdout": "", "stderr": ""}
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        return command

    def compare(parsed: Any, raw: bytes, path: str) -> dict[str, Any]:
        comparisons.append((parsed, raw, path))
        return {"synthetic": True, "normative_export_allowed": False}

    def run(catalog: str = "baseline") -> dict[str, Any]:
        nonlocal selected
        selected = catalog
        return cast(dict[str, Any], runner.run_coverage(
            catalog=catalog, source=source, private_runtime=runtime, output=tmp_path / "output"))

    monkeypatch.setattr(runner, "_execute", execute)
    monkeypatch.setattr(runner, "compare_legacy_table", compare)
    return SimpleNamespace(root=root, source=source, runtime=runtime, pilot=pilot,
                           output=tmp_path / "output", calls=calls, comparisons=comparisons,
                           execute=execute, run=run, catalogs=all_catalogs)


@pytest.mark.parametrize("catalog,count,nodes", [("baseline", 30, 107), ("upper", 28, 123)])
def test_fixed_legacy_catalog_and_units(runner: ModuleType, catalog: str, count: int, nodes: int
                                      ) -> None:
    cases = runner.CATALOGS[catalog]
    assert len(cases) == count
    assert len({case.name for case in cases}) == count
    assert {case.ion_z for case in cases} == (
        {2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 25, 26, 27, 28}
        - ({2} if catalog == "upper" else set()))
    for case in cases:
        assert runner.expected_control_grid(case) == _grid(case)
        assert len(_grid(case)) == nodes
        assert case.energy_max_mev < 10000 * case.ion_mass_u
        lines = runner.input_bytes(case).decode("ascii").splitlines()
        assert len(lines) == 15 and lines[12] == "5"
        assert tuple(map(Decimal, lines[14].split())) == (
            case.energy_min_mev * 1000, case.energy_max_mev * 1000)
        assert b"\r\n" in runner.input_bytes(case)
        scope = runner._scope(case)
        assert scope["package_trim_help_high_energy_caution"] is (
            case.energy_max_mev > 1000 * case.ion_mass_u)
        assert scope["physical_accuracy_verified"] is False


def test_boundary_catalog_is_explicit_and_not_hze(runner: ModuleType) -> None:
    cases = runner.CATALOGS["boundary"]
    assert [(c.name, c.target_z, c.energy_max_mev) for c in cases] == [
        ("h_al_2000", 13, Decimal(2000)), ("h_al_4000", 13, Decimal(4000)),
        ("h_al_6000", 13, Decimal(6000)), ("h_si_4000", 14, Decimal(4000)),
    ]
    for case in cases:
        assert case.ion_z == 1 and case.ion_mass_u == Decimal("1.008")
        assert case.energy_min_mev == Decimal("0.1")
        assert "proton_" in case.resource_path
        assert runner.expected_control_grid(case) == _grid(case)


@pytest.mark.parametrize("field,value", [("ion_z", True), ("ion_mass_u", 4.0),
    ("energy_max_mev", Decimal(10000)), ("density_g_cm3", Decimal(3))])
def test_undeclared_or_weakly_typed_controls_rejected(
    runner: ModuleType, field: str, value: Any,
) -> None:
    bad = replace(runner.CATALOGS["baseline"][0], **{field: value})
    for function in (runner.input_bytes, runner.expected_control_grid):
        with pytest.raises(ValueError, match="fixed legacy-ion"):
            function(bad)


@pytest.mark.parametrize("catalog", ["baseline", "upper", "boundary"])
def test_complete_mock_catalog_preserves_pins_and_never_qualifies(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, catalog: str,
) -> None:
    monkeypatch.setattr(runner, "CATALOGS", isolated.catalogs)
    report = isolated.run(catalog)
    count = {"baseline": 30, "upper": 28, "boundary": 4}[catalog]
    assert report["case_count"] == count and len(isolated.calls) == 2 * count
    assert len(set(isolated.calls)) == len(isolated.comparisons) == 2 * count
    assert report["protected_sha256_before"] == report["protected_sha256_after"]
    assert report["all_native_controls_repeated"] is True
    assert report["diagnostic_status"] == "INCOMPLETE"
    assert report["native_control_status_counts"]["CONTROL_COMPLETE"] == count
    assert report["catalog_kind"] == ("proton_boundary" if catalog == "boundary"
                                       else "legacy_hze_coverage")
    for key in ("required_runtime_species_and_masses_fully_validated", "physical_accuracy_verified",
                "normative_export_allowed", "production_data_modified", "release_status_changed"):
        assert report[key] is False
    for case in report["cases"]:
        assert case["status"] == "CONTROL_COMPLETE"
        assert case["semantic_outputs_identical"] and case["raw_output_bytes_identical"]
        assert (isolated.output / f"{case['case']['name']}.json").is_file()
        for repeat in case["repeats"]:
            assert repeat["parsed"]["range_kind"] == "projected"
            work = Path(repeat["input"]["path"]).parent
            assert (work / "result.json").is_file()
            assert not any(path.name.startswith("_") for path in work.iterdir())
    assert json.loads((isolated.output / "report.json").read_text(encoding="utf-8"))[
        "diagnostic_status"] == \
        "INCOMPLETE"


@pytest.mark.parametrize("target", ["pilot", "source", "runtime", "production", "implementation"])
def test_pins_fail_before_output_creation(runner: ModuleType, isolated: Any, target: str) -> None:
    path = {"pilot": isolated.pilot, "source": isolated.source / "VERSION",
            "runtime": isolated.runtime, "production": isolated.root / runner.PRODUCTION[0],
            "implementation": isolated.root / runner.pilot.IMPLEMENTATION[0]}[target]
    path.write_bytes(b"TAMPERED")
    with pytest.raises(ValueError):
        isolated.run()
    assert not isolated.calls and not isolated.output.exists()


def test_existing_output_and_unknown_catalog_rejected(runner: ModuleType, isolated: Any) -> None:
    with pytest.raises(ValueError, match="Unknown fixed"):
        isolated.run("arbitrary")
    assert not isolated.output.exists()
    isolated.output.mkdir()
    with pytest.raises(FileExistsError):
        isolated.run()
    assert not isolated.calls


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "missing", "truncated", "grid",
                                     "metadata", "repeat"])
def test_explicit_failure_is_retained_and_other_cases_continue(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    first_name = runner.CATALOGS["baseline"][0].name
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        applies = work.name == f"{first_name}_r{2 if failure == 'repeat' else 1}"
        if not applies:
            return result
        path = work / "srim_output.txt"
        if failure in ("timeout", "nonzero"):
            result.update(returncode=1, timed_out=failure == "timeout")
            (work / "command.json").write_text(json.dumps(result))
            raise RuntimeError("Synthetic native failure")
        if failure == "missing":
            path.unlink()
        elif failure == "truncated":
            path.write_bytes(b"TRUNCATED")
        elif failure == "grid":
            lines = path.read_text().splitlines()
            del lines[22]
            path.write_text("\r\n".join(lines) + "\r\n")
        elif failure == "metadata":
            path.write_bytes(path.read_bytes().replace(b"2.70 g/cm3", b"2.71 g/cm3"))
        else:
            path.write_bytes(path.read_bytes().replace(b"MeV 0.2 ", b"MeV 0.3 ", 1))
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    failed, success = report["cases"]
    expected = ("NATIVE_FAILED" if failure in ("timeout", "nonzero") else
                "REPEAT_MISMATCH" if failure == "repeat" else "INVALID_OUTPUT")
    assert failed["status"] == expected and not failed["semantic_outputs_identical"]
    assert len(failed["repeats"]) == (2 if failure == "repeat" else 1)
    assert success["status"] == "CONTROL_COMPLETE"
    assert not report["all_native_controls_repeated"]
    assert len(isolated.comparisons) == (4 if failure == "repeat" else 2)
    assert (isolated.output / f"{first_name}_r1/result.json").is_file()
    if failure != "repeat":
        assert "parsed" not in failed["repeats"][0]
        assert "comparison" not in failed["repeats"][0]


@pytest.mark.parametrize("target", ["copied", "source", "input", "runtime", "implementation",
                                    "production", "pilot"])
@pytest.mark.parametrize("native_failed", [False, True])
def test_mutation_hard_fails_even_after_native_failure(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch,
    target: str, native_failed: bool,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        path = {"copied": work / "SR.IN", "source": isolated.source / "VERSION",
                "input": isolated.source / "SR.IN", "runtime": isolated.runtime,
                "implementation": isolated.root / runner.IMPLEMENTATION[-1],
                "production": isolated.root / runner.PRODUCTION[0], "pilot": isolated.pilot}[target]
        path.write_bytes(b"MUTATION")
        if native_failed:
            result.update(returncode=1, timed_out=True)
            (work / "command.json").write_text(json.dumps(result))
            raise RuntimeError("Synthetic timeout with mutation")
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="Protected source"):
        isolated.run()
    assert not (isolated.output / "report.json").exists()
    assert len(isolated.calls) == 1


@pytest.mark.parametrize("mode", ["no_record", "success_record", "malformed_record", "wrong_cwd"])
def test_unrelated_runtimeerror_is_not_misdiagnosed_as_native_failure(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        if mode != "no_record":
            result = cast(dict[str, Any], isolated.execute(work))
            if mode == "malformed_record":
                result["returncode"] = True
            elif mode == "wrong_cwd":
                result.update(returncode=1, cwd="SOME OTHER PROCESS")
            (work / "command.json").write_text(json.dumps(result))
        raise RuntimeError("Unexpected launcher error")
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        isolated.run()
    assert not (isolated.output / "report.json").exists()


def test_comparison_errors_are_not_silently_turned_into_failed_native_outputs(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_compare(*args: Any) -> dict[str, Any]:
        raise ValueError("Synthetic corrupt legacy resource")
    monkeypatch.setattr(runner, "compare_legacy_table", bad_compare)
    with pytest.raises(ValueError, match="corrupt legacy"):
        isolated.run()
    assert not (isolated.output / "report.json").exists()


def test_success_return_must_match_retained_command(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        result["pid"] += 1
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="disagrees with retained"):
        isolated.run()
    assert not (isolated.output / "report.json").exists()
