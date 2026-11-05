"""Isolated runner controls using synthetic text only; never executes SRIM."""

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
    script = Path(__file__).resolve().parents[1] / "scripts/run_srim_controls.py"
    spec = importlib.util.spec_from_file_location("srim_controls_test_subject", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_grid(case: Any) -> tuple[Decimal, ...]:
    # Independently enumerate the retained normal-grid coordinates; never obtain
    # the expected test data by calling the runner's grid implementation.
    hundredths = (100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 225, 250, 275,
                  300, 325, 350, 375, 400, 450, 500, 550, 600, 650, 700, 800, 900)
    energies = (Decimal(value) * Decimal(10) ** exponent
                for exponent in range(-3, 3) for value in hundredths)
    return tuple(energy for energy in energies
                 if case.energy_min_mev <= energy <= case.energy_max_mev)


def _synthetic_output(case: Any) -> bytes:
    """Preserve the genuine format but use explicitly synthetic numeric values."""
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC TEST DATA"
    header[5] = "Disk File Name = srim_output.txt"
    ion = {1: "Hydrogen", 2: "Helium", 26: "Iron"}[case.ion_z]
    header[6] = header[17] = f"Ion = {ion} [{case.ion_z}] , Mass = {case.ion_mass_u} amu"
    header[7] = f"Density = {case.density_g_cm3} g/cm3 = 6.0000E+22 atoms/cm3"
    symbol = "Al" if case.target_z == 13 else "Si"
    header[12] = f"{symbol} {case.target_z} 100.00 100.00"
    # Full control-grid coverage, but explicitly fake stopping and monotonic
    # projected range. These values test orchestration, not stopping physics.
    rows = [f"{energy} MeV 0.2 0.001 {index} um 1 A 1 A"
            for index, energy in enumerate(_fixture_grid(case), 1)]
    footer = lines[lines.index("-" * 59):]
    text = re.sub(r"(?<=\d),(?=\d)", ".", "\r\n".join(header + rows + footer[:-1]))
    return (text + "\r\n" + footer[-1] + "\r\n").encode("ascii")


@pytest.fixture
def isolated(runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root, source = tmp_path / "repository", tmp_path / "source"
    root.mkdir()
    source.mkdir()
    for name in runner.SOURCE_HASHES:
        (source / name).write_bytes(f"SYNTHETIC SOURCE {name}".encode("ascii"))
    (source / "SR.IN").write_bytes(b"UNTOUCHED USER INPUT")
    runtime = tmp_path / "private.dll"
    runtime.write_bytes(b"SYNTHETIC PRIVATE RUNTIME; NOT EXECUTABLE")
    for name in (*runner.IMPLEMENTATION, *runner.PRODUCTION):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC PROTECTED {name}".encode("ascii"))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "SOURCE_HASHES", {
        name: runner._sha((source / name).read_bytes()) for name in runner.SOURCE_HASHES
    })
    monkeypatch.setattr(runner, "PRIVATE_RUNTIME_SHA256", runner._sha(runtime.read_bytes()))
    calls: list[Path] = []
    comparisons: list[tuple[Any, bytes, str]] = []

    def fake_execute(work: Path) -> dict[str, Any]:
        calls.append(work)
        name = work.name.rsplit("_r", 1)[0]
        case = next(case for case in runner.CONTROL_CASES if case.name == name)
        (work / "srim_output.txt").write_bytes(_synthetic_output(case))
        record = {"synthetic": True, "returncode": 0, "timed_out": False}
        (work / "command.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    def fake_compare(parsed: Any, raw: bytes, path: str) -> dict[str, Any]:
        comparisons.append((parsed, raw, path))
        return {"synthetic": True, "normative_export_allowed": False}

    monkeypatch.setattr(runner, "_execute", fake_execute)
    monkeypatch.setattr(runner, "compare_legacy_table", fake_compare)
    return SimpleNamespace(
        root=root, source=source, runtime=runtime, output=tmp_path / "outputs",
        calls=calls, comparisons=comparisons, execute=fake_execute,
        run=lambda: runner.run_controls(source=source, private_runtime=runtime,
                                       output=tmp_path / "outputs"),
    )


def test_six_fixed_normal_mode_controls_and_total_energy_input(runner: ModuleType) -> None:
    expected = {
        "h_al": (1, "1.008", 13, "26.981539", "2.70", "0.1", "1000"),
        "h_si": (1, "1.008", 14, "28.0855", "2.3290", "0.1", "1000"),
        "he4_al": (2, "4", 13, "26.981539", "2.70", "1", "12000"),
        "he4_si": (2, "4", 14, "28.0855", "2.3290", "1", "12000"),
        "fe56_al": (26, "56", 13, "26.981539", "2.70", "1", "12000"),
        "fe56_si": (26, "56", 14, "28.0855", "2.3290", "1", "12000"),
    }
    assert len(runner.CONTROL_CASES) == len(expected)
    for case in runner.CONTROL_CASES:
        z, mass, tz, target_mass, density, minimum, maximum = expected[case.name]
        assert (case.ion_z, case.ion_mass_u, case.target_z, case.target_mass_u,
                case.density_g_cm3, case.energy_min_mev, case.energy_max_mev) == (
                    z, Decimal(mass), tz, Decimal(target_mass), Decimal(density),
                    Decimal(minimum), Decimal(maximum))
        raw = runner.input_bytes(case)
        assert raw.isascii() and b"\r\n" in raw
        lines = raw.decode("ascii").splitlines()
        assert lines[11:14] == ["---Output Stopping Units (1-8)", "5",
                                "---Ion Energy : E-Min(keV), E-Max(keV)"]
        assert tuple(map(Decimal, lines[14].split())) == (
            case.energy_min_mev * 1000, case.energy_max_mev * 1000)
        assert len(lines) == 15  # One target; no special-energy-list or terminating-zero mode.
        grid = runner.expected_control_grid(case)
        assert grid == _fixture_grid(case)
        assert len(grid) == (105 if case.ion_z == 1 else 107)
    with pytest.raises(ValueError, match="Only the six"):
        runner.input_bytes(replace(runner.CONTROL_CASES[0], energy_max_mev=Decimal("9999")))


def test_complete_synthetic_run_preserves_evidence_without_qualification(
    runner: ModuleType, isolated: Any,
) -> None:
    report = isolated.run()
    assert report["case_count"] == 6 and report["runs_per_case"] == 2
    assert report["diagnostic_status"] == "INCOMPLETE"
    assert report["all_planned_control_ranges_covered"] is False
    assert len(isolated.calls) == len(set(isolated.calls)) == len(isolated.comparisons) == 12
    assert report["selected_files_sha256_before"] == report["selected_files_sha256_after"]
    for key in ("production_data_modified", "release_status_changed", "physical_accuracy_verified",
                "normative_export_allowed"):
        assert report[key] is False
    assert report["source_files_unchanged"] is True
    for record in report["cases"]:
        assert record["semantic_outputs_identical"] is True
        assert record["raw_output_bytes_identical"] is True
        case = next(case for case in runner.CONTROL_CASES if case.name == record["case"]["name"])
        assert record["expected_control_grid_mev"] == _fixture_grid(case)
        assert record["complete_native_control_grid_verified"] is True
        if record["case"]["name"] == "h_al":
            assert record["case"]["energy_max_mev"] == Decimal("1000")
            assert record["planned_control_energy_max_mev"] == Decimal("10000")
            assert record["planned_control_range_covered"] is False
            assert "Overflow" in record["native_upper_range_limitation"]
            assert "maximum supported energy is not established" in record[
                "native_upper_range_limitation"]
        else:
            assert record["planned_control_energy_max_mev"] == record["case"]["energy_max_mev"]
            assert record["planned_control_range_covered"] is True
            assert record["native_upper_range_limitation"] is None
        for repeat in record["repeats"]:
            assert repeat["parsed"]["range_kind"] == "projected"
            assert repeat["parsed"]["energy_basis"] == "total_ion_kinetic_energy"
            assert repeat["parsed"]["normative_export_allowed"] is False
            work = Path(repeat["input"]["path"]).parent
            assert (work / "MSVBVM50.dll").read_bytes() == isolated.runtime.read_bytes()
            assert not any(path.name.startswith("_") for path in work.iterdir())
    assert (isolated.source / "SR.IN").read_bytes() == b"UNTOUCHED USER INPUT"
    assert json.loads((isolated.output / "report.json").read_text(encoding="utf-8"))[
        "normative_export_allowed"] is False


@pytest.mark.parametrize("name", ["SRModule.exe", "SCOEF03.dat", "SNUC03.dat", "VERSION",
                                 "HELP- SR Module.rtf"])
def test_source_pin_mismatch_fails_before_launch(isolated: Any, name: str) -> None:
    (isolated.source / name).write_bytes(b"TAMPERED")
    with pytest.raises(ValueError, match="Pinned normal-mode SRIM source changed"):
        isolated.run()
    assert not isolated.calls and not isolated.output.exists()


def test_private_runtime_pin_mismatch_fails_before_launch(isolated: Any) -> None:
    isolated.runtime.write_bytes(b"DIFFERENT RUNTIME")
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        isolated.run()
    assert not isolated.calls and not isolated.output.exists()


def test_existing_output_is_not_overwritten(isolated: Any) -> None:
    isolated.output.mkdir()
    sentinel = isolated.output / "existing-result.txt"
    sentinel.write_bytes(b"PRESERVE")
    with pytest.raises(FileExistsError):
        isolated.run()
    assert sentinel.read_bytes() == b"PRESERVE" and not isolated.calls


@pytest.mark.parametrize("name", ["SRModule.exe", "SCOEF03.dat", "SNUC03.dat", "VERSION",
                                 "HELP- SR Module.rtf", "MSVBVM50.dll", "SR.IN"])
def test_copied_input_or_runtime_mutation_is_rejected(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        (work / name).write_bytes(b"MODIFIED BY SYNTHETIC CHILD")
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="changed a copied runtime/input file"):
        isolated.run()
    assert not (isolated.output / "report.json").exists()


@pytest.mark.parametrize("target", ["source", "input", "runtime", "production", "implementation"])
def test_original_or_protected_mutation_prevents_report(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    paths = {"source": isolated.source / "VERSION", "input": isolated.source / "SR.IN",
             "runtime": isolated.runtime, "production": isolated.root / runner.PRODUCTION[0],
             "implementation": isolated.root / runner.IMPLEMENTATION[0]}
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        if len(isolated.calls) == 1:
            paths[target].write_bytes(b"SYNTHETIC CONCURRENT MUTATION")
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="changed during controls"):
        isolated.run()
    assert not (isolated.output / "report.json").exists()


@pytest.mark.parametrize("field,value", [
    ("ion_z", 2), ("ion_mass_u", Decimal("1")), ("target_z", 14),
    ("density_g_cm3", Decimal("2.6989")), ("bragg_correction_percent", Decimal("1")),
    ("engine_label", "SRIM-2003"), ("version_label", "SRIM-2013.00"),
    ("output_name", "stale.txt"), ("energy_min", Decimal("0.09")),
    ("energy_max", Decimal("9999")),
])
def test_case_metadata_and_endpoint_mismatch_fail_closed(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        path = work / "srim_output.txt"
        lines = path.read_bytes().decode("ascii").splitlines()
        if field in ("energy_min", "energy_max"):
            index = 21 if field == "energy_min" else lines.index("-" * 59) - 1
            fields = lines[index].split()
            fields[0] = str(value)
            lines[index] = " ".join(fields)
        elif field == "ion_z":
            lines[6] = lines[17] = lines[6].replace("Hydrogen [1]", "Helium [2]")
        elif field == "ion_mass_u":
            lines[6] = lines[17] = lines[6].replace("1.008 amu", f"{value} amu")
        elif field == "target_z":
            lines[12] = "Si 14 100.00 100.00"
        elif field == "density_g_cm3":
            lines[7] = lines[7].replace("2.70 g/cm3", f"{value} g/cm3")
        elif field == "bragg_correction_percent":
            lines[14] = f"Bragg Correction = {value}%"
        elif field == "engine_label":
            lines[1] = f"Calculation using {value}"
        elif field == "version_label":
            lines[2] = f"SRIM version ---> {value}"
        else:
            assert field == "output_name"
            lines[5] = f"Disk File Name = {value}"
        path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="metadata or coverage mismatch"):
        isolated.run()
    assert not (isolated.output / "report.json").exists()


@pytest.mark.parametrize("mutation", ["missing_interior", "extra_interior", "wrong_same_count"])
def test_full_native_grid_rejects_interior_changes_with_unchanged_endpoints(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        path = work / "srim_output.txt"
        lines = path.read_bytes().decode("ascii").splitlines()
        if mutation == "missing_interior":
            del lines[22]
        elif mutation == "extra_interior":
            lines.insert(22, "0.105 MeV 0.2 0.001 1.5 um 1 A 1 A")
        else:
            fields = lines[22].split()
            fields[0] = "0.115"
            lines[22] = " ".join(fields)
        changed = ("\r\n".join(lines) + "\r\n").encode("ascii")
        # All variants remain well-formed, monotonic native tables. They must be
        # rejected by the fixed-grid runner contract, not by malformed text.
        parsed = runner.parse_srim_range_output(changed)
        assert parsed.rows[0].energy_total_mev == Decimal("0.1")
        assert parsed.rows[-1].energy_total_mev == Decimal("1000")
        assert len(parsed.rows) == {"missing_interior": 104, "extra_interior": 106,
                                    "wrong_same_count": 105}[mutation]
        path.write_bytes(changed)
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="full control energy grid mismatch"):
        isolated.run()
    assert not isolated.comparisons
    assert not (isolated.output / "report.json").exists()


def test_numeric_repeat_mismatch_prevents_report(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        result = cast(dict[str, Any], isolated.execute(work))
        if work.name.endswith("_r2"):
            path = work / "srim_output.txt"
            path.write_bytes(path.read_bytes().replace(b"MeV 0.2 ", b"MeV 0.3 ", 1))
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="repeated numeric/metadata result changed"):
        isolated.run()
    assert len(isolated.calls) == 2
    assert not (isolated.output / "report.json").exists()


@pytest.mark.parametrize("failure", ["native", "missing_output", "malformed_output"])
def test_failed_or_incomplete_native_output_is_not_a_success(
    runner: ModuleType, isolated: Any, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    def execute(work: Path) -> dict[str, Any]:
        if failure == "native":
            raise RuntimeError("SYNTHETIC native failure")
        result = cast(dict[str, Any], isolated.execute(work))
        if failure == "missing_output":
            (work / "srim_output.txt").unlink()
        else:
            (work / "srim_output.txt").write_bytes(b"TRUNCATED")
        return result
    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        isolated.run()
    assert not (isolated.output / "report.json").exists()


@pytest.mark.parametrize("mode", ["success", "nonzero", "timeout"])
def test_native_wrapper_hides_and_bounds_only_its_mock_child(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    waits: list[int] = []
    child = SimpleNamespace(pid=54321, returncode=1 if mode == "nonzero" else 0,
                            killed=False)

    def communicate(*, timeout: int) -> tuple[bytes, bytes]:
        waits.append(timeout)
        if mode == "timeout" and len(waits) == 1:
            raise runner.subprocess.TimeoutExpired("SYNTHETIC CHILD", timeout)
        return b"SYNTHETIC STDOUT", b"SYNTHETIC STDERR"

    def kill() -> None:
        child.killed = True
        child.returncode = -9

    def popen(argv: list[str], **kwargs: Any) -> Any:
        calls.append((argv, kwargs))
        child.communicate, child.kill = communicate, kill
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.sys, "platform", "win32")
    monkeypatch.setattr(runner.subprocess, "STARTUPINFO",
                        lambda: SimpleNamespace(dwFlags=0, wShowWindow=99), raising=False)
    monkeypatch.setattr(runner.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr(runner.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    if mode == "success":
        assert runner._execute(tmp_path)["timed_out"] is False
    else:
        with pytest.raises(RuntimeError, match="SRIM control failed"):
            runner._execute(tmp_path)
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv == [str(tmp_path / "SRModule.exe")]
    assert options["cwd"] == tmp_path and options["startupinfo"].wShowWindow == 0
    assert options["creationflags"] == runner.subprocess.CREATE_NO_WINDOW
    assert options["stdin"] == runner.subprocess.DEVNULL
    assert waits == ([30, 5] if mode == "timeout" else [30])
    assert child.killed is (mode == "timeout")
    record = json.loads((tmp_path / "command.json").read_text(encoding="utf-8"))
    assert record["timed_out"] is (mode == "timeout")
    assert record["pid"] == child.pid and record["returncode"] == child.returncode


def test_native_wrapper_rejects_non_windows_before_launch(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="approved Windows environment"):
        runner._execute(tmp_path)
    assert not (tmp_path / "command.json").exists()
