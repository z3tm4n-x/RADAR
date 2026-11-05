"""Lower-native-energy pilot safety/identity tests; never run an executable."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import pytest

from scripts import run_srim_low_energy as subject
from scripts import run_srim_controls as launcher
from radar.shielding.srim_dataset import load_srim_hze_dataset
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output

D = Decimal


def control(z: int = 2, target: int = 13) -> subject.LowControl:
    mass = {2: D(4), 26: D("55.8"), 92: D(238)}[z]
    return subject.LowControl(f"z{z}_t{target}_low10ev", z, mass, target,
                              D("26.981539" if target == 13 else "28.0855"),
                              D("2.70" if target == 13 else "2.3290"), D(10))


def text(case: subject.LowControl, *, low: bool = True,
         range_factor: Decimal = D(1), stopping: str = "0.2") -> bytes:
    template = (Path(__file__).parent / "fixtures/srim_h_al_smoke.txt").read_bytes().decode("cp1251")
    template = template.replace("probe_h_al.txt", "srim_output.txt")
    template = re.sub(r"Ion = Hydrogen\s+\[1\] , Mass = 1,008 amu",
                      f"Ion = Test [{case.ion_z}] , Mass = {case.ion_mass_u} amu", template)
    template = re.sub(r"Density =\s*[^\r\n]+", f"Density = {case.density_g_cm3} g/cm3 = 6E22 atoms/cm3",
                      template)
    if case.target_z == 14:
        template = re.sub(r"Al\s+13\s+100,00\s+100,00", "Si 14 100,00 100,00", template)
    lines = template.splitlines()
    indexes = [i for i, line in enumerate(lines) if re.match(r"\s*[\d.,]+\s+MeV\s+", line)]
    values = subject.expected_grid(case.energy_max_mev)
    if not low:
        values = tuple(v for v in values if v >= 1)
    rows = [f"{e} MeV {stopping} 0.001 {e*1000*range_factor} um {e*10} um {e*5} um"
            for e in values]
    result = lines[:indexes[0]] + rows + lines[indexes[-1]+1:]
    result = [line if line.strip().startswith("(C)") else re.sub(r"(?<=\d),(?=\d)", ".", line)
              for line in result]
    return ("\r\n".join(result) + "\r\n").encode("cp1251")


def baseline(case: subject.LowControl) -> SrimRangeOutput:
    return parse_srim_range_output(text(case, low=False))


@pytest.mark.parametrize("maximum,rows", [(D(10),157), (D(12000),237), (D(55000),255), (D(225000),271)])
def test_full_fixed_grid(maximum: Decimal, rows: int) -> None:
    grid = subject.expected_grid(maximum)
    assert len(grid) == rows
    assert grid[0] == D("0.00001") and grid[-1] == maximum
    assert grid[130] == 1
    assert all(a < b for a, b in zip(grid, grid[1:]))


@pytest.mark.parametrize("maximum", [D(0),D("NaN"),D("Infinity"),D("1.25"),D(225001),10])
def test_invalid_upper(maximum: Any) -> None:
    with pytest.raises(ValueError):
        subject.expected_grid(maximum)


@pytest.mark.parametrize("z", subject.PILOT_Z)
@pytest.mark.parametrize("target", (13,14))
def test_input_units_and_identity(z: int, target: int) -> None:
    case = control(z,target)
    raw = subject.input_bytes(case)
    assert raw.endswith(b"0.01    10000\r\n")
    assert b"---Ion Energy : E-Min(keV), E-Max(keV)" in raw
    assert f"{z}   {case.ion_mass_u}".encode() in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


@pytest.mark.parametrize("field,value", [("ion_mass_u",D(56)),("target_mass_u",D(27)),
                                       ("density_g_cm3",D(3)),("ion_z",3), ("target_z",12)])
def test_noncanonical_inputs(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        subject.input_bytes(replace(control(), **{field:value}))


def test_native_grid_shared_stopping_and_separate_projected_audit() -> None:
    case = control()
    old = baseline(case)
    new = parse_srim_range_output(text(case))
    report = subject.validate_native(new, old, case)
    assert report["new_lower_rows"] == 130
    assert report["shared_rows"] == 27
    assert report["shared_complete_rows_identical"] is True
    changed = parse_srim_range_output(text(case,range_factor=D("1.1")))
    report = subject.validate_native(changed, old, case)
    assert report["shared_stopping_identical"] is True
    assert report["shared_complete_rows_identical"] is False
    assert report["projected_range_and_straggling_changes"][0]["changed_rows"] == 27


def test_one_quantum_observed_energy_is_preserved_not_snapped() -> None:
    case = control()
    raw = text(case).replace(b"0.00001 MeV", b"9.99999 eV").replace(b"0.000011 MeV", b"10.9999 eV")
    native = parse_srim_range_output(raw)
    report = subject.validate_native(native, baseline(case), case)
    assert native.rows[0].energy_total_mev == D("0.00000999999")
    assert report["observed_minimum_total_ion_energy_mev"] != subject.MINIMUM_MEV
    assert report["printed_energy_shift_count"] == 2
    assert report["max_printed_energy_shift_mev"] == D("0.0000000001")


@pytest.mark.parametrize("token", (b"9.99998 eV", b"10.00002 eV", b"9 eV"))
def test_printed_grid_allowance_is_bounded(token: bytes) -> None:
    case = control()
    native = parse_srim_range_output(text(case).replace(b"0.00001 MeV",token))
    with pytest.raises(ValueError,match="printed-grid"):
        subject.validate_native(native,baseline(case),case)


@pytest.mark.parametrize("mutation", ("no_low", "missing_node", "stopping", "mass", "target"))
def test_invalid_native(mutation: str) -> None:
    case = control()
    raw = text(case)
    if mutation == "no_low":
        raw = text(case,low=False)
    elif mutation == "missing_node":
        raw = re.sub(rb"0.000011 MeV[^\r\n]+\r\n",b"",raw)
    elif mutation == "stopping":
        raw = text(case,stopping="0.201")
    elif mutation == "mass":
        raw = raw.replace(b"Mass = 4 amu",b"Mass = 5 amu")
    else:
        raw = raw.replace(b"Al     13",b"Si     14")
    new = parse_srim_range_output(raw)
    with pytest.raises(ValueError):
        subject.validate_native(new,baseline(case),case)


def mock_execution(monkeypatch: pytest.MonkeyPatch, cases: tuple[subject.LowControl,...],
                   mutation: str = "") -> list[Path]:
    calls: list[Path] = []
    def execute(work: Path) -> dict[str,Any]:
        calls.append(work)
        case = next(c for c in cases if work.name.startswith(c.name + "_r"))
        command = {"argv":[str(work/"SRModule.exe")], "cwd":str(work), "pid":123,
                   "returncode":0,"timed_out":False,"elapsed_seconds":0.01,"stdout":"","stderr":""}
        if mutation == "timeout":
            command.update(returncode=-1,timed_out=True)
        (work/"command.json").write_text(json.dumps(command),encoding="utf-8")
        if mutation == "timeout":
            raise RuntimeError("mock timeout")
        raw = text(case, range_factor=D("1.1") if mutation == "repeat" and len(calls)==2 else D(1))
        if mutation == "invalid":
            raw = b"not native output"
        (work/"srim_output.txt").write_bytes(raw)
        if mutation == "input":
            (work/"SR.IN").write_bytes(b"changed")
        if mutation == "command":
            return {**command,"pid":124}
        return command
    monkeypatch.setattr(subject,"_execute",execute)
    return calls


@pytest.mark.parametrize("mutation,status,count", [("","CONTROL_COMPLETE",2),
    ("timeout","NATIVE_FAILED",1),("invalid","INVALID_OUTPUT",1),("repeat","REPEAT_MISMATCH",2)])
def test_bounded_case_retains_success_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  mutation: str, status: str, count: int) -> None:
    case = control()
    calls = mock_execution(monkeypatch,(case,),mutation)
    record = subject._case(case,baseline(case),output=tmp_path,copied_source={"SRModule.exe":b"fake"},
                           runtime=b"fake",pins={})
    assert record["status"] == status and len(calls) == count
    assert not record["normative_export_allowed"] and not record["absolute_range_origin_established"]
    assert len(record["repeats"]) == count
    assert (tmp_path/f"{case.name}.json").exists()
    for repeat in record["repeats"]:
        assert Path(repeat["command_file"]["path"]).exists()
        assert Path(repeat["input"]["path"]).exists()


@pytest.mark.parametrize("mutation",("input","command"))
def test_fail_closed_on_native_input_or_command_tamper(tmp_path: Path,monkeypatch: pytest.MonkeyPatch,
                                                       mutation: str) -> None:
    case = control()
    mock_execution(monkeypatch,(case,),mutation)
    with pytest.raises(ValueError):
        subject._case(case,baseline(case),output=tmp_path,copied_source={"SRModule.exe":b"fake"},
                      runtime=b"fake",pins={})


def test_protected_inputs(tmp_path: Path) -> None:
    source = tmp_path/"protected"
    source.write_bytes(b"initial")
    pins = subject._pins([source])
    subject._verify(pins)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Protected"):
        subject._verify(pins)


def test_six_pair_package_is_loadable_and_text_only(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    cases = tuple(control(z,target) for z in subject.PILOT_Z for target in (13,14))
    mock_execution(monkeypatch,cases)
    records = [subject._case(c,baseline(c),output=tmp_path,copied_source={"SRModule.exe":b"fake"},
                            runtime=b"fake",pins={}) for c in cases]
    report = {"all_native_controls_repeated":True,"cases":records,
              "baseline_bundle":{"sha256":subject.BASELINE_SHA256}}
    path = tmp_path/"report.json"
    path.write_text(subject._json(report),encoding="utf-8")
    output = tmp_path/"observed.zip"
    identity = subject.package(path,report,output)
    loaded = load_srim_hze_dataset(output,expected_sha256=identity["sha256"])
    assert tuple(loaded.tables_by_z) == subject.PILOT_Z
    assert not loaded.normative_export_allowed
    for table in loaded.tables_by_z.values():
        assert table.native_al.rows[0].energy_total_mev == subject.MINIMUM_MEV
    with ZipFile(output) as archive:
        assert len(archive.namelist()) == 7
        assert all(name.endswith(".txt") or name == "manifest.json" for name in archive.namelist())
    with pytest.raises(FileExistsError):
        subject.package(path,report,output)
    report["all_native_controls_repeated"] = False
    with pytest.raises(ValueError,match="complete"):
        subject.package(path,report,tmp_path/"bad.zip")


@pytest.mark.parametrize("output_kind",("exists","outside"))
def test_run_rejects_unsafe_output_before_native_reads(tmp_path: Path, output_kind: str) -> None:
    output = tmp_path if output_kind=="exists" else subject.ROOT/"forbidden_low_pilot_test"
    expected = FileExistsError if output_kind=="exists" else ValueError
    with pytest.raises(expected):
        subject.run(source=tmp_path/"missing",private_runtime=tmp_path/"missing",
                    output=output,bundle=tmp_path/"missing")


@pytest.mark.parametrize("mutation", ("", "timeout", "source_hash", "runtime_hash"))
def test_full_orchestration_with_fake_process_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 mutation: str) -> None:
    # This exercises exclusive output, pre-execution authentication, stopping
    # after failure, protected snapshots and successful text-only packaging.
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    (tmp_path/"outputs").mkdir()
    (tmp_path/"scripts").mkdir()
    for name in ("run_srim_controls.py", "run_srim_coverage.py"):
        (tmp_path/"scripts"/name).write_bytes(b"pinned test implementation")
    source = tmp_path/"source"
    source.mkdir()
    native = b"this is a test, never an executable"
    (source/"SRModule.exe").write_bytes(native)
    (source/"SR.IN").write_bytes(b"user original input")
    monkeypatch.setattr(launcher,"SOURCE_HASHES",{"SRModule.exe":subject._sha(native)})
    runtime = tmp_path/"private.dll"
    runtime.write_bytes(b"test runtime")
    monkeypatch.setattr(launcher,"PRIVATE_RUNTIME_SHA256",subject._sha(runtime.read_bytes()))
    cases = tuple(control(z,target) for z in subject.PILOT_Z for target in (13,14))
    monkeypatch.setattr(subject,"controls",lambda _:tuple((c,baseline(c)) for c in cases))
    bundle = tmp_path/"baseline.zip"
    bundle.write_bytes(b"authenticated fake baseline")
    sha = subject._sha(bundle.read_bytes())
    monkeypatch.setattr(subject,"BASELINE_SHA256",sha)
    def load(path: Path, *, expected_sha256: str) -> Any:
        if path == bundle:
            assert expected_sha256 == sha
            return None  # controls() is the test's explicitly mocked catalog.
        return load_srim_hze_dataset(path,expected_sha256=expected_sha256)
    monkeypatch.setattr(subject,"load_srim_hze_dataset",load)
    calls = mock_execution(monkeypatch,cases,"timeout" if mutation=="timeout" else "")
    output = tmp_path/"outputs/new"
    if mutation in ("source_hash","runtime_hash"):
        (source/"SRModule.exe" if mutation=="source_hash" else runtime).write_bytes(b"changed")
        with pytest.raises(ValueError,match="changed|identity"):
            subject.run(source=source,private_runtime=runtime,output=output,bundle=bundle)
        assert not calls and not output.exists()
        return
    report = subject.run(source=source,private_runtime=runtime,output=output,bundle=bundle)
    assert (source/"SR.IN").read_bytes() == b"user original input"
    assert (source/"SRModule.exe").read_bytes() == native
    assert report["protected_sha256_before"] == report["protected_sha256_after"]
    assert not report["absolute_range_origin_established"]
    if mutation == "timeout":
        assert len(calls)==1 and len(report["not_attempted_cases"])==5
        assert not report["all_native_controls_repeated"]
        assert not (output/"observed_pilot.zip").exists()
    else:
        assert len(calls)==12 and report["all_native_controls_repeated"]
        assert (output/"observed_pilot.zip").exists()
        identity = json.loads((output/"bundle_identity.json").read_text(encoding="utf-8"))
        assert identity["sha256"] == subject._sha((output/"observed_pilot.zip").read_bytes())
    with pytest.raises(FileExistsError):
        subject.run(source=source,private_runtime=runtime,output=output,bundle=bundle)
