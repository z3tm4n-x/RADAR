"""Bounded upper SRIM pilot contracts; native process execution is mocked."""
from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pytest

from scripts import run_srim_upper_pilot as subject
from scripts import run_srim_low_energy as low
from scripts import run_srim_controls as launcher
from radar.shielding.srim_dataset import load_srim_hze_dataset
from radar.validation.srim_output import parse_srim_range_output
from test_srim_low_energy_runner import text as low_text

D = Decimal


def control(z: int = 2, target: int = 13) -> subject.UpperControl:
    suffix, mass, density = subject._TARGETS[target]
    maximum = subject.endpoint(subject._MASSES[z])
    return subject.UpperControl(f"{subject._NAMES[z]}_{suffix}_upper{maximum:f}", z,
                                subject._MASSES[z], target, mass, density, maximum)


def text(case: subject.UpperControl, *, maximum: Decimal | None = None) -> bytes:
    # Synthetic table with a complete normal grid; not physical stopping data.
    base = low.LowControl("synthetic", case.ion_z, case.ion_mass_u, case.target_z,
                          case.target_mass_u, case.density_g_cm3, D(10))
    template = low_text(base).decode("cp1251")
    lines = template.splitlines()
    indices = [i for i, line in enumerate(lines) if re.match(r"\s*[\d.]+\s+MeV\s+", line)]
    grid = subject.normal_grid(maximum or case.energy_max_mev)
    rows = [f"{e} MeV 0.2 0.001 {e*1000} um {e*10} um {e*5} um" for e in grid]
    rows[0] = rows[0].replace("0.00001 MeV", "9.99999 eV")
    return ("\r\n".join(lines[:indices[0]] + rows + lines[indices[-1]+1:]) + "\r\n").encode("cp1251")


def baseline(case: subject.UpperControl) -> Any:
    maximum = {2: D(12000), 10: D(20000), 26: D(55000), 92: D(225000)}[case.ion_z]
    return parse_srim_range_output(text(case, maximum=maximum))


@pytest.mark.parametrize("z,maximum", [(2,20000),(10,100000),(26,275000),(92,1100000)])
@pytest.mark.parametrize("target", (13,14))
def test_exact_pilot_input_and_whole_extension(z: int, maximum: int, target: int) -> None:
    case = control(z,target)
    assert case.energy_max_mev == maximum
    assert maximum <= 5000 * case.ion_mass_u
    assert maximum * 1000 <= 2147483647
    raw = subject.input_bytes(case)
    assert raw.endswith(f"0.01    {maximum*1000}\r\n".encode())
    assert f"{z}   {case.ion_mass_u}".encode() in raw
    new = parse_srim_range_output(text(case))
    old = baseline(case)
    result = subject.validate_native(new, old, case)
    assert result["parent_prefix_identical"] is True
    assert result["new_upper_rows"] > 0
    assert result["observed_minimum_total_ion_energy_mev"] == D(".00000999999")
    assert new.rows[:len(old.rows)] == old.rows


@pytest.mark.parametrize("field,value", [("ion_mass_u",D("4.1")),("ion_mass_u",4.),
    ("ion_z",True),("ion_z",3),("target_z",12),("target_mass_u",D(27)),
    ("density_g_cm3",D(3)),("energy_max_mev",D(30000)),("name","../escape")])
def test_noncanonical_case_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(control(), **{field:value})


@pytest.mark.parametrize("bad", [D(0),D(-1),D("NaN"),D("Infinity"),D("2147483.648"),100])
def test_grid_engineering_guard(bad: Any) -> None:
    with pytest.raises(ValueError):
        subject.normal_grid(bad)


@pytest.mark.parametrize("mutation", ["prefix_stopping", "prefix_range", "prefix_token",
                                       "missing_tail", "missing_middle", "metadata", "upper"])
def test_bad_native_never_accepted(mutation: str) -> None:
    c = control()
    raw = text(c)
    if mutation == "prefix_stopping":
        raw = raw.replace(b"1 MeV 0.2", b"1 MeV 0.3")
    elif mutation == "prefix_range":
        raw = raw.replace(b"1 MeV 0.2 0.001 1000 um", b"1 MeV 0.2 0.001 1001 um")
    elif mutation == "prefix_token":
        raw = raw.replace(b"1 MeV 0.2", b"1.0 MeV 0.2")
    elif mutation == "metadata":
        raw = raw.replace(b"Mass = 4 amu", b"Mass = 5 amu")
    elif mutation == "missing_tail":
        raw = text(c, maximum=D(18000))
    elif mutation == "upper":
        raw = text(c, maximum=D(22500))
    else:
        raw = re.sub(rb"15000(?:\.0)? MeV[^\r\n]+\r\n", b"", raw)
    with pytest.raises(ValueError):
        subject.validate_native(parse_srim_range_output(raw), baseline(c), c)


def fake_catalog(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    bases = []
    tables = {}
    for z in subject.PILOT_Z:
        pair = {}
        for target in (13,14):
            c = control(z,target)
            bases.append(SimpleNamespace(name=f"{subject._NAMES[z]}_{subject._TARGETS[target][0]}",
                ion_z=z, ion_mass_u=c.ion_mass_u, target_z=target,
                target_mass_u=c.target_mass_u, density_g_cm3=c.density_g_cm3))
            pair["native_al" if target == 13 else "native_si"] = baseline(c)
        tables[z] = SimpleNamespace(**pair)
    monkeypatch.setattr(subject,"build_source_catalog",lambda root:tuple(bases))
    return SimpleNamespace(tables_by_z=tables), bases


def test_catalog_all_eight_and_no_other_species(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, bases = fake_catalog(monkeypatch)
    result = subject.controls(dataset)
    assert len(result) == 8
    assert {c.ion_z for c,_ in result} == set(subject.PILOT_Z)
    bases.append(SimpleNamespace(ion_z=6))
    assert subject.controls(dataset) == result
    bases.pop()
    bases.pop()
    with pytest.raises(ValueError,match="eight"):
        subject.controls(dataset)


def command(work: Path, *, failure: bool = False) -> dict[str, Any]:
    return {"argv":[str(work/"SRModule.exe")],"cwd":str(work),"pid":123,
            "returncode":1 if failure else 0,"timed_out":failure,
            "elapsed_seconds":.01,"stdout":"","stderr":""}


@pytest.mark.parametrize("outcome", ["success","failed","invalid","mismatch","copied_changed"])
def test_run_case_repeats_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    c = control()
    calls = []
    def execute(work: Path) -> dict[str, Any]:
        calls.append(work)
        record = command(work,failure=outcome=="failed")
        (work/"command.json").write_text(json.dumps(record),encoding="utf-8")
        if outcome == "failed":
            raise RuntimeError("bounded timeout")
        raw = b"invalid" if outcome == "invalid" else text(c)
        if outcome == "mismatch" and len(calls)==2:
            raw = raw.replace(b"20000 MeV 0.2",b"20000 MeV 0.3")
        (work/"srim_output.txt").write_bytes(raw)
        if outcome == "copied_changed":
            (work/"SR.IN").write_bytes(b"modified")
        return record
    monkeypatch.setattr(subject,"_execute",execute)
    kwargs = dict(output=tmp_path,originals={"SRModule.exe":b"not executable"},runtime=b"mock",pins={})
    if outcome == "copied_changed":
        with pytest.raises(ValueError,match="copied"):
            subject.run_case(c,baseline(c),**kwargs)  # type: ignore[arg-type]
        return
    result = subject.run_case(c,baseline(c),**kwargs)  # type: ignore[arg-type]
    wanted = {"success":"CONTROL_COMPLETE","failed":"NATIVE_FAILED",
              "invalid":"INVALID_OUTPUT","mismatch":"REPEAT_MISMATCH"}[outcome]
    assert result["status"] == wanted
    assert len(calls) == (2 if outcome in ("success","mismatch") else 1)
    assert not result["normative_export_allowed"]
    assert not result["source_energy_domain_covered"]
    assert (tmp_path/f"{c.name}.json").exists()


def package_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    records = []
    for z in subject.PILOT_Z:
        for target in (13,14):
            c = control(z,target)
            path = tmp_path/f"{c.name}.txt"
            path.write_bytes(text(c))
            records.append({"case":asdict(c),"status":"CONTROL_COMPLETE",
                            "repeats":[{"output":subject._identity(path)}]})
    report = {"cases":records,"all_native_controls_repeated":True}
    path = tmp_path/"report.json"
    path.write_text(subject._json(report),encoding="utf-8")
    return path,report


def test_package_text_only_complete_compatible_and_exclusive(tmp_path: Path) -> None:
    report_path, report = package_fixture(tmp_path)
    output = tmp_path/"bundle.zip"
    result = subject.package(report_path,report,output)
    assert result["sha256"] == subject._sha(output.read_bytes())
    loaded = load_srim_hze_dataset(output,expected_sha256=result["sha256"])
    assert tuple(loaded.tables_by_z)==subject.PILOT_Z
    assert loaded.normative_export_allowed is False
    with ZipFile(output) as archive:
        assert len(archive.namelist())==9
        assert all(n.endswith((".txt",".json")) for n in archive.namelist())
    with pytest.raises(FileExistsError):
        subject.package(report_path,report,output)


@pytest.mark.parametrize("mutation", ["incomplete","missing","failed","tamper","duplicate"])
def test_package_rejects_bad_evidence(tmp_path: Path, mutation: str) -> None:
    path,report = package_fixture(tmp_path)
    if mutation == "incomplete":
        report["all_native_controls_repeated"]=False
    elif mutation == "missing":
        report["cases"].pop()
    elif mutation == "failed":
        report["cases"][0]["status"]="NATIVE_FAILED"
    elif mutation == "tamper":
        Path(report["cases"][0]["repeats"][0]["output"]["path"]).write_bytes(b"changed")
    else:
        report["cases"][1]=report["cases"][0]
    with pytest.raises(ValueError):
        subject.package(path,report,tmp_path/"bad.zip")


def test_run_rejects_existing_or_outside_directory_before_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    with pytest.raises(FileExistsError):
        subject.run(source=tmp_path,private_runtime=tmp_path/"dll",output=tmp_path,bundle=tmp_path/"zip")
    with pytest.raises(ValueError,match="outputs"):
        subject.run(source=tmp_path,private_runtime=tmp_path/"dll",output=tmp_path/"outside",bundle=tmp_path/"zip")


@pytest.mark.parametrize("failure", [False,True])
def test_campaign_stops_on_first_failure_and_only_packages_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool,
) -> None:
    monkeypatch.setattr(subject,"ROOT",tmp_path)
    (tmp_path/"outputs").mkdir()
    (tmp_path/"scripts").mkdir()
    (tmp_path/"scripts/run_srim_coverage.py").write_bytes(b"protected test helper")
    source = tmp_path/"source"
    source.mkdir()
    (source/"SRModule.exe").write_bytes(b"never executed")
    (source/"SR.IN").write_bytes(b"original protected input")
    runtime = tmp_path/"runtime.dll"
    runtime.write_bytes(b"mock runtime")
    bundle = tmp_path/"parent.zip"
    bundle.write_bytes(b"mock parent bundle")
    monkeypatch.setattr(launcher,"SOURCE_HASHES",{"SRModule.exe":subject._sha(b"never executed")})
    monkeypatch.setattr(launcher,"PRIVATE_RUNTIME_SHA256",subject._sha(runtime.read_bytes()))
    monkeypatch.setattr(subject,"load_srim_hze_dataset",lambda *a,**k:object())
    selected = tuple((control(z,t),baseline(control(z,t))) for z in subject.PILOT_Z for t in (13,14))
    monkeypatch.setattr(subject,"controls",lambda dataset:selected)
    calls = []
    def case(c: subject.UpperControl, old: Any, **kwargs: Any) -> dict[str,Any]:
        calls.append(c.name)
        return {"case":asdict(c),"status":"NATIVE_FAILED" if failure else "CONTROL_COMPLETE"}
    monkeypatch.setattr(subject,"run_case",case)
    packages = []
    def package(path: Path, report: Any, output: Path) -> dict[str, Any]:
        packages.append(output)
        output.write_bytes(b"mocked bundle")
        return subject._identity(output)
    monkeypatch.setattr(subject,"package",package)
    out = tmp_path/"outputs/run"
    result = subject.run(source=source,private_runtime=runtime,output=out,bundle=bundle)
    assert len(calls)==(1 if failure else 8)
    assert len(packages)==(0 if failure else 1)
    assert len(result["not_attempted_cases"])==(7 if failure else 0)
    assert result["all_native_controls_repeated"] is not failure
    assert result["protected_sha256_before"]==result["protected_sha256_after"]
    assert (source/"SR.IN").read_bytes()==b"original protected input"
    assert (out/"report.json").exists()
    assert result["normative_export_allowed"] is False
