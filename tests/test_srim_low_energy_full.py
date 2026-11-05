"""Full lower-energy campaign contracts; native execution is always mocked."""
from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_srim_low_energy_full as subject
from scripts import run_srim_low_energy as pilot
from scripts import run_srim_controls as launcher
from radar.validation.srim_output import parse_srim_range_output
from test_srim_low_energy_runner import baseline, control, text


def _command(work: Path, code: int = 0) -> dict[str, Any]:
    return {"argv": [str(work / "SRModule.exe")], "cwd": str(work), "pid": 123,
            "returncode": code, "timed_out": False, "elapsed_seconds": .01,
            "stdout": "", "stderr": ""}


@pytest.mark.parametrize("z", (2, 26, 92))
@pytest.mark.parametrize("target", (13, 14))
def test_input_is_identical_to_reviewed_pilot(z: int, target: int) -> None:
    c = control(z, target)
    assert subject.input_bytes(c, (c,)) == pilot.input_bytes(c)


@pytest.mark.parametrize("field,value", [("name", "../../outside"), ("ion_mass_u", Decimal(56)),
                                       ("ion_z", 3), ("energy_max_mev", Decimal(12000)),
                                       ("density_g_cm3", Decimal(9))])
def test_inputs_outside_validated_catalog_rejected(field: str, value: Any) -> None:
    c = control()
    with pytest.raises(ValueError, match="catalog"):
        subject.input_bytes(replace(c, **{field: value}), (c,))


def test_nonpilot_input_keeps_literal_mass_and_total_energy() -> None:
    c = pilot.LowControl("c12_al_low10ev", 6, Decimal(12), 13,
                                Decimal("26.981539"), Decimal("2.70"), Decimal(12000))
    raw = subject.input_bytes(c, (c,))
    assert b"6   12\r\n" in raw
    assert raw.endswith(b"0.01    12000000\r\n")


def roster(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    bases = []
    tables = {}
    for z in range(2, 86):
        native_pair = {}
        for target in (13, 14):
            density = Decimal("2.70" if target == 13 else "2.3290")
            bases.append(SimpleNamespace(name=f"z{z}t{target}", ion_z=z,
                                         ion_mass_u=Decimal(z*2), target_z=target,
                                         target_mass_u=Decimal(target*2), density_g_cm3=density))
            native_pair["native_al" if target == 13 else "native_si"] = SimpleNamespace(
                ion_mass_u=Decimal(z*2), target_z=target, density_g_cm3=density,
                rows=(SimpleNamespace(energy_total_mev=Decimal(1)),
                      SimpleNamespace(energy_total_mev=Decimal(12000))))
        tables[z] = SimpleNamespace(**native_pair)
    monkeypatch.setattr(subject, "build_source_catalog", lambda root: tuple(bases))
    return SimpleNamespace(tables_by_z=tables), bases


def test_complete_roster_and_unchanged_upper(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _ = roster(monkeypatch)
    cases = subject.controls(dataset)
    assert len(cases) == 168
    assert len({c.ion_z for c, _ in cases}) == 84
    assert all(c.energy_max_mev == 12000 for c, _ in cases)
    assert all(c.name.endswith("_low10ev") for c, _ in cases)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "mass", "density", "lower", "upper"))
def test_bad_roster_rejected(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    dataset, bases = roster(monkeypatch)
    if mutation == "missing":
        bases.pop()
    elif mutation == "duplicate":
        bases[-1] = bases[-2]
    else:
        native = dataset.tables_by_z[2].native_al
        if mutation == "mass":
            native.ion_mass_u += 1
        elif mutation == "density":
            native.density_g_cm3 += 1
        else:
            native.rows[0 if mutation == "lower" else 1].energy_total_mev = Decimal("1.25")
    with pytest.raises(ValueError):
        subject.controls(dataset)


def test_read_identity_authenticates_path_hash_size(tmp_path: Path) -> None:
    path = tmp_path / "native.txt"
    path.write_bytes(b"retained")
    record = subject._identity(path)
    pins: dict[str, str] = {}
    assert subject._read_identity(record, path, pins) == b"retained"
    assert pins[str(path.resolve())] == subject._sha(b"retained")
    for changed, expected in (({**record, "bytes": 1}, path),
                              ({**record, "sha256": "0"*64}, path),
                              (record, tmp_path / "other")):
        with pytest.raises(ValueError):
            subject._read_identity(changed, expected, {})


@pytest.mark.parametrize("outcome", ("valid", "nonzero", "invalid", "repeat_mismatch", "mutate_input"))
def test_case_repeats_and_failure_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     outcome: str) -> None:
    c = control()
    old = baseline(c)
    calls = []
    def execute(work: Path) -> dict[str, Any]:
        calls.append(work)
        command = _command(work, 1 if outcome == "nonzero" else 0)
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        if outcome == "nonzero":
            raise RuntimeError("native failure")
        raw = b"bad" if outcome == "invalid" else text(c)
        if outcome == "repeat_mismatch" and len(calls) == 2:
            # A lower-only stopping alteration keeps the shared suffix intact.
            raw = raw.replace(b"0.00001 MeV 0.2", b"0.00001 MeV 0.3")
        (work / "srim_output.txt").write_bytes(raw)
        if outcome == "mutate_input":
            (work / "SR.IN").write_bytes(b"changed")
        return command
    monkeypatch.setattr(subject, "_execute", execute)
    kwargs = dict(output=tmp_path, input_raw=subject.input_bytes(c, (c,)),
                  originals={"SRModule.exe": b"not executed"}, runtime=b"mock", pins={})
    if outcome == "mutate_input":
        with pytest.raises(ValueError, match="copied"):
            subject.run_case(c, old, **kwargs)  # type: ignore[arg-type]
        return
    result = subject.run_case(c, old, **kwargs)  # type: ignore[arg-type]
    wanted = {"valid": "CONTROL_COMPLETE", "nonzero": "NATIVE_FAILED",
              "invalid": "INVALID_OUTPUT", "repeat_mismatch": "REPEAT_MISMATCH"}
    assert result["status"] == wanted[outcome]
    assert len(calls) == (2 if outcome in ("valid", "repeat_mismatch") else 1)
    assert result["origin"] == "executed_here"
    assert result["normative_export_allowed"] is False
    assert result["absolute_range_origin_established"] is False
    assert (tmp_path / f"{c.name}.json").exists()


def pilot_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Any, Any]:
    selected = tuple((control(z, target), baseline(control(z, target)))
                     for z in (2, 26, 92) for target in (13, 14))
    source_files = {"SRModule.exe": b"not executable", "VERSION": b"version"}
    monkeypatch.setattr(launcher, "SOURCE_HASHES", {k: subject._sha(v) for k,v in source_files.items()})
    runtime = b"runtime"
    monkeypatch.setattr(launcher, "PRIVATE_RUNTIME_SHA256", subject._sha(runtime))
    cases = []
    tables: dict[int, Any] = {}
    for c, old in selected:
        raw = text(c)
        native = parse_srim_range_output(raw)
        table = tables.setdefault(c.ion_z, SimpleNamespace())
        setattr(table, "native_al" if c.target_z == 13 else "native_si", native)
        repeats = []
        for number in (1, 2):
            work = tmp_path / f"{c.name}_r{number}"
            work.mkdir()
            for name, data in {**source_files, "MSVBVM50.dll": runtime,
                               "SR.IN": subject.input_bytes(c, tuple(c for c,_ in selected)),
                               "srim_output.txt": raw}.items():
                (work / name).write_bytes(data)
            command = _command(work)
            (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
            repeats.append({"repeat": number, "status": "VALID_NATIVE_OUTPUT",
                            "input": subject._identity(work / "SR.IN"),
                            "output": subject._identity(work / "srim_output.txt"),
                            "command": command, "command_file": subject._identity(work / "command.json"),
                            "semantic_sha256": native.semantic_sha256,
                            "comparison": pilot.validate_native(native, old, c), **subject.FLAGS})
        cases.append({"case": asdict(c), "baseline_raw_sha256": old.raw_sha256,
                      "status": "CONTROL_COMPLETE", "repeats": repeats, **subject.FLAGS})
    report = {"schema_version": "radar.srim_low_energy_pilot.v1", "scope": "pilot",
              "all_native_controls_repeated": True, "not_attempted_cases": [], "cases": cases,
              "baseline_bundle": {"sha256": pilot.BASELINE_SHA256},
              "protected_sha256_before": {}, "protected_sha256_after": {}, **subject.FLAGS}
    path = tmp_path / "report.json"
    path.write_text(subject._json(report), encoding="utf-8")
    monkeypatch.setattr(subject, "PILOT_REPORT_SHA256", subject._sha(path.read_bytes()))
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"mocked bundle")
    monkeypatch.setattr(subject, "load_srim_hze_dataset", lambda *a, **k: SimpleNamespace(tables_by_z=tables))
    return path, bundle, selected, report


def test_reuse_authenticates_both_repeats_and_keeps_decimal_case(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    path, bundle, selected, _ = pilot_fixture(tmp_path, monkeypatch)
    pins: dict[str, str] = {}
    reused = subject.authenticate_pilot(path, bundle, selected, pins)
    assert len(reused) == 6
    assert len(pins) == 75
    assert all(r["origin"] == "reviewed_pilot_reused" for r in reused.values())
    assert all(isinstance(r["case"]["ion_mass_u"], Decimal) for r in reused.values())
    pilot._verify(pins)


@pytest.mark.parametrize("mutation", ("report_hash", "code_hash", "second_output", "input", "binary",
                                      "runtime", "command", "duplicate", "flag", "incomplete", "comparison"))
def test_pilot_reuse_rejects_changed_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            mutation: str) -> None:
    path, bundle, selected, report = pilot_fixture(tmp_path, monkeypatch)
    record = report["cases"][0]
    repeat = record["repeats"][1]
    work = tmp_path / f"{selected[0][0].name}_r2"
    if mutation == "report_hash":
        monkeypatch.setattr(subject, "PILOT_REPORT_SHA256", "0"*64)
    elif mutation == "code_hash":
        monkeypatch.setattr(subject, "PILOT_CODE_SHA256", "0"*64)
    elif mutation in ("second_output", "input", "binary", "runtime"):
        name = {"second_output":"srim_output.txt", "input":"SR.IN", "binary":"SRModule.exe",
                "runtime":"MSVBVM50.dll"}[mutation]
        (work / name).write_bytes(b"tampered")
    else:
        if mutation == "command":
            repeat["command"]["returncode"] = 1
        elif mutation == "duplicate":
            report["cases"][1] = record
        elif mutation == "flag":
            repeat["normative_export_allowed"] = True
        elif mutation == "incomplete":
            record["repeats"].pop()
        else:
            repeat["comparison"]["new_lower_rows"] = 129
        path.write_text(subject._json(report), encoding="utf-8")
        monkeypatch.setattr(subject, "PILOT_REPORT_SHA256", subject._sha(path.read_bytes()))
    with pytest.raises(ValueError):
        subject.authenticate_pilot(path, bundle, selected, {})


@pytest.mark.parametrize("complete,count", [(False,168), (True,167), (True,0)])
def test_package_requires_full_success(tmp_path: Path, complete: bool, count: int) -> None:
    with pytest.raises(ValueError):
        subject.package(tmp_path / "report.json", {"all_native_controls_repeated":complete,
                                                    "cases":[{}]*count}, tmp_path / "full.zip")
    assert not (tmp_path / "full.zip").exists()


@pytest.mark.parametrize("existing", (False,True))
def test_output_is_exclusive_and_repo_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           existing: bool) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path / "repository")
    destination = tmp_path if existing else tmp_path / "outside"
    with pytest.raises(FileExistsError if existing else ValueError):
        subject.run(source=tmp_path, private_runtime=tmp_path / "runtime", output=destination,
                    bundle=tmp_path / "bundle", pilot_report=tmp_path / "report",
                    pilot_bundle=tmp_path / "pilot")
