"""Full upper campaign guards; native execution is always mocked."""

from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_srim_upper_full as subject
from scripts import run_srim_upper_pilot as pilot
from scripts import run_srim_controls as launcher
from scripts import run_srim_low_energy as low
from radar.validation.srim_output import parse_srim_range_output
from radar.validation.srim_source_catalog import MANIFEST_PATH
from test_srim_upper_pilot import baseline, command, control, text


def full_control(z: int = 2, target: int = 13) -> subject.FullUpperControl:
    return subject.FullUpperControl(**asdict(control(z, target)))


@pytest.mark.parametrize("z", pilot.PILOT_Z)
@pytest.mark.parametrize("target", (13, 14))
def test_pilot_inputs_and_native_checks_identical(z: int, target: int) -> None:
    c = full_control(z, target)
    p = control(z, target)
    assert subject.input_bytes(c, (c,)) == pilot.input_bytes(p)
    native = parse_srim_range_output(text(p))
    assert subject.validate_native(native, baseline(p), c) == pilot.validate_native(
        native, baseline(p), p
    )


def test_ambient_decimal_context_cannot_move_ceiling_or_input() -> None:
    c = full_control(26)
    expected_input = subject.input_bytes(c, (c,))
    native, old = parse_srim_range_output(text(control(26))), baseline(control(26))
    expected = subject.validate_native(native, old, c)
    with localcontext() as context:
        context.prec = 2
        assert subject.endpoint(Decimal("6.9")) == 32500
        assert subject.endpoint(Decimal("20.2")) == 100000
        assert subject.input_bytes(c, (c,)) == expected_input
        assert subject.validate_native(native, old, c) == expected


@pytest.mark.parametrize(
    "bad", (Decimal(0), Decimal("NaN"), Decimal("Infinity"), Decimal(239), 4.0)
)
def test_endpoint_rejects_unbounded_mass(bad: Any) -> None:
    with pytest.raises(ValueError):
        subject.endpoint(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "../escape"),
        ("ion_z", 6),
        ("ion_mass_u", Decimal(5)),
        ("density_g_cm3", Decimal(3)),
        ("energy_max_mev", Decimal(22500)),
    ],
)
def test_input_outside_authenticated_roster_rejected(field: str, value: Any) -> None:
    c = full_control()
    with pytest.raises(ValueError, match="catalog"):
        subject.input_bytes(replace(c, **{field: value}), (c,))


def roster(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    bases, tables = [], {}
    for z in (*range(2, 85), 92):
        mass = pilot._MASSES.get(z, Decimal(z * 2))
        pair = {}
        for target in (13, 14):
            suffix, tmass, density = pilot._TARGETS[target]
            name = pilot._NAMES.get(z, f"z{z}")
            bases.append(
                SimpleNamespace(
                    name=f"{name}_{suffix}",
                    ion_z=z,
                    ion_mass_u=mass,
                    target_z=target,
                    target_mass_u=tmass,
                    density_g_cm3=density,
                )
            )
            pair["native_al" if target == 13 else "native_si"] = SimpleNamespace(
                ion_z=z,
                ion_mass_u=mass,
                target_z=target,
                density_g_cm3=density,
                rows=(
                    SimpleNamespace(energy_total_mev=Decimal(".00000999999")),
                    SimpleNamespace(energy_total_mev=Decimal(12000)),
                ),
            )
        tables[z] = SimpleNamespace(**pair)
    monkeypatch.setattr(subject, "build_source_catalog", lambda root: tuple(bases))
    return SimpleNamespace(tables_by_z=tables), bases


def test_full_roster_exact_bounded_and_pilot_names(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _ = roster(monkeypatch)
    selected = subject.controls(dataset)
    assert len(selected) == 168 and len({c.ion_z for c, _ in selected}) == 84
    for c, _ in selected:
        assert c.energy_max_mev <= c.ion_mass_u * 5000
        assert c.energy_max_mev * 1000 <= 2147483647
    assert {c.name for c, _ in selected if c.ion_z in pilot.PILOT_Z} == {
        control(z, t).name for z in pilot.PILOT_Z for t in (13, 14)
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "duplicate",
        "extra_pair",
        "missing_pair",
        "z",
        "mass",
        "density",
        "lower",
        "upper",
    ),
)
def test_invalid_full_roster_rejected(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    dataset, bases = roster(monkeypatch)
    if mutation == "missing":
        bases.pop()
    elif mutation == "duplicate":
        bases[-1] = bases[-2]
    elif mutation == "extra_pair":
        dataset.tables_by_z[1] = dataset.tables_by_z[2]
    elif mutation == "missing_pair":
        dataset.tables_by_z.pop(92)
    else:
        native = dataset.tables_by_z[2].native_al
        if mutation in ("z", "mass", "density"):
            key = {"z": "ion_z", "mass": "ion_mass_u", "density": "density_g_cm3"}[mutation]
            setattr(native, key, getattr(native, key) + 1)
        else:
            native.rows[0 if mutation == "lower" else 1].energy_total_mev = Decimal(20000)
    with pytest.raises(ValueError):
        subject.controls(dataset)


@pytest.mark.parametrize(
    "mutation", ("prefix_token", "prefix_stopping", "metadata", "missing_tail")
)
def test_native_prefix_and_metadata_rechecked(mutation: str) -> None:
    c, p = full_control(), control()
    raw = text(p)
    if mutation == "prefix_token":
        raw = raw.replace(b"1 MeV 0.2", b"1.0 MeV 0.2")
    elif mutation == "prefix_stopping":
        raw = raw.replace(b"1 MeV 0.2", b"1 MeV 0.3")
    elif mutation == "metadata":
        raw = raw.replace(b"Mass = 4 amu", b"Mass = 5 amu")
    else:
        raw = text(p, maximum=Decimal(18000))
    with pytest.raises(ValueError):
        subject.validate_native(parse_srim_range_output(raw), baseline(p), c)


@pytest.mark.parametrize(
    "outcome", ("success", "failed", "invalid", "mismatch", "raw_mismatch", "copied_changed")
)
def test_case_repeat_and_failure_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    c, p, calls = full_control(), control(), []

    def execute(work: Path) -> dict[str, Any]:
        calls.append(work)
        record = command(work, failure=outcome == "failed")
        (work / "command.json").write_text(json.dumps(record), encoding="utf-8")
        if outcome == "failed":
            raise RuntimeError("bounded timeout")
        raw = b"bad" if outcome == "invalid" else text(p)
        if len(calls) == 2:
            if outcome == "mismatch":
                raw = raw.replace(b"20000 MeV 0.2", b"20000 MeV 0.3")
            elif outcome == "raw_mismatch":
                raw += b"\r\n"
        (work / "srim_output.txt").write_bytes(raw)
        if outcome == "copied_changed":
            (work / "SR.IN").write_bytes(b"changed")
        return record

    monkeypatch.setattr(subject, "_execute", execute)
    kwargs = dict(
        output=tmp_path,
        input_raw=subject.input_bytes(c, (c,)),
        originals={"SRModule.exe": b"never executed"},
        runtime=b"mock",
        pins={},
    )
    if outcome == "copied_changed":
        with pytest.raises(ValueError, match="copied"):
            subject.run_case(c, baseline(p), **kwargs)  # type: ignore[arg-type]
        return
    result = subject.run_case(c, baseline(p), **kwargs)  # type: ignore[arg-type]
    assert (
        result["status"]
        == {
            "success": "CONTROL_COMPLETE",
            "failed": "NATIVE_FAILED",
            "invalid": "INVALID_OUTPUT",
            "mismatch": "REPEAT_MISMATCH",
            "raw_mismatch": "REPEAT_MISMATCH",
        }[outcome]
    )
    assert len(calls) == (2 if outcome in ("success", "mismatch", "raw_mismatch") else 1)
    assert result["origin"] == "executed_here" and result["diagnostic_status"] == "INCOMPLETE"
    assert not result["normative_export_allowed"]


def pilot_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Any, Any]:
    selected = tuple(
        (full_control(z, t), baseline(control(z, t))) for z in pilot.PILOT_Z for t in (13, 14)
    )
    source_files = {"SRModule.exe": b"not executable", "VERSION": b"version"}
    monkeypatch.setattr(
        launcher, "SOURCE_HASHES", {k: subject._sha(v) for k, v in source_files.items()}
    )
    runtime = b"runtime"
    monkeypatch.setattr(launcher, "PRIVATE_RUNTIME_SHA256", subject._sha(runtime))
    cases: list[dict[str, Any]] = []
    tables: dict[int, Any] = {}
    for c, old in selected:
        raw = text(control(c.ion_z, c.target_z))
        native = parse_srim_range_output(raw)
        pair = tables.setdefault(c.ion_z, SimpleNamespace())
        setattr(pair, "native_al" if c.target_z == 13 else "native_si", native)
        repeats = []
        for number in (1, 2):
            work = tmp_path / f"{c.name}_r{number}"
            work.mkdir()
            for name, data in {
                **source_files,
                "MSVBVM50.dll": runtime,
                "SR.IN": subject.input_bytes(c, (c,)),
                "srim_output.txt": raw,
            }.items():
                (work / name).write_bytes(data)
            cmd = command(work)
            (work / "command.json").write_text(json.dumps(cmd), encoding="utf-8")
            repeats.append(
                {
                    "repeat": number,
                    "status": "VALID_NATIVE_OUTPUT",
                    "input": subject._identity(work / "SR.IN"),
                    "output": subject._identity(work / "srim_output.txt"),
                    "command": cmd,
                    "command_file": subject._identity(work / "command.json"),
                    "semantic_sha256": native.semantic_sha256,
                    "comparison": subject.validate_native(native, old, c),
                    **subject.FLAGS,
                }
            )
        cases.append(
            {
                "case": asdict(c),
                "baseline_raw_sha256": old.raw_sha256,
                "status": "CONTROL_COMPLETE",
                "repeats": repeats,
                "semantic_outputs_identical": True,
                "raw_output_bytes_identical": True,
                **subject.FLAGS,
            }
        )
    protected = tmp_path / "source.txt"
    protected.write_bytes(b"source calculation unchanged")
    pins = {str(protected): subject._sha(protected.read_bytes())}
    report = {
        "schema_version": "radar.srim_upper_pilot.v1",
        "scope": "pilot",
        "runs_per_case": 2,
        "all_native_controls_repeated": True,
        "not_attempted_cases": [],
        "cases": cases,
        "baseline_bundle": {"sha256": pilot.BASELINE_SHA256},
        "protected_sha256_before": pins,
        "protected_sha256_after": pins,
        **subject.FLAGS,
    }
    path = tmp_path / "report.json"
    path.write_text(subject._json(report), encoding="utf-8")
    monkeypatch.setattr(subject, "PILOT_REPORT_SHA256", subject._sha(path.read_bytes()))
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"mock bundle")
    monkeypatch.setattr(
        subject, "load_srim_hze_dataset", lambda *a, **k: SimpleNamespace(tables_by_z=tables)
    )
    return path, bundle, selected, report


def test_pilot_reused_only_after_both_repeats_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, bundle, selected, _ = pilot_fixture(tmp_path, monkeypatch)
    pins: dict[str, str] = {}
    reused = subject.authenticate_pilot(path, bundle, selected, pins)
    assert len(reused) == 8 and len(pins) > 90
    assert all(r["origin"] == "reviewed_pilot_reused" for r in reused.values())
    low._verify(pins)


@pytest.mark.parametrize(
    "mutation",
    (
        "report_hash",
        "code_hash",
        "source",
        "second_output",
        "input",
        "binary",
        "runtime",
        "command",
        "duplicate",
        "flag",
        "incomplete",
        "comparison",
    ),
)
def test_changed_pilot_evidence_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path, bundle, selected, report = pilot_fixture(tmp_path, monkeypatch)
    record = report["cases"][0]
    repeat = record["repeats"][1]
    work = tmp_path / f"{selected[0][0].name}_r2"
    if mutation == "report_hash":
        monkeypatch.setattr(subject, "PILOT_REPORT_SHA256", "0" * 64)
    elif mutation == "code_hash":
        monkeypatch.setattr(subject, "PILOT_CODE_SHA256", "0" * 64)
    elif mutation in ("second_output", "input", "binary", "runtime", "source"):
        name = {
            "second_output": "srim_output.txt",
            "input": "SR.IN",
            "binary": "SRModule.exe",
            "runtime": "MSVBVM50.dll",
            "source": "source.txt",
        }[mutation]
        ((tmp_path if mutation == "source" else work) / name).write_bytes(b"tampered")
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
            repeat["comparison"]["new_upper_rows"] = 129
        path.write_text(subject._json(report), encoding="utf-8")
        monkeypatch.setattr(subject, "PILOT_REPORT_SHA256", subject._sha(path.read_bytes()))
    with pytest.raises(ValueError):
        subject.authenticate_pilot(path, bundle, selected, {})


@pytest.mark.parametrize("complete,count", ((False, 168), (True, 167), (True, 0)))
def test_incomplete_campaign_not_packaged(tmp_path: Path, complete: bool, count: int) -> None:
    with pytest.raises(ValueError):
        subject.package(
            tmp_path / "report",
            {"all_native_controls_repeated": complete, "cases": [{}] * count},
            tmp_path / "bundle.zip",
        )
    assert not (tmp_path / "bundle.zip").exists()


@pytest.mark.parametrize("existing", (False, True))
def test_output_exclusive_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path / "repository")
    with pytest.raises(FileExistsError if existing else ValueError):
        subject.run(
            source=tmp_path,
            private_runtime=tmp_path / "dll",
            bundle=tmp_path / "zip",
            pilot_report=tmp_path / "report",
            pilot_bundle=tmp_path / "pilot",
            output=tmp_path if existing else tmp_path / "outside",
        )


@pytest.mark.parametrize("failure", (False, True))
def test_campaign_reuses_eight_runs_only_160_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    dataset, _ = roster(monkeypatch)
    selected = subject.controls(dataset)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "HELPER_SHA256", {})
    (tmp_path / "outputs").mkdir()
    (tmp_path / MANIFEST_PATH).write_bytes(b"mock manifest")
    source = tmp_path / "source"
    source.mkdir()
    (source / "SRModule.exe").write_bytes(b"never executable")
    (source / "SR.IN").write_bytes(b"original")
    runtime, bundle, pr, pb = (
        tmp_path / s for s in ("runtime", "bundle", "pilotreport", "pilotbundle")
    )
    for p in (runtime, bundle, pr, pb):
        p.write_bytes(b"mock")
    monkeypatch.setattr(
        launcher, "SOURCE_HASHES", {"SRModule.exe": subject._sha(b"never executable")}
    )
    monkeypatch.setattr(launcher, "PRIVATE_RUNTIME_SHA256", subject._sha(b"mock"))
    monkeypatch.setattr(subject, "load_srim_hze_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(
        subject,
        "authenticate_pilot",
        lambda *a: {
            c.name: {
                "case": asdict(c),
                "status": "CONTROL_COMPLETE",
                "origin": "reviewed_pilot_reused",
            }
            for c, _ in selected
            if c.ion_z in pilot.PILOT_Z
        },
    )
    calls, packages = [], []

    def case(c: subject.FullUpperControl, old: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(c.name)
        return {
            "case": asdict(c),
            "status": "NATIVE_FAILED" if failure else "CONTROL_COMPLETE",
            "origin": "executed_here",
        }

    def package(path: Path, report: Any, output: Path) -> dict[str, Any]:
        packages.append(output)
        output.write_bytes(b"mock zip")
        return subject._identity(output)

    monkeypatch.setattr(subject, "run_case", case)
    monkeypatch.setattr(subject, "package", package)
    result = subject.run(
        source=source,
        private_runtime=runtime,
        bundle=bundle,
        pilot_report=pr,
        pilot_bundle=pb,
        output=tmp_path / "outputs/run",
    )
    assert len(calls) == (1 if failure else 160)
    assert len(packages) == (0 if failure else 1)
    assert result["reused_pilot_cases"] == (2 if failure else 8)
    assert len(result["not_attempted_cases"]) == (165 if failure else 0)
    assert result["protected_sha256_before"] == result["protected_sha256_after"]
    assert not result["normative_export_allowed"]
