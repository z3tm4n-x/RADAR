"""Synthetic candidate replay controls; no native execution or production writes."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.validation.srim_anchored_range import AnchoredSrimRangeTable
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output


D = Decimal


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/build_srim_range_candidate.py"
    spec = importlib.util.spec_from_file_location("srim_candidate_runner_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_lines() -> list[str]:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    return [line.strip() if "(C)" in line else re.sub(r"(?<=[0-9]),(?=[0-9])", ".", line.strip())
            for line in fixture.read_bytes().decode("cp1251").splitlines() if line.strip()]


def native(*, target: int = 13, mass: str = "12", altered: bool = False) -> SrimRangeOutput:
    lines = _fixture_lines()
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[6] = header[17] = f"Ion = Carbon [6] , Mass = {mass} amu"
    header[7] = "Density = 2 g/cm3 = 6E22 atoms/cm3"
    header[12] = "Al 13 100 100" if target == 13 else "Si 14 100 100"
    rows = [f"{e} MeV {'0.002' if altered else '0.001'} 0.001 {D(e)/4} cm 1 A 1 A"
            for e in (1, 2, 4, 8)]
    footer = lines[lines.index("-" * 59):]
    return parse_srim_range_output(("\r\n".join(header + rows + footer) + "\r\n").encode("ascii"))


def raw_native(*, target: int = 13) -> bytes:
    parsed = native(target=target)
    lines = _fixture_lines()
    rows = [" ".join(row.printed_tokens) for row in parsed.rows]
    footer = lines[lines.index("-" * 59):]
    return ("\r\n".join((*parsed.metadata_lines, *rows, *footer)) + "\r\n").encode("ascii")


@pytest.fixture
def pinned(runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path / "repo"
    root.mkdir()

    def write(relative: str, raw: bytes) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    for relative in runner.IMPLEMENTATION:
        write(relative, b"SYNTHETIC IMPLEMENTATION PIN")
    parent_implementation = write("parent_implementation.py", b"SYNTHETIC PARENT IMPLEMENTATION")
    source = write("source_parameters.py", b"SYNTHETIC SOURCE PIN")
    protected = write("protected_native.dat", b"SYNTHETIC NATIVE DEPENDENCY")
    source_pins = {"source_parameters.py": runner.replay._sha(source)}
    mass_path = write(runner.MASS_MANIFEST, runner.native_controls._json({
        "source_files_sha256_before": source_pins,
        "source_files_sha256_after": source_pins.copy(),
    }).encode())
    cases = []
    for name, target in (("c12_al", 13), ("c12_si", 14)):
        repeats = []
        for number in (1, 2):
            data: dict[str, Any] = {}
            for key, raw in (("input", b"SYNTHETIC INPUT"),
                             ("output", raw_native(target=target)),
                             ("command_file", b"SYNTHETIC COMMAND NEVER EXECUTED")):
                path = write(f"native/{name}_{number}_{key}.txt", raw)
                data[key] = {"path": str(path), "bytes": len(raw),
                             "sha256": runner.replay._sha(path)}
            parsed = parse_srim_range_output(raw_native(target=target))
            repeats.append({**data, "status": "VALID_NATIVE_OUTPUT",
                            "semantic_sha256": parsed.semantic_sha256,
                            "parsed": json.loads(runner.native_controls._json(asdict(parsed))),
                            "comparison": {}})
        cases.append({"case": {"name": name}, "status": "CONTROL_COMPLETE", "repeats": repeats})
    identity = {str(protected): runner.replay._sha(protected)}
    report = {"protected_sha256_before": identity.copy(),
              "protected_sha256_after": identity.copy(), "cases": cases}
    catalog_path = root / "outputs/srim_legacy_baseline_final_20261101/report.json"
    parent_path = root / runner.PARENT_REPORT
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner.replay, "ROOT", root)
    monkeypatch.setattr(runner.replay, "REPORT_PINS", {})
    monkeypatch.setattr(runner, "MASS_MANIFEST_SHA256", runner.replay._sha(mass_path))
    monkeypatch.setattr(runner, "SELECTION", (("baseline", "c12_al"), ("baseline", "c12_si")))
    monkeypatch.setattr(runner.native_controls, "CATALOGS", {
        "baseline": (SimpleNamespace(name="c12_al"), SimpleNamespace(name="c12_si")),
    })
    monkeypatch.setattr(runner.native_controls, "expected_control_grid",
                        lambda case: (D(1), D(2), D(4), D(8)))

    def save() -> None:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(runner.native_controls._json(report), encoding="utf-8")
        runner.replay.REPORT_PINS["baseline"] = runner.replay._sha(catalog_path)
        parent_path.write_text(runner.native_controls._json({
            "report_sha256": runner.replay.REPORT_PINS.copy(),
            "implementation_sha256": {"parent_implementation.py": runner.replay._sha(parent_implementation)},
        }), encoding="utf-8")
        monkeypatch.setattr(runner, "PARENT_SHA256", runner.replay._sha(parent_path))

    save()
    return SimpleNamespace(root=root, source=source, protected=protected, mass_path=mass_path,
                           parent_path=parent_path, parent_implementation=parent_implementation,
                           catalog_path=catalog_path, report=report, save=save)


def test_fixed_selection_and_shields(runner: ModuleType) -> None:
    assert len(runner.SELECTION) == 32 and len(set(runner.SELECTION)) == 32
    assert runner.SELECTION[-2:] == (("boundary", "h_al_4000"), ("boundary", "h_si_4000"))
    assert runner.SHIELDS == tuple(D(v) for v in ("0", ".01", ".1", "1", "3", "10"))
    assert runner.INTERPOLATIONS == ("linear_stopping", "loglog_stopping")


@pytest.mark.parametrize("interpolation", ["linear_stopping", "loglog_stopping"])
def test_transport_retains_total_energy_and_unknown_exits(runner: ModuleType, interpolation: str) -> None:
    n = native()
    table = AnchoredSrimRangeTable.from_native(n, interpolation=interpolation)  # type: ignore[arg-type]
    result = runner._transport_checks(table, n)
    assert len(result["off_node_roundtrips"]) == 3
    assert not result["source_spectrum_convolution_performed"]
    assert not result["below_support_means_physically_stopped"]
    zero = result["shields"][0]
    assert zero["candidate_supported"] == 4 and zero["candidate_unresolved"] == 0
    for row in zero["rows"]:
        assert row["candidate_exit_total_mev"] == row["incident_total_mev"]
        assert row["projected_comparison_exit_total_mev"] == row["incident_total_mev"]
    one = next(s for s in result["shields"] if s["thickness_g_cm2"] == 1)
    assert one["rows"][0]["candidate_exit_total_mev"] is None
    assert one["rows"][0]["candidate_status"] == "UNRESOLVED_BELOW_SUPPORT"
    assert one["rows"][0]["candidate_minus_projected_exit_mev"] is None
    assert abs(one["rows"][-1]["candidate_exit_total_mev"] - 6) < D("1e-70")
    assert one["rows"][-1]["projected_comparison_exit_total_mev"] == 6
    assert result["shields"][-1]["candidate_unresolved"] == 4


def test_transport_rejects_mismatched_native_identity(runner: ModuleType) -> None:
    table = AnchoredSrimRangeTable.from_native(native(), interpolation="linear_stopping")
    with pytest.raises(ValueError):
        runner._transport_checks(table, native(altered=True))


def test_candidate_retains_source_gaps_and_unqualified_status(runner: ModuleType, pinned: Any) -> None:
    result = runner.build_candidate()
    assert len(result["selected_native_controls"]) == 2
    assert len(result["range_candidates"]) == 2
    assert result["protected_sha256_before"] == result["protected_sha256_after"]
    assert result["diagnostic_status"] == "INCOMPLETE"
    for key in ("normative_export_allowed", "production_tables_replaced", "physical_accuracy_established",
                "source_spectra_changed", "isotope_policy_chosen"):
        assert result[key] is False
    for table in result["range_candidates"]:
        candidate = table["candidate"]
        assert candidate["anchored_range_g_cm2"][0] == 0
        assert candidate["energy_total_mev"][0] == 1
        assert candidate["range_kind"] == "derived_anchored_csda_path"
        assert candidate["native_range_kind"] == "projected"
        assert not candidate["absolute_range_origin_established"]
    summaries = {s["source"]: s for s in result["source_native_coverage"]["summaries"]}
    assert summaries["GCR"]["missing_positive_species"] == 83
    assert summaries["SEP"]["missing_positive_species"] == 23


@pytest.mark.parametrize("target", ["parent", "mass", "source", "parent_impl", "catalog", "protected",
                                    "input", "output", "command_file"])
def test_tampered_parent_native_or_source_is_rejected(runner: ModuleType, pinned: Any, target: str) -> None:
    paths = {"parent": pinned.parent_path, "mass": pinned.mass_path, "source": pinned.source,
             "parent_impl": pinned.parent_implementation, "catalog": pinned.catalog_path,
             "protected": pinned.protected}
    path = paths[target] if target in paths else Path(pinned.report["cases"][0]["repeats"][0][target]["path"])
    path.write_bytes(path.read_bytes() + b"TAMPER")
    with pytest.raises(ValueError):
        runner.build_candidate()


def test_required_incomplete_control_is_rejected(runner: ModuleType, pinned: Any) -> None:
    pinned.report["cases"][0]["status"] = "REPEAT_MISMATCH"
    pinned.save()
    with pytest.raises(ValueError, match="Required native candidate control is incomplete"):
        runner.build_candidate()


def test_unselected_failed_control_remains_unavailable(runner: ModuleType, pinned: Any) -> None:
    repeat = pinned.report["cases"][0]["repeats"][0]
    failed = {k: repeat[k] for k in ("input", "command_file")}
    failed.update({"output": None, "status": "NATIVE_FAILED"})
    pinned.report["cases"].append({"case": {"name": "failed_control"},
                                  "status": "NATIVE_FAILED", "repeats": [failed]})
    pinned.save()
    result = runner.build_candidate()
    assert result["unavailable_controls"] == [{"catalog": "baseline",
                                               "case": {"name": "failed_control"},
                                               "status": "NATIVE_FAILED"}]
    assert len(result["range_candidates"]) == 2


@pytest.mark.parametrize("target", ["source", "implementation", "catalog", "protected",
                                    "input", "output", "command_file"])
def test_mid_build_mutation_is_rejected(
    runner: ModuleType, pinned: Any, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    original = runner._transport_checks
    paths = {"source": pinned.source, "implementation": pinned.root / runner.IMPLEMENTATION[0],
             "catalog": pinned.catalog_path, "protected": pinned.protected}
    path = paths[target] if target in paths else Path(pinned.report["cases"][0]["repeats"][0][target]["path"])

    def mutate(*args: Any) -> Any:
        result = original(*args)
        path.write_bytes(path.read_bytes() + b"TAMPER")
        return result

    monkeypatch.setattr(runner, "_transport_checks", mutate)
    with pytest.raises(ValueError):
        runner.build_candidate()


def test_cli_writes_once_and_refuses_existing_output(
    runner: ModuleType, pinned: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "new" / "candidate.json"
    monkeypatch.setattr(sys, "argv", ["build", "--output", str(output)])
    assert runner.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["diagnostic_status"] == "INCOMPLETE"
    before = output.read_bytes()
    monkeypatch.setattr(runner, "build_candidate", lambda: pytest.fail("must reject before audit"))
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2 and output.read_bytes() == before


def test_cli_race_preserves_other_writer(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raced.json"
    monkeypatch.setattr(sys, "argv", ["build", "--output", str(output)])

    def race() -> dict[str, Any]:
        output.write_bytes(b"OTHER WRITER")
        return {"range_candidates": []}

    monkeypatch.setattr(runner, "build_candidate", race)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"OTHER WRITER"


def test_failed_build_does_not_publish(
    runner: ModuleType, pinned: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "not_created" / "candidate.json"
    pinned.source.unlink()
    monkeypatch.setattr(sys, "argv", ["build", "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        runner.main()
    assert not output.exists() and not output.parent.exists()
