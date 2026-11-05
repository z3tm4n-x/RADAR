"""The mass experiment validator rejects relabelled, incomplete or changed inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar.validation.ion_mass_contract import CandidateIsotope
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


PILOTS = (CandidateIsotope(10, 20, "Ne"), CandidateIsotope(26, 56, "Fe"))


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("parameterized_validator_test", directory /
                                                 "validate_parameterized_stopping.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, raw: dict[str, Any]) -> None:
    path.write_text(json.dumps(raw), encoding="utf-8")


def _repin_manifest(path: Path, raw: dict[str, Any]) -> None:
    manifest = Path(raw["manifest_path"])
    _write(manifest, raw["manifest"])
    raw["manifest_sha256"] = _sha(manifest)
    _write(path, raw)


@pytest.fixture
def candidate(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner.experiment, "ROOT", tmp_path)
    monkeypatch.setattr(runner.manifest_builder, "_ROOT", tmp_path)
    # A two-node analytic grid keeps corruption tests small; production uses
    # the real generator grid and this patch is confined to this fixture.
    monkeypatch.setattr(runner.experiment, "experiment_grid", lambda _density: (1.0, 2.0))
    resources = {}
    for relative in runner.experiment.MANIFEST_SOURCE_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"inert source fingerprint fixture")
        resources[relative] = _sha(target)
    source = tmp_path / "research/ion_stopping_20261031/source.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"inert backend source fixture, never executed")
    pins = {"source.bin": _sha(source)}
    monkeypatch.setattr(runner.native, "PINS", pins)
    generators = {}
    for relative in runner._GENERATOR_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"# inert pinned generator fixture")
        generators[relative] = _sha(target)
    for relative in runner._VALIDATION_PATHS:
        target = tmp_path / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"# inert validator implementation fingerprint fixture")
    pilot_path = tmp_path / "original_pilot.json"
    _write(pilot_path, {"schema_version": "radar.ion_stopping_candidate.raw.v1",
                        "ions": [asdict(pilot) for pilot in PILOTS]})
    manifest = runner.manifest_builder.build_manifest(pilot_path,
                                                      expected_candidate_sha256=_sha(pilot_path))
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest)
    datasets = {}
    for policy, field in zip(runner._POLICIES, ("projectiles", "pilot_isotopes"), strict=True):
        ions = manifest[field]
        datasets[policy] = {
            "mass_policy": policy, "projectiles": deepcopy(ions),
            # Constant whole-projectile S=10 gives R=m*e/10 exactly.
            "rows": [asdict(ParameterizedStoppingRow(ion["z"], ion["mass_parameter_u"], material,
                        energy, 7.0, 2.0, ion["mass_parameter_u"] * energy / 10, 1.0))
                     for ion in ions for material in (13, 14) for energy in (1.0, 2.0)],
            "materials": [{"z": 13, "atomic_mass_u": 27.0, "density_g_cm3": 2.7},
                          {"z": 14, "atomic_mass_u": 28.0, "density_g_cm3": 2.33}],
            "mass_diagnostics": [{"z": ion["z"], "mass_parameter_u": ion["mass_parameter_u"],
                                  "backend_reference_mass_u": ion["mass_parameter_u"],
                                  "ls_mass_linear_weight": 0.0, "ls_mass_extrapolation": False,
                                  "physical_isotope_assignment_verified": False} for ion in ions],
            "generation_checks": {"low_energy_closure_count": len(ions) * 4,
                "low_energy_closure_max_relative_error": 0.0,
                "range_derivative_worst": {"z": ions[0]["z"],
                    "mass_parameter_u": ions[0]["mass_parameter_u"], "material_z": 13,
                    "energy_mev_per_nucleon": 1.0, "relative_error": 0.0}},
            "backend_configuration": dict(runner._CONFIG),
            "nonradiative_configuration": {**runner._CONFIG, "corrections": 8},
            "normative_export_allowed": False,
        }
    raw = {
        "schema_version": "radar.parameterized_stopping.raw.v1",
        "manifest": manifest, "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path),
        "energies_mev_per_nucleon": [1.0, 2.0], "material_atomic_numbers": [13, 14],
        "points_per_decade": 100, "datasets": datasets, "production_resources_sha256": resources,
        "source_pins": pins, "generator_hashes": generators, "python": "analytic fixture",
        "backend_info_literal": "native backend not executed",
        "range_origin_energy_mev_per_nucleon": 0.001,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "projectile_charge_input": "Q=Z; pinned PierceBlann uses equilibrium charge, not SEP A/Q",
        "let_convention": "electronic_collision_mass_stopping_divided_by_1000_in_Si",
        **{flag: False for flag in runner._FALSE_RAW_FLAGS},
        "limitations": ["Synthetic test fixture only"],
    }
    path = tmp_path / "candidate.json"
    _write(path, raw)
    return path, raw


def test_analytic_full_roster_validation_and_paired_mass_denominator(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    report = runner.run_validator(path)
    assert report["numerical_consistency_passed"]
    assert report["source_inventory_independently_reconstructed"] is True
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert len(report["dataset_validations"][runner._POLICIES[0]]["groups"]) == 168
    assert len(report["paired_mass_comparisons"]) == 4
    neon = report["paired_mass_comparisons"][0]
    assert neon["metrics"]["range_g_cm2"]["signed_relative_change_at_max"] == pytest.approx(0.01)
    assert neon["metrics"]["stopping_electronic_mev_cm2_g"]["max_absolute_relative_change"] == 0
    assert neon["metrics"]["stopping_total_mev_cm2_g"]["max_absolute_relative_change"] == 0
    assert report["production_resources_sha256_before"] == raw["production_resources_sha256"]
    assert len(report["validation_implementation_sha256_before"]) == 3
    assert (report["validation_implementation_sha256_before"]
            == report["validation_implementation_sha256_after"])
    for flag in ("absolute_range_origin_verified", "physical_mass_policy_qualified",
                 "normative_export_allowed", "production_tables_replaced",
                 "generation_checks_independently_replayed"):
        assert report[flag] is False


@pytest.mark.parametrize("field", ["manifest", "datasets", "source_pins", "generator_hashes",
                                  "production_resources_sha256", "let_convention"])
def test_missing_raw_fields_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], field: str,
) -> None:
    path, raw = candidate
    del raw[field]
    _write(path, raw)
    with pytest.raises(ValueError, match="schema"):
        runner.run_validator(path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unknown"), ("unexpected", 1), ("points_per_decade", True),
    ("range_origin_energy_mev_per_nucleon", 0), ("normative_export_allowed", True),
    ("physical_mass_policy_qualified", 0), ("material_atomic_numbers", [13]),
    ("material_atomic_numbers", [13.0, 14.0]), ("let_convention", "total_stopping"),
    ("energies_mev_per_nucleon", [1.0, 3.0]), ("source_pins", {}), ("generator_hashes", {}),
])
def test_invalid_raw_values_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], field: str, value: Any,
) -> None:
    path, raw = candidate
    raw[field] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path)


@pytest.mark.parametrize("alteration", ["mass", "zero", "charge", "membership"])
def test_repinned_manifest_semantic_corruption_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], alteration: str,
) -> None:
    path, raw = candidate
    manifest = raw["manifest"]
    if alteration == "mass":
        next(ion for ion in manifest["projectiles"] if ion["z"] == 26)["mass_parameter_u"] = 56
    elif alteration == "zero":
        manifest["declared_zero_sources"][0]["source_weight_parameter"] = 1
    elif alteration == "charge":
        next(row for row in manifest["source_rows"] if row["source"] == "SEP" and row["z"] == 26)[
            "source_mass_to_charge"] = 55.8 / 26
    else:
        # Preserve the declared count while assigning SEP to the wrong element.
        next(ion for ion in manifest["projectiles"] if ion["z"] == 2)["source_membership"] = ["GCR"]
        next(ion for ion in manifest["projectiles"] if ion["z"] == 3)["source_membership"] = ["GCR", "SEP"]
    _repin_manifest(path, raw)
    with pytest.raises(ValueError, match="independently reconstructed"):
        runner.run_validator(path)


@pytest.mark.parametrize("alteration", ["missing_policy", "identity", "row", "duplicate", "missing_row",
                                      "config", "materials", "diagnostic", "qualification"])
def test_dataset_corruption_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], alteration: str,
) -> None:
    path, raw = candidate
    dataset = raw["datasets"][runner._POLICIES[1]]
    if alteration == "missing_policy":
        del raw["datasets"][runner._POLICIES[1]]
    elif alteration == "identity":
        dataset["projectiles"][0]["mass_parameter_u"] += 1
    elif alteration == "row":
        del dataset["rows"][0]["stopping_radiative_mev_cm2_g"]
    elif alteration == "duplicate":
        dataset["rows"].append(dataset["rows"][0])
    elif alteration == "missing_row":
        dataset["rows"].pop()
    elif alteration == "config":
        dataset["backend_configuration"]["z_effective"] = True
    elif alteration == "materials":
        dataset["materials"][0]["atomic_mass_u"] += 0.1
    elif alteration == "diagnostic":
        dataset["mass_diagnostics"][0]["ls_mass_extrapolation"] = True
    else:
        dataset["normative_export_allowed"] = True
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_duplicate_and_nonfinite_json_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], text: str,
) -> None:
    path, _ = candidate
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        runner.run_validator(path)


@pytest.mark.parametrize("relative", ["../outside.txt", "C:/outside.txt", "/outside.txt", "x:stream"])
def test_unsafe_resource_paths_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], relative: str,
) -> None:
    path, raw = candidate
    raw["production_resources_sha256"] = {relative: "0" * 64}
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path)


@pytest.mark.parametrize("which", ["candidate", "manifest", "pilot", "source", "generator", "resource",
                                  "validator", "hash_helper"])
def test_concurrent_change_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    path, raw = candidate
    original = runner.validate_parameterized_rows

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        targets = {"candidate": path, "manifest": Path(raw["manifest_path"]),
                   "pilot": Path(raw["manifest"]["candidate_path"]),
                   "source": path.parent / "research/ion_stopping_20261031/source.bin",
                   "generator": path.parent / "scripts/generate_parameterized_stopping.py",
                   "validator": path.parent / "scripts/validate_parameterized_stopping.py",
                   "hash_helper": path.parent / "scripts/validate_ion_stopping_candidate.py",
                   "resource": path.parent / "src/radar/gcr/parameters.py"}
        target = targets[which]
        target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(runner, "validate_parameterized_rows", mutate)
    with pytest.raises(ValueError, match="changed|mismatch|does not match"):
        runner.run_validator(path)


def test_numeric_failure_is_reported_without_acceptance(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    for row in raw["datasets"][runner._POLICIES[0]]["rows"]:
        row["range_g_cm2"] *= 1.1
    _write(path, raw)
    report = runner.run_validator(path)
    assert not report["numerical_consistency_passed"]
    assert not report["normative_export_allowed"]


def test_exclusive_write_and_cli_do_not_overwrite(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = candidate
    output = path.parent / "validation.json"
    assert runner.main([str(path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["normative_export_allowed"] is False
    monkeypatch.setattr(runner, "run_validator", lambda *_args, **_kwargs: pytest.fail("ran twice"))
    with pytest.raises(SystemExit):
        runner.main([str(path), "--output", str(output)])
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({}, output)
    finite_output = path.parent / "nonfinite.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"value": float("nan")}, finite_output)
    assert not finite_output.exists()


def test_repinned_fabricated_pilot_identity_is_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    raw["manifest"]["pilot_isotopes"][0]["mass_parameter_u"] = 21
    _repin_manifest(path, raw)
    with pytest.raises(ValueError, match="candidate-backed"):
        runner.run_validator(path)


def test_changed_pilot_fingerprint_is_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    raw["manifest"]["candidate_expected_sha256"] = "0" * 64
    _repin_manifest(path, raw)
    with pytest.raises(ValueError, match="SHA256"):
        runner.run_validator(path)
