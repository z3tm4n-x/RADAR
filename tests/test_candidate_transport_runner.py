"""Runner guards and comparison semantics; native table validation is mocked explicitly."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_candidate_transport.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("candidate_transport_runner_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _change(path: Path, operation: Any) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    operation(raw)
    _write(path, raw)


def _source(*, peak: bool) -> Spectrum1D:
    return Spectrum1D(
        x=(1., 2., 4.), y=(2., 2., 2.), x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
                else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        quantity=(SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX if peak
                  else SpectrumQuantity.DIFFERENTIAL_FLUENCE),
        particle=Particle.HZE, source=RadiationSource.GCR, model="analytic",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.SOURCE_MAXIMUM if peak else TemporalBasis.MISSION_TOTAL,
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies("He", atomic_number=2, mass_number=4),
        assessment_mode=AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        mission_duration_s=None if peak else 12345.,
        model_reference=ModelReference("analytic", "1", ("analytic",), ("synthetic",)),
        transformation_chain=(TransformationStep("synthetic_source", "1"),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


@pytest.fixture
def inputs(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Only the prior raw-table validator is replaced; runner hashing/adapters are real."""
    paths = (tmp_path / "coarse.json", tmp_path / "dense.json")
    for name in ("implementation.py", "production.txt", "generator.py"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    source_root = tmp_path / "research/ion_stopping_20261031"
    source_root.mkdir(parents=True)
    (source_root / "source.txt").write_text("source fixture\n", encoding="utf-8")
    source_hashes = {"source.txt": _sha(source_root / "source.txt")}
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {"fixture": True})
    inventory = [{"z": 2, "symbol": "He", "mass_parameter_u": 4.,
                  "source_membership": ["GCR"]}]
    policy = runner.MASS_POLICIES[0]
    for path, energies in zip(paths, ((1., 4.), (1., 2., 4.)), strict=True):
        raw = {
            "manifest": {"fixture": True}, "manifest_sha256": _sha(manifest_path),
            "manifest_path": str(manifest_path),
            "source_pins": source_hashes, "energy_coordinate_compatibility": "synthetic",
            "let_convention": "electronic", "projectile_charge_input": "Z",
            "range_origin_energy_mev_per_nucleon": .001,
            "energies_mev_per_nucleon": energies,
            "datasets": {policy: {
                "projectiles": inventory, "backend_configuration": {"test": 1},
                "nonradiative_configuration": {"test": 2}, "materials": [13, 14],
                "rows": [asdict(ParameterizedStoppingRow(
                    2, 4., material, energy, 1000 * energy, 0., energy,
                )) for material in (13, 14) for energy in energies],
            }},
        }
        _write(path, raw)
    production = {"production.txt": _sha(tmp_path / "production.txt")}
    generators = {"generator.py": _sha(tmp_path / "generator.py")}

    def validator(path: Path) -> dict[str, Any]:
        return {
            "input_sha256_after": _sha(path), "numerical_consistency_passed": True,
            "production_resources_sha256_after": production,
            "generator_hashes_after": generators, "source_pins_after": source_hashes,
        }

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "IMPLEMENTATION_PATHS", ("implementation.py",))
    monkeypatch.setattr(runner, "SHIELDS", (0., .5))
    monkeypatch.setattr(runner, "THRESHOLDS", (1., 2.))
    monkeypatch.setattr(runner, "ENERGY_EDGES", (.01, 1., 2., 4., 100.))
    monkeypatch.setattr(runner, "LET_EDGES", (.1, 1., 2., 4., 10.))
    monkeypatch.setattr(runner, "run_validator", validator)
    monkeypatch.setattr(runner, "_source_inputs", lambda _: (_source(peak=True), _source(peak=False)))
    return paths


def test_analytic_runner_preserves_pins_scope_and_nonrelease(
    runner: ModuleType, inputs: tuple[Path, Path],
) -> None:
    report = runner.run_experiment(*inputs)
    assert report["case_count_per_grid"] == 4
    assert report["source_mode_count"] == 2
    assert report["selected_integral_grid_check_passed"]
    assert report["all_zero_shield_identities_passed"]
    assert report["candidate_sha256"] == [_sha(path) for path in inputs]
    assert report["implementation_sha256_before"] == report["implementation_sha256_after"]
    for flag in ("physical_accuracy_verified", "normative_export_allowed",
                 "production_tables_replaced", "stage4_acceptance_matrix_completed"):
        assert report[flag] is False
    assert "no_orbit_no_physical_qualification" in report["scope"]
    assert report["source_policy"]["source_grid_refined_here"] is False
    assert report["source_policy"]["mission_time_factor_in_adapter"] == "none"
    assert all(case["incomplete_reasons"] for case in report["dense_cases"])
    json.dumps(report, allow_nan=False)


def test_subselection_never_claims_full_inventory(
    runner: ModuleType, inputs: tuple[Path, Path],
) -> None:
    report = runner.run_experiment(*inputs, symbols=("He",))
    assert report["full_positive_inventory_selected"] is False
    assert report["case_count_per_grid"] == 4


@pytest.mark.parametrize("symbols", [(), ("He", "He"), ("Unknown",), ("",)])
def test_empty_duplicate_unknown_symbols_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], symbols: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="unique positive source entries"):
        runner.run_experiment(*inputs, symbols=symbols)


@pytest.mark.parametrize("key", [
    "manifest", "manifest_sha256", "source_pins", "energy_coordinate_compatibility",
    "let_convention", "projectile_charge_input", "range_origin_energy_mev_per_nucleon",
])
def test_candidate_semantics_mismatch_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], key: str,
) -> None:
    _change(inputs[1], lambda raw: raw.__setitem__(key, "mismatch"))
    with pytest.raises(ValueError, match=f"Candidates differ in {key}"):
        runner.run_experiment(*inputs)


@pytest.mark.parametrize("key", [
    "projectiles", "backend_configuration", "nonradiative_configuration", "materials",
])
def test_candidate_policy_mismatch_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], key: str,
) -> None:
    policy = runner.MASS_POLICIES[0]
    _change(inputs[1], lambda raw: raw["datasets"][policy].__setitem__(key, "mismatch"))
    with pytest.raises(ValueError, match=f"Source-policy candidates differ in {key}"):
        runner.run_experiment(*inputs)


@pytest.mark.parametrize("energies", [(1., 4.), (1., 2., 3.), (1., 3., 5., 6.)])
def test_nonrefinement_or_nonnested_grid_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], energies: tuple[float, ...],
) -> None:
    _change(inputs[1], lambda raw: raw.__setitem__("energies_mev_per_nucleon", energies))
    with pytest.raises(ValueError, match="strictly denser nested"):
        runner.run_experiment(*inputs)


def test_candidate_changed_during_computation_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sources(_: Any) -> tuple[Spectrum1D, ...]:
        _change(inputs[0], lambda raw: raw.__setitem__("changed", True))
        return (_source(peak=True), _source(peak=False))

    monkeypatch.setattr(runner, "_source_inputs", sources)
    with pytest.raises(ValueError, match="Candidate changed during"):
        runner.run_experiment(*inputs)


@pytest.mark.parametrize("relative", [
    "production.txt", "generator.py", "research/ion_stopping_20261031/source.txt",
])
def test_pinned_resource_changed_during_computation_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, relative: str,
) -> None:
    def sources(_: Any) -> tuple[Spectrum1D, ...]:
        (inputs[0].parent / relative).write_text("changed", encoding="utf-8")
        return (_source(peak=True), _source(peak=False))

    monkeypatch.setattr(runner, "_source_inputs", sources)
    with pytest.raises(ValueError, match="resource hash mismatch"):
        runner.run_experiment(*inputs)


def test_implementation_changed_during_computation_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sources(_: Any) -> tuple[Spectrum1D, ...]:
        (inputs[0].parent / "implementation.py").write_text("changed", encoding="utf-8")
        return (_source(peak=True), _source(peak=False))

    monkeypatch.setattr(runner, "_source_inputs", sources)
    with pytest.raises(ValueError, match="implementation changed"):
        runner.run_experiment(*inputs)


def _comparison_case(value: float) -> dict[str, Any]:
    return {"key": ["gcr", 2, "peak", 0.1],
            "mass_parameter_u": 4., "energy_y_unit": "energy_flux", "let_y_unit": "let_flux",
            "primary_ledger": {"supported_exit_integral": value},
            "let_ledger": {"mapped_integral": value},
            "let_tail_integrals": {"1.0": value},
            "energy_bin_integrals": [value], "let_bin_integrals": [value]}


@pytest.mark.parametrize("left, right, expected", [(0., 0., 0.), (0., 1., None), (2., 1., .5)])
def test_zero_reference_is_not_relabelled_as_relative_zero(
    runner: ModuleType, left: float, right: float, expected: float | None,
) -> None:
    assert runner._relative(left, right) == expected
    comparison = runner._compare(_comparison_case(left), _comparison_case(right))
    assert comparison["relative_integral_changes"]["supported_primary"] == expected
    assert comparison["histogram_l1_over_coarse_integral"]["energy"] == expected
    assert comparison["selected_integral_grid_check_passed"] is (expected == 0.)


def test_exact_one_percent_is_not_below_one_percent(runner: ModuleType) -> None:
    result = runner._compare(_comparison_case(100.), _comparison_case(101.))
    assert result["relative_integral_changes"]["supported_primary"] == .01
    assert not result["selected_integral_grid_check_passed"]


def test_unmatched_case_rejected(runner: ModuleType) -> None:
    changed = _comparison_case(1.)
    changed["key"] = ["sep", 2, "peak", .1]
    with pytest.raises(ValueError, match="Unmatched"):
        runner._compare(_comparison_case(1.), changed)


def test_different_histogram_lengths_rejected(runner: ModuleType) -> None:
    changed = _comparison_case(1.)
    changed["energy_bin_integrals"] = [1., 2.]
    with pytest.raises(ValueError):
        runner._compare(_comparison_case(1.), changed)


def test_histogram_shape_change_is_reported_separately_from_integrals(runner: ModuleType) -> None:
    left, right = _comparison_case(2.), _comparison_case(2.)
    left["energy_bin_integrals"] = [2., 0.]
    right["energy_bin_integrals"] = [0., 2.]
    result = runner._compare(left, right)
    assert result["selected_integral_grid_check_passed"]
    assert result["histogram_l1_over_coarse_integral"]["energy"] == 2.


def test_real_source_helper_has_expected_positive_roster_and_both_modes(runner: ModuleType) -> None:
    sources = runner._source_inputs([
        {"symbol": "He", "z": 2, "source_membership": ["GCR", "SEP"]},
        {"symbol": "Kr", "z": 36, "source_membership": ["GCR"]},
    ])
    assert len(sources) == 6
    assert {(s.source, s.species.atomic_number, s.assessment_mode)
            for s in sources if s.species is not None} == {
        (source, z, mode) for source, z in ((RadiationSource.GCR, 2),
                                         (RadiationSource.GCR, 36), (RadiationSource.SEP, 2))
        for mode in AssessmentMode
    }


def test_real_source_helper_rejects_missing_membership(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="Missing, duplicate or unexpected"):
        runner._source_inputs([{"symbol": "He", "z": 2, "source_membership": ["GCR"]}])


def test_report_writer_never_overwrites(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"replacement": True}, path)
    assert path.read_text(encoding="utf-8") == "keep"


def test_cli_existing_output_fails_before_calculation(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    path.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runner", "coarse", "dense", "--output", str(path)])
    monkeypatch.setattr(runner, "run_experiment", lambda *_args, **_kwargs: pytest.fail("must not run"))
    with pytest.raises(FileExistsError):
        runner.main()
    assert path.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("field, value", [
    ("mass_parameter_u", 4.1), ("energy_y_unit", "energy_fluence"),
    ("let_y_unit", "let_fluence"),
])
def test_mismatched_mass_or_units_cannot_be_compared(
    runner: ModuleType, field: str, value: float | str,
) -> None:
    changed = _comparison_case(1.)
    changed[field] = value
    with pytest.raises(ValueError, match="Unmatched"):
        runner._compare(_comparison_case(1.), changed)


def test_manifest_changed_during_computation_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sources(_: Any) -> tuple[Spectrum1D, ...]:
        (inputs[0].parent / "manifest.json").write_text("changed", encoding="utf-8")
        return (_source(peak=True), _source(peak=False))

    monkeypatch.setattr(runner, "_source_inputs", sources)
    with pytest.raises(ValueError, match="manifest changed"):
        runner.run_experiment(*inputs)


def test_input_pin_mismatch_after_individual_validation_rejected(
    runner: ModuleType, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = runner.run_validator

    def stale_pin(path: Path) -> dict[str, Any]:
        result = dict(validator(path))
        result["input_sha256_after"] = "0" * 64
        return result

    monkeypatch.setattr(runner, "run_validator", stale_pin)
    with pytest.raises(ValueError, match="Input changed after individual validation"):
        runner.run_experiment(*inputs)


def test_empty_cli_symbols_fail_without_expanding_to_all_ions(
    runner: ModuleType, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = inputs[0].parent / "never_created.json"
    monkeypatch.setattr(sys, "argv", [
        "runner", str(inputs[0]), str(inputs[1]), "--symbols", "", "--output", str(output),
    ])
    with pytest.raises(ValueError, match="unique positive source entries"):
        runner.main()
    assert not output.exists()
