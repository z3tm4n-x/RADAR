"""Independent analytic witnesses for bounded He4 table-substitution impacts."""

from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import importlib.util
import json
from math import inf, nan
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
from radar.validation.candidate_transport import CandidateIonTable
from radar.validation.nist_astar import AstarArchive, AstarMaterial, AstarRow
from radar.validation.nist_astar_live import CURRENT_DEFAULT_ENERGIES_MEV_TOTAL
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


POLICY = "source_tabulated_parameter_diagnostic"


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_astar_helium_impact.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("astar_helium_impact_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def _source(
    *, x: tuple[float, ...] = (.01, .5, 1.5), y: tuple[float, ...] = (2., 2., 2.),
    peak: bool = True, source: RadiationSource = RadiationSource.GCR,
) -> Spectrum1D:
    return Spectrum1D(
        x=x, y=y, x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
                else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        quantity=((SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if source is RadiationSource.SEP
                   else SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX) if peak
                  else SpectrumQuantity.DIFFERENTIAL_FLUENCE),
        particle=Particle.HZE, source=source, model="analytic_helium",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=((TemporalBasis.SEP_PEAK if source is RadiationSource.SEP
                         else TemporalBasis.SOURCE_MAXIMUM) if peak else TemporalBasis.MISSION_TOTAL),
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies("He", atomic_number=2, mass_number=4),
        assessment_mode=AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        mission_duration_s=None if peak else 123456789.,
        model_reference=ModelReference("analytic_helium", "1", ("analytic witness",), ("synthetic",)),
        transformation_chain=(TransformationStep("analytic_source", "1", (
            ("source_tabulated_mass_parameter", "4.0"),
        )),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


def _analytic_table(*, range_scale: float = 1., range_offset: float = 0.,
                    let_scale: float = 1.) -> CandidateIonTable:
    nodes = (.01, .5, 1., 1.5, 2., 10., 250.)
    return CandidateIonTable(
        z=2, mass_parameter_u=4., mass_policy=POLICY,
        dataset_ref="synthetic:analytic-He4",
        al_energy=nodes, al_range=tuple(range_scale * e + range_offset for e in nodes),
        si_energy=nodes, si_electronic_let=tuple(let_scale * min(e, 2.) for e in nodes),
    )


def _reference_and_controls() -> tuple[AstarArchive, dict[str, Any]]:
    # Explicit full archive-like grid, not a fabricated online 22.5-MeV datum.
    energies = tuple(sorted((*CURRENT_DEFAULT_ENERGIES_MEV_TOTAL, 22.5)))
    archive = AstarArchive(
        materials=(
            AstarMaterial(13, "ALUMINUM", tuple(
                AstarRow(e, 1.9, .1, e / 2 + 137., .99) for e in energies
            )),
            AstarMaterial(14, "SILICON", tuple(
                AstarRow(e, 2000. * min(e / 4, 2.), .1, e / 4 + 37., .99) for e in energies
            )),
        ), archive_sha256="a" * 64, member_sha256=(("synthetic", "b" * 64),),
    )
    controls = {
        "projectile": {"z": 2, "mass_parameter_u": 4., "charge_input": 2},
        "archive_sha256": archive.archive_sha256,
        "rows": [{
            "material_z": material, "energy_mev_total": e,
            "energy_mev_per_nucleon": e / 4,
            "stopping_electronic_mev_cm2_g": 3.9 if material == 13 else 1000. * min(e / 4, 2.),
            "stopping_nuclear_mev_cm2_g": .1,
            "stopping_radiative_mev_cm2_g": 0., "range_g_cm2": e / 4,
        } for material in (13, 14) for e in energies if e >= .04],
    }
    return archive, controls


def _candidate_payload() -> dict[str, Any]:
    energies = (.01, .5, 1., 2., 100., 300., 1000.)
    return {"manifest_sha256": "e" * 64, "datasets": {POLICY: {
        "mass_policy": POLICY,
        "rows": [asdict(ParameterizedStoppingRow(
            2, 4., material, energy, 1000 * energy, .1, energy + 7.,
        )) for material in (13, 14) for energy in energies],
    }}}


def test_factorial_uses_exact_whole_alpha_conversion_and_independent_operator_choices(
    runner: ModuleType,
) -> None:
    archive, controls = _reference_and_controls()
    tables = runner._factorial_tables(controls, archive, native_sha256="c" * 64)
    assert set(tables) == {"CC", "AC", "CA", "AA"}
    common = tables["CC"].al_energy
    assert len(common) == 101
    assert common[0] == .01 and common[-1] == 250.
    assert all(table.al_energy == common and table.si_energy == common for table in tables.values())
    assert tables["CC"].al_range == pytest.approx(common)
    assert tables["AC"].al_range == pytest.approx(tuple(2 * e + 137. for e in common))
    assert tables["CA"].al_range == tables["CC"].al_range
    assert tables["AA"].al_range == tables["AC"].al_range
    assert tables["CC"].si_electronic_let == pytest.approx(tuple(min(e, 2.) for e in common))
    assert tables["CA"].si_electronic_let == pytest.approx(tuple(2 * min(e, 2.) for e in common))
    assert tables["AC"].si_electronic_let == tables["CC"].si_electronic_let
    assert tables["AA"].si_electronic_let == tables["CA"].si_electronic_let
    assert all(table.mass_parameter_u == 4. and not table.normative_export_allowed
               for table in tables.values())


def test_candidate_domain_clipping_inserts_supported_upper_boundary_without_extrapolation(
    runner: ModuleType,
) -> None:
    table = runner._bounded_candidate_table(
        _candidate_payload(), dataset_sha256="d" * 64, label="catima_dense",
    )
    assert table.al_energy == (.01, .5, 1., 2., 100., 250.)
    assert table.si_energy == table.al_energy
    assert table.al_range[-1] == pytest.approx(257.)
    assert table.si_electronic_let[-1] == pytest.approx(250., rel=0, abs=1e-12)
    assert table.mass_parameter_u == 4.
    assert "d" * 64 in table.dataset_ref
    assert not table.normative_export_allowed


def test_candidate_missing_exact_mass4_is_rejected(runner: ModuleType) -> None:
    payload = _candidate_payload()
    for row in payload["datasets"][POLICY]["rows"]:
        row["mass_parameter_u"] = 4.1
    with pytest.raises(ValueError):
        runner._bounded_candidate_table(payload, dataset_sha256="d" * 64, label="catima_dense")


@pytest.mark.parametrize("material", [13, 14])
def test_candidate_missing_material_is_not_an_implicit_zero(runner: ModuleType, material: int) -> None:
    payload = _candidate_payload()
    payload["datasets"][POLICY]["rows"] = [row for row in payload["datasets"][POLICY]["rows"]
                                           if row["material_z"] != material]
    with pytest.raises(ValueError):
        runner._bounded_candidate_table(payload, dataset_sha256="d" * 64, label="catima_dense")


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_candidate_insufficient_boundary_support_is_rejected(runner: ModuleType, side: str) -> None:
    payload = _candidate_payload()
    payload["datasets"][POLICY]["rows"] = [row for row in payload["datasets"][POLICY]["rows"]
                                           if (row["energy_mev_per_nucleon"] > .01 if side == "lower"
                                               else row["energy_mev_per_nucleon"] < 250.)]
    with pytest.raises(ValueError):
        runner._bounded_candidate_table(payload, dataset_sha256="d" * 64, label="catima_dense")


def test_factorial_analytic_counts_and_let_tails_have_separate_effects(runner: ModuleType) -> None:
    archive, controls = _reference_and_controls()
    tables = runner._factorial_tables(controls, archive, native_sha256="c" * 64)
    cases = {name: runner._evaluate_case(_source(), table, .5) for name, table in tables.items()}
    for name, expected in {"CC": 1.98, "CA": 1.98, "AC": 2.48, "AA": 2.48}.items():
        assert cases[name]["source_integral"] == pytest.approx(2.98)
        assert cases[name]["metrics"]["supported_primary_integral"] == pytest.approx(expected)
        assert cases[name]["metrics"]["si_mapped_integral"] == pytest.approx(expected)
    for name, expected in {"CC": 0., "AC": 0., "CA": .6, "AA": 1.1}.items():
        assert cases[name]["metrics"]["let_ge_1.4_to10"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("peak", [True, False])
def test_zero_shield_retains_full_source_and_reports_common_domain_separately(
    runner: ModuleType, peak: bool,
) -> None:
    source = _source(x=(.001, 1., 500.), peak=peak)
    case = runner._evaluate_case(source, _analytic_table(), 0.)
    assert case["zero_shield_identity"] is True
    assert case["source_integral"] == pytest.approx(2 * (500 - .001))
    assert case["metrics"]["energy_common_integral"] == pytest.approx(2 * (250 - .01))
    assert case["metrics"]["supported_primary_integral"] == pytest.approx(case["source_integral"])
    assert case["metrics"]["outside_si_domain_integral"] == pytest.approx(
        2 * (.01 - .001 + 500 - 250),
    )


def test_positive_shield_retains_unknown_outside_source_and_low_residual_counts(
    runner: ModuleType,
) -> None:
    source = _source(x=(.001, 1., 500.))
    case = runner._evaluate_case(source, _analytic_table(), .1)
    metrics = case["metrics"]
    assert metrics["outside_al_domain_integral"] == pytest.approx(2 * (.01 - .001 + 500 - 250))
    assert metrics["unresolved_low_residual_integral"] == pytest.approx(.2)
    assert metrics["supported_primary_integral"] == pytest.approx(2 * (250 - .11))
    assert sum(metrics[name] for name in (
        "outside_al_domain_integral", "unresolved_low_residual_integral", "supported_primary_integral",
    )) == pytest.approx(case["source_integral"])


def test_zero_valued_declared_outside_tails_do_not_become_missing_positive_counts(
    runner: ModuleType,
) -> None:
    source = _source(x=(.001, .01, 1., 250., 500.), y=(0., 0., 2., 0., 0.))
    case = runner._evaluate_case(source, _analytic_table(), .1)
    assert case["metrics"]["outside_al_domain_integral"] == 0.
    assert case["source_integral"] == pytest.approx(249.99)
    zero = runner._evaluate_case(source, _analytic_table(), 0.)
    assert zero["metrics"]["outside_si_domain_integral"] == 0.
    assert zero["metrics"]["energy_common_integral"] == pytest.approx(zero["source_integral"])


def test_entire_declared_source_can_be_unresolved_without_physical_stopping_claim(
    runner: ModuleType,
) -> None:
    case = runner._evaluate_case(_source(), _analytic_table(), 1000.)
    assert case["metrics"]["unresolved_low_residual_integral"] == pytest.approx(2.98)
    assert case["metrics"]["supported_primary_integral"] == 0.
    assert case["metrics"]["si_mapped_integral"] == 0.
    assert not case["primary_ledger"]["stopped_particle_classification_established"]


def test_additive_range_origin_does_not_change_supported_impacts(runner: ModuleType) -> None:
    ordinary = runner._evaluate_case(_source(), _analytic_table(), .5)
    shifted = runner._evaluate_case(_source(), _analytic_table(range_offset=137.), .5)
    assert shifted["metrics"] == pytest.approx(ordinary["metrics"], abs=1e-12)
    assert shifted["energy_bin_integrals"] == pytest.approx(ordinary["energy_bin_integrals"], abs=1e-12)
    assert shifted["let_bin_integrals"] == pytest.approx(ordinary["let_bin_integrals"], abs=1e-12)


@pytest.mark.parametrize("source", [RadiationSource.GCR, RadiationSource.SEP])
def test_peak_and_cumulative_do_not_receive_extra_time_multiplier(
    runner: ModuleType, source: RadiationSource,
) -> None:
    peak = runner._evaluate_case(_source(source=source), _analytic_table(), .5)
    cumulative = runner._evaluate_case(_source(source=source, peak=False), _analytic_table(), .5)
    assert peak["metrics"] == cumulative["metrics"]
    assert peak["source_integral"] == cumulative["source_integral"]
    assert peak["energy_y_unit"] == Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY.value
    assert peak["let_y_unit"] == Unit.DIFFERENTIAL_LET_FLUX.value
    assert cumulative["energy_y_unit"] == Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY.value
    assert cumulative["let_y_unit"] == Unit.DIFFERENTIAL_LET_FLUENCE.value


def test_linear_source_scaling_applies_once(runner: ModuleType) -> None:
    ordinary = runner._evaluate_case(_source(), _analytic_table(), .5)
    scaled = runner._evaluate_case(_source(y=(6., 6., 6.)), _analytic_table(), .5)
    assert scaled["metrics"] == pytest.approx({name: 3 * value
                                              for name, value in ordinary["metrics"].items()})


def test_wrong_energy_units_rejected(runner: ModuleType) -> None:
    with pytest.raises(ValueError):
        runner._evaluate_case(replace(_source(), x_unit=Unit.MEV), _analytic_table(), .5)


def test_wrong_mass4_table_rejected(runner: ModuleType) -> None:
    with pytest.raises(ValueError):
        runner._evaluate_case(_source(), replace(_analytic_table(), mass_parameter_u=4.1), .5)


def _cases(runner: ModuleType, source: Spectrum1D | None = None) -> dict[str, dict[str, Any]]:
    source = _source() if source is None else source
    tables = {"CC": _analytic_table(), "AC": _analytic_table(range_scale=2., range_offset=137.),
              "CA": _analytic_table(let_scale=2.),
              "AA": _analytic_table(range_scale=2., range_offset=137., let_scale=2.)}
    return {label: dict(runner._evaluate_case(source, table, .5)) for label, table in tables.items()}


def test_factorial_interaction_is_nonadditive_tail_change_not_physical_error(runner: ModuleType) -> None:
    cases = _cases(runner)
    result = runner._factorial_interaction(cases)
    metric = result["metrics"]["let_ge_1.4_to10"]
    assert result["formula"] == "AA-AC-CA+CC"
    assert metric["signed_interaction"] == pytest.approx(.5, abs=1e-12)
    assert metric["source_normalized_signed_interaction"] == pytest.approx(.5 / 2.98)
    assert metric["common_source_normalized_signed_interaction"] == pytest.approx(.5 / 2.98)
    assert result["metrics"]["energy_common_integral"]["signed_interaction"] == pytest.approx(0.)
    assert all(result["energy_independent_of_si_table"].values())
    assert all(result["full_window_count_independent_of_si_table"].values())
    assert all(case["normative_export_allowed"] is False for case in cases.values())


def test_positive_response_over_zero_baseline_has_no_ratio(runner: ModuleType) -> None:
    cases = _cases(runner)
    result = runner._impact(cases["CC"], cases["CA"])
    metric = result["metrics"]["let_ge_1.4_to10"]
    assert metric["left"] == 0.
    assert metric["right"] == pytest.approx(.6)
    assert metric["signed_difference"] == pytest.approx(.6)
    assert metric["source_normalized_signed_difference"] == pytest.approx(.6 / 2.98)
    assert metric["ratio_right_over_left"] is None
    assert metric["relative_change"] is None
    assert metric["left_denominator_zero"] is True
    assert metric["both_zero"] is False


def test_both_zero_and_zero_source_denominators_remain_none(runner: ModuleType) -> None:
    cases = _cases(runner, _source(y=(0., 0., 0.)))
    comparison = runner._impact(cases["CC"], cases["AA"])
    for metric in comparison["metrics"].values():
        assert metric["signed_difference"] == 0.
        assert metric["ratio_right_over_left"] is None
        assert metric["relative_change"] is None
        assert metric["source_normalized_signed_difference"] is None
        assert metric["common_source_normalized_signed_difference"] is None
        assert metric["both_zero"] is True
    interaction = runner._factorial_interaction(cases)
    assert all(metric["source_normalized_signed_interaction"] is None
               for metric in interaction["metrics"].values())


def test_source_and_common_domain_normalization_have_different_denominators(runner: ModuleType) -> None:
    cases = _cases(runner, _source(x=(.001, 1., 500.)))
    comparison = runner._impact(cases["CC"], cases["AC"])
    metric = comparison["metrics"]["supported_primary_integral"]
    assert metric["signed_difference"] == pytest.approx(.5)
    assert metric["source_normalized_signed_difference"] == pytest.approx(.5 / 999.998)
    assert metric["common_source_normalized_signed_difference"] == pytest.approx(.5 / 499.98)


@pytest.mark.parametrize("field", [
    "key", "source_sha256", "source_integral", "source_common_domain_coverage",
    "energy_y_unit", "let_y_unit", "mass_parameter_u",
])
def test_mismatched_source_units_mass_or_domains_rejected(runner: ModuleType, field: str) -> None:
    cases = _cases(runner)
    changed = deepcopy(cases["CA"])
    changed[field] = "mismatched"
    with pytest.raises(ValueError, match="Unmatched"):
        runner._impact(cases["CC"], changed)
    cases["CA"] = changed
    with pytest.raises(ValueError, match="Unmatched"):
        runner._factorial_interaction(cases)


def test_missing_metric_is_not_assumed_zero(runner: ModuleType) -> None:
    cases = _cases(runner)
    del cases["CA"]["metrics"]["let_ge_1.4_to10"]
    with pytest.raises(ValueError, match="coverage differs"):
        runner._impact(cases["CC"], cases["CA"])


@pytest.mark.parametrize("value", [True, -1., nan, inf])
def test_invalid_impact_metric_is_rejected(runner: ModuleType, value: float) -> None:
    cases = _cases(runner)
    cases["CA"]["metrics"]["let_ge_1.4_to10"] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        runner._impact(cases["CC"], cases["CA"])


def test_missing_factorial_variant_rejected(runner: ModuleType) -> None:
    cases = _cases(runner)
    del cases["AA"]
    with pytest.raises(ValueError, match="All four"):
        runner._factorial_interaction(cases)


@pytest.mark.parametrize("change", ["energy", "mapped_count", "swapped_range_and_let"])
def test_factorial_independence_failure_is_not_a_valid_interaction(
    runner: ModuleType, change: str,
) -> None:
    cases = _cases(runner)
    if change == "energy":
        cases["CA"]["energy_bin_integrals"][0] += 1.
    elif change == "mapped_count":
        cases["CA"]["metrics"]["si_mapped_integral"] += 1.
    else:
        cases["CA"], cases["AC"] = cases["AC"], cases["CA"]
    with pytest.raises(ValueError, match="Changing only Si LET"):
        runner._factorial_interaction(cases)


def test_partial_let_window_is_rejected_instead_of_hiding_tail_loss(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="LET window"):
        runner._evaluate_case(_source(), _analytic_table(let_scale=6.), .5)


@pytest.mark.parametrize("domain", ["al", "si"])
def test_different_table_domain_is_rejected(runner: ModuleType, domain: str) -> None:
    table = _analytic_table()
    nodes = (.02, *table.al_energy[1:])
    changed = replace(table, al_energy=nodes) if domain == "al" else replace(table, si_energy=nodes)
    with pytest.raises(ValueError, match="same.*domains"):
        runner._evaluate_case(_source(), changed, .5)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def inputs(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    """Mock only preceding archive/control/raw validators; hashing and physics helpers are real."""
    archive, controls = _reference_and_controls()
    paths = (tmp_path / "coarse.json", tmp_path / "dense.json",
             tmp_path / "native.json", tmp_path / "archive.fixture")
    for name in ("implementation.py", "production.txt", "generator.py", "source.txt"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    paths[3].write_text("synthetic archive bytes", encoding="utf-8")
    archive = replace(archive, archive_sha256=_sha(paths[3]))
    inventory = [{"symbol": "He", "z": 2, "mass_parameter_u": 4.,
                  "source_membership": ["GCR", "SEP"]}]
    manifest = {"projectiles": inventory}
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest)
    materials = [{"z": z, "atomic_mass_u": 27. if z == 13 else 28., "density_g_cm3": 2.}
                 for z in (13, 14)]
    production = {"production.txt": _sha(tmp_path / "production.txt")}
    generator = {"generator.py": _sha(tmp_path / "generator.py")}
    sources = {"source.txt": _sha(tmp_path / "source.txt")}
    implementation = {"implementation.py": _sha(tmp_path / "implementation.py")}
    for path, density in zip(paths[:2], (100, 200), strict=True):
        payload = _candidate_payload()
        payload.update({"manifest": manifest, "manifest_path": str(manifest_path),
                        "manifest_sha256": _sha(manifest_path), "points_per_decade": density})
        dataset = payload["datasets"][POLICY]
        dataset.update({"materials": materials, "projectiles": inventory,
                        "backend_configuration": runner.CONFIG,
                        "nonradiative_configuration": {**runner.CONFIG, "corrections": 8}})
        for row in dataset["rows"]:
            row["stopping_electronic_mev_cm2_g"] = 1000 * min(row["energy_mev_per_nucleon"], 2.)
        _write(path, payload)
    controls.update({"archive_sha256": archive.archive_sha256,
                     "archive_member_sha256": dict(archive.member_sha256), "materials": materials,
                     "native_source_pins_after": sources,
                     "production_resources_sha256_after": production,
                     "implementation_sha256_after": implementation})
    _write(paths[2], controls)

    def control_loader(path: Path, expected_sha: str, archive_sha: str,
                       energies: tuple[float, ...]) -> dict[str, Any]:
        assert _sha(path) == expected_sha
        assert archive_sha == archive.archive_sha256
        assert len(energies) == 101
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def validator(path: Path) -> dict[str, Any]:
        return {"input_sha256_after": _sha(path), "numerical_consistency_passed": True,
                "dataset_validations": {POLICY: {"max_relative_discrepancy": 0.}},
                "production_resources_sha256_after": production,
                "generator_hashes_after": generator, "source_pins_after": sources}

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "IMPLEMENTATION_PATHS", ("implementation.py",))
    monkeypatch.setattr(runner.native, "SOURCE", tmp_path)
    monkeypatch.setattr(runner, "SHIELDS", (0., .5))
    monkeypatch.setattr(runner, "run_validator", validator)
    monkeypatch.setattr(runner, "load_astar_archive", lambda _: archive)
    monkeypatch.setattr(runner, "_native_controls", control_loader)
    monkeypatch.setattr(runner, "_source_inputs", lambda _: tuple(
        _source(source=source, peak=peak) for source in (RadiationSource.GCR, RadiationSource.SEP)
        for peak in (True, False)
    ))
    return paths


def _run(runner: ModuleType, paths: tuple[Path, Path, Path, Path]) -> dict[str, Any]:
    return dict(runner.run_experiment(*paths[:3], native_sha256=_sha(paths[2]), archive_path=paths[3]))


def test_full_runner_fixture_is_diagnostic_not_acceptance(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path],
) -> None:
    report = _run(runner, inputs)
    assert report["case_count_per_variant"] == 8
    assert report["total_case_count"] == 48
    assert report["all_zero_shield_identities_passed"]
    assert report["implementation_sha256_before"] == report["implementation_sha256_after"]
    assert report["candidate_sha256"] == [_sha(path) for path in inputs[:2]]
    for flag in ("normative_export_allowed", "production_tables_replaced", "physical_accuracy_verified",
                 "physical_mass_policy_qualified", "stage4_acceptance_matrix_completed",
                 "astar_interpolation_error_bounded"):
        assert report[flag] is False
    assert report["energy_domain_mev_per_nucleon"] == [.01, 250.]
    assert "not_physical_acceptance" in report["scope"]
    assert len(report["factorial_interactions"]) == 8
    assert set(report["cases"]) == set(runner.VARIANTS)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("filename", [
    "coarse.json", "dense.json", "manifest.json", "native.json", "archive.fixture",
    "implementation.py", "production.txt", "generator.py", "source.txt",
])
def test_changed_pinned_input_resource_or_implementation_rejected(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch, filename: str,
) -> None:
    original = runner._source_inputs

    def sources(inventory: Any) -> tuple[Spectrum1D, ...]:
        selected = tuple(original(inventory))
        (inputs[0].parent / filename).write_text("changed during calculation", encoding="utf-8")
        return selected

    monkeypatch.setattr(runner, "_source_inputs", sources)
    with pytest.raises(ValueError):
        _run(runner, inputs)


def test_mismatched_candidate_manifests_are_rejected(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path],
) -> None:
    payload = json.loads(inputs[1].read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "f" * 64
    _write(inputs[1], payload)
    with pytest.raises(ValueError, match="Matched 100/200"):
        _run(runner, inputs)


def test_native_control_archive_members_must_match(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path],
) -> None:
    payload = json.loads(inputs[2].read_text(encoding="utf-8"))
    payload["archive_member_sha256"] = {"different": "f" * 64}
    _write(inputs[2], payload)
    with pytest.raises(ValueError, match="same ASTAR source members"):
        _run(runner, inputs)


def test_report_writer_never_overwrites_existing_file(runner: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"replacement": True}, output)
    assert output.read_text(encoding="utf-8") == "keep"


def test_cli_existing_output_fails_before_calculations(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["impact", "coarse", "dense", "native",
                                     "--native-sha256", "a" * 64, "--output", str(output)])
    monkeypatch.setattr(runner, "run_experiment", lambda *_args, **_kwargs: pytest.fail("must not run"))
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("flag", [False, None, 1, "true"])
def test_unpassed_candidate_numerical_check_is_not_ignored(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch, flag: Any,
) -> None:
    validator = runner.run_validator

    def incomplete(path: Path) -> dict[str, Any]:
        result = dict(validator(path))
        result["numerical_consistency_passed"] = flag
        return result

    monkeypatch.setattr(runner, "run_validator", incomplete)
    with pytest.raises(ValueError, match="must pass their declared"):
        _run(runner, inputs)


@pytest.mark.parametrize("discrepancy", [True, -1., nan, inf])
def test_invalid_reported_numerical_discrepancy_rejected(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch, discrepancy: float,
) -> None:
    validator = runner.run_validator

    def invalid(path: Path) -> dict[str, Any]:
        result = dict(validator(path))
        result["dataset_validations"][POLICY]["max_relative_discrepancy"] = discrepancy
        return result

    monkeypatch.setattr(runner, "run_validator", invalid)
    with pytest.raises(ValueError):
        _run(runner, inputs)


def _four_sources() -> tuple[Spectrum1D, ...]:
    return tuple(_source(source=source, peak=peak)
                 for source in (RadiationSource.GCR, RadiationSource.SEP) for peak in (True, False))


def test_four_unique_he_source_modes_are_required(runner: ModuleType) -> None:
    sources = _four_sources()
    assert runner._validate_source_roster(sources) is None
    for changed in ((), sources[:-1], (sources[0],) * 4):
        with pytest.raises(ValueError, match="Four unique"):
            runner._validate_source_roster(changed)


@pytest.mark.parametrize("species", [
    ParticleSpecies("He3", atomic_number=2, mass_number=3),
    ParticleSpecies("Li", atomic_number=3, mass_number=7),
    ParticleSpecies("aggregate", is_aggregate=True), None,
])
def test_source_roster_cannot_mask_wrong_or_aggregate_species(
    runner: ModuleType, species: ParticleSpecies | None,
) -> None:
    sources = _four_sources()
    with pytest.raises(ValueError, match="Four unique"):
        runner._validate_source_roster((replace(sources[0], species=species), *sources[1:]))


def test_runner_rejects_duplicate_sources_not_just_wrong_count(
    runner: ModuleType, inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_source_inputs", lambda _: (_source(),) * 4)
    with pytest.raises(ValueError, match="Four unique"):
        _run(runner, inputs)
