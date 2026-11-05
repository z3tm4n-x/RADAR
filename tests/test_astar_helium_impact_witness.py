"""Analytic controls for the independent incident-energy midpoint witness."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumOrdinateBasis, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.candidate_transport import CandidateIonTable


@pytest.fixture(scope="module")
def witness() -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        spec = importlib.util.spec_from_file_location(
            "astar_impact_witness_test", scripts / "check_astar_helium_impact_witness.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _source(*, cumulative: bool = False, scale: float = 1.0) -> Spectrum1D:
    return Spectrum1D(
        x=(1., 2., 4.), y=(2 * scale,) * 3,
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY if cumulative
                else Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY),
        quantity=(SpectrumQuantity.DIFFERENTIAL_FLUENCE if cumulative
                  else SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX),
        particle=Particle.HZE, source=RadiationSource.GCR, model="analytic_witness_test",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.MISSION_TOTAL if cumulative else TemporalBasis.SOURCE_MAXIMUM,
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies("He", atomic_number=2, mass_number=4),
        assessment_mode=AssessmentMode.CUMULATIVE if cumulative else AssessmentMode.PEAK,
        mission_duration_s=1234567.0 if cumulative else None,
        model_reference=ModelReference("analytic_witness_test", "1", ("synthetic",), ("test",)),
        transformation_chain=(TransformationStep("synthetic", "1", (
            ("source_tabulated_mass_parameter", "4.0"),
        )),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


def _table(*, offset: float = 17.0) -> CandidateIonTable:
    return CandidateIonTable(
        z=2, mass_parameter_u=4.0, mass_policy="source_tabulated_parameter_diagnostic",
        dataset_ref="synthetic-analytic-witness-test", al_energy=(.5, 1., 2., 3.),
        al_range=tuple(e + offset for e in (.5, 1., 2., 3.)),
        si_energy=(.5, 1., 2., 3.), si_electronic_let=(.5, 1., 2., 3.),
    )


def test_midpoint_measure_matches_analytic_range_and_tail_partition(witness: ModuleType) -> None:
    observed = witness._midpoint_metrics(_source(), _table(), .5, 200_000)
    expected = {
        "source_integral": 6., "inside_al_source_integral": 4.,
        "outside_al_source_integral": 2., "outside_al_domain_integral": 2.,
        "unresolved_low_residual_integral": 0., "supported_exit_integral": 4.,
        "outside_si_domain_integral": 0., "mapped_let_integral": 4.,
        "exit_below_source_min_integral": 1., "let_above_1.5": 2.,
        "energy_above_1.0": 3.,
    }
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value, abs=2e-4), key


def test_zero_shield_preserves_full_energy_but_not_missing_si_data(witness: ModuleType) -> None:
    observed = witness._midpoint_metrics(_source(), _table(), 0., 200_000)
    assert observed["source_integral"] == pytest.approx(6., abs=1e-8)
    assert observed["supported_exit_integral"] == observed["source_integral"]
    assert observed["outside_al_domain_integral"] == 0.
    assert observed["outside_al_source_integral"] == pytest.approx(2., abs=2e-4)
    assert observed["outside_si_domain_integral"] == pytest.approx(2., abs=2e-4)
    assert observed["mapped_let_integral"] == pytest.approx(4., abs=2e-4)
    assert observed["unresolved_low_residual_integral"] == 0.


def test_thick_shield_unresolved_is_not_claimed_stopped(witness: ModuleType) -> None:
    observed = witness._midpoint_metrics(_source(), _table(), 100., 200_000)
    assert observed["unresolved_low_residual_integral"] == pytest.approx(4., abs=2e-4)
    assert observed["outside_al_domain_integral"] == pytest.approx(2., abs=2e-4)
    assert observed["supported_exit_integral"] == observed["mapped_let_integral"] == 0.
    assert not any("stopped" in key for key in observed)


@pytest.mark.parametrize("shield", [0., .5, 1.5, 100.])
def test_cumulative_measure_has_no_extra_time_multiplier(witness: ModuleType, shield: float) -> None:
    peak = witness._midpoint_metrics(_source(), _table(), shield, 20_000)
    cumulative = witness._midpoint_metrics(_source(cumulative=True), _table(), shield, 20_000)
    assert peak == cumulative
    scaled = witness._midpoint_metrics(_source(cumulative=True, scale=7.), _table(), shield, 20_000)
    for name, value in peak.items():
        assert scaled[name] == pytest.approx(value * 7, rel=2e-14, abs=1e-14), name


@pytest.mark.parametrize("offset", [1., 137., 2**40])
def test_common_range_origin_cancels_before_shield_subtraction(
    witness: ModuleType, offset: float,
) -> None:
    baseline = witness._midpoint_metrics(_source(), _table(), .5, 20_000)
    shifted = witness._midpoint_metrics(_source(), _table(offset=offset), .5, 20_000)
    assert baseline == shifted


def test_zero_source_does_not_generate_counts(witness: ModuleType) -> None:
    observed = witness._midpoint_metrics(_source(scale=0.), _table(), .5, 20_000)
    assert all(value == 0. for value in observed.values())


def test_nonmonotone_let_two_preimages_are_counted_independently(witness: ModuleType) -> None:
    table = replace(_table(), si_energy=(.5, 1.5, 2.5), si_electronic_let=(.5, 2.5, .5))
    observed = witness._midpoint_metrics(_source(), table, .5, 200_000)
    # Exit density remains 2 on [.5, 2.5]; L >= 1.5 occurs on [1, 2].
    assert observed["let_above_1.5"] == pytest.approx(2., abs=2e-4)
    assert observed["mapped_let_integral"] == pytest.approx(4., abs=2e-4)


@pytest.mark.parametrize("count", [True, False, 0, 1, -1, 2.5, 20_000_001])
def test_invalid_midpoint_count_rejected(witness: ModuleType, count: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        witness._midpoint_metrics(_source(), _table(), .5, count)


def test_bin_average_input_is_not_interpreted_as_point_samples(witness: ModuleType) -> None:
    source = replace(_source(), x=(1.5, 3.), y=(2., 6.), bin_edges=(1., 2., 4.),
                     ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE)
    observed = witness._midpoint_metrics(source, _table(), .5, 200_000)
    # Integrals are 2*(2-1)+6*(4-2), with the part above E=3 unresolved.
    assert observed["source_integral"] == pytest.approx(14., abs=4e-4)
    assert observed["outside_al_domain_integral"] == pytest.approx(6., abs=4e-4)
    assert observed["supported_exit_integral"] == pytest.approx(8., abs=4e-4)
    assert observed["let_above_1.5"] == pytest.approx(6., abs=4e-4)


@pytest.mark.parametrize("shield", [0., .5])
def test_entire_source_outside_table_remains_in_coverage_ledger(
    witness: ModuleType, shield: float,
) -> None:
    source = replace(_source(), x=(4., 5., 6.))
    observed = witness._midpoint_metrics(source, _table(), shield, 20_000)
    assert observed["source_integral"] == pytest.approx(4., abs=1e-8)
    assert observed["inside_al_source_integral"] == 0.
    assert observed["outside_al_source_integral"] == observed["source_integral"]
    assert observed["mapped_let_integral"] == 0.
    assert observed["unresolved_low_residual_integral"] == 0.
    assert observed["supported_exit_integral"] == (observed["source_integral"] if shield == 0 else 0.)


@pytest.mark.parametrize("shield", [True, -1., float("nan"), float("inf"), ".1"])
def test_invalid_shield_rejected(witness: ModuleType, shield: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        witness._midpoint_metrics(_source(), _table(), shield, 100)


@pytest.mark.parametrize("counts", [(), (2, 2), (4, 2), (2, True), [2, 4], (2, 20_000_001)])
def test_invalid_resolution_ladder_rejected(witness: ModuleType, counts: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        witness._validate_samples(counts)


_FALSE_FLAGS = (
    "normative_export_allowed", "physical_accuracy_verified", "production_tables_replaced",
    "physical_mass_policy_qualified", "stage4_acceptance_matrix_completed",
    "astar_interpolation_error_bounded",
)


def _report() -> dict[str, object]:
    return {"schema_version": "radar.astar_helium_spectral_impact.v1",
            **dict.fromkeys(_FALSE_FLAGS, False)}


def test_report_hash_and_nonrelease_flags_are_checked(
    witness: ModuleType, tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert witness._read_report(path, digest) == _report()
    with pytest.raises(ValueError, match="SHA256"):
        witness._read_report(path, "a" * 64)


@pytest.mark.parametrize("flag", _FALSE_FLAGS)
@pytest.mark.parametrize("value", [True, 0, None])
def test_report_cannot_silently_upgrade_qualification(
    witness: ModuleType, tmp_path: Path, flag: str, value: object,
) -> None:
    raw = _report()
    raw[flag] = value
    path = tmp_path / "report.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification"):
        witness._read_report(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("payload", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_duplicate_and_nonfinite_json_are_rejected(
    witness: ModuleType, tmp_path: Path, payload: str,
) -> None:
    path = tmp_path / "report.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        witness._read_report(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_changed_file_invalidates_witness_pin(witness: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "pinned.txt"
    path.write_text("original", encoding="utf-8")
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    assert witness._check_pins(pins) == pins
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        witness._check_pins(pins)


def test_input_path_must_resolve_to_a_file_inside_repo(
    witness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(witness, "ROOT", root)
    inside = root / "input.json"
    outside = tmp_path / "outside.json"
    inside.write_text("{}", encoding="utf-8")
    outside.write_text("{}", encoding="utf-8")
    assert witness._inside_repo(inside) == inside.resolve()
    for path in (outside, root, root / ".." / "outside.json"):
        with pytest.raises(ValueError, match="inside this repository"):
            witness._inside_repo(path)


def _localized_table() -> CandidateIonTable:
    return replace(_table(), al_energy=(.01, .5, 1., 2.),
                   al_range=(17.01, 17.5, 18., 19.),
                   si_energy=(.01, .25, .5, 1., 2.),
                   si_electronic_let=(.5, 1.5, 2.5, .5, .25))


def test_localized_forward_check_matches_two_branch_analytic_tails(witness: ModuleType) -> None:
    source = replace(_source(), x=(.1, 1., 3.))
    result = witness._localized_high_let_metrics(source, _localized_table(), 1., 100_000)
    assert result["exit_energy_band_mev_per_nucleon"] == pytest.approx((.01, 1.))
    assert result["incident_energy_band_mev_per_nucleon"] == pytest.approx((1.01, 2.))
    assert result["outside_exit_band_let_upper_bound"] == pytest.approx(.5)
    assert result["outside_exit_band_let_bound_below_threshold"] is True
    assert result["band_source_integral"] == pytest.approx(1.98)
    for threshold, expected in ((1.25, 1.245), (1.4, 1.098), (1.45, 1.049), (1.5, 1.)):
        assert result["threshold_integrals"][f"let_above_{threshold}"] == pytest.approx(
            expected, abs=5e-5,
        )


def test_localized_band_is_not_normalized_or_multiplied_by_mission_time(witness: ModuleType) -> None:
    source = replace(_source(), x=(.1, 1., 3.))
    cumulative = replace(_source(cumulative=True, scale=7.), x=source.x)
    left = witness._localized_high_let_metrics(source, _localized_table(), 1., 20_000)
    right = witness._localized_high_let_metrics(cumulative, _localized_table(), 1., 20_000)
    assert right["band_source_integral"] == pytest.approx(7 * left["band_source_integral"])
    for name, value in left["threshold_integrals"].items():
        assert right["threshold_integrals"][name] == pytest.approx(7 * value)


@pytest.mark.parametrize("upper_let", [1.25, 2., 10.])
def test_localized_band_cannot_ignore_high_let_outside_it(
    witness: ModuleType, upper_let: float,
) -> None:
    source = replace(_source(), x=(.1, 1., 3.))
    table = replace(_localized_table(), si_electronic_let=(.5, 1.5, 2.5, .5, upper_let))
    with pytest.raises(ValueError):
        witness._localized_high_let_metrics(source, table, 1., 100)


@pytest.mark.parametrize("shield", [0., .01, 100.])
def test_localized_probe_rejects_zero_shield_or_unsupported_incident_band(
    witness: ModuleType, shield: float,
) -> None:
    source = replace(_source(), x=(.1, 1., 3.))
    with pytest.raises(ValueError):
        witness._localized_high_let_metrics(source, _localized_table(), shield, 100)
