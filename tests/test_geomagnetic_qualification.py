"""Independent access provenance, time-unit preservation and release quarantine."""

from dataclasses import replace

import pytest

from radar.core.products import SpectrumProduct, select_products_for_assessment_mode
from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import scale_spectrum
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationProductKind, RadiationSource,
    SpectrumAxisKind, SpectrumCompleteness, SpectrumLocation, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.geomagnetic.ost_orbit import OstOrbitSample
from radar.geomagnetic.ost_penetration import build_ost_penetration_function_from_samples
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid
from radar.geomagnetic.spectrum import apply_hze_penetration, apply_proton_penetration


def _spectrum(source: RadiationSource, particle: Particle, mode: AssessmentMode) -> Spectrum1D:
    peak = mode is AssessmentMode.PEAK
    ion = particle is Particle.HZE
    quantity = (
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if source is RadiationSource.SEP
        else SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    ) if peak else SpectrumQuantity.DIFFERENTIAL_FLUENCE
    y_unit = (
        Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
        else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY
    ) if ion else (Unit.DIFFERENTIAL_FLUX if peak else Unit.DIFFERENTIAL_FLUENCE)
    return Spectrum1D(
        x=(1.0, 10.0, 100.0), y=(3.0, 2.0, 1.0),
        x_unit=Unit.MEV_PER_NUCLEON if ion else Unit.MEV, y_unit=y_unit,
        quantity=quantity, particle=particle, source=source, model="synthetic_source",
        axis_kind=(SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON if ion
                   else SpectrumAxisKind.KINETIC_ENERGY),
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=(TemporalBasis.SEP_PEAK if source is RadiationSource.SEP
                        else TemporalBasis.SOURCE_MAXIMUM) if peak else TemporalBasis.MISSION_TOTAL,
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies(symbol="He" if ion else "p", atomic_number=2 if ion else 1,
                                mass_number=4 if ion else 1),
        assessment_mode=mode, mission_duration_s=None if peak else 31500000.0,
        model_reference=ModelReference(
            name="synthetic_source", version="test-v1",
            document_refs=("synthetic-test-only",), dataset_refs=("synthetic-test-data",),
        ),
        transformation_chain=(TransformationStep(name="synthetic_source", version="1"),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


def _apply(spectrum: Spectrum1D, penetration: PenetrationFunction) -> Spectrum1D:
    if spectrum.particle is Particle.PROTON:
        return apply_proton_penetration(spectrum, penetration)
    return apply_hze_penetration(spectrum, penetration, mass_to_charge=2.0)


@pytest.mark.parametrize("source", [RadiationSource.SEP, RadiationSource.GCR])
@pytest.mark.parametrize("particle", [Particle.PROTON, Particle.HZE])
@pytest.mark.parametrize("mode", list(AssessmentMode))
def test_access_preserves_source_statistic_units_and_release_quarantine(
    source: RadiationSource, particle: Particle, mode: AssessmentMode,
) -> None:
    source_spectrum = _spectrum(source, particle, mode)
    source_spectrum.validate_production_contract()
    operator = PenetrationFunction(
        rigidity_grid=RigidityGrid((0.001, 100.0)), values=(0.25, 0.25),
        model="synthetic_access", model_version="pointwise-test-v2",
        parameters=(("averaging", "weighted_mean_of_pointwise_access"),),
        incomplete_reasons=("unqualified synthetic access",),
    )
    output = _apply(source_spectrum, operator)
    assert output.y == pytest.approx((0.75, 0.5, 0.25))
    assert output.quantity is source_spectrum.quantity
    assert output.temporal_basis is source_spectrum.temporal_basis
    assert output.assessment_mode is mode
    assert output.y_unit is source_spectrum.y_unit
    assert output.angular_basis is source_spectrum.angular_basis
    assert output.mission_duration_s == source_spectrum.mission_duration_s
    assert output.completeness is SpectrumCompleteness.INCOMPLETE
    assert output.incomplete_reasons == ("unqualified synthetic access",)
    assert source_spectrum.completeness is SpectrumCompleteness.COMPLETE
    step = output.transformation_chain[-1]
    assert step.version == "pointwise-test-v2"
    metadata = dict(step.parameters)
    assert metadata["time_multiplier"] == "none"
    assert metadata["operator_averaging"] == "weighted_mean_of_pointwise_access"
    if mode is AssessmentMode.PEAK:
        assert metadata["source_statistic_application"].endswith(
            "not_instantaneous_orbit_maximum"
        )
        kind = (RadiationProductKind.PEAK_FLUX if source is RadiationSource.SEP
                else RadiationProductKind.MAXIMUM_FLUX)
    else:
        assert metadata["source_statistic_application"] == "mission_fluence_no_time_rescaling"
        kind = RadiationProductKind.MISSION_FLUENCE
        longer_metadata = replace(source_spectrum, mission_duration_s=63000000.0)
        assert _apply(longer_metadata, operator).y == output.y
    assert _apply(scale_spectrum(source_spectrum, 2.0), operator).y == pytest.approx(
        tuple(2.0 * value for value in output.y)
    )
    with pytest.raises(ValueError, match="incomplete spectra"):
        output.validate_production_contract()
    with pytest.raises(ValueError, match="incomplete spectra"):
        select_products_for_assessment_mode(
            (SpectrumProduct(kind=kind, spectrum=output, label="diagnostic"),),
            mode, validate_for_release=True,
        )


def test_real_ost_builder_exposes_shadow_and_default_mlt_assumptions() -> None:
    sample = OstOrbitSample(
        index=0, time_s=0.0, true_anomaly_deg=0.0, radius_km=42157.2,
        altitude_km=35786.0, latitude_deg=0.0, longitude_deg=0.0,
    )
    operator = build_ost_penetration_function_from_samples(
        samples=(sample,), rigidity_grid=RigidityGrid((0.001, 100.0)),
        apply_disturbance=True,
    )
    metadata = dict(operator.parameters)
    assert metadata["cutoff_condition"] == "R>Rc_star"
    assert metadata["shadow_model"] == "opaque_sphere_geometric"
    assert metadata["mlt_policy"] == "uniform_ensemble"
    assert any("Zh.9" in reason for reason in operator.incomplete_reasons)
    assert any("MLT" in reason for reason in operator.incomplete_reasons)
    explicit = build_ost_penetration_function_from_samples(
        samples=(sample,), rigidity_grid=operator.rigidity_grid, apply_disturbance=True,
        magnetic_local_times_hours=(12.0,),
    )
    assert dict(explicit.parameters)["mlt_policy"] == "explicit_per_sample"
    assert any("Zh.9" in reason for reason in explicit.incomplete_reasons)
    assert not any("Uniform MLT" in reason for reason in explicit.incomplete_reasons)


def test_existing_incompleteness_is_not_erased_by_access() -> None:
    source = replace(
        _spectrum(RadiationSource.SEP, Particle.PROTON, AssessmentMode.PEAK),
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("missing source data",),
    )
    operator = PenetrationFunction(
        rigidity_grid=RigidityGrid((0.001, 100.0)), values=(1.0, 1.0), model="test",
        incomplete_reasons=("unqualified geometry",),
    )
    assert _apply(source, operator).incomplete_reasons == (
        "missing source data", "unqualified geometry",
    )


@pytest.mark.parametrize("parameters", [(("", "x"),), (("x", ""),), (("x", "a"), ("x", "b"))])
def test_penetration_provenance_rejects_ambiguous_parameters(
    parameters: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError, match="parameter"):
        PenetrationFunction(
            rigidity_grid=RigidityGrid((1.0,)), values=(1.0,), model="test", parameters=parameters,
        )
