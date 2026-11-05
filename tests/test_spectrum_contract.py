import pytest

from radar.core.products import SpectrumProduct, select_products_for_assessment_mode
from radar.core.spectra import (
    ModelReference,
    ParticleSpecies,
    ShieldingDescriptor,
    Spectrum1D,
    TransformationStep,
)
from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit


def _model_reference(name: str = "OST SEP") -> ModelReference:
    return ModelReference(
        name=name,
        version="134-1044-2007 amendment 2",
        document_refs=("OST 134-1044-2007, appendix B",),
        dataset_refs=("sep_protons/ost_134_1044_2007_coefficients.csv@sha256:test",),
    )


def _peak_sep_proton_spectrum(**overrides: object) -> Spectrum1D:
    values: dict[str, object] = {
        "x": (1.0, 10.0, 100.0),
        "y": (3.0, 2.0, 1.0),
        "x_unit": Unit.MEV,
        "y_unit": Unit.DIFFERENTIAL_FLUX,
        "quantity": SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        "particle": Particle.PROTON,
        "source": RadiationSource.SEP,
        "model": "OST SEP",
        "axis_kind": SpectrumAxisKind.KINETIC_ENERGY,
        "angular_basis": AngularBasis.OMNIDIRECTIONAL_4PI,
        "temporal_basis": TemporalBasis.SEP_PEAK,
        "location": SpectrumLocation.ON_ORBIT,
        "species": ParticleSpecies(symbol="p", atomic_number=1, mass_number=1),
        "assessment_mode": AssessmentMode.PEAK,
        "model_reference": _model_reference(),
        "transformation_chain": (
            TransformationStep(name="source_model", version="1"),
        ),
        "completeness": SpectrumCompleteness.COMPLETE,
    }
    values.update(overrides)
    return Spectrum1D(**values)  # type: ignore[arg-type]


def test_contract_enums_and_per_nucleon_units_are_explicit() -> None:
    assert AssessmentMode.PEAK.value == "peak"
    assert AssessmentMode.CUMULATIVE.value == "cumulative"
    assert AngularBasis.OMNIDIRECTIONAL_4PI.value == "omnidirectional_4pi"
    assert Unit.MEV_PER_NUCLEON.value == "MeV/nucleon"
    assert (
        Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY.value
        == "cm^-2 (MeV/nucleon)^-1"
    )


def test_legacy_spectrum_construction_remains_compatible_but_not_production_ready() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 2.0),
        y=(2.0, 1.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="legacy",
    )

    violations = spectrum.production_contract_violations()

    assert "axis_kind is required" in violations
    assert "angular_basis is required" in violations
    assert "model_reference is required" in violations
    with pytest.raises(ValueError, match="production contract"):
        spectrum.validate_production_contract()


def test_complete_peak_sep_product_satisfies_production_contract() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.PEAK_FLUX,
        spectrum=_peak_sep_proton_spectrum(),
    )

    assert product.production_contract_violations() == ()
    product.validate_production_contract()


def test_complete_cumulative_hze_product_uses_mev_per_nucleon_and_mission_time() -> None:
    spectrum = Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(3.0, 2.0, 1.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE,
        source=RadiationSource.GCR,
        model="OST GCR",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.MISSION_TOTAL,
        location=SpectrumLocation.BEHIND_SHIELD,
        species=ParticleSpecies(symbol="Fe", atomic_number=26, mass_number=56),
        shielding=ShieldingDescriptor(material="Al", areal_density_g_cm2=1.5),
        assessment_mode=AssessmentMode.CUMULATIVE,
        mission_duration_s=3.0 * 365.25 * 86400.0,
        model_reference=ModelReference(
            name="OST GCR",
            version="134-1044-2007 amendment 2",
            document_refs=("OST 134-1044-2007, appendix A",),
            dataset_refs=("gcr/ost_coefficients.csv@sha256:test",),
        ),
        transformation_chain=(
            TransformationStep(
                name="aluminium_transport",
                version="1",
                parameters=(("thickness_g_cm2", "1.5"),),
            ),
        ),
        completeness=SpectrumCompleteness.COMPLETE,
    )
    product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=spectrum,
    )

    assert product.production_contract_violations() == ()
    product.validate_production_contract()


def test_cumulative_contract_requires_mission_duration() -> None:
    spectrum = _peak_sep_proton_spectrum(
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        temporal_basis=TemporalBasis.MISSION_TOTAL,
        assessment_mode=AssessmentMode.CUMULATIVE,
    )
    product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=spectrum,
    )

    assert "cumulative spectra require mission_duration_s" in (
        product.production_contract_violations()
    )
    with pytest.raises(ValueError, match="mission_duration_s"):
        product.validate_production_contract()


def test_product_kind_rejects_wrong_temporal_semantics_at_production_boundary() -> None:
    spectrum = _peak_sep_proton_spectrum(
        temporal_basis=TemporalBasis.SOURCE_MAXIMUM,
    )
    product = SpectrumProduct(
        kind=RadiationProductKind.PEAK_FLUX,
        spectrum=spectrum,
    )

    violations = product.production_contract_violations()

    assert (
        "product peak_flux requires temporal_basis sep_peak"
        in violations
    )


def test_internal_mean_product_has_no_third_user_assessment_mode() -> None:
    spectrum = _peak_sep_proton_spectrum(
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        temporal_basis=TemporalBasis.MEAN_INTERNAL,
        assessment_mode=None,
    )
    product = SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=spectrum,
    )

    assert product.production_contract_violations() == ()


def test_mev_per_nucleon_axis_rejects_per_mev_ordinate_unit() -> None:
    spectrum = Spectrum1D(
        x=(10.0, 100.0),
        y=(2.0, 1.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE,
        source=RadiationSource.GCR,
        model="OST GCR",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
    )

    assert (
        "quantity differential_fluence on axis kinetic_energy_per_nucleon "
        "requires y_unit cm^-2 (MeV/nucleon)^-1"
        in spectrum.production_contract_violations()
    )
    with pytest.raises(ValueError, match="MeV/nucleon"):
        SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=spectrum,
        )


def test_behind_shield_location_requires_material_and_areal_density() -> None:
    spectrum = _peak_sep_proton_spectrum(
        location=SpectrumLocation.BEHIND_SHIELD,
    )

    assert "behind-shield spectra require shielding metadata" in (
        spectrum.production_contract_violations()
    )
    with pytest.raises(ValueError, match="non-negative"):
        ShieldingDescriptor(material="Al", areal_density_g_cm2=-0.1)


def test_particle_species_z_and_a_are_structural_and_consistent() -> None:
    iron = ParticleSpecies(symbol="Fe", atomic_number=26, mass_number=56)

    assert iron.atomic_number == 26
    assert iron.mass_number == 56
    with pytest.raises(ValueError, match="both be set"):
        ParticleSpecies(symbol="Fe", atomic_number=26)
    with pytest.raises(ValueError, match="smaller than Z"):
        ParticleSpecies(symbol="invalid", atomic_number=26, mass_number=25)


def test_model_traceability_requires_document_and_dataset_references() -> None:
    spectrum = _peak_sep_proton_spectrum(
        model_reference=ModelReference(name="OST SEP", version="1"),
    )

    violations = spectrum.production_contract_violations()

    assert "at least one normative document reference is required" in violations
    assert "at least one dataset reference is required" in violations


def test_model_reference_name_must_match_legacy_model_identifier() -> None:
    spectrum = _peak_sep_proton_spectrum(
        model_reference=_model_reference(name="different model"),
    )

    assert "model_reference name must match the legacy model name" in (
        spectrum.production_contract_violations()
    )


def test_incomplete_spectrum_is_explicit_and_cannot_pass_release_boundary() -> None:
    spectrum = _peak_sep_proton_spectrum(
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("missing Fe transport table",),
    )

    violations = spectrum.production_contract_violations()

    assert "incomplete spectra cannot be released as production results" in violations
    with pytest.raises(ValueError, match="incomplete spectra"):
        spectrum.validate_production_contract()

    SpectrumProduct(
        kind=RadiationProductKind.PEAK_FLUX,
        spectrum=spectrum,
    ).validate_calculation_contract()


def test_peak_contract_rejects_mission_duration_field() -> None:
    spectrum = _peak_sep_proton_spectrum(mission_duration_s=365.25 * 86400.0)

    assert "peak spectra must not carry mission_duration_s" in (
        spectrum.production_contract_violations()
    )


@pytest.mark.parametrize("placeholder", ("unversioned", "not_implemented"))
def test_production_contract_rejects_placeholder_model_version(
    placeholder: str,
) -> None:
    spectrum = _peak_sep_proton_spectrum(
        model_reference=ModelReference(
            name="OST SEP",
            version=placeholder,
            document_refs=("OST 134-1044-2007",),
            dataset_refs=("sep_protons/ost_134_1044_2007_coefficients.csv",),
        )
    )

    assert "production model_reference requires a real version" in (
        spectrum.production_contract_violations()
    )


def test_peak_selector_returns_only_canonical_differential_products() -> None:
    differential = SpectrumProduct(
        kind=RadiationProductKind.MAXIMUM_FLUX,
        spectrum=_peak_sep_proton_spectrum(
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            temporal_basis=TemporalBasis.SOURCE_MAXIMUM,
            source=RadiationSource.GCR,
        ),
    )
    integral = SpectrumProduct(
        kind=RadiationProductKind.MAXIMUM_FLUX,
        spectrum=_peak_sep_proton_spectrum(
            quantity=SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
            y_unit=Unit.INTEGRAL_FLUX,
            temporal_basis=TemporalBasis.SOURCE_MAXIMUM,
            source=RadiationSource.GCR,
        ),
    )
    mean = SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=_peak_sep_proton_spectrum(
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            temporal_basis=TemporalBasis.MEAN_INTERNAL,
            assessment_mode=None,
            source=RadiationSource.GCR,
        ),
    )

    assert select_products_for_assessment_mode(
        (differential, integral, mean),
        AssessmentMode.PEAK,
    ) == (differential,)


def test_release_selector_checks_normative_dataset_digest() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.PEAK_FLUX,
        spectrum=_peak_sep_proton_spectrum(),
    )

    with pytest.raises(ValueError, match="digest does not match"):
        select_products_for_assessment_mode(
            (product,),
            AssessmentMode.PEAK,
            validate_for_release=True,
        )
