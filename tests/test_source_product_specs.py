from radar.core.source_product_specs import (
    RadiationProductLocation,
    RadiationProductNormativeStatus,
    RadiationProductPurpose,
    SourceRadiationProductSpec,
    has_source_radiation_product_spec,
    source_radiation_product_specs_for_purpose,
    source_radiation_product_specs_for_source,
    source_radiation_product_specs_for_source_and_particle,
    source_radiation_product_specs_for_status,
)
from radar.core.types import Particle, RadiationProductKind, RadiationSource


def test_sep_product_specs_include_dose_and_single_event_products() -> None:
    sep_specs = source_radiation_product_specs_for_source(RadiationSource.SEP)

    assert has_source_radiation_product_spec(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        purpose=RadiationProductPurpose.ACCUMULATED_DOSE,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.PEAK_FLUX,
        purpose=RadiationProductPurpose.SINGLE_EVENT_EFFECT,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.PEAK_LET_FLUX,
        purpose=RadiationProductPurpose.SINGLE_EVENT_EFFECT,
    )
    assert all(spec.source is RadiationSource.SEP for spec in sep_specs)


def test_gcr_product_specs_include_mean_maximum_and_let_products() -> None:
    assert has_source_radiation_product_spec(
        source=RadiationSource.GCR,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        purpose=RadiationProductPurpose.ACCUMULATED_DOSE,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.GCR,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MAXIMUM_FLUX,
        purpose=RadiationProductPurpose.SINGLE_EVENT_EFFECT,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MAXIMUM_LET_FLUX,
        purpose=RadiationProductPurpose.SINGLE_EVENT_EFFECT,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MEAN_LET_FLUX,
        purpose=RadiationProductPurpose.SINGLE_EVENT_EFFECT,
    )


def test_erb_product_specs_include_proton_electron_and_no_heavy_ion_products() -> None:
    assert has_source_radiation_product_spec(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        purpose=RadiationProductPurpose.ACCUMULATED_DOSE,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        purpose=RadiationProductPurpose.ACCUMULATED_DOSE,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MAXIMUM_FLUX,
        purpose=RadiationProductPurpose.SINGLE_EVENT_EFFECT,
    )
    assert not source_radiation_product_specs_for_source_and_particle(
        RadiationSource.ERB,
        Particle.HZE,
    )


def test_ost_minimum_accumulated_dose_products_are_separated_from_extended_balance() -> None:
    ost_minimum_dose_specs = tuple(
        spec
        for spec in source_radiation_product_specs_for_purpose(
            RadiationProductPurpose.ACCUMULATED_DOSE
        )
        if spec.normative_status is RadiationProductNormativeStatus.OST_MINIMUM
    )

    assert {
        (spec.source, spec.particle, spec.product_kind)
        for spec in ost_minimum_dose_specs
    } == {
        (
            RadiationSource.SEP,
            Particle.PROTON,
            RadiationProductKind.MISSION_FLUENCE,
        ),
        (
            RadiationSource.ERB,
            Particle.PROTON,
            RadiationProductKind.MISSION_FLUENCE,
        ),
        (
            RadiationSource.ERB,
            Particle.ELECTRON,
            RadiationProductKind.MISSION_FLUENCE,
        ),
    }


def test_single_event_product_specs_cover_five_mechanisms() -> None:
    single_event_specs = source_radiation_product_specs_for_purpose(
        RadiationProductPurpose.SINGLE_EVENT_EFFECT
    )

    required = {
        (
            RadiationSource.SEP,
            Particle.PROTON,
            RadiationProductKind.PEAK_FLUX,
        ),
        (
            RadiationSource.GCR,
            Particle.PROTON,
            RadiationProductKind.MAXIMUM_FLUX,
        ),
        (
            RadiationSource.ERB,
            Particle.PROTON,
            RadiationProductKind.MAXIMUM_FLUX,
        ),
        (
            RadiationSource.SEP,
            Particle.HZE,
            RadiationProductKind.PEAK_LET_FLUX,
        ),
        (
            RadiationSource.GCR,
            Particle.HZE,
            RadiationProductKind.MAXIMUM_LET_FLUX,
        ),
    }

    actual = {
        (spec.source, spec.particle, spec.product_kind)
        for spec in single_event_specs
    }

    assert required <= actual


def test_let_output_form_specs_are_behind_shielding_let_products() -> None:
    let_product_kinds = {
        RadiationProductKind.PEAK_LET_FLUX,
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
    }
    let_output_form_specs = tuple(
        spec
        for spec in source_radiation_product_specs_for_status(
            RadiationProductNormativeStatus.OUTPUT_FORM
        )
        if spec.product_kind in let_product_kinds
    )

    assert let_output_form_specs
    assert all(
        spec.location is RadiationProductLocation.BEHIND_SHIELDING
        for spec in let_output_form_specs
    )
    assert all(spec.particle is Particle.HZE for spec in let_output_form_specs)
    assert all(
        spec.product_kind in let_product_kinds
        for spec in let_output_form_specs
    )


def test_source_radiation_product_spec_rejects_empty_purpose() -> None:
    try:
        SourceRadiationProductSpec(
            source=RadiationSource.SEP,
            particle=Particle.PROTON,
            product_kind=RadiationProductKind.MISSION_FLUENCE,
            location=RadiationProductLocation.BEFORE_SHIELDING,
            purposes=(),
            normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
            title="Флюенс протонов СКЛ за САС",
        )
    except ValueError as exc:
        assert "at least one purpose" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty purposes")


def test_source_radiation_product_spec_rejects_empty_title() -> None:
    try:
        SourceRadiationProductSpec(
            source=RadiationSource.SEP,
            particle=Particle.PROTON,
            product_kind=RadiationProductKind.MISSION_FLUENCE,
            location=RadiationProductLocation.BEFORE_SHIELDING,
            purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
            normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
            title="",
        )
    except ValueError as exc:
        assert "title" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty title")



def test_hze_intermediate_products_are_defined_before_let_output_forms() -> None:
    assert has_source_radiation_product_spec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.PEAK_FLUX,
        purpose=RadiationProductPurpose.INTERMEDIATE,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MAXIMUM_FLUX,
        purpose=RadiationProductPurpose.INTERMEDIATE,
    )
    assert has_source_radiation_product_spec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MEAN_FLUX,
        purpose=RadiationProductPurpose.INTERMEDIATE,
    )



def test_erb_electron_mean_flux_is_intermediate_product() -> None:
    assert has_source_radiation_product_spec(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        purpose=RadiationProductPurpose.INTERMEDIATE,
    )



def test_sep_proton_mean_flux_is_output_characteristic() -> None:
    assert has_source_radiation_product_spec(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        purpose=RadiationProductPurpose.OUTPUT_CHARACTERISTIC,
    )
