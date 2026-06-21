import pytest

from radar.core.source_products import (
    allowed_product_kinds_for_source,
    is_product_kind_allowed_for_source,
    validate_product_allowed_for_source,
    validate_product_kind_allowed_for_source,
    validate_product_spectra_match_spectra,
    validate_products_allowed_for_source,
    validate_products_match_spectra_and_source,
    validate_spectra_match_expected,
)
from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit


def _sep_mission_fluence_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(4.0, 5.0, 6.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.SEP,
            model="sep",
        ),
    )


def _gcr_mission_fluence_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(4.0, 5.0, 6.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.GCR,
            model="gcr",
        ),
    )


def _sep_peak_let_product(
    particle: Particle = Particle.HZE,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.PEAK_LET_FLUX,
        spectrum=Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(4.0, 5.0, 6.0),
            x_unit=Unit.LET,
            y_unit=Unit.DIFFERENTIAL_LET_FLUX,
            quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
            particle=particle,
            source=RadiationSource.SEP,
            model="sep_let",
        ),
    )


def test_sep_allowed_products() -> None:
    assert set(allowed_product_kinds_for_source(RadiationSource.SEP)) == {
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.PEAK_FLUX,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.PEAK_LET_FLUX,
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MISSION_LET_FLUENCE,
    }


def test_gcr_allowed_products() -> None:
    assert set(allowed_product_kinds_for_source(RadiationSource.GCR)) == {
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_LET_FLUENCE,
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
    }


def test_erb_allowed_products() -> None:
    assert set(allowed_product_kinds_for_source(RadiationSource.ERB)) == {
        RadiationProductKind.ORBIT_AVERAGED_FLUX,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MAXIMUM_FLUX,
    }


def test_product_kind_allowed_for_source_predicate() -> None:
    assert (
        is_product_kind_allowed_for_source(
            product_kind=RadiationProductKind.PEAK_FLUX,
            source=RadiationSource.SEP,
        )
        is True
    )
    assert (
        is_product_kind_allowed_for_source(
            product_kind=RadiationProductKind.PEAK_LET_FLUX,
            source=RadiationSource.SEP,
        )
        is True
    )


def test_validate_product_kind_allowed_for_source() -> None:
    validate_product_kind_allowed_for_source(
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        source=RadiationSource.SEP,
    )

    with pytest.raises(ValueError, match="not allowed"):
        validate_product_kind_allowed_for_source(
            product_kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
            source=RadiationSource.SEP,
        )


def test_validate_product_allowed_for_source() -> None:
    validate_product_allowed_for_source(
        product=_sep_mission_fluence_product(),
        source=RadiationSource.SEP,
    )


def test_validate_product_allowed_for_source_rejects_wrong_spectrum_source() -> None:
    with pytest.raises(ValueError, match="spectrum source"):
        validate_product_allowed_for_source(
            product=_gcr_mission_fluence_product(),
            source=RadiationSource.SEP,
        )

def test_validate_products_allowed_for_source() -> None:
    validate_products_allowed_for_source(
        products=(_sep_mission_fluence_product(),),
        source=RadiationSource.SEP,
    )


def test_validate_products_allowed_for_source_rejects_wrong_source() -> None:
    with pytest.raises(ValueError, match="spectrum source"):
        validate_products_allowed_for_source(
            products=(_gcr_mission_fluence_product(),),
            source=RadiationSource.SEP,
        )


def test_validate_spectra_match_expected() -> None:
    product = _sep_mission_fluence_product()

    validate_spectra_match_expected(
        spectra=(product.spectrum,),
        expected_spectra=(product.spectrum,),
        mismatch_message="spectra mismatch",
    )


def test_validate_spectra_match_expected_rejects_mismatch() -> None:
    sep_product = _sep_mission_fluence_product()
    gcr_product = _gcr_mission_fluence_product()

    with pytest.raises(ValueError, match="spectra mismatch"):
        validate_spectra_match_expected(
            spectra=(sep_product.spectrum,),
            expected_spectra=(gcr_product.spectrum,),
            mismatch_message="spectra mismatch",
        )


def test_validate_product_spectra_match_spectra() -> None:
    product = _sep_mission_fluence_product()

    validate_product_spectra_match_spectra(
        products=(product,),
        spectra=(product.spectrum,),
        mismatch_message="product spectra mismatch",
    )


def test_validate_product_spectra_match_spectra_rejects_mismatch() -> None:
    sep_product = _sep_mission_fluence_product()
    gcr_product = _gcr_mission_fluence_product()

    with pytest.raises(ValueError, match="product spectra mismatch"):
        validate_product_spectra_match_spectra(
            products=(sep_product,),
            spectra=(gcr_product.spectrum,),
            mismatch_message="product spectra mismatch",
        )


def test_validate_products_match_spectra_and_source() -> None:
    product = _sep_mission_fluence_product()

    validate_products_match_spectra_and_source(
        products=(product,),
        spectra=(product.spectrum,),
        source=RadiationSource.SEP,
        mismatch_message="combined mismatch",
    )


def test_validate_products_match_spectra_and_source_rejects_wrong_source() -> None:
    product = _gcr_mission_fluence_product()

    with pytest.raises(ValueError, match="spectrum source"):
        validate_products_match_spectra_and_source(
            products=(product,),
            spectra=(product.spectrum,),
            source=RadiationSource.SEP,
            mismatch_message="combined mismatch",
        )



def test_validate_product_allowed_for_source_accepts_hze_specific_product() -> None:
    validate_product_allowed_for_source(
        product=_sep_peak_let_product(),
        source=RadiationSource.SEP,
    )


def test_validate_product_allowed_for_source_rejects_wrong_particle_for_spec() -> None:
    with pytest.raises(ValueError, match="not specified"):
        validate_product_allowed_for_source(
            product=_sep_peak_let_product(particle=Particle.PROTON),
            source=RadiationSource.SEP,
        )
