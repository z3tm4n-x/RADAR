import pytest

from radar.core.products import (
    SpectrumProduct,
    allowed_spectrum_quantities_for_product_kind,
    default_dose_unit_for_product_kind,
    dose_quantity_for_product_kind,
    radiation_quantity_mode_for_product_kind,
    validate_spectrum_matches_product_kind,
)
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    DoseQuantity,
    Particle,
    RadiationProductKind,
    RadiationQuantityMode,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit


def _spectrum(quantity: SpectrumQuantity) -> Spectrum1D:
    if quantity in (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ):
        x_unit = Unit.LET
        y_unit = (
            Unit.DIFFERENTIAL_LET_FLUENCE
            if quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
            else Unit.DIFFERENTIAL_LET_FLUX
        )
    else:
        x_unit = Unit.MEV
        y_unit = (
            Unit.DIFFERENTIAL_FLUENCE
            if quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
            else Unit.DIFFERENTIAL_FLUX
        )

    return Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(4.0, 5.0, 6.0),
        x_unit=x_unit,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="test",
    )


@pytest.mark.parametrize(
    ("product_kind", "expected_quantity"),
    [
        (
            RadiationProductKind.MISSION_FLUENCE,
            SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        ),
        (
            RadiationProductKind.ORBIT_AVERAGED_FLUX,
            SpectrumQuantity.DIFFERENTIAL_FLUX,
        ),
        (
            RadiationProductKind.MEAN_FLUX,
            SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        ),
        (
            RadiationProductKind.MAXIMUM_FLUX,
            SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        ),
        (
            RadiationProductKind.PEAK_FLUX,
            SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        ),
        (
            RadiationProductKind.MISSION_LET_FLUENCE,
            SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
        ),
        (
            RadiationProductKind.MEAN_LET_FLUX,
            SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        ),
        (
            RadiationProductKind.MAXIMUM_LET_FLUX,
            SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        ),
        (
            RadiationProductKind.PEAK_LET_FLUX,
            SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        ),
    ],
)
def test_product_kind_validates_matching_spectrum_quantity(
    product_kind: RadiationProductKind,
    expected_quantity: SpectrumQuantity,
) -> None:
    spectrum = _spectrum(expected_quantity)

    validate_spectrum_matches_product_kind(
        spectrum=spectrum,
        product_kind=product_kind,
    )

    product = SpectrumProduct(kind=product_kind, spectrum=spectrum)

    assert product.kind is product_kind
    assert product.spectrum is spectrum
    assert expected_quantity in allowed_spectrum_quantities_for_product_kind(product_kind)


@pytest.mark.parametrize(
    "product_kind",
    [
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MISSION_LET_FLUENCE,
    ],
)
def test_fluence_products_map_to_accumulated_dose(
    product_kind: RadiationProductKind,
) -> None:
    assert (
        radiation_quantity_mode_for_product_kind(product_kind)
        is RadiationQuantityMode.FLUENCE
    )
    assert dose_quantity_for_product_kind(product_kind) is DoseQuantity.ACCUMULATED_DOSE
    assert default_dose_unit_for_product_kind(product_kind) is Unit.RAD


@pytest.mark.parametrize(
    "product_kind",
    [
        RadiationProductKind.ORBIT_AVERAGED_FLUX,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.PEAK_FLUX,
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
        RadiationProductKind.PEAK_LET_FLUX,
    ],
)
def test_flux_products_map_to_dose_rate(
    product_kind: RadiationProductKind,
) -> None:
    assert radiation_quantity_mode_for_product_kind(product_kind) is RadiationQuantityMode.FLUX
    assert dose_quantity_for_product_kind(product_kind) is DoseQuantity.DOSE_RATE
    assert default_dose_unit_for_product_kind(product_kind) is Unit.RAD_PER_SECOND


def test_spectrum_product_exposes_derived_quantity_metadata() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.PEAK_FLUX,
        spectrum=_spectrum(SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX),
        label="SEP peak proton flux",
    )

    assert product.label == "SEP peak proton flux"
    assert product.radiation_quantity_mode is RadiationQuantityMode.FLUX
    assert product.dose_quantity is DoseQuantity.DOSE_RATE
    assert product.default_dose_unit is Unit.RAD_PER_SECOND


def test_product_rejects_mismatched_spectrum_quantity() -> None:
    spectrum = _spectrum(SpectrumQuantity.DIFFERENTIAL_FLUX)

    with pytest.raises(ValueError, match="does not match"):
        SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=spectrum,
        )
