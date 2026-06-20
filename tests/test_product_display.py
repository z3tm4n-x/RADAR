import pytest

from radar.core.product_display import (
    average_daily_spectrum_for_mission_years,
    average_daily_spectrum_for_product,
    daily_display_unit_for_product_kind,
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


def _mission_fluence_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(365.0, 730.0, 1095.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.SEP,
            model="test_sep",
        ),
    )


def _mission_let_fluence_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_LET_FLUENCE,
        spectrum=Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(365.0, 730.0, 1095.0),
            x_unit=Unit.LET,
            y_unit=Unit.DIFFERENTIAL_LET_FLUENCE,
            quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.GCR,
            model="test_gcr_let",
        ),
    )


def _mean_flux_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(1.0, 2.0, 3.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            particle=Particle.PROTON,
            source=RadiationSource.GCR,
            model="test_gcr_flux",
        ),
    )


def test_daily_display_unit_for_accumulated_products() -> None:
    assert (
        daily_display_unit_for_product_kind(RadiationProductKind.MISSION_FLUENCE)
        is Unit.DIFFERENTIAL_FLUENCE_PER_DAY
    )
    assert (
        daily_display_unit_for_product_kind(RadiationProductKind.MISSION_LET_FLUENCE)
        is Unit.DIFFERENTIAL_LET_FLUENCE_PER_DAY
    )


def test_daily_display_unit_rejects_flux_product() -> None:
    with pytest.raises(ValueError, match="cannot be converted"):
        daily_display_unit_for_product_kind(RadiationProductKind.MEAN_FLUX)


def test_average_daily_spectrum_for_mission_fluence_product() -> None:
    daily_spectrum = average_daily_spectrum_for_product(
        product=_mission_fluence_product(),
        duration_days=365.0,
    )

    assert daily_spectrum.y == pytest.approx((1.0, 2.0, 3.0))
    assert daily_spectrum.x_unit is Unit.MEV
    assert daily_spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE_PER_DAY
    assert daily_spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert daily_spectrum.particle is Particle.PROTON
    assert daily_spectrum.source is RadiationSource.SEP
    assert daily_spectrum.model == "test_sep"


def test_average_daily_spectrum_for_mission_let_fluence_product() -> None:
    daily_spectrum = average_daily_spectrum_for_product(
        product=_mission_let_fluence_product(),
        duration_days=365.0,
    )

    assert daily_spectrum.y == pytest.approx((1.0, 2.0, 3.0))
    assert daily_spectrum.x_unit is Unit.LET
    assert daily_spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE_PER_DAY
    assert daily_spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE


def test_average_daily_spectrum_for_mission_years() -> None:
    daily_spectrum = average_daily_spectrum_for_mission_years(
        product=_mission_fluence_product(),
        lifetime_years=1,
    )

    assert daily_spectrum.y == pytest.approx((1.0, 2.0, 3.0))
    assert daily_spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE_PER_DAY


def test_average_daily_spectrum_rejects_flux_product() -> None:
    with pytest.raises(ValueError, match="cannot be converted"):
        average_daily_spectrum_for_product(
            product=_mean_flux_product(),
            duration_days=365.0,
        )


def test_average_daily_spectrum_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        average_daily_spectrum_for_product(
            product=_mission_fluence_product(),
            duration_days=0.0,
        )
