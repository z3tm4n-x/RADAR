"""Display/export conversions for radiation products."""

from __future__ import annotations

from radar.core.display_units import (
    accumulated_values_to_average_daily_values,
    days_for_mission_years,
)
from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import RadiationProductKind
from radar.core.units import Unit

DAILY_DISPLAY_UNITS_BY_PRODUCT_KIND: dict[RadiationProductKind, Unit] = {
    RadiationProductKind.MISSION_FLUENCE: Unit.DIFFERENTIAL_FLUENCE_PER_DAY,
    RadiationProductKind.MISSION_LET_FLUENCE: Unit.DIFFERENTIAL_LET_FLUENCE_PER_DAY,
}


def daily_display_unit_for_product_kind(product_kind: RadiationProductKind) -> Unit:
    """Return average daily display spectrum unit for an accumulated product."""

    try:
        return DAILY_DISPLAY_UNITS_BY_PRODUCT_KIND[product_kind]
    except KeyError as exc:
        msg = (
            f"Radiation product {product_kind.value} cannot be converted to "
            "average daily fluence display units."
        )
        raise ValueError(msg) from exc


def average_daily_spectrum_for_product(
    product: SpectrumProduct,
    *,
    duration_days: float,
) -> Spectrum1D:
    """Convert accumulated fluence product to average daily display spectrum.

    The returned spectrum is for display/export only. It intentionally keeps the
    source spectrum quantity while changing the y-unit to the daily display unit.
    It must not be wrapped back into SpectrumProduct.
    """

    daily_unit = daily_display_unit_for_product_kind(product.kind)

    spectrum = product.spectrum

    return Spectrum1D(
        x=spectrum.x,
        y=accumulated_values_to_average_daily_values(
            values=spectrum.y,
            duration_days=duration_days,
        ),
        x_unit=spectrum.x_unit,
        y_unit=daily_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=spectrum.model,
    )


def average_daily_spectrum_for_mission_years(
    product: SpectrumProduct,
    *,
    lifetime_years: int,
) -> Spectrum1D:
    """Convert accumulated product to average daily display spectrum for SAS years."""

    return average_daily_spectrum_for_product(
        product=product,
        duration_days=days_for_mission_years(lifetime_years),
    )
