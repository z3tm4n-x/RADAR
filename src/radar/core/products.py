"""Radiation product metadata and validation."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.dose_units import (
    default_dose_unit_for_radiation_quantity_mode,
    dose_quantity_for_radiation_quantity_mode,
)
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    DoseQuantity,
    RadiationProductKind,
    RadiationQuantityMode,
    SpectrumQuantity,
)
from radar.core.units import Unit

FLUENCE_PRODUCT_KINDS = (
    RadiationProductKind.MISSION_FLUENCE,
    RadiationProductKind.MISSION_LET_FLUENCE,
)

FLUX_PRODUCT_KINDS = (
    RadiationProductKind.ORBIT_AVERAGED_FLUX,
    RadiationProductKind.MEAN_FLUX,
    RadiationProductKind.MAXIMUM_FLUX,
    RadiationProductKind.PEAK_FLUX,
    RadiationProductKind.MEAN_LET_FLUX,
    RadiationProductKind.MAXIMUM_LET_FLUX,
    RadiationProductKind.PEAK_LET_FLUX,
)

PRODUCT_KIND_SPECTRUM_QUANTITIES: dict[
    RadiationProductKind,
    tuple[SpectrumQuantity, ...],
] = {
    RadiationProductKind.MISSION_FLUENCE: (
        SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    ),
    RadiationProductKind.ORBIT_AVERAGED_FLUX: (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.MEAN_FLUX: (
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.MAXIMUM_FLUX: (
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.PEAK_FLUX: (
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.MISSION_LET_FLUENCE: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    ),
    RadiationProductKind.MEAN_LET_FLUX: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.MAXIMUM_LET_FLUX: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.PEAK_LET_FLUX: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ),
}

PRODUCT_KIND_SPECTRUM_UNITS: dict[RadiationProductKind, Unit] = {
    RadiationProductKind.MISSION_FLUENCE: Unit.DIFFERENTIAL_FLUENCE,
    RadiationProductKind.ORBIT_AVERAGED_FLUX: Unit.DIFFERENTIAL_FLUX,
    RadiationProductKind.MEAN_FLUX: Unit.DIFFERENTIAL_FLUX,
    RadiationProductKind.MAXIMUM_FLUX: Unit.DIFFERENTIAL_FLUX,
    RadiationProductKind.PEAK_FLUX: Unit.DIFFERENTIAL_FLUX,
    RadiationProductKind.MISSION_LET_FLUENCE: Unit.DIFFERENTIAL_LET_FLUENCE,
    RadiationProductKind.MEAN_LET_FLUX: Unit.DIFFERENTIAL_LET_FLUX,
    RadiationProductKind.MAXIMUM_LET_FLUX: Unit.DIFFERENTIAL_LET_FLUX,
    RadiationProductKind.PEAK_LET_FLUX: Unit.DIFFERENTIAL_LET_FLUX,
}


def radiation_quantity_mode_for_product_kind(
    product_kind: RadiationProductKind,
) -> RadiationQuantityMode:
    """Return Flux/Fluence class for a named radiation product."""

    if product_kind in FLUENCE_PRODUCT_KINDS:
        return RadiationQuantityMode.FLUENCE

    if product_kind in FLUX_PRODUCT_KINDS:
        return RadiationQuantityMode.FLUX

    msg = f"Unsupported radiation product kind: {product_kind}"
    raise ValueError(msg)


def dose_quantity_for_product_kind(product_kind: RadiationProductKind) -> DoseQuantity:
    """Return dose quantity implied by a radiation product kind."""

    mode = radiation_quantity_mode_for_product_kind(product_kind)

    return dose_quantity_for_radiation_quantity_mode(mode)


def default_dose_unit_for_product_kind(product_kind: RadiationProductKind) -> Unit:
    """Return default dose unit implied by a radiation product kind."""

    mode = radiation_quantity_mode_for_product_kind(product_kind)

    return default_dose_unit_for_radiation_quantity_mode(mode)


def allowed_spectrum_quantities_for_product_kind(
    product_kind: RadiationProductKind,
) -> tuple[SpectrumQuantity, ...]:
    """Return allowed spectrum quantities for a radiation product kind."""

    try:
        return PRODUCT_KIND_SPECTRUM_QUANTITIES[product_kind]
    except KeyError as exc:
        msg = f"Unsupported radiation product kind: {product_kind}"
        raise ValueError(msg) from exc


def expected_spectrum_unit_for_product_kind(
    product_kind: RadiationProductKind,
) -> Unit:
    """Return required model spectrum y-unit for a radiation product kind."""

    try:
        return PRODUCT_KIND_SPECTRUM_UNITS[product_kind]
    except KeyError as exc:
        msg = f"Unsupported radiation product kind: {product_kind}"
        raise ValueError(msg) from exc


def validate_spectrum_matches_product_kind(
    spectrum: Spectrum1D,
    product_kind: RadiationProductKind,
) -> None:
    """Validate that spectrum quantity and unit match a named radiation product."""

    allowed_quantities = allowed_spectrum_quantities_for_product_kind(product_kind)

    if spectrum.quantity not in allowed_quantities:
        allowed_values = ", ".join(quantity.value for quantity in allowed_quantities)
        msg = (
            f"Spectrum quantity {spectrum.quantity.value} does not match "
            f"radiation product {product_kind.value}. "
            f"Allowed quantities: {allowed_values}."
        )
        raise ValueError(msg)

    expected_unit = expected_spectrum_unit_for_product_kind(product_kind)

    if spectrum.y_unit is not expected_unit:
        msg = (
            f"Spectrum unit {spectrum.y_unit.value} does not match "
            f"radiation product {product_kind.value}. "
            f"Expected unit: {expected_unit.value}."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class SpectrumProduct:
    """A named radiation product with explicit product semantics."""

    kind: RadiationProductKind
    spectrum: Spectrum1D
    label: str = ""

    def __post_init__(self) -> None:
        validate_spectrum_matches_product_kind(
            spectrum=self.spectrum,
            product_kind=self.kind,
        )

    @property
    def radiation_quantity_mode(self) -> RadiationQuantityMode:
        """Flux/Fluence class of this product."""

        return radiation_quantity_mode_for_product_kind(self.kind)

    @property
    def dose_quantity(self) -> DoseQuantity:
        """Dose quantity implied by this product."""

        return dose_quantity_for_product_kind(self.kind)

    @property
    def default_dose_unit(self) -> Unit:
        """Default dose unit implied by this product."""

        return default_dose_unit_for_product_kind(self.kind)
