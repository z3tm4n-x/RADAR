"""Dose unit validation and Flux/Fluence mapping."""

from __future__ import annotations

from radar.core.types import DoseQuantity, RadiationQuantityMode, SpectrumQuantity
from radar.core.units import Unit

ACCUMULATED_DOSE_UNITS = (Unit.RAD,)

DOSE_RATE_UNITS = (Unit.RAD_PER_SECOND,)

FLUENCE_SPECTRUM_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
)

FLUX_SPECTRUM_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)


def radiation_quantity_mode_for_spectrum_quantity(
    spectrum_quantity: SpectrumQuantity,
) -> RadiationQuantityMode:
    """Return Flux/Fluence mode implied by a spectrum quantity."""

    if spectrum_quantity in FLUENCE_SPECTRUM_QUANTITIES:
        return RadiationQuantityMode.FLUENCE

    if spectrum_quantity in FLUX_SPECTRUM_QUANTITIES:
        return RadiationQuantityMode.FLUX

    msg = f"Unsupported spectrum quantity: {spectrum_quantity}"
    raise ValueError(msg)


def dose_quantity_for_radiation_quantity_mode(
    radiation_quantity_mode: RadiationQuantityMode,
) -> DoseQuantity:
    """Return dose quantity required by Flux/Fluence mode."""

    if radiation_quantity_mode is RadiationQuantityMode.FLUENCE:
        return DoseQuantity.ACCUMULATED_DOSE

    if radiation_quantity_mode is RadiationQuantityMode.FLUX:
        return DoseQuantity.DOSE_RATE

    msg = f"Unsupported radiation quantity mode: {radiation_quantity_mode}"
    raise ValueError(msg)


def default_dose_unit_for_radiation_quantity_mode(
    radiation_quantity_mode: RadiationQuantityMode,
) -> Unit:
    """Return default dose unit required by Flux/Fluence mode."""

    if radiation_quantity_mode is RadiationQuantityMode.FLUENCE:
        return Unit.RAD

    if radiation_quantity_mode is RadiationQuantityMode.FLUX:
        return Unit.RAD_PER_SECOND

    msg = f"Unsupported radiation quantity mode: {radiation_quantity_mode}"
    raise ValueError(msg)


def dose_quantity_for_spectrum_quantity(
    spectrum_quantity: SpectrumQuantity,
) -> DoseQuantity:
    """Return dose quantity implied by a source spectrum quantity."""

    mode = radiation_quantity_mode_for_spectrum_quantity(spectrum_quantity)

    return dose_quantity_for_radiation_quantity_mode(mode)


def validate_dose_quantity_for_radiation_quantity_mode(
    radiation_quantity_mode: RadiationQuantityMode,
    dose_quantity: DoseQuantity,
) -> None:
    """Validate that dose quantity matches Flux/Fluence mode."""

    expected_dose_quantity = dose_quantity_for_radiation_quantity_mode(
        radiation_quantity_mode
    )

    if dose_quantity is expected_dose_quantity:
        return

    if radiation_quantity_mode is RadiationQuantityMode.FLUENCE:
        msg = "Fluence mode requires accumulated dose."
        raise ValueError(msg)

    if radiation_quantity_mode is RadiationQuantityMode.FLUX:
        msg = "Flux mode requires dose rate."
        raise ValueError(msg)

    msg = f"Unsupported radiation quantity mode: {radiation_quantity_mode}"
    raise ValueError(msg)


def validate_dose_unit_for_quantity(
    dose_quantity: DoseQuantity,
    dose_unit: Unit,
) -> None:
    """Validate that dose unit matches accumulated-dose or dose-rate quantity."""

    if dose_quantity is DoseQuantity.ACCUMULATED_DOSE:
        if dose_unit not in ACCUMULATED_DOSE_UNITS:
            msg = "Accumulated dose unit must be rad."
            raise ValueError(msg)

        return

    if dose_quantity is DoseQuantity.DOSE_RATE:
        if dose_unit not in DOSE_RATE_UNITS:
            msg = "Dose rate unit must be rad/s."
            raise ValueError(msg)

        return

    msg = f"Unsupported dose quantity: {dose_quantity}"
    raise ValueError(msg)


def validate_dose_settings_for_radiation_quantity_mode(
    radiation_quantity_mode: RadiationQuantityMode,
    dose_quantity: DoseQuantity,
    dose_unit: Unit,
) -> None:
    """Validate Flux/Fluence mode, dose quantity, and dose unit together."""

    validate_dose_quantity_for_radiation_quantity_mode(
        radiation_quantity_mode=radiation_quantity_mode,
        dose_quantity=dose_quantity,
    )
    validate_dose_unit_for_quantity(
        dose_quantity=dose_quantity,
        dose_unit=dose_unit,
    )


def is_accumulated_dose_unit(unit: Unit) -> bool:
    """Return True if unit is an accumulated dose unit."""

    return unit in ACCUMULATED_DOSE_UNITS


def is_dose_rate_unit(unit: Unit) -> bool:
    """Return True if unit is a dose rate unit."""

    return unit in DOSE_RATE_UNITS
