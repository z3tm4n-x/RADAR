import pytest

from radar.core.dose_units import (
    dose_quantity_for_spectrum_quantity,
    is_accumulated_dose_unit,
    is_dose_rate_unit,
    radiation_quantity_mode_for_spectrum_quantity,
    validate_dose_unit_for_quantity,
)
from radar.core.types import DoseQuantity, RadiationQuantityMode, SpectrumQuantity
from radar.core.units import Unit


@pytest.mark.parametrize(
    "spectrum_quantity",
    [
        SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    ],
)
def test_fluence_spectrum_quantity_maps_to_accumulated_dose(
    spectrum_quantity: SpectrumQuantity,
) -> None:
    assert (
        radiation_quantity_mode_for_spectrum_quantity(spectrum_quantity)
        is RadiationQuantityMode.FLUENCE
    )
    assert (
        dose_quantity_for_spectrum_quantity(spectrum_quantity)
        is DoseQuantity.ACCUMULATED_DOSE
    )


@pytest.mark.parametrize(
    "spectrum_quantity",
    [
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ],
)
def test_flux_spectrum_quantity_maps_to_dose_rate(
    spectrum_quantity: SpectrumQuantity,
) -> None:
    assert (
        radiation_quantity_mode_for_spectrum_quantity(spectrum_quantity)
        is RadiationQuantityMode.FLUX
    )
    assert dose_quantity_for_spectrum_quantity(spectrum_quantity) is DoseQuantity.DOSE_RATE


def test_validate_accumulated_dose_unit() -> None:
    validate_dose_unit_for_quantity(
        dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
        dose_unit=Unit.RAD,
    )

    with pytest.raises(ValueError, match="Accumulated"):
        validate_dose_unit_for_quantity(
            dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
            dose_unit=Unit.RAD_PER_SECOND,
        )


@pytest.mark.parametrize(
    "unit",
    [
        Unit.RAD_PER_SECOND,
        Unit.RAD_PER_DAY,
        Unit.RAD_PER_YEAR,
    ],
)
def test_validate_dose_rate_units(unit: Unit) -> None:
    validate_dose_unit_for_quantity(
        dose_quantity=DoseQuantity.DOSE_RATE,
        dose_unit=unit,
    )

    with pytest.raises(ValueError, match="Dose rate"):
        validate_dose_unit_for_quantity(
            dose_quantity=DoseQuantity.DOSE_RATE,
            dose_unit=Unit.RAD,
        )


def test_dose_unit_predicates() -> None:
    assert is_accumulated_dose_unit(Unit.RAD) is True
    assert is_accumulated_dose_unit(Unit.RAD_PER_SECOND) is False

    assert is_dose_rate_unit(Unit.RAD) is False
    assert is_dose_rate_unit(Unit.RAD_PER_SECOND) is True
    assert is_dose_rate_unit(Unit.RAD_PER_DAY) is True
    assert is_dose_rate_unit(Unit.RAD_PER_YEAR) is True
