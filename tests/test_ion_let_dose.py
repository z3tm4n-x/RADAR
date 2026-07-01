from __future__ import annotations

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.ion_let import (
    MEV_PER_MG_TO_MEV_PER_G,
    MEV_TO_RAD,
    calculate_ion_let_dose_from_fluence,
    integrate_let_energy_deposition_MeV_per_mg,
)


def _let_product(
    *,
    source: RadiationSource = RadiationSource.SEP,
    kind: RadiationProductKind = RadiationProductKind.MISSION_LET_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    x: tuple[float, ...] = (1.0, 4.0),
    y: tuple[float, ...] = (10.0, 0.0),
    x_unit: Unit = Unit.LET,
    y_unit: Unit = Unit.DIFFERENTIAL_LET_FLUENCE,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=x,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
            quantity=quantity,
            particle=Particle.HZE,
            source=source,
            model="combined_proton_hze_let",
        ),
    )


def test_integrates_let_fluence_with_mg_to_g_conversion() -> None:
    product = _let_product()

    # LET centers 1 and 4 have logarithmic bin edges 0.5, 2, 8.
    # Only the first bin is nonzero: 1 * 10 * (2 - 0.5) = 15 MeV/mg.
    expected_MeV_per_mg = 15.0
    expected_MeV_per_g = expected_MeV_per_mg * MEV_PER_MG_TO_MEV_PER_G

    result = calculate_ion_let_dose_from_fluence(product)

    assert integrate_let_energy_deposition_MeV_per_mg(product) == pytest.approx(
        expected_MeV_per_mg,
    )
    assert result.energy_deposition_MeV_per_mg == pytest.approx(expected_MeV_per_mg)
    assert result.energy_deposition_MeV_per_g == pytest.approx(expected_MeV_per_g)
    assert result.dose_rad == pytest.approx(expected_MeV_per_g * MEV_TO_RAD)
    assert result.included_in_total is False


def test_zero_let_fluence_gives_zero_dose() -> None:
    product = _let_product(y=(0.0, 0.0))

    result = calculate_ion_let_dose_from_fluence(product)

    assert result.energy_deposition_MeV_per_mg == 0.0
    assert result.energy_deposition_MeV_per_g == 0.0
    assert result.dose_rad == 0.0


def test_let_dose_scales_linearly_with_fluence() -> None:
    product = _let_product(y=(10.0, 2.0))
    doubled = _let_product(y=(20.0, 4.0))

    dose = calculate_ion_let_dose_from_fluence(product)
    doubled_dose = calculate_ion_let_dose_from_fluence(doubled)

    assert doubled_dose.dose_rad == pytest.approx(2.0 * dose.dose_rad)


def test_rejects_erb_let_product() -> None:
    product = _let_product(source=RadiationSource.ERB)

    with pytest.raises(ValueError, match="SEP and GCR"):
        calculate_ion_let_dose_from_fluence(product)


def test_rejects_let_flux_product() -> None:
    product = _let_product(
        kind=RadiationProductKind.MEAN_LET_FLUX,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_LET_FLUX,
    )

    with pytest.raises(ValueError, match="mission LET fluence"):
        calculate_ion_let_dose_from_fluence(product)


def test_rejects_non_let_x_unit() -> None:
    product = _let_product(x_unit=Unit.MEV)

    with pytest.raises(ValueError, match="LET x unit"):
        calculate_ion_let_dose_from_fluence(product)


def test_rejects_single_point_let_spectrum() -> None:
    product = _let_product(x=(1.0,), y=(10.0,))

    with pytest.raises(ValueError, match="at least two LET points"):
        calculate_ion_let_dose_from_fluence(product)
