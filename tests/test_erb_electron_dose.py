from __future__ import annotations

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.electron_shieldose2 import load_normative_elbrbas2_database
from radar.dose.erb_electron import (
    ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL,
    ERB_ELECTRON_SHIELDOSE2_GEOMETRY,
    calculate_erb_electron_dose_with_shieldose2,
)


def _erb_electron_product(
    *,
    particle: Particle = Particle.ELECTRON,
    source: RadiationSource = RadiationSource.ERB,
    kind: RadiationProductKind = RadiationProductKind.MISSION_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    model: str = "erb_electron_on_orbit",
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(0.1, 1.0, 10.0),
            y=(1.0e6, 1.0e5, 1.0e4),
            x_unit=x_unit,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=source,
            model=model,
        ),
    )


def test_calculates_erb_electron_shieldose2_dose() -> None:
    product = _erb_electron_product()
    database = load_normative_elbrbas2_database()

    result = calculate_erb_electron_dose_with_shieldose2(
        product,
        thicknesses_g_cm2=(1.0, 0.1),
        database=database,
        n_energy_points=101,
    )

    assert result.product is product
    assert result.model == ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL
    assert result.geometry == ERB_ELECTRON_SHIELDOSE2_GEOMETRY
    assert result.thicknesses_g_cm2 == (0.1, 1.0)
    assert len(result.total_rad) == 2
    assert all(value >= 0.0 for value in result.electron_rad)
    assert all(value >= 0.0 for value in result.brems_rad)
    assert all(value >= 0.0 for value in result.total_rad)
    assert result.total_rad == pytest.approx(
        tuple(e + b for e, b in zip(result.electron_rad, result.brems_rad)),
    )
    assert any(value > 0.0 for value in result.total_slab_rad)
    assert result.included_in_total is True


def test_calculates_single_requested_thickness_with_internal_depth_grid() -> None:
    product = _erb_electron_product()
    database = load_normative_elbrbas2_database()

    result = calculate_erb_electron_dose_with_shieldose2(
        product,
        thicknesses_g_cm2=(0.5,),
        database=database,
        n_energy_points=101,
    )

    assert result.thicknesses_g_cm2 == (0.5,)
    assert len(result.total_rad) == 1
    assert result.total_rad[0] >= 0.0


def test_rejects_shielded_erb_electron_spectrum() -> None:
    product = _erb_electron_product(
        model="erb_electron_on_orbit+erb_electron_al_shielding_primary_csda",
    )

    with pytest.raises(ValueError, match="on-orbit mission fluence"):
        calculate_erb_electron_dose_with_shieldose2(
            product,
            thicknesses_g_cm2=(0.1, 1.0),
            n_energy_points=51,
        )


def test_rejects_proton_product() -> None:
    product = _erb_electron_product(particle=Particle.PROTON)

    with pytest.raises(ValueError, match="on-orbit mission fluence"):
        calculate_erb_electron_dose_with_shieldose2(
            product,
            thicknesses_g_cm2=(0.1, 1.0),
            n_energy_points=51,
        )


def test_rejects_flux_product() -> None:
    product = _erb_electron_product(
        kind=RadiationProductKind.MEAN_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="erb_electron_mean_flux",
    )

    with pytest.raises(ValueError, match="on-orbit mission fluence"):
        calculate_erb_electron_dose_with_shieldose2(
            product,
            thicknesses_g_cm2=(0.1, 1.0),
            n_energy_points=51,
        )


def test_rejects_zero_thickness() -> None:
    product = _erb_electron_product()

    with pytest.raises(ValueError, match="positive"):
        calculate_erb_electron_dose_with_shieldose2(
            product,
            thicknesses_g_cm2=(0.0, 1.0),
            n_energy_points=51,
        )


def test_rejects_duplicate_thicknesses() -> None:
    product = _erb_electron_product()

    with pytest.raises(ValueError, match="duplicates"):
        calculate_erb_electron_dose_with_shieldose2(
            product,
            thicknesses_g_cm2=(1.0, 1.0),
            n_energy_points=51,
        )
