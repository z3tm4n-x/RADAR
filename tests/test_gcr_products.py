from math import inf, nan

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.products import (
    GCR_MISSION_PRODUCTS_MODEL_SUFFIX,
    GcrFluxTimeBin,
    GcrMissionProducts,
    calculate_gcr_mission_products,
)


def _gcr_flux_spectrum(
    *,
    x: tuple[float, ...] = (10.0, 20.0),
    y: tuple[float, ...] = (1.0, 2.0),
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.GCR,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUX,
    model: str = "gcr_test",
) -> Spectrum1D:
    return Spectrum1D(
        x=x,
        y=y,
        x_unit=x_unit,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model=model,
    )


def test_gcr_mission_products_calculate_mean_maximum_and_fluence() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(y=(1.0, 2.0)),
                duration_seconds=10.0,
            ),
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(y=(3.0, 6.0)),
                duration_seconds=20.0,
            ),
        )
    )

    assert result.mean_flux.spectrum.y == pytest.approx((70.0 / 30.0, 140.0 / 30.0))
    assert result.maximum_flux.spectrum.y == pytest.approx((3.0, 6.0))
    assert result.mission_fluence.spectrum.y == pytest.approx((70.0, 140.0))


def test_gcr_mission_products_use_product_contract_quantities_and_units() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(),
                duration_seconds=10.0,
            ),
        )
    )

    assert result.mean_flux.kind is RadiationProductKind.MEAN_FLUX
    assert result.mean_flux.spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert result.mean_flux.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX

    assert result.maximum_flux.kind is RadiationProductKind.MAXIMUM_FLUX
    assert (
        result.maximum_flux.spectrum.quantity
        is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    )
    assert result.maximum_flux.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX

    assert result.mission_fluence.kind is RadiationProductKind.MISSION_FLUENCE
    assert result.mission_fluence.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert result.mission_fluence.spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE


def test_gcr_mission_products_preserve_grid_particle_source_and_order() -> None:
    source_spectrum = _gcr_flux_spectrum(
        x=(1.0, 10.0, 100.0),
        y=(4.0, 5.0, 6.0),
        particle=Particle.HZE,
        model="gcr_o_total",
    )

    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=source_spectrum,
                duration_seconds=2.0,
            ),
        )
    )

    assert result.products == (
        result.mean_flux,
        result.maximum_flux,
        result.mission_fluence,
    )

    for product in result.products:
        assert product.spectrum.x == source_spectrum.x
        assert product.spectrum.x_unit is source_spectrum.x_unit
        assert product.spectrum.particle is Particle.HZE
        assert product.spectrum.source is RadiationSource.GCR
        assert product.spectrum.model.startswith(
            f"gcr_o_total:{GCR_MISSION_PRODUCTS_MODEL_SUFFIX}:"
        )


def test_gcr_mission_products_accept_gev_per_nucleon_input_grid() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(
                    x=(0.1, 1.0),
                    x_unit=Unit.GEV_PER_NUCLEON,
                ),
                duration_seconds=10.0,
            ),
        )
    )

    assert result.mean_flux.spectrum.x_unit is Unit.GEV_PER_NUCLEON
    assert result.mission_fluence.spectrum.x_unit is Unit.GEV_PER_NUCLEON


@pytest.mark.parametrize("bad_duration", [0.0, -1.0, inf, nan])
def test_gcr_flux_time_bin_rejects_bad_duration(bad_duration: float) -> None:
    with pytest.raises(ValueError, match="finite|positive"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(),
            duration_seconds=bad_duration,
        )


def test_gcr_mission_products_reject_empty_time_bins() -> None:
    with pytest.raises(ValueError, match="at least one"):
        calculate_gcr_mission_products(())


def test_gcr_mission_products_reject_non_gcr_input() -> None:
    with pytest.raises(ValueError, match="source must be GCR"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(source=RadiationSource.SEP),
            duration_seconds=1.0,
        )


def test_gcr_mission_products_reject_non_flux_input_quantity() -> None:
    with pytest.raises(ValueError, match="quantity"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(
                y_unit=Unit.DIFFERENTIAL_FLUENCE,
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            ),
            duration_seconds=1.0,
        )


def test_gcr_mission_products_reject_wrong_flux_unit() -> None:
    with pytest.raises(ValueError, match="unit"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(y_unit=Unit.DIFFERENTIAL_FLUENCE),
            duration_seconds=1.0,
        )


def test_gcr_mission_products_reject_mismatched_grids() -> None:
    with pytest.raises(ValueError, match="same energy grid"):
        calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x=(10.0, 20.0)),
                    duration_seconds=1.0,
                ),
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x=(10.0, 30.0)),
                    duration_seconds=1.0,
                ),
            )
        )


def test_gcr_mission_products_reject_mismatched_particles() -> None:
    with pytest.raises(ValueError, match="same particle"):
        calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(particle=Particle.PROTON),
                    duration_seconds=1.0,
                ),
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(particle=Particle.HZE),
                    duration_seconds=1.0,
                ),
            )
        )


def test_gcr_mission_products_reject_mismatched_energy_units() -> None:
    with pytest.raises(ValueError, match="same energy unit"):
        calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x_unit=Unit.MEV),
                    duration_seconds=1.0,
                ),
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x_unit=Unit.GEV_PER_NUCLEON),
                    duration_seconds=1.0,
                ),
            )
        )


def test_gcr_mission_products_reject_invalid_result_product_kinds() -> None:
    valid = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(),
                duration_seconds=1.0,
            ),
        )
    )

    with pytest.raises(ValueError, match="MEAN_FLUX"):
        GcrMissionProducts(
            mean_flux=SpectrumProduct(
                kind=RadiationProductKind.MAXIMUM_FLUX,
                spectrum=valid.maximum_flux.spectrum,
            ),
            maximum_flux=valid.maximum_flux,
            mission_fluence=valid.mission_fluence,
        )
