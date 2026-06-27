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
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid
from radar.gcr.penetration import (
    apply_gcr_penetration_to_products,
    gcr_product_mass_to_charge,
    gcr_rigidity_grid_for_products,
)
from radar.physics.rigidity import (
    ion_kinetic_energy_per_nucleon_to_rigidity_gv,
    proton_kinetic_energy_to_rigidity_gv,
)


def _product(
    *,
    particle: Particle,
    model: str,
    kind: RadiationProductKind = RadiationProductKind.MEAN_FLUX,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    x: tuple[float, ...] = (10.0, 100.0),
    y: tuple[float, ...] = (2.0, 4.0),
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=x,
            y=y,
            x_unit=Unit.MEV,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=RadiationSource.GCR,
            model=model,
        ),
        label="test product",
    )


def test_gcr_product_mass_to_charge_uses_proton_and_hze_symbol_tags() -> None:
    proton = _product(
        particle=Particle.PROTON,
        model="gost_gcr_source_outside_magnetosphere:H:total:mission_products:mean_flux",
    )
    helium = _product(
        particle=Particle.HZE,
        model="gost_gcr_source_outside_magnetosphere:He:total:mission_products:mean_flux",
    )

    assert gcr_product_mass_to_charge(proton) == pytest.approx(1.0)
    assert gcr_product_mass_to_charge(helium) == pytest.approx(2.0)


def test_gcr_rigidity_grid_for_products_includes_proton_and_hze_points() -> None:
    proton = _product(
        particle=Particle.PROTON,
        model="gost_gcr_source_outside_magnetosphere:H:total:mission_products:mean_flux",
    )
    helium = _product(
        particle=Particle.HZE,
        model="gost_gcr_source_outside_magnetosphere:He:total:mission_products:mean_flux",
    )

    grid = gcr_rigidity_grid_for_products((proton, helium))

    expected_values = tuple(
        sorted(
            {
                proton_kinetic_energy_to_rigidity_gv(10.0),
                proton_kinetic_energy_to_rigidity_gv(100.0),
                ion_kinetic_energy_per_nucleon_to_rigidity_gv(
                    kinetic_energy_mev_per_nucleon=10.0,
                    mass_to_charge=2.0,
                ),
                ion_kinetic_energy_per_nucleon_to_rigidity_gv(
                    kinetic_energy_mev_per_nucleon=100.0,
                    mass_to_charge=2.0,
                ),
            }
        )
    )

    assert grid.values_gv == pytest.approx(expected_values)


def test_apply_gcr_penetration_to_products_preserves_product_semantics() -> None:
    proton = _product(
        particle=Particle.PROTON,
        model="gost_gcr_source_outside_magnetosphere:H:total:mission_products:mean_flux",
        y=(2.0, 4.0),
    )
    helium = _product(
        particle=Particle.HZE,
        model="gost_gcr_source_outside_magnetosphere:He:total:mission_products:maximum_flux",
        kind=RadiationProductKind.MAXIMUM_FLUX,
        quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        y=(6.0, 8.0),
    )
    penetration = PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(0.001, 1000.0)),
        values=(0.5, 0.5),
        model="test_penetration",
    )

    transformed = apply_gcr_penetration_to_products(
        products=(proton, helium),
        penetration=penetration,
    )

    assert len(transformed) == 4

    assert transformed[0].kind is RadiationProductKind.MEAN_FLUX
    assert transformed[0].label == proton.label
    assert transformed[0].spectrum.y == pytest.approx((1.0, 2.0))
    assert transformed[0].spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert transformed[0].spectrum.model == (
        "gost_gcr_source_outside_magnetosphere:H:total:mission_products:mean_flux"
        "+test_penetration"
    )

    assert transformed[1].kind is RadiationProductKind.MEAN_FLUX
    assert transformed[1].spectrum.quantity is SpectrumQuantity.MEAN_INTEGRAL_FLUX
    assert transformed[1].spectrum.y_unit is Unit.INTEGRAL_FLUX
    assert transformed[1].spectrum.x == transformed[0].spectrum.x
    assert transformed[1].spectrum.model.startswith(
        "gost_gcr_source_outside_magnetosphere:H:total:mission_products:mean_flux"
        "+test_penetration+derived_from_"
    )

    assert transformed[2].kind is RadiationProductKind.MAXIMUM_FLUX
    assert transformed[2].label == helium.label
    assert transformed[2].spectrum.y == pytest.approx((3.0, 4.0))
    assert transformed[2].spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    assert transformed[2].spectrum.model == (
        "gost_gcr_source_outside_magnetosphere:He:total:mission_products:maximum_flux"
        "+test_penetration"
    )

    assert transformed[3].kind is RadiationProductKind.MAXIMUM_FLUX
    assert transformed[3].spectrum.quantity is SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX
    assert transformed[3].spectrum.y_unit is Unit.INTEGRAL_FLUX
    assert transformed[3].spectrum.x == transformed[2].spectrum.x
    assert transformed[3].spectrum.model.startswith(
        "gost_gcr_source_outside_magnetosphere:He:total:mission_products:maximum_flux"
        "+test_penetration+derived_from_"
    )


def test_apply_gcr_penetration_to_products_supports_mission_fluence() -> None:
    product = _product(
        particle=Particle.PROTON,
        model="gost_gcr_source_outside_magnetosphere:H:total:mission_products:mission_fluence",
        kind=RadiationProductKind.MISSION_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        y=(10.0, 20.0),
    )
    penetration = PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(0.001, 1000.0)),
        values=(0.25, 0.25),
        model="test_penetration",
    )

    transformed = apply_gcr_penetration_to_products(
        products=(product,),
        penetration=penetration,
    )

    assert transformed[0].kind is RadiationProductKind.MISSION_FLUENCE
    assert transformed[0].spectrum.y == pytest.approx((2.5, 5.0))
    assert transformed[0].spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE


def test_gcr_penetration_rejects_hze_product_without_parseable_ion_symbol() -> None:
    product = _product(
        particle=Particle.HZE,
        model="static_hze_product",
    )

    with pytest.raises(ValueError, match="ion symbol"):
        gcr_product_mass_to_charge(product)
