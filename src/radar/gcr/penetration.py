"""Apply geomagnetic penetration to GCR products."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.products import SpectrumProduct
from radar.core.project import CalculationConfig
from radar.core.types import Particle
from radar.geomagnetic.ost_penetration import build_ost_penetration_function_for_config
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid
from radar.geomagnetic.spectrum import apply_hze_penetration, apply_proton_penetration
from radar.gcr.parameters import get_gost_gcr_ion_parameters
from radar.physics.rigidity import (
    ion_kinetic_energy_per_nucleon_to_rigidity_gv,
    proton_kinetic_energy_to_rigidity_gv,
)


@dataclass(frozen=True)
class GcrGeomagneticPenetrationOutput:
    """Output of applying geomagnetic penetration to GCR products."""

    products: tuple[SpectrumProduct, ...]
    penetration: PenetrationFunction
    rigidity_grid: RigidityGrid


def _gcr_ion_symbol_from_product(product: SpectrumProduct) -> str:
    parts = product.spectrum.model.split(":")

    if len(parts) < 2:
        msg = "GCR HZE product model must contain an ion symbol tag."
        raise ValueError(msg)

    symbol = parts[1]

    if symbol not in get_gost_gcr_ion_parameters():
        msg = f"GCR HZE product model contains unknown ion symbol {symbol!r}."
        raise ValueError(msg)

    return symbol


def gcr_product_mass_to_charge(product: SpectrumProduct) -> float:
    """Return A/Z for one GCR product."""

    if product.spectrum.particle is Particle.PROTON:
        return 1.0

    if product.spectrum.particle is Particle.HZE:
        symbol = _gcr_ion_symbol_from_product(product)
        return get_gost_gcr_ion_parameters()[symbol].mass_to_charge

    msg = "GCR geomagnetic penetration supports only proton and HZE products."
    raise ValueError(msg)


def gcr_rigidity_grid_for_products(
    products: tuple[SpectrumProduct, ...],
) -> RigidityGrid:
    """Return a sorted rigidity grid covering GCR product energy grids."""

    values_gv: set[float] = set()

    for product in products:
        if product.spectrum.particle is Particle.PROTON:
            values_gv.update(
                proton_kinetic_energy_to_rigidity_gv(energy_mev)
                for energy_mev in product.spectrum.x
            )
            continue

        if product.spectrum.particle is Particle.HZE:
            mass_to_charge = gcr_product_mass_to_charge(product)
            values_gv.update(
                ion_kinetic_energy_per_nucleon_to_rigidity_gv(
                    kinetic_energy_mev_per_nucleon=energy_mev_per_nucleon,
                    mass_to_charge=mass_to_charge,
                )
                for energy_mev_per_nucleon in product.spectrum.x
            )
            continue

        msg = "GCR geomagnetic penetration requires proton or HZE products."
        raise ValueError(msg)

    if not values_gv:
        msg = "GCR geomagnetic penetration requires at least one product."
        raise ValueError(msg)

    return RigidityGrid(values_gv=tuple(sorted(values_gv)))


def apply_gcr_penetration_to_products(
    *,
    products: tuple[SpectrumProduct, ...],
    penetration: PenetrationFunction,
) -> tuple[SpectrumProduct, ...]:
    """Apply an existing penetration function to GCR products."""

    transformed_products: list[SpectrumProduct] = []

    for product in products:
        if product.spectrum.particle is Particle.PROTON:
            transformed_products.append(
                SpectrumProduct(
                    kind=product.kind,
                    spectrum=apply_proton_penetration(
                        spectrum=product.spectrum,
                        penetration=penetration,
                    ),
                    label=product.label,
                )
            )
            continue

        if product.spectrum.particle is Particle.HZE:
            transformed_products.append(
                SpectrumProduct(
                    kind=product.kind,
                    spectrum=apply_hze_penetration(
                        spectrum=product.spectrum,
                        penetration=penetration,
                        mass_to_charge=gcr_product_mass_to_charge(product),
                    ),
                    label=product.label,
                )
            )
            continue

        msg = "GCR geomagnetic penetration supports only proton and HZE products."
        raise ValueError(msg)

    return tuple(transformed_products)


def apply_ost_geomagnetic_penetration_to_gcr_products(
    *,
    config: CalculationConfig,
    products: tuple[SpectrumProduct, ...],
) -> GcrGeomagneticPenetrationOutput:
    """Build OST penetration from config and apply it to GCR products."""

    rigidity_grid = gcr_rigidity_grid_for_products(products)
    penetration = build_ost_penetration_function_for_config(
        config,
        rigidity_grid=rigidity_grid,
    )

    return GcrGeomagneticPenetrationOutput(
        products=apply_gcr_penetration_to_products(
            products=products,
            penetration=penetration,
        ),
        penetration=penetration,
        rigidity_grid=rigidity_grid,
    )


__all__ = [
    "GcrGeomagneticPenetrationOutput",
    "apply_gcr_penetration_to_products",
    "apply_ost_geomagnetic_penetration_to_gcr_products",
    "gcr_product_mass_to_charge",
    "gcr_rigidity_grid_for_products",
]
