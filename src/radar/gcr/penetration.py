"""Apply geomagnetic penetration to GCR products."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.products import SpectrumProduct
from radar.core.project import CalculationConfig
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import integrate_differential_spectrum_tail_power_law
from radar.core.types import Particle, SpectrumQuantity
from radar.core.units import Unit
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


_GCR_DIFFERENTIAL_TO_INTEGRAL_QUANTITY: dict[SpectrumQuantity, SpectrumQuantity] = {
    SpectrumQuantity.DIFFERENTIAL_FLUX: SpectrumQuantity.INTEGRAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX: SpectrumQuantity.MEAN_INTEGRAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX: SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX: SpectrumQuantity.PEAK_INTEGRAL_FLUX,
    SpectrumQuantity.DIFFERENTIAL_FLUENCE: SpectrumQuantity.INTEGRAL_FLUENCE,
}


def _is_gcr_differential_energy_product(product: SpectrumProduct) -> bool:
    return product.spectrum.quantity in _GCR_DIFFERENTIAL_TO_INTEGRAL_QUANTITY


def _integral_unit_for_quantity(quantity: SpectrumQuantity) -> Unit:
    if quantity is SpectrumQuantity.INTEGRAL_FLUENCE:
        return Unit.INTEGRAL_FLUENCE

    return Unit.INTEGRAL_FLUX


def _integral_label_for_product(product: SpectrumProduct) -> str:
    label = product.label or product.kind.value

    if "fluence" in label:
        return label.replace("fluence", "integral fluence", 1)

    if "flux" in label:
        return label.replace("flux", "integral flux", 1)

    return f"{label} integral"


def _matching_integral_source_product(
    *,
    products: tuple[SpectrumProduct, ...],
    differential_index: int,
    differential_product: SpectrumProduct,
) -> SpectrumProduct | None:
    integral_quantity = _GCR_DIFFERENTIAL_TO_INTEGRAL_QUANTITY[
        differential_product.spectrum.quantity
    ]

    candidate_index = differential_index + 1
    if candidate_index >= len(products):
        return None

    candidate = products[candidate_index]

    if candidate.kind is not differential_product.kind:
        return None

    if candidate.spectrum.quantity is not integral_quantity:
        return None

    if candidate.spectrum.particle is not differential_product.spectrum.particle:
        return None

    if candidate.spectrum.source is not differential_product.spectrum.source:
        return None

    if candidate.spectrum.x != differential_product.spectrum.x:
        return None

    if candidate.spectrum.x_unit is not differential_product.spectrum.x_unit:
        return None

    return candidate


def _integral_product_from_differential_product(
    product: SpectrumProduct,
    *,
    source_integral_product: SpectrumProduct | None = None,
) -> SpectrumProduct:
    integral_quantity = _GCR_DIFFERENTIAL_TO_INTEGRAL_QUANTITY[product.spectrum.quantity]
    integral_values = integrate_differential_spectrum_tail_power_law(
        energies=product.spectrum.x,
        differential_values=product.spectrum.y,
        context="GCR geomagnetic penetration",
    )
    source_model = (
        source_integral_product.spectrum.model
        if source_integral_product is not None
        else product.spectrum.model
    )
    label = (
        source_integral_product.label
        if source_integral_product is not None
        else _integral_label_for_product(product)
    )
    integral_spectrum = Spectrum1D(
        x=product.spectrum.x,
        y=integral_values,
        x_unit=product.spectrum.x_unit,
        y_unit=_integral_unit_for_quantity(integral_quantity),
        quantity=integral_quantity,
        particle=product.spectrum.particle,
        source=product.spectrum.source,
        model=f"{source_model}+derived_from_{product.spectrum.model}",
    )

    return SpectrumProduct(
        kind=product.kind,
        spectrum=integral_spectrum,
        label=label,
    )


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
        if not _is_gcr_differential_energy_product(product):
            continue

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

    for index, product in enumerate(products):
        if not _is_gcr_differential_energy_product(product):
            continue

        source_integral_product = _matching_integral_source_product(
            products=products,
            differential_index=index,
            differential_product=product,
        )

        if product.spectrum.particle is Particle.PROTON:
            penetrated_product = SpectrumProduct(
                kind=product.kind,
                spectrum=apply_proton_penetration(
                    spectrum=product.spectrum,
                    penetration=penetration,
                ),
                label=product.label,
            )
            transformed_products.extend(
                (
                    penetrated_product,
                    _integral_product_from_differential_product(
                        penetrated_product,
                        source_integral_product=source_integral_product,
                    ),
                )
            )
            continue

        if product.spectrum.particle is Particle.HZE:
            penetrated_product = SpectrumProduct(
                kind=product.kind,
                spectrum=apply_hze_penetration(
                    spectrum=product.spectrum,
                    penetration=penetration,
                    mass_to_charge=gcr_product_mass_to_charge(product),
                ),
                label=product.label,
            )
            transformed_products.extend(
                (
                    penetrated_product,
                    _integral_product_from_differential_product(
                        penetrated_product,
                        source_integral_product=source_integral_product,
                    ),
                )
            )
            continue

        msg = "GCR geomagnetic penetration supports only proton and HZE products."
        raise ValueError(msg)

    if not transformed_products:
        msg = "GCR geomagnetic penetration requires at least one differential energy product."
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
