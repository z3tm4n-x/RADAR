"""Input product selection for single event effect calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from radar.core.products import SpectrumProduct
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit

ERB_PROTON_SEE_MARKER = "erb_proton_al_shielding_primary_survival_secondary"
SEP_PROTON_SEE_MARKER = "sep_proton_al_shielding_primary_survival_secondary"
GCR_PROTON_SEE_MARKER = "gcr_proton_al_shielding_primary_survival_secondary"

_PROTON_SEE_MARKERS: dict[RadiationSource, str] = {
    RadiationSource.ERB: ERB_PROTON_SEE_MARKER,
    RadiationSource.SEP: SEP_PROTON_SEE_MARKER,
    RadiationSource.GCR: GCR_PROTON_SEE_MARKER,
}

_HEAVY_ION_LET_SEE_SOURCES = (
    RadiationSource.SEP,
    RadiationSource.GCR,
)


@dataclass(frozen=True)
class SingleEventEffectInputSelection:
    """Selected spectral products for SEE calculations."""

    proton_products: tuple[SpectrumProduct, ...]
    heavy_ion_let_products: tuple[SpectrumProduct, ...]


def _model_has_marker(product: SpectrumProduct, marker: str) -> bool:
    return marker in product.spectrum.model.split("+")


def _is_proton_see_product(product: SpectrumProduct) -> bool:
    spectrum = product.spectrum
    marker = _PROTON_SEE_MARKERS.get(spectrum.source)

    return (
        marker is not None
        and product.kind is RadiationProductKind.MISSION_FLUENCE
        and spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
        and spectrum.particle is Particle.PROTON
        and spectrum.x_unit is Unit.MEV
        and spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE
        and _model_has_marker(product, marker)
    )


def _is_heavy_ion_let_see_product(product: SpectrumProduct) -> bool:
    spectrum = product.spectrum

    return (
        spectrum.source in _HEAVY_ION_LET_SEE_SOURCES
        and product.kind is RadiationProductKind.MISSION_LET_FLUENCE
        and spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
        and spectrum.particle is Particle.HZE
        and spectrum.x_unit is Unit.LET
        and spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    )


def select_proton_see_inputs(
    products: Iterable[SpectrumProduct],
) -> tuple[SpectrumProduct, ...]:
    """Select SEP/GCR/ERB shielded proton mission fluence products for SEE."""

    return tuple(product for product in products if _is_proton_see_product(product))


def select_heavy_ion_let_see_inputs(
    products: Iterable[SpectrumProduct],
) -> tuple[SpectrumProduct, ...]:
    """Select SEP/GCR mission LET fluence products for SEE.

    ERB LET products are intentionally not selected.
    """

    return tuple(
        product
        for product in products
        if _is_heavy_ion_let_see_product(product)
    )


def select_single_event_effect_inputs(
    products: Iterable[SpectrumProduct],
) -> SingleEventEffectInputSelection:
    """Select all spectral inputs used by SEE calculations."""

    product_tuple = tuple(products)

    return SingleEventEffectInputSelection(
        proton_products=select_proton_see_inputs(product_tuple),
        heavy_ion_let_products=select_heavy_ion_let_see_inputs(product_tuple),
    )


__all__ = [
    "ERB_PROTON_SEE_MARKER",
    "GCR_PROTON_SEE_MARKER",
    "SEP_PROTON_SEE_MARKER",
    "SingleEventEffectInputSelection",
    "select_heavy_ion_let_see_inputs",
    "select_proton_see_inputs",
    "select_single_event_effect_inputs",
]
