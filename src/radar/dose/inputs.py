"""Selection of radiation products used by dose backends."""

from __future__ import annotations

from radar.core.products import SpectrumProduct
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity

_ERB_ELECTRON_AL_SHIELDING_MARKER = "erb_electron_al_shielding_primary_csda"

_PROTON_AL_SHIELDING_MARKERS = {
    RadiationSource.ERB: "erb_proton_al_shielding_primary_survival_secondary",
    RadiationSource.SEP: "sep_proton_al_shielding_primary_survival_secondary",
    RadiationSource.GCR: "gcr_proton_al_shielding_primary_survival_secondary",
}


def _is_mission_fluence(product: SpectrumProduct) -> bool:
    return (
        product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    )


def _is_mission_let_fluence(product: SpectrumProduct) -> bool:
    return (
        product.kind is RadiationProductKind.MISSION_LET_FLUENCE
        and product.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    )


def _contains_model_marker(product: SpectrumProduct, marker: str) -> bool:
    return marker in product.spectrum.model.split("+")


def _is_proton_behind_al_mission_fluence(product: SpectrumProduct) -> bool:
    if not _is_mission_fluence(product):
        return False

    if product.spectrum.particle is not Particle.PROTON:
        return False

    marker = _PROTON_AL_SHIELDING_MARKERS.get(product.spectrum.source)
    if marker is None:
        return False

    return _contains_model_marker(product, marker)


def select_erb_electron_shieldose2_inputs(
    products: tuple[SpectrumProduct, ...],
) -> tuple[SpectrumProduct, ...]:
    """Return ERB electron on-orbit mission fluence products for SHIELDOSE-2.

    Electron SHIELDOSE-2 dose uses incident on-orbit electron fluence, not the
    CSDA electron spectrum behind aluminium shielding.
    """

    return tuple(
        product
        for product in products
        if _is_mission_fluence(product)
        and product.spectrum.source is RadiationSource.ERB
        and product.spectrum.particle is Particle.ELECTRON
        and not _contains_model_marker(product, _ERB_ELECTRON_AL_SHIELDING_MARKER)
    )


def select_proton_dose_inputs(
    products: tuple[SpectrumProduct, ...],
) -> tuple[SpectrumProduct, ...]:
    """Return proton behind-aluminium mission fluence products for dose."""

    return tuple(
        product
        for product in products
        if _is_proton_behind_al_mission_fluence(product)
    )


def select_hze_let_dose_inputs(
    products: tuple[SpectrumProduct, ...],
) -> tuple[SpectrumProduct, ...]:
    """Return SEP/GCR mission LET fluence products for optional HZE dose."""

    return tuple(
        product
        for product in products
        if _is_mission_let_fluence(product)
        and product.spectrum.source in (RadiationSource.SEP, RadiationSource.GCR)
    )


__all__ = [
    "select_erb_electron_shieldose2_inputs",
    "select_hze_let_dose_inputs",
    "select_proton_dose_inputs",
]
