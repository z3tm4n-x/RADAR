"""Ionizing dose from LET fluence spectra."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

from radar.core.products import SpectrumProduct
from radar.core.types import RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit

MEV_TO_RAD = 1.6021892e-8
MEV_PER_MG_TO_MEV_PER_G = 1000.0
ION_LET_DOSE_MODEL = "ion_let_fluence_dose"


@dataclass(frozen=True)
class IonLetDoseResult:
    """Dose contribution calculated from a mission LET fluence spectrum."""

    product: SpectrumProduct
    dose_rad: float
    energy_deposition_MeV_per_mg: float
    energy_deposition_MeV_per_g: float
    included_in_total: bool = False
    model: str = ION_LET_DOSE_MODEL


def _log_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    if len(centers) < 2:
        msg = "LET dose calculation requires at least two LET centers."
        raise ValueError(msg)

    if any(center <= 0.0 for center in centers):
        msg = "LET centers must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(centers, centers[1:])):
        msg = "LET centers must be strictly increasing."
        raise ValueError(msg)

    log_centers = tuple(log(center) for center in centers)
    edges: list[float] = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (log_centers[index - 1] + log_centers[index]))

    edges[0] = exp(log_centers[0] - 0.5 * (log_centers[1] - log_centers[0]))
    edges[-1] = exp(log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2]))

    return tuple(edges)


def _validate_ion_let_dose_product(product: SpectrumProduct) -> None:
    spectrum = product.spectrum

    if product.kind is not RadiationProductKind.MISSION_LET_FLUENCE:
        msg = "Ion LET dose requires a mission LET fluence product."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE:
        msg = "Ion LET dose requires a differential LET fluence spectrum."
        raise ValueError(msg)

    if spectrum.source not in (RadiationSource.SEP, RadiationSource.GCR):
        msg = "Ion LET dose is only defined for SEP and GCR LET spectra."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.LET:
        msg = "Ion LET dose requires LET x unit MeV*cm^2/mg."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_LET_FLUENCE:
        msg = "Ion LET dose requires differential LET fluence y unit."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "Ion LET dose requires at least two LET points."
        raise ValueError(msg)

    if any(not isfinite(value) for value in spectrum.x):
        msg = "LET centers must be finite."
        raise ValueError(msg)

    if any(value <= 0.0 for value in spectrum.x):
        msg = "LET centers must be positive."
        raise ValueError(msg)


def integrate_let_energy_deposition_MeV_per_mg(product: SpectrumProduct) -> float:
    """Return ? L * Phi_L(L) dL in MeV/mg."""

    _validate_ion_let_dose_product(product)

    spectrum = product.spectrum
    edges = _log_edges_from_centers(spectrum.x)

    return sum(
        let_center * fluence_density * (edges[index + 1] - edges[index])
        for index, (let_center, fluence_density) in enumerate(
            zip(spectrum.x, spectrum.y),
        )
    )


def calculate_ion_let_dose_from_fluence(product: SpectrumProduct) -> IonLetDoseResult:
    """Calculate accumulated ionizing dose from mission LET fluence."""

    energy_deposition_MeV_per_mg = integrate_let_energy_deposition_MeV_per_mg(
        product,
    )
    energy_deposition_MeV_per_g = (
        energy_deposition_MeV_per_mg * MEV_PER_MG_TO_MEV_PER_G
    )
    dose_rad = energy_deposition_MeV_per_g * MEV_TO_RAD

    return IonLetDoseResult(
        product=product,
        dose_rad=dose_rad,
        energy_deposition_MeV_per_mg=energy_deposition_MeV_per_mg,
        energy_deposition_MeV_per_g=energy_deposition_MeV_per_g,
        included_in_total=False,
    )


__all__ = [
    "ION_LET_DOSE_MODEL",
    "IonLetDoseResult",
    "calculate_ion_let_dose_from_fluence",
    "integrate_let_energy_deposition_MeV_per_mg",
]
