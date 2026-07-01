"""Proton ionizing dose from shielded mission fluence spectra."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

from radar.core.products import SpectrumProduct
from radar.core.types import Particle, RadiationProductKind, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.constants import MEV_PER_MG_TO_MEV_PER_G, MEV_TO_RAD
from radar.dose.inputs import select_proton_dose_inputs
from radar.shielding.proton_si_let import ProtonSiLetTable
from radar.shielding.resources import (
    load_normative_proton_si_let_table,
)

PROTON_SHIELDED_DOSE_MODEL = "proton_shielded_fluence_si_electronic_stopping_dose"


@dataclass(frozen=True)
class ProtonDoseResult:
    """Dose contribution calculated from a shielded proton mission fluence."""

    product: SpectrumProduct
    dose_rad: float
    energy_deposition_MeV_per_mg: float
    energy_deposition_MeV_per_g: float
    included_in_total: bool = True
    model: str = PROTON_SHIELDED_DOSE_MODEL


def _log_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    if len(centers) < 2:
        msg = "Proton dose calculation requires at least two energy centers."
        raise ValueError(msg)

    if any(center <= 0.0 for center in centers):
        msg = "Proton energy centers must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(centers, centers[1:])):
        msg = "Proton energy centers must be strictly increasing."
        raise ValueError(msg)

    log_centers = tuple(log(center) for center in centers)
    edges: list[float] = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (log_centers[index - 1] + log_centers[index]))

    edges[0] = exp(log_centers[0] - 0.5 * (log_centers[1] - log_centers[0]))
    edges[-1] = exp(log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2]))

    return tuple(edges)


def _interpolate_linear_strict(
    *,
    x: float,
    xp: tuple[float, ...],
    fp: tuple[float, ...],
    quantity: str,
) -> float:
    if not isfinite(x):
        msg = f"{quantity} interpolation coordinate must be finite."
        raise ValueError(msg)

    if x < xp[0] or x > xp[-1]:
        msg = f"{quantity} interpolation coordinate is outside the tabulated range."
        raise ValueError(msg)

    if x == xp[0]:
        return fp[0]

    if x == xp[-1]:
        return fp[-1]

    for index in range(len(xp) - 1):
        left_x = xp[index]
        right_x = xp[index + 1]

        if left_x <= x <= right_x:
            fraction = (x - left_x) / (right_x - left_x)
            return fp[index] + fraction * (fp[index + 1] - fp[index])

    msg = f"Could not interpolate {quantity}."
    raise ValueError(msg)


def _electronic_let_at_energy(
    *,
    table: ProtonSiLetTable,
    energy_mev: float,
) -> float:
    return _interpolate_linear_strict(
        x=energy_mev,
        xp=table.energy_mev,
        fp=table.let_electronic_mev_cm2_mg,
        quantity="proton Si electronic LET",
    )


def _validate_proton_dose_product(product: SpectrumProduct) -> None:
    spectrum = product.spectrum

    if select_proton_dose_inputs((product,)) != (product,):
        msg = (
            "Proton dose requires an ERB/SEP/GCR proton mission fluence "
            "spectrum behind aluminium shielding."
        )
        raise ValueError(msg)

    if product.kind is not RadiationProductKind.MISSION_FLUENCE:
        msg = "Proton dose requires a mission fluence product."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        msg = "Proton dose requires a differential fluence spectrum."
        raise ValueError(msg)

    if spectrum.particle is not Particle.PROTON:
        msg = "Proton dose requires a proton spectrum."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "Proton dose requires an energy grid in MeV."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
        msg = "Proton dose requires differential fluence y unit."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "Proton dose requires at least two energy points."
        raise ValueError(msg)


def integrate_proton_energy_deposition_MeV_per_mg(
    *,
    product: SpectrumProduct,
    table: ProtonSiLetTable,
) -> float:
    """Return ? Phi(E) * S_e,Si(E) dE in MeV/mg."""

    _validate_proton_dose_product(product)

    spectrum = product.spectrum
    edges = _log_edges_from_centers(spectrum.x)

    return sum(
        fluence_density
        * _electronic_let_at_energy(table=table, energy_mev=energy_mev)
        * (edges[index + 1] - edges[index])
        for index, (energy_mev, fluence_density) in enumerate(
            zip(spectrum.x, spectrum.y),
        )
    )


def calculate_proton_dose_from_shielded_fluence(
    product: SpectrumProduct,
    *,
    table: ProtonSiLetTable | None = None,
) -> ProtonDoseResult:
    """Calculate accumulated proton dose from shielded mission fluence."""

    proton_si_let = table or load_normative_proton_si_let_table()
    energy_deposition_MeV_per_mg = integrate_proton_energy_deposition_MeV_per_mg(
        product=product,
        table=proton_si_let,
    )
    energy_deposition_MeV_per_g = (
        energy_deposition_MeV_per_mg * MEV_PER_MG_TO_MEV_PER_G
    )
    dose_rad = energy_deposition_MeV_per_g * MEV_TO_RAD

    return ProtonDoseResult(
        product=product,
        dose_rad=dose_rad,
        energy_deposition_MeV_per_mg=energy_deposition_MeV_per_mg,
        energy_deposition_MeV_per_g=energy_deposition_MeV_per_g,
        included_in_total=True,
    )


__all__ = [
    "PROTON_SHIELDED_DOSE_MODEL",
    "ProtonDoseResult",
    "calculate_proton_dose_from_shielded_fluence",
    "integrate_proton_energy_deposition_MeV_per_mg",
]
