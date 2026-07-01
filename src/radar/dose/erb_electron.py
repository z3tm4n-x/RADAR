"""ERB electron dose calculation through SHIELDOSE-2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10
from typing import Sequence

import numpy as np

from radar.core.products import SpectrumProduct
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.electron_shieldose2 import (
    DEFAULT_DETECTOR,
    Shieldose2ElectronDatabase,
    calculate_electron_dose,
    load_normative_elbrbas2_database,
)
from radar.dose.inputs import select_erb_electron_shieldose2_inputs

ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL = "erb_electron_shieldose2_sphere_center_dose"
ERB_ELECTRON_SHIELDOSE2_GEOMETRY = "sphere_center_half"
_ERB_ELECTRON_AL_SHIELDING_MARKER = "erb_electron_al_shielding_primary_csda"


@dataclass(frozen=True)
class ErbElectronShieldose2DoseResult:
    """Dose contribution from ERB incident electron fluence via SHIELDOSE-2."""

    product: SpectrumProduct
    thicknesses_g_cm2: tuple[float, ...]
    electron_rad: tuple[float, ...]
    brems_rad: tuple[float, ...]
    total_rad: tuple[float, ...]
    electron_slab_rad: tuple[float, ...]
    brems_slab_rad: tuple[float, ...]
    total_slab_rad: tuple[float, ...]
    included_in_total: bool = True
    model: str = ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL
    geometry: str = ERB_ELECTRON_SHIELDOSE2_GEOMETRY


def _validate_erb_electron_shieldose2_product(product: SpectrumProduct) -> None:
    spectrum = product.spectrum

    if select_erb_electron_shieldose2_inputs((product,)) != (product,):
        msg = (
            "ERB electron SHIELDOSE-2 dose requires an ERB electron on-orbit "
            "mission fluence product."
        )
        raise ValueError(msg)

    if product.kind is not RadiationProductKind.MISSION_FLUENCE:
        msg = "ERB electron SHIELDOSE-2 dose requires mission fluence."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        msg = "ERB electron SHIELDOSE-2 dose requires differential fluence."
        raise ValueError(msg)

    if spectrum.source is not RadiationSource.ERB:
        msg = "ERB electron SHIELDOSE-2 dose requires ERB source."
        raise ValueError(msg)

    if spectrum.particle is not Particle.ELECTRON:
        msg = "ERB electron SHIELDOSE-2 dose requires electron spectrum."
        raise ValueError(msg)

    if _ERB_ELECTRON_AL_SHIELDING_MARKER in spectrum.model.split("+"):
        msg = "ERB electron SHIELDOSE-2 dose must use on-orbit, not shielded, electrons."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "ERB electron SHIELDOSE-2 dose requires energy grid in MeV."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
        msg = "ERB electron SHIELDOSE-2 dose requires differential fluence y unit."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "ERB electron SHIELDOSE-2 dose requires at least two energy points."
        raise ValueError(msg)


def _validate_thicknesses(thicknesses_g_cm2: Sequence[float]) -> tuple[float, ...]:
    thicknesses = tuple(float(value) for value in thicknesses_g_cm2)

    if not thicknesses:
        msg = "ERB electron SHIELDOSE-2 dose requires at least one shielding thickness."
        raise ValueError(msg)

    if any(not isfinite(value) for value in thicknesses):
        msg = "ERB electron SHIELDOSE-2 shielding thicknesses must be finite."
        raise ValueError(msg)

    if any(value <= 0.0 for value in thicknesses):
        msg = "ERB electron SHIELDOSE-2 shielding thicknesses must be positive."
        raise ValueError(msg)

    if len(set(thicknesses)) != len(thicknesses):
        msg = "ERB electron SHIELDOSE-2 shielding thicknesses must not contain duplicates."
        raise ValueError(msg)

    return tuple(sorted(thicknesses))


def _internal_depth_grid(thicknesses_g_cm2: tuple[float, ...]) -> np.ndarray:
    min_depth = min(thicknesses_g_cm2)
    max_depth = max(thicknesses_g_cm2)

    if len(thicknesses_g_cm2) == 1:
        lower = min_depth / 3.0
        upper = max_depth * 3.0
    else:
        lower = min_depth / 2.0
        upper = max_depth * 2.0

    n_points = max(61, 6 * len(thicknesses_g_cm2) + 7)
    log_grid = np.logspace(log10(lower), log10(upper), num=n_points)
    combined = np.concatenate(
        (
            log_grid,
            np.asarray(thicknesses_g_cm2, dtype=float),
        ),
    )

    return np.unique(combined)


def _interpolate_to_requested(
    *,
    requested: tuple[float, ...],
    source_depths: np.ndarray,
    source_values: np.ndarray,
) -> tuple[float, ...]:
    values = np.interp(
        np.asarray(requested, dtype=float),
        source_depths,
        source_values,
    )

    return tuple(float(value) for value in values)


def calculate_erb_electron_dose_with_shieldose2(
    product: SpectrumProduct,
    *,
    thicknesses_g_cm2: Sequence[float],
    database: Shieldose2ElectronDatabase | None = None,
    detector: str = DEFAULT_DETECTOR,
    n_energy_points: int = 1001,
) -> ErbElectronShieldose2DoseResult:
    """Calculate ERB electron dose with SHIELDOSE-2.

    The input product is the incident ERB electron on-orbit mission fluence.
    The CSDA electron spectrum behind aluminium shielding is not used here.
    """

    _validate_erb_electron_shieldose2_product(product)
    requested_thicknesses = _validate_thicknesses(thicknesses_g_cm2)
    internal_depths = _internal_depth_grid(requested_thicknesses)
    shieldose_database = database or load_normative_elbrbas2_database(detector=detector)

    raw = calculate_electron_dose(
        energy_MeV=product.spectrum.x,
        spectrum=product.spectrum.y,
        depths_gcm2=tuple(float(value) for value in internal_depths),
        database=shieldose_database,
        detector=detector,
        quantity="fluence",
        n_energy_points=n_energy_points,
    )

    if raw.output_unit != "rad":
        msg = "ERB electron SHIELDOSE-2 wrapper expected accumulated dose in rad."
        raise ValueError(msg)

    return ErbElectronShieldose2DoseResult(
        product=product,
        thicknesses_g_cm2=requested_thicknesses,
        electron_rad=_interpolate_to_requested(
            requested=requested_thicknesses,
            source_depths=raw.depths_gcm2,
            source_values=raw.electron_sphere_center_half,
        ),
        brems_rad=_interpolate_to_requested(
            requested=requested_thicknesses,
            source_depths=raw.depths_gcm2,
            source_values=raw.brems_sphere_center_half,
        ),
        total_rad=_interpolate_to_requested(
            requested=requested_thicknesses,
            source_depths=raw.depths_gcm2,
            source_values=raw.total_sphere_center_half,
        ),
        electron_slab_rad=_interpolate_to_requested(
            requested=requested_thicknesses,
            source_depths=raw.depths_gcm2,
            source_values=raw.electron_slab,
        ),
        brems_slab_rad=_interpolate_to_requested(
            requested=requested_thicknesses,
            source_depths=raw.depths_gcm2,
            source_values=raw.brems_slab,
        ),
        total_slab_rad=_interpolate_to_requested(
            requested=requested_thicknesses,
            source_depths=raw.depths_gcm2,
            source_values=raw.total_slab,
        ),
        included_in_total=True,
    )


__all__ = [
    "ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL",
    "ERB_ELECTRON_SHIELDOSE2_GEOMETRY",
    "ErbElectronShieldose2DoseResult",
    "calculate_erb_electron_dose_with_shieldose2",
]
