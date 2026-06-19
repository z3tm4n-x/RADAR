"""Apply geomagnetic penetration functions to particle spectra."""

from __future__ import annotations

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, SpectrumQuantity
from radar.core.units import Unit
from radar.geomagnetic.penetration import PenetrationFunction
from radar.physics.rigidity import proton_kinetic_energy_to_rigidity_gv

ENERGY_SPECTRUM_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)


def _validate_proton_energy_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.particle is not Particle.PROTON:
        msg = "Geomagnetic penetration can be applied only to proton spectra."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "Proton spectrum energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.quantity not in ENERGY_SPECTRUM_QUANTITIES:
        msg = "Geomagnetic penetration requires an energy spectrum quantity."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "Proton spectrum energies must be positive."
        raise ValueError(msg)


def proton_penetration_factors(
    spectrum: Spectrum1D,
    penetration: PenetrationFunction,
) -> tuple[float, ...]:
    """Return geomagnetic penetration factors for a proton energy spectrum."""

    _validate_proton_energy_spectrum(spectrum)

    factors: list[float] = []

    for kinetic_energy_mev in spectrum.x:
        rigidity_gv = proton_kinetic_energy_to_rigidity_gv(kinetic_energy_mev)
        factors.append(penetration.value_at(rigidity_gv))

    return tuple(factors)


def apply_proton_penetration(
    spectrum: Spectrum1D,
    penetration: PenetrationFunction,
    model: str | None = None,
) -> Spectrum1D:
    """Return a proton spectrum multiplied by the geomagnetic penetration function."""

    factors = proton_penetration_factors(
        spectrum=spectrum,
        penetration=penetration,
    )

    output_model = model

    if output_model is None:
        output_model = f"{spectrum.model}+{penetration.model}"

    return Spectrum1D(
        x=spectrum.x,
        y=tuple(value * factor for value, factor in zip(spectrum.y, factors)),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=output_model,
    )
