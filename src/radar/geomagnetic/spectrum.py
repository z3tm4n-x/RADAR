"""Apply geomagnetic penetration functions to particle spectra."""

from __future__ import annotations

from radar.core.spectra import Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.core.types import Particle, SpectrumLocation, SpectrumQuantity
from radar.core.units import Unit
from radar.geomagnetic.penetration import PenetrationFunction
from radar.physics.rigidity import (
    ion_kinetic_energy_per_nucleon_to_rigidity_gv,
    proton_kinetic_energy_to_rigidity_gv,
)

ENERGY_SPECTRUM_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)


def _penetration_transformation(
    spectrum: Spectrum1D,
    penetration: PenetrationFunction,
    particle_parameters: tuple[tuple[str, str], ...],
) -> TransformationStep:
    """Retain the source statistic: a dimensionless access factor is not time.

    OST Zh.1/Zh.7 supplies one orbit-averaged access operator for the B/V
    source spectra. It does not define a maximum of instantaneous access.
    In particular, a source-peak spectrum multiplied by this operator must
    not be relabelled as a reconstructed instantaneous orbital maximum.
    """
    temporal_application = {
        SpectrumQuantity.DIFFERENTIAL_FLUENCE: "mission_fluence_no_time_rescaling",
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX: "source_peak_not_instantaneous_orbit_maximum",
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX: "source_maximum_not_instantaneous_orbit_maximum",
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX: "source_mean_rate_no_time_rescaling",
        SpectrumQuantity.DIFFERENTIAL_FLUX: "source_rate_no_time_rescaling",
    }[spectrum.quantity]
    return TransformationStep(
        name="geomagnetic_penetration",
        version=penetration.model_version,
        parameters=(
            ("penetration_model", penetration.model),
            ("kp", str(penetration.kp)),
            *particle_parameters,
            *((f"operator_{key}", value) for key, value in penetration.parameters),
            ("source_statistic_application", temporal_application),
            ("time_multiplier", "none"),
        ),
    )


def _retain_penetration_qualification(
    spectrum: Spectrum1D,
    penetration: PenetrationFunction,
) -> Spectrum1D:
    if penetration.incomplete_reasons:
        return mark_spectrum_incomplete(spectrum, *penetration.incomplete_reasons)
    return spectrum


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

    transformed = derive_spectrum(
        spectrum,
        y=tuple(value * factor for value, factor in zip(spectrum.y, factors)),
        model=output_model,
        location=SpectrumLocation.ON_ORBIT,
        shielding=None,
        transformation=_penetration_transformation(
            spectrum, penetration, (("particle", "proton"),),
        ),
    )
    return _retain_penetration_qualification(transformed, penetration)



def _validate_hze_energy_spectrum(
    spectrum: Spectrum1D,
    mass_to_charge: float,
) -> None:
    if spectrum.particle is not Particle.HZE:
        msg = "Geomagnetic penetration can be applied only to HZE spectra."
        raise ValueError(msg)

    if spectrum.x_unit not in (Unit.MEV, Unit.MEV_PER_NUCLEON):
        msg = "HZE spectrum energy-per-nucleon grid must be in MeV/nucleon."
        raise ValueError(msg)

    if spectrum.quantity not in ENERGY_SPECTRUM_QUANTITIES:
        msg = "Geomagnetic penetration requires an energy spectrum quantity."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "HZE spectrum energies per nucleon must be positive."
        raise ValueError(msg)

    # Also validates the A/Z argument.
    ion_kinetic_energy_per_nucleon_to_rigidity_gv(
        kinetic_energy_mev_per_nucleon=spectrum.x[0],
        mass_to_charge=mass_to_charge,
    )


def hze_penetration_factors(
    spectrum: Spectrum1D,
    penetration: PenetrationFunction,
    mass_to_charge: float,
) -> tuple[float, ...]:
    """Return geomagnetic penetration factors for an HZE energy spectrum."""

    _validate_hze_energy_spectrum(
        spectrum=spectrum,
        mass_to_charge=mass_to_charge,
    )

    factors: list[float] = []

    for kinetic_energy_mev_per_nucleon in spectrum.x:
        rigidity_gv = ion_kinetic_energy_per_nucleon_to_rigidity_gv(
            kinetic_energy_mev_per_nucleon=kinetic_energy_mev_per_nucleon,
            mass_to_charge=mass_to_charge,
        )
        factors.append(penetration.value_at(rigidity_gv))

    return tuple(factors)


def apply_hze_penetration(
    spectrum: Spectrum1D,
    penetration: PenetrationFunction,
    mass_to_charge: float,
    model: str | None = None,
) -> Spectrum1D:
    """Return an HZE spectrum multiplied by the geomagnetic penetration function."""

    factors = hze_penetration_factors(
        spectrum=spectrum,
        penetration=penetration,
        mass_to_charge=mass_to_charge,
    )

    output_model = model

    if output_model is None:
        output_model = f"{spectrum.model}+{penetration.model}"

    transformed = derive_spectrum(
        spectrum,
        y=tuple(value * factor for value, factor in zip(spectrum.y, factors)),
        model=output_model,
        location=SpectrumLocation.ON_ORBIT,
        shielding=None,
        transformation=_penetration_transformation(
            spectrum, penetration, (
                ("particle", "hze"),
                ("mass_to_charge", f"{mass_to_charge:.17g}"),
            ),
        ),
    )
    return _retain_penetration_qualification(transformed, penetration)
