"""SEP proton shielding product construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource
from radar.shielding.proton_al import shield_proton_spectrum_primary_through_al
from radar.shielding.proton_al_secondary import (
    calculate_secondary_proton_spectrum_through_al,
    combine_primary_and_secondary_proton_spectra,
)
from radar.shielding.proton_al_survival import (
    apply_proton_nonelastic_survival_to_primary_spectrum,
)
from radar.shielding.resources import ShieldingNormativeTables

SEP_PROTON_AL_SHIELDING_MODEL = "sep_proton_al_shielding_primary_survival_secondary"
SEP_PROTON_AL_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"
SEP_PROTON_AL_NO_HIDDEN_4PI = "no_hidden_4pi_factor"


@dataclass(frozen=True)
class SepProtonShieldingResult:
    """Intermediate and total SEP proton spectra behind aluminium shielding."""

    primary: Spectrum1D
    primary_survived: Spectrum1D
    secondary: Spectrum1D
    total: Spectrum1D
    thickness_g_cm2: float
    nonelastic_survival: bool
    secondary_protons: bool

    def __post_init__(self) -> None:
        if not isfinite(self.thickness_g_cm2):
            msg = "SEP proton shielding thickness must be finite."
            raise ValueError(msg)

        if self.thickness_g_cm2 < 0.0:
            msg = "SEP proton shielding thickness must be non-negative."
            raise ValueError(msg)

        for spectrum in (self.primary, self.primary_survived, self.secondary, self.total):
            _validate_sep_proton_spectrum(spectrum)


def _validate_sep_proton_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source is not RadiationSource.SEP:
        msg = "SEP proton shielding requires a SEP spectrum."
        raise ValueError(msg)

    if spectrum.particle is not Particle.PROTON:
        msg = "SEP proton shielding requires a proton spectrum."
        raise ValueError(msg)


def _zero_spectrum_like(spectrum: Spectrum1D, *, model: str) -> Spectrum1D:
    return Spectrum1D(
        x=spectrum.x,
        y=tuple(0.0 for _ in spectrum.x),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=Particle.PROTON,
        source=spectrum.source,
        model=model,
    )


def _total_model(
    *,
    input_model: str,
    include_nonelastic_survival: bool,
    include_secondary_protons: bool,
) -> str:
    survival = (
        "proton_nonelastic_survival_included"
        if include_nonelastic_survival
        else "proton_nonelastic_survival_not_included"
    )
    secondary = (
        "secondary_protons_included"
        if include_secondary_protons
        else "secondary_protons_not_included"
    )

    return (
        f"{input_model}+{SEP_PROTON_AL_SHIELDING_MODEL}+"
        f"{SEP_PROTON_AL_CENTERED_SPHERICAL_SHELL}+"
        f"{survival}+{secondary}+{SEP_PROTON_AL_NO_HIDDEN_4PI}"
    )


def calculate_sep_proton_shielding_spectrum(
    *,
    spectrum: Spectrum1D,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> SepProtonShieldingResult:
    """Calculate SEP proton primary, survived primary, secondary and total spectra.

    This function performs only proton energy-spectrum shielding through a
    centered spherical aluminium shell. It does not calculate LET, dose or SEE.
    """

    if not isfinite(thickness_g_cm2):
        msg = "SEP proton shielding thickness must be finite."
        raise ValueError(msg)

    if thickness_g_cm2 < 0.0:
        msg = "SEP proton shielding thickness must be non-negative."
        raise ValueError(msg)

    _validate_sep_proton_spectrum(spectrum)

    primary = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=tables.proton_al_range,
        thickness_g_cm2=thickness_g_cm2,
    )

    if include_nonelastic_survival:
        primary_survived = apply_proton_nonelastic_survival_to_primary_spectrum(
            spectrum=spectrum,
            stopping_table=tables.proton_al_range,
            cross_section_table=tables.al27_nonelastic_xs,
            thickness_g_cm2=thickness_g_cm2,
            integration_steps=survival_integration_steps,
        )
    else:
        primary_survived = primary

    if include_secondary_protons:
        secondary = calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=spectrum,
            stopping_table=tables.proton_al_range,
            kernel=tables.secondary_proton_kernel,
            thickness_g_cm2=thickness_g_cm2,
            depth_steps=secondary_depth_steps,
        )
    else:
        secondary = _zero_spectrum_like(
            spectrum,
            model=f"{spectrum.model}+secondary_protons_not_included",
        )

    total = combine_primary_and_secondary_proton_spectra(
        primary=primary_survived,
        secondary=secondary,
        model=_total_model(
            input_model=spectrum.model,
            include_nonelastic_survival=include_nonelastic_survival,
            include_secondary_protons=include_secondary_protons,
        ),
    )

    return SepProtonShieldingResult(
        primary=primary,
        primary_survived=primary_survived,
        secondary=secondary,
        total=total,
        thickness_g_cm2=thickness_g_cm2,
        nonelastic_survival=include_nonelastic_survival,
        secondary_protons=include_secondary_protons,
    )


def calculate_sep_proton_shielding_product(
    *,
    product: SpectrumProduct,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> SpectrumProduct:
    """Calculate a SEP proton behind-shielding product from a source product."""

    _validate_sep_proton_spectrum(product.spectrum)

    result = calculate_sep_proton_shielding_spectrum(
        spectrum=product.spectrum,
        tables=tables,
        thickness_g_cm2=thickness_g_cm2,
        include_nonelastic_survival=include_nonelastic_survival,
        include_secondary_protons=include_secondary_protons,
        survival_integration_steps=survival_integration_steps,
        secondary_depth_steps=secondary_depth_steps,
    )

    source_label = product.label or product.kind.value
    label = f"{source_label} behind Al shield {thickness_g_cm2:g} g/cm^2"

    return SpectrumProduct(
        kind=product.kind,
        spectrum=result.total,
        label=label,
    )


__all__ = [
    "SEP_PROTON_AL_CENTERED_SPHERICAL_SHELL",
    "SEP_PROTON_AL_NO_HIDDEN_4PI",
    "SEP_PROTON_AL_SHIELDING_MODEL",
    "SepProtonShieldingResult",
    "calculate_sep_proton_shielding_product",
    "calculate_sep_proton_shielding_spectrum",
]
