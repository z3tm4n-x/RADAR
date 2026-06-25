"""SEP HZE shielding product construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource
from radar.shielding.hze_al import HzeAlRangeTable, shield_hze_spectrum_primary_through_al
from radar.shielding.resources import ShieldingNormativeTables

SEP_HZE_AL_SHIELDING_MODEL = "sep_hze_al_shielding_primary_csda"
SEP_HZE_AL_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"
SEP_HZE_AL_NUCLEAR_ATTENUATION_NOT_INCLUDED = "heavy_ion_nuclear_attenuation_not_included"
SEP_HZE_AL_FRAGMENTATION_NOT_INCLUDED = "heavy_ion_fragmentation_not_included"
SEP_HZE_AL_NO_HIDDEN_4PI = "no_hidden_4pi_factor"


@dataclass(frozen=True)
class SepHzeShieldingResult:
    """SEP HZE spectrum for one ion behind aluminium shielding."""

    z: int
    symbol: str
    spectrum: Spectrum1D
    thickness_g_cm2: float
    nuclear_attenuation: bool
    fragmentation: bool

    def __post_init__(self) -> None:
        if self.z < 2:
            msg = "SEP HZE shielding ion charge Z must be at least 2."
            raise ValueError(msg)

        if not self.symbol:
            msg = "SEP HZE shielding ion symbol must not be empty."
            raise ValueError(msg)

        if not isfinite(self.thickness_g_cm2):
            msg = "SEP HZE shielding thickness must be finite."
            raise ValueError(msg)

        if self.thickness_g_cm2 < 0.0:
            msg = "SEP HZE shielding thickness must be non-negative."
            raise ValueError(msg)

        if self.nuclear_attenuation:
            msg = "SEP HZE shielding result must not claim included nuclear attenuation."
            raise ValueError(msg)

        if self.fragmentation:
            msg = "SEP HZE shielding result must not claim included fragmentation."
            raise ValueError(msg)

        _validate_sep_hze_spectrum(self.spectrum)


def _validate_sep_hze_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source is not RadiationSource.SEP:
        msg = "SEP HZE shielding requires a SEP spectrum."
        raise ValueError(msg)

    if spectrum.particle is not Particle.HZE:
        msg = "SEP HZE shielding requires an HZE spectrum."
        raise ValueError(msg)


def _table_for_z(
    *,
    tables: ShieldingNormativeTables,
    z: int,
) -> HzeAlRangeTable:
    try:
        return tables.hze_al_ranges_by_z[z]
    except KeyError as exc:
        msg = f"No HZE aluminium range table is available for Z={z}."
        raise ValueError(msg) from exc


def _hze_model(*, input_model: str, z: int, symbol: str) -> str:
    return (
        f"{input_model}+Z{z}_{symbol}+{SEP_HZE_AL_SHIELDING_MODEL}+"
        f"{SEP_HZE_AL_CENTERED_SPHERICAL_SHELL}+"
        f"{SEP_HZE_AL_NUCLEAR_ATTENUATION_NOT_INCLUDED}+"
        f"{SEP_HZE_AL_FRAGMENTATION_NOT_INCLUDED}+"
        f"{SEP_HZE_AL_NO_HIDDEN_4PI}"
    )


def calculate_sep_hze_shielding_spectrum(
    *,
    z: int,
    spectrum: Spectrum1D,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> SepHzeShieldingResult:
    """Calculate one SEP HZE ion spectrum behind aluminium shielding.

    This is a primary-only CSDA transport calculation. Heavy-ion nuclear
    attenuation and fragmentation are intentionally not included.
    """

    if not isfinite(thickness_g_cm2):
        msg = "SEP HZE shielding thickness must be finite."
        raise ValueError(msg)

    if thickness_g_cm2 < 0.0:
        msg = "SEP HZE shielding thickness must be non-negative."
        raise ValueError(msg)

    _validate_sep_hze_spectrum(spectrum)

    table = _table_for_z(tables=tables, z=z)
    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=thickness_g_cm2,
        model=_hze_model(input_model=spectrum.model, z=table.z, symbol=table.symbol),
    )

    return SepHzeShieldingResult(
        z=table.z,
        symbol=table.symbol,
        spectrum=shielded,
        thickness_g_cm2=thickness_g_cm2,
        nuclear_attenuation=False,
        fragmentation=False,
    )


def calculate_sep_hze_shielding_spectra(
    *,
    spectra_by_z: Mapping[int, Spectrum1D],
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> dict[int, SepHzeShieldingResult]:
    """Calculate SEP HZE spectra behind aluminium shielding for several ions."""

    if not spectra_by_z:
        msg = "At least one SEP HZE spectrum is required."
        raise ValueError(msg)

    results: dict[int, SepHzeShieldingResult] = {}

    for z, spectrum in sorted(spectra_by_z.items()):
        results[z] = calculate_sep_hze_shielding_spectrum(
            z=z,
            spectrum=spectrum,
            tables=tables,
            thickness_g_cm2=thickness_g_cm2,
        )

    return results


def calculate_sep_hze_shielding_product(
    *,
    z: int,
    product: SpectrumProduct,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> SpectrumProduct:
    """Calculate one SEP HZE behind-shielding product from a source product."""

    _validate_sep_hze_spectrum(product.spectrum)

    result = calculate_sep_hze_shielding_spectrum(
        z=z,
        spectrum=product.spectrum,
        tables=tables,
        thickness_g_cm2=thickness_g_cm2,
    )

    source_label = product.label or product.kind.value
    label = (
        f"{source_label} Z={result.z} {result.symbol} "
        f"behind Al shield {thickness_g_cm2:g} g/cm^2"
    )

    return SpectrumProduct(
        kind=product.kind,
        spectrum=result.spectrum,
        label=label,
    )


__all__ = [
    "SEP_HZE_AL_CENTERED_SPHERICAL_SHELL",
    "SEP_HZE_AL_FRAGMENTATION_NOT_INCLUDED",
    "SEP_HZE_AL_NO_HIDDEN_4PI",
    "SEP_HZE_AL_NUCLEAR_ATTENUATION_NOT_INCLUDED",
    "SEP_HZE_AL_SHIELDING_MODEL",
    "SepHzeShieldingResult",
    "calculate_sep_hze_shielding_product",
    "calculate_sep_hze_shielding_spectra",
    "calculate_sep_hze_shielding_spectrum",
]
