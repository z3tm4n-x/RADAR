"""SEP LET product construction from shielded proton and HZE spectra."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import RadiationProductKind, SpectrumQuantity
from radar.shielding.hze_si_let import HzeSiLetTable, hze_energy_spectrum_to_let_spectrum
from radar.shielding.let_combined import (
    combine_proton_and_hze_let_spectra,
    rebin_let_spectrum_to_grid,
)
from radar.shielding.proton_si_let import proton_energy_spectrum_to_let_spectrum
from radar.shielding.resources import ShieldingNormativeTables
from radar.shielding.sep_hze import SepHzeShieldingResult
from radar.shielding.sep_proton import SepProtonShieldingResult

SEP_LET_PRODUCTS_MODEL = "sep_shielded_let_products"
SEP_LET_PROTON_TOTAL_AFTER_AL = "proton_total_after_al"
SEP_LET_HZE_PRIMARY_AFTER_AL = "hze_primary_after_al"
SEP_LET_SILICON_TARGET = "silicon_let"
SEP_LET_NO_HIDDEN_4PI = "no_hidden_4pi_factor"


@dataclass(frozen=True)
class SepLetProductsResult:
    """SEP LET spectra after shielding and conversion in silicon."""

    proton_let: Spectrum1D
    hze_let_by_z: dict[int, Spectrum1D]
    combined_let: Spectrum1D
    let_centers: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.let_centers) < 2:
            msg = "SEP LET result must contain at least two LET centers."
            raise ValueError(msg)

        if self.proton_let.x != self.let_centers:
            msg = "SEP proton LET spectrum must use the common LET grid."
            raise ValueError(msg)

        if self.combined_let.x != self.let_centers:
            msg = "SEP combined LET spectrum must use the common LET grid."
            raise ValueError(msg)

        for spectrum in self.hze_let_by_z.values():
            if spectrum.x != self.let_centers:
                msg = "SEP HZE LET spectra must use the common LET grid."
                raise ValueError(msg)


def _log_grid(min_value: float, max_value: float, count: int) -> tuple[float, ...]:
    if count < 2:
        msg = "LET grid must contain at least two centers."
        raise ValueError(msg)

    if min_value <= 0.0 or max_value <= 0.0:
        msg = "LET grid bounds must be positive."
        raise ValueError(msg)

    if not isfinite(min_value) or not isfinite(max_value):
        msg = "LET grid bounds must be finite."
        raise ValueError(msg)

    if max_value < min_value:
        msg = "LET grid upper bound must not be below lower bound."
        raise ValueError(msg)

    if min_value == max_value:
        return (min_value / 2.0, min_value, min_value * 2.0)

    log_min = log(min_value)
    log_max = log(max_value)

    return tuple(exp(log_min + (log_max - log_min) * index / (count - 1)) for index in range(count))


def _default_common_let_centers(spectra: tuple[Spectrum1D, ...]) -> tuple[float, ...]:
    if not spectra:
        msg = "At least one LET spectrum is required to build a common LET grid."
        raise ValueError(msg)

    min_let = min(spectrum.x[0] for spectrum in spectra)
    max_let = max(spectrum.x[-1] for spectrum in spectra)
    count = max(len(spectrum.x) for spectrum in spectra)

    return _log_grid(min_let, max_let, count)


def _validate_energy_quantities_match(
    *,
    proton: SepProtonShieldingResult,
    hze_by_z: Mapping[int, SepHzeShieldingResult],
) -> None:
    quantity = proton.total.quantity

    for z, result in hze_by_z.items():
        if result.spectrum.quantity is not quantity:
            msg = (
                "SEP LET construction requires proton and HZE spectra with the same "
                f"energy quantity; Z={z} has {result.spectrum.quantity.value}, "
                f"expected {quantity.value}."
            )
            raise ValueError(msg)


def _product_kind_from_energy_quantity(quantity: SpectrumQuantity) -> RadiationProductKind:
    if quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        return RadiationProductKind.MISSION_LET_FLUENCE

    if quantity is SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX:
        return RadiationProductKind.PEAK_LET_FLUX

    if quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MEAN_LET_FLUX

    if quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MAXIMUM_LET_FLUX

    msg = (
        "Cannot infer LET product kind from generic or unsupported energy quantity "
        f"{quantity.value}."
    )
    raise ValueError(msg)


def _combined_model(input_model: str) -> str:
    return (
        f"{input_model}+{SEP_LET_PRODUCTS_MODEL}+"
        f"{SEP_LET_PROTON_TOTAL_AFTER_AL}+{SEP_LET_HZE_PRIMARY_AFTER_AL}+"
        f"{SEP_LET_SILICON_TARGET}+{SEP_LET_NO_HIDDEN_4PI}"
    )


def _hze_let_table_for_z(
    *,
    tables: ShieldingNormativeTables,
    z: int,
) -> HzeSiLetTable:
    try:
        return tables.hze_si_let_by_z[z]
    except KeyError as exc:
        msg = f"No HZE silicon LET table is available for Z={z}."
        raise ValueError(msg) from exc


def calculate_sep_let_products(
    *,
    proton: SepProtonShieldingResult,
    hze_by_z: Mapping[int, SepHzeShieldingResult],
    tables: ShieldingNormativeTables,
    let_centers: tuple[float, ...] | None = None,
) -> SepLetProductsResult:
    """Build proton, HZE and combined SEP LET spectra in silicon."""

    if not hze_by_z:
        msg = "SEP LET construction requires at least one HZE shielding result."
        raise ValueError(msg)

    _validate_energy_quantities_match(proton=proton, hze_by_z=hze_by_z)

    proton_let_raw = proton_energy_spectrum_to_let_spectrum(
        spectrum=proton.total,
        table=tables.proton_si_let,
    )

    hze_let_raw_by_z: dict[int, Spectrum1D] = {}

    for z, hze_result in sorted(hze_by_z.items()):
        table = _hze_let_table_for_z(tables=tables, z=z)
        hze_let_raw_by_z[z] = hze_energy_spectrum_to_let_spectrum(
            spectrum=hze_result.spectrum,
            table=table,
        )

    raw_spectra = (proton_let_raw, *tuple(hze_let_raw_by_z.values()))
    output_centers = let_centers or _default_common_let_centers(raw_spectra)

    proton_let = rebin_let_spectrum_to_grid(
        spectrum=proton_let_raw,
        let_centers=output_centers,
    )
    hze_let_by_z = {
        z: rebin_let_spectrum_to_grid(
            spectrum=spectrum,
            let_centers=output_centers,
        )
        for z, spectrum in hze_let_raw_by_z.items()
    }

    combined_let = combine_proton_and_hze_let_spectra(
        proton_let=proton_let,
        hze_let_spectra=tuple(hze_let_by_z[z] for z in sorted(hze_let_by_z)),
        let_centers=output_centers,
        model=_combined_model(proton.total.model),
    )

    return SepLetProductsResult(
        proton_let=proton_let,
        hze_let_by_z=hze_let_by_z,
        combined_let=combined_let,
        let_centers=output_centers,
    )


def calculate_sep_combined_let_product(
    *,
    proton: SepProtonShieldingResult,
    hze_by_z: Mapping[int, SepHzeShieldingResult],
    tables: ShieldingNormativeTables,
    let_centers: tuple[float, ...] | None = None,
    product_kind: RadiationProductKind | None = None,
    label: str = "",
) -> SpectrumProduct:
    """Return the combined SEP LET spectrum as a SpectrumProduct."""

    result = calculate_sep_let_products(
        proton=proton,
        hze_by_z=hze_by_z,
        tables=tables,
        let_centers=let_centers,
    )
    output_kind = product_kind or _product_kind_from_energy_quantity(proton.total.quantity)

    return SpectrumProduct(
        kind=output_kind,
        spectrum=result.combined_let,
        label=label or f"SEP combined LET behind Al shield {proton.thickness_g_cm2:g} g/cm^2",
    )


__all__ = [
    "SEP_LET_HZE_PRIMARY_AFTER_AL",
    "SEP_LET_NO_HIDDEN_4PI",
    "SEP_LET_PRODUCTS_MODEL",
    "SEP_LET_PROTON_TOTAL_AFTER_AL",
    "SEP_LET_SILICON_TARGET",
    "SepLetProductsResult",
    "calculate_sep_combined_let_product",
    "calculate_sep_let_products",
]
