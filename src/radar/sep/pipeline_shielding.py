"""SEP pipeline assembly for shielding and LET products."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Mapping

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, SpectrumQuantity
from radar.gcr.shielding import (
    GCR_AL_CENTERED_SPHERICAL_SHELL,
    GCR_HZE_FRAGMENTATION_NOT_INCLUDED,
    GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED,
    GCR_NO_HIDDEN_4PI,
)
from radar.shielding.hze_al import HzeAlRangeTable, shield_hze_spectrum_primary_through_al
from radar.shielding.hze_si_let import HzeSiLetTable, hze_energy_spectrum_to_let_spectrum
from radar.shielding.let_combined import (
    combine_proton_and_hze_let_spectra,
    rebin_let_spectrum_to_grid,
)
from radar.shielding.proton_si_let import proton_energy_spectrum_to_let_spectrum
from radar.shielding.resources import ShieldingNormativeTables
from radar.shielding.sep_proton import (
    SepProtonShieldingResult,
    calculate_sep_proton_shielding_spectrum,
)

SEP_HZE_AL_SHIELDING_MODEL = "sep_hze_al_shielding_primary_csda"
SEP_LET_PRODUCTS_MODEL = "sep_shielded_let_products"
SEP_SILICON_LET_TARGET = "silicon_let"

SEP_SHIELDING_LET_PRODUCT_KIND_ORDER: tuple[RadiationProductKind, ...] = (
    RadiationProductKind.PEAK_FLUX,
    RadiationProductKind.MEAN_FLUX,
    RadiationProductKind.MISSION_FLUENCE,
)

SEP_SHIELDING_LET_DIFFERENTIAL_QUANTITIES: tuple[SpectrumQuantity, ...] = (
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
)

_SEP_LET_KIND_BY_ENERGY_PRODUCT_KIND: dict[
    RadiationProductKind,
    RadiationProductKind,
] = {
    RadiationProductKind.PEAK_FLUX: RadiationProductKind.PEAK_LET_FLUX,
    RadiationProductKind.MEAN_FLUX: RadiationProductKind.MEAN_LET_FLUX,
    RadiationProductKind.MISSION_FLUENCE: RadiationProductKind.MISSION_LET_FLUENCE,
}


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

        _validate_thickness(self.thickness_g_cm2, "SEP HZE shielding thickness")

        if self.nuclear_attenuation:
            msg = "SEP HZE shielding result must not claim included nuclear attenuation."
            raise ValueError(msg)

        if self.fragmentation:
            msg = "SEP HZE shielding result must not claim included fragmentation."
            raise ValueError(msg)

        _validate_sep_hze_energy_spectrum(self.spectrum)


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


@dataclass(frozen=True)
class SepShieldingLetPipelineProducts:
    """SEP shielding and LET products for one aluminium thickness."""

    thickness_g_cm2: float
    proton_by_kind: dict[RadiationProductKind, SepProtonShieldingResult]
    hze_by_kind: dict[RadiationProductKind, dict[int, SepHzeShieldingResult]]
    shielded_products: tuple[SpectrumProduct, ...]
    let_products: tuple[SpectrumProduct, ...]
    let_results_by_kind: dict[
        RadiationProductKind,
        SepLetProductsResult,
    ] = field(default_factory=dict)

    @property
    def products(self) -> tuple[SpectrumProduct, ...]:
        """Return shielded energy products followed by LET products."""

        return (*self.shielded_products, *self.let_products)

    def __post_init__(self) -> None:
        _validate_thickness(self.thickness_g_cm2, "SEP pipeline shielding thickness")

        if not self.shielded_products:
            msg = "SEP pipeline shielding must produce at least one shielded product."
            raise ValueError(msg)


def _validate_thickness(value: float, name: str) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def _validate_sep_energy_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source.value != "sep":
        msg = "SEP shielding requires a SEP spectrum."
        raise ValueError(msg)

    if spectrum.quantity not in SEP_SHIELDING_LET_DIFFERENTIAL_QUANTITIES:
        msg = "SEP shielding requires mission fluence, peak flux or mean flux."
        raise ValueError(msg)


def _validate_sep_hze_energy_spectrum(spectrum: Spectrum1D) -> None:
    _validate_sep_energy_spectrum(spectrum)

    if spectrum.particle is not Particle.HZE:
        msg = "SEP HZE shielding requires an HZE spectrum."
        raise ValueError(msg)


def _shielded_label(
    *,
    product: SpectrumProduct,
    thickness_g_cm2: float,
) -> str:
    source_label = product.label or product.kind.value
    return f"{source_label} behind Al shield {thickness_g_cm2:g} g/cm^2"


def _let_label(
    *,
    product_kind: RadiationProductKind,
    thickness_g_cm2: float,
) -> str:
    return (
        f"SEP {product_kind.value} combined LET "
        f"behind Al shield {thickness_g_cm2:g} g/cm^2"
    )


def _shielded_product(
    *,
    source_product: SpectrumProduct,
    spectrum_result: SepProtonShieldingResult | SepHzeShieldingResult,
) -> SpectrumProduct:
    if isinstance(spectrum_result, SepProtonShieldingResult):
        spectrum = spectrum_result.total
    else:
        spectrum = spectrum_result.spectrum

    return SpectrumProduct(
        kind=source_product.kind,
        spectrum=spectrum,
        label=_shielded_label(
            product=source_product,
            thickness_g_cm2=spectrum_result.thickness_g_cm2,
        ),
    )


def _is_supported_energy_product(product: SpectrumProduct) -> bool:
    return (
        product.kind in SEP_SHIELDING_LET_PRODUCT_KIND_ORDER
        and product.spectrum.quantity in SEP_SHIELDING_LET_DIFFERENTIAL_QUANTITIES
    )


def _sep_hze_charge_number_from_product(product: SpectrumProduct) -> int:
    parts = product.spectrum.model.split(":")

    if len(parts) < 3 or parts[1] != "hze":
        msg = "SEP HZE product model must contain an hze ion tag."
        raise ValueError(msg)

    tag = parts[2]

    if not tag.startswith("z"):
        msg = "SEP HZE product ion tag must start with 'z'."
        raise ValueError(msg)

    digits: list[str] = []

    for character in tag[1:]:
        if not character.isdigit():
            break

        digits.append(character)

    if not digits:
        msg = "SEP HZE product ion tag must contain a charge number."
        raise ValueError(msg)

    return int("".join(digits))


def _hze_table_for_z(
    *,
    tables: ShieldingNormativeTables,
    z: int,
) -> HzeAlRangeTable:
    try:
        return tables.hze_al_ranges_by_z[z]
    except KeyError as exc:
        msg = f"No SEP HZE aluminium range table is available for Z={z}."
        raise ValueError(msg) from exc


def _hze_let_table_for_z(
    *,
    tables: ShieldingNormativeTables,
    z: int,
) -> HzeSiLetTable:
    try:
        return tables.hze_si_let_by_z[z]
    except KeyError as exc:
        msg = f"No SEP HZE silicon LET table is available for Z={z}."
        raise ValueError(msg) from exc


def _hze_model(*, input_model: str, z: int, symbol: str) -> str:
    return (
        f"{input_model}+Z{z}_{symbol}+{SEP_HZE_AL_SHIELDING_MODEL}+"
        f"{GCR_AL_CENTERED_SPHERICAL_SHELL}+"
        f"{GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED}+"
        f"{GCR_HZE_FRAGMENTATION_NOT_INCLUDED}+{GCR_NO_HIDDEN_4PI}"
    )


def calculate_sep_hze_shielding_spectrum(
    *,
    z: int,
    spectrum: Spectrum1D,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> SepHzeShieldingResult:
    """Calculate one SEP HZE ion spectrum behind aluminium shielding."""

    _validate_thickness(thickness_g_cm2, "SEP HZE shielding thickness")
    _validate_sep_hze_energy_spectrum(spectrum)

    table = _hze_table_for_z(tables=tables, z=z)
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


def _combined_let_model(input_model: str) -> str:
    return (
        f"{input_model}+{SEP_LET_PRODUCTS_MODEL}+"
        f"proton_total_after_al+hze_primary_after_al+"
        f"{SEP_SILICON_LET_TARGET}+{GCR_NO_HIDDEN_4PI}"
    )


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
        model=_combined_let_model(proton.total.model),
    )

    return SepLetProductsResult(
        proton_let=proton_let,
        hze_let_by_z=hze_let_by_z,
        combined_let=combined_let,
        let_centers=output_centers,
    )


def calculate_sep_shielding_let_products_for_thickness(
    *,
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> SepShieldingLetPipelineProducts:
    """Calculate SEP shielding and combined LET products for one thickness."""

    if not products:
        msg = "SEP pipeline shielding requires at least one input product."
        raise ValueError(msg)

    proton_by_kind: dict[RadiationProductKind, SepProtonShieldingResult] = {}
    hze_by_kind: dict[RadiationProductKind, dict[int, SepHzeShieldingResult]] = {}
    shielded_products: list[SpectrumProduct] = []

    for product in products:
        if not _is_supported_energy_product(product):
            continue

        if product.spectrum.particle is Particle.PROTON:
            proton_result = calculate_sep_proton_shielding_spectrum(
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=thickness_g_cm2,
                include_nonelastic_survival=include_nonelastic_survival,
                include_secondary_protons=include_secondary_protons,
                survival_integration_steps=survival_integration_steps,
                secondary_depth_steps=secondary_depth_steps,
            )
            proton_by_kind[product.kind] = proton_result
            shielded_products.append(
                _shielded_product(
                    source_product=product,
                    spectrum_result=proton_result,
                )
            )
            continue

        if product.spectrum.particle is Particle.HZE:
            z = _sep_hze_charge_number_from_product(product)

            if z not in tables.hze_al_ranges_by_z:
                continue

            hze_result = calculate_sep_hze_shielding_spectrum(
                z=z,
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            hze_by_kind.setdefault(product.kind, {})[z] = hze_result
            shielded_products.append(
                _shielded_product(
                    source_product=product,
                    spectrum_result=hze_result,
                )
            )
            continue

        msg = "SEP pipeline shielding supports only proton and HZE products."
        raise ValueError(msg)

    let_products: list[SpectrumProduct] = []
    let_results_by_kind: dict[RadiationProductKind, SepLetProductsResult] = {}

    for product_kind in SEP_SHIELDING_LET_PRODUCT_KIND_ORDER:
        proton_for_let = proton_by_kind.get(product_kind)
        if proton_for_let is None:
            continue

        hze_for_let = {
            z: result
            for z, result in hze_by_kind.get(product_kind, {}).items()
            if z in tables.hze_si_let_by_z
        }

        if not hze_for_let:
            continue

        let_result = calculate_sep_let_products(
            proton=proton_for_let,
            hze_by_z=hze_for_let,
            tables=tables,
        )
        let_results_by_kind[product_kind] = let_result

        let_products.append(
            SpectrumProduct(
                kind=_SEP_LET_KIND_BY_ENERGY_PRODUCT_KIND[product_kind],
                spectrum=let_result.combined_let,
                label=_let_label(
                    product_kind=product_kind,
                    thickness_g_cm2=thickness_g_cm2,
                ),
            )
        )

    return SepShieldingLetPipelineProducts(
        thickness_g_cm2=thickness_g_cm2,
        proton_by_kind=proton_by_kind,
        hze_by_kind=hze_by_kind,
        shielded_products=tuple(shielded_products),
        let_products=tuple(let_products),
        let_results_by_kind=let_results_by_kind,
    )


def calculate_sep_shielding_let_products(
    *,
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
    thicknesses_g_cm2: tuple[float, ...],
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> tuple[SepShieldingLetPipelineProducts, ...]:
    """Calculate SEP shielding and LET products for all configured thicknesses."""

    if not thicknesses_g_cm2:
        msg = "SEP pipeline shielding requires at least one thickness."
        raise ValueError(msg)

    return tuple(
        calculate_sep_shielding_let_products_for_thickness(
            products=products,
            tables=tables,
            thickness_g_cm2=thickness_g_cm2,
            include_nonelastic_survival=include_nonelastic_survival,
            include_secondary_protons=include_secondary_protons,
            survival_integration_steps=survival_integration_steps,
            secondary_depth_steps=secondary_depth_steps,
        )
        for thickness_g_cm2 in thicknesses_g_cm2
    )


__all__ = [
    "SEP_HZE_AL_SHIELDING_MODEL",
    "SEP_LET_PRODUCTS_MODEL",
    "SEP_SHIELDING_LET_PRODUCT_KIND_ORDER",
    "SepHzeShieldingResult",
    "SepLetProductsResult",
    "SepShieldingLetPipelineProducts",
    "calculate_sep_hze_shielding_spectrum",
    "calculate_sep_let_products",
    "calculate_sep_shielding_let_products",
    "calculate_sep_shielding_let_products_for_thickness",
]
