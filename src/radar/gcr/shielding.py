"""GCR shielding and LET product construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.shielding.hze_al import HzeAlRangeTable, shield_hze_spectrum_primary_through_al
from radar.shielding.hze_si_let import HzeSiLetTable, hze_energy_spectrum_to_let_spectrum
from radar.shielding.let_combined import (
    combine_proton_and_hze_let_spectra,
    rebin_let_spectrum_to_grid,
)
from radar.shielding.proton_al import shield_proton_spectrum_primary_through_al
from radar.shielding.proton_al_secondary import (
    calculate_secondary_proton_spectrum_through_al,
    combine_primary_and_secondary_proton_spectra,
)
from radar.shielding.proton_al_survival import (
    apply_proton_nonelastic_survival_to_primary_spectrum,
)
from radar.shielding.proton_si_let import proton_energy_spectrum_to_let_spectrum
from radar.shielding.resources import ShieldingNormativeTables

from radar.gcr.parameters import get_gost_gcr_ion_parameters

GCR_PROTON_AL_SHIELDING_MODEL = "gcr_proton_al_shielding_primary_survival_secondary"
GCR_HZE_AL_SHIELDING_MODEL = "gcr_hze_al_shielding_primary_csda"
GCR_LET_PRODUCTS_MODEL = "gcr_shielded_let_products"

GCR_AL_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"
GCR_PROTON_NONELASTIC_SURVIVAL_INCLUDED = "proton_nonelastic_survival_included"
GCR_PROTON_NONELASTIC_SURVIVAL_NOT_INCLUDED = "proton_nonelastic_survival_not_included"
GCR_SECONDARY_PROTONS_INCLUDED = "secondary_protons_included"
GCR_SECONDARY_PROTONS_NOT_INCLUDED = "secondary_protons_not_included"
GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED = "heavy_ion_nuclear_attenuation_not_included"
GCR_HZE_FRAGMENTATION_NOT_INCLUDED = "heavy_ion_fragmentation_not_included"
GCR_SILICON_LET_TARGET = "silicon_let"
GCR_NO_HIDDEN_4PI = "no_hidden_4pi_factor"


@dataclass(frozen=True)
class GcrProtonShieldingResult:
    """Intermediate and total GCR proton spectra behind aluminium shielding."""

    primary: Spectrum1D
    primary_survived: Spectrum1D
    secondary: Spectrum1D
    total: Spectrum1D
    thickness_g_cm2: float
    nonelastic_survival: bool
    secondary_protons: bool

    def __post_init__(self) -> None:
        _validate_thickness(self.thickness_g_cm2, "GCR proton shielding thickness")

        for spectrum in (self.primary, self.primary_survived, self.secondary, self.total):
            _validate_gcr_proton_energy_spectrum(spectrum)


@dataclass(frozen=True)
class GcrHzeShieldingResult:
    """GCR HZE spectrum for one ion behind aluminium shielding."""

    z: int
    symbol: str
    spectrum: Spectrum1D
    thickness_g_cm2: float
    nuclear_attenuation: bool
    fragmentation: bool

    def __post_init__(self) -> None:
        if self.z < 2:
            msg = "GCR HZE shielding ion charge Z must be at least 2."
            raise ValueError(msg)

        if not self.symbol:
            msg = "GCR HZE shielding ion symbol must not be empty."
            raise ValueError(msg)

        _validate_thickness(self.thickness_g_cm2, "GCR HZE shielding thickness")

        if self.nuclear_attenuation:
            msg = "GCR HZE shielding result must not claim included nuclear attenuation."
            raise ValueError(msg)

        if self.fragmentation:
            msg = "GCR HZE shielding result must not claim included fragmentation."
            raise ValueError(msg)

        _validate_gcr_hze_energy_spectrum(self.spectrum)


@dataclass(frozen=True)
class GcrLetProductsResult:
    """GCR LET spectra after shielding and conversion in silicon."""

    proton_let: Spectrum1D
    hze_let_by_z: dict[int, Spectrum1D]
    combined_let: Spectrum1D
    let_centers: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.let_centers) < 2:
            msg = "GCR LET result must contain at least two LET centers."
            raise ValueError(msg)

        if self.proton_let.x != self.let_centers:
            msg = "GCR proton LET spectrum must use the common LET grid."
            raise ValueError(msg)

        if self.combined_let.x != self.let_centers:
            msg = "GCR combined LET spectrum must use the common LET grid."
            raise ValueError(msg)

        for spectrum in self.hze_let_by_z.values():
            if spectrum.x != self.let_centers:
                msg = "GCR HZE LET spectra must use the common LET grid."
                raise ValueError(msg)


def _validate_thickness(value: float, name: str) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def _validate_gcr_energy_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source is not RadiationSource.GCR:
        msg = "GCR shielding requires a GCR spectrum."
        raise ValueError(msg)

    if spectrum.quantity not in (
        SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    ):
        msg = "GCR shielding requires mission fluence, mean flux or maximum flux."
        raise ValueError(msg)


def _validate_gcr_proton_energy_spectrum(spectrum: Spectrum1D) -> None:
    _validate_gcr_energy_spectrum(spectrum)

    if spectrum.particle is not Particle.PROTON:
        msg = "GCR proton shielding requires a proton spectrum."
        raise ValueError(msg)


def _validate_gcr_hze_energy_spectrum(spectrum: Spectrum1D) -> None:
    _validate_gcr_energy_spectrum(spectrum)

    if spectrum.particle is not Particle.HZE:
        msg = "GCR HZE shielding requires an HZE spectrum."
        raise ValueError(msg)


def _zero_spectrum_like(spectrum: Spectrum1D, *, model: str) -> Spectrum1D:
    return Spectrum1D(
        x=spectrum.x,
        y=tuple(0.0 for _ in spectrum.x),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=model,
    )


def _gcr_ion_symbol_from_product(product: SpectrumProduct) -> str:
    parts = product.spectrum.model.split(":")

    if len(parts) < 2:
        msg = "GCR HZE product model must contain an ion symbol tag."
        raise ValueError(msg)

    symbol = parts[1].split("+", maxsplit=1)[0]
    parameters = get_gost_gcr_ion_parameters().get(symbol)

    if parameters is None:
        msg = f"GCR HZE product model contains unknown ion symbol {symbol!r}."
        raise ValueError(msg)

    if parameters.z < 2:
        msg = "GCR HZE product model must identify a heavy ion symbol."
        raise ValueError(msg)

    return symbol


def gcr_hze_charge_number_from_product(product: SpectrumProduct) -> int:
    """Return charge number Z encoded by a GCR HZE product model string."""

    symbol = _gcr_ion_symbol_from_product(product)

    return get_gost_gcr_ion_parameters()[symbol].z


def _proton_total_model(
    *,
    input_model: str,
    include_nonelastic_survival: bool,
    include_secondary_protons: bool,
) -> str:
    survival = (
        GCR_PROTON_NONELASTIC_SURVIVAL_INCLUDED
        if include_nonelastic_survival
        else GCR_PROTON_NONELASTIC_SURVIVAL_NOT_INCLUDED
    )
    secondary = (
        GCR_SECONDARY_PROTONS_INCLUDED
        if include_secondary_protons
        else GCR_SECONDARY_PROTONS_NOT_INCLUDED
    )

    return (
        f"{input_model}+{GCR_PROTON_AL_SHIELDING_MODEL}+"
        f"{GCR_AL_CENTERED_SPHERICAL_SHELL}+{survival}+{secondary}+"
        f"{GCR_NO_HIDDEN_4PI}"
    )


def calculate_gcr_proton_shielding_spectrum(
    *,
    spectrum: Spectrum1D,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> GcrProtonShieldingResult:
    """Calculate GCR proton primary, survived primary, secondary and total spectra."""

    _validate_thickness(thickness_g_cm2, "GCR proton shielding thickness")
    _validate_gcr_proton_energy_spectrum(spectrum)

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
            model=f"{spectrum.model}+{GCR_SECONDARY_PROTONS_NOT_INCLUDED}",
        )

    total = combine_primary_and_secondary_proton_spectra(
        primary=primary_survived,
        secondary=secondary,
        model=_proton_total_model(
            input_model=spectrum.model,
            include_nonelastic_survival=include_nonelastic_survival,
            include_secondary_protons=include_secondary_protons,
        ),
    )

    return GcrProtonShieldingResult(
        primary=primary,
        primary_survived=primary_survived,
        secondary=secondary,
        total=total,
        thickness_g_cm2=thickness_g_cm2,
        nonelastic_survival=include_nonelastic_survival,
        secondary_protons=include_secondary_protons,
    )


def calculate_gcr_proton_shielding_product(
    *,
    product: SpectrumProduct,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> SpectrumProduct:
    """Calculate one GCR proton behind-shielding product."""

    result = calculate_gcr_proton_shielding_spectrum(
        spectrum=product.spectrum,
        tables=tables,
        thickness_g_cm2=thickness_g_cm2,
        include_nonelastic_survival=include_nonelastic_survival,
        include_secondary_protons=include_secondary_protons,
        survival_integration_steps=survival_integration_steps,
        secondary_depth_steps=secondary_depth_steps,
    )

    source_label = product.label or product.kind.value

    return SpectrumProduct(
        kind=product.kind,
        spectrum=result.total,
        label=f"{source_label} behind Al shield {thickness_g_cm2:g} g/cm^2",
    )


def _hze_table_for_z(
    *,
    tables: ShieldingNormativeTables,
    z: int,
) -> HzeAlRangeTable:
    try:
        return tables.hze_al_ranges_by_z[z]
    except KeyError as exc:
        msg = f"No GCR HZE aluminium range table is available for Z={z}."
        raise ValueError(msg) from exc


def _hze_model(*, input_model: str, z: int, symbol: str) -> str:
    return (
        f"{input_model}+Z{z}_{symbol}+{GCR_HZE_AL_SHIELDING_MODEL}+"
        f"{GCR_AL_CENTERED_SPHERICAL_SHELL}+"
        f"{GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED}+"
        f"{GCR_HZE_FRAGMENTATION_NOT_INCLUDED}+{GCR_NO_HIDDEN_4PI}"
    )


def calculate_gcr_hze_shielding_spectrum(
    *,
    z: int,
    spectrum: Spectrum1D,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> GcrHzeShieldingResult:
    """Calculate one GCR HZE ion spectrum behind aluminium shielding."""

    _validate_thickness(thickness_g_cm2, "GCR HZE shielding thickness")
    _validate_gcr_hze_energy_spectrum(spectrum)

    table = _hze_table_for_z(tables=tables, z=z)
    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=thickness_g_cm2,
        model=_hze_model(input_model=spectrum.model, z=table.z, symbol=table.symbol),
    )

    return GcrHzeShieldingResult(
        z=table.z,
        symbol=table.symbol,
        spectrum=shielded,
        thickness_g_cm2=thickness_g_cm2,
        nuclear_attenuation=False,
        fragmentation=False,
    )


def calculate_gcr_hze_shielding_product(
    *,
    product: SpectrumProduct,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> SpectrumProduct:
    """Calculate one GCR HZE behind-shielding product."""

    z = gcr_hze_charge_number_from_product(product)
    result = calculate_gcr_hze_shielding_spectrum(
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


def calculate_gcr_shielding_products(
    *,
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
) -> tuple[SpectrumProduct, ...]:
    """Calculate GCR behind-shielding products for protons and HZE ions."""

    if not products:
        msg = "GCR shielding requires at least one product."
        raise ValueError(msg)

    shielded: list[SpectrumProduct] = []

    for product in products:
        if product.spectrum.particle is Particle.PROTON:
            shielded.append(
                calculate_gcr_proton_shielding_product(
                    product=product,
                    tables=tables,
                    thickness_g_cm2=thickness_g_cm2,
                    include_nonelastic_survival=include_nonelastic_survival,
                    include_secondary_protons=include_secondary_protons,
                    survival_integration_steps=survival_integration_steps,
                    secondary_depth_steps=secondary_depth_steps,
                )
            )
            continue

        if product.spectrum.particle is Particle.HZE:
            shielded.append(
                calculate_gcr_hze_shielding_product(
                    product=product,
                    tables=tables,
                    thickness_g_cm2=thickness_g_cm2,
                )
            )
            continue

        msg = "GCR shielding supports only proton and HZE products."
        raise ValueError(msg)

    return tuple(shielded)


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
    proton: GcrProtonShieldingResult,
    hze_by_z: Mapping[int, GcrHzeShieldingResult],
) -> None:
    quantity = proton.total.quantity

    for z, result in hze_by_z.items():
        if result.spectrum.quantity is not quantity:
            msg = (
                "GCR LET construction requires proton and HZE spectra with the same "
                f"energy quantity; Z={z} has {result.spectrum.quantity.value}, "
                f"expected {quantity.value}."
            )
            raise ValueError(msg)


def _product_kind_from_energy_quantity(quantity: SpectrumQuantity) -> RadiationProductKind:
    if quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        return RadiationProductKind.MISSION_LET_FLUENCE

    if quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MEAN_LET_FLUX

    if quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MAXIMUM_LET_FLUX

    msg = f"Cannot infer GCR LET product kind from energy quantity {quantity.value}."
    raise ValueError(msg)


def _combined_let_model(input_model: str) -> str:
    return (
        f"{input_model}+{GCR_LET_PRODUCTS_MODEL}+"
        f"proton_total_after_al+hze_primary_after_al+"
        f"{GCR_SILICON_LET_TARGET}+{GCR_NO_HIDDEN_4PI}"
    )


def _hze_let_table_for_z(
    *,
    tables: ShieldingNormativeTables,
    z: int,
) -> HzeSiLetTable:
    try:
        return tables.hze_si_let_by_z[z]
    except KeyError as exc:
        msg = f"No GCR HZE silicon LET table is available for Z={z}."
        raise ValueError(msg) from exc


def calculate_gcr_let_products(
    *,
    proton: GcrProtonShieldingResult,
    hze_by_z: Mapping[int, GcrHzeShieldingResult],
    tables: ShieldingNormativeTables,
    let_centers: tuple[float, ...] | None = None,
) -> GcrLetProductsResult:
    """Build proton, HZE and combined GCR LET spectra in silicon."""

    if not hze_by_z:
        msg = "GCR LET construction requires at least one HZE shielding result."
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

    return GcrLetProductsResult(
        proton_let=proton_let,
        hze_let_by_z=hze_let_by_z,
        combined_let=combined_let,
        let_centers=output_centers,
    )


def calculate_gcr_combined_let_product(
    *,
    proton: GcrProtonShieldingResult,
    hze_by_z: Mapping[int, GcrHzeShieldingResult],
    tables: ShieldingNormativeTables,
    let_centers: tuple[float, ...] | None = None,
    product_kind: RadiationProductKind | None = None,
    label: str = "",
) -> SpectrumProduct:
    """Return the combined GCR LET spectrum as a SpectrumProduct."""

    result = calculate_gcr_let_products(
        proton=proton,
        hze_by_z=hze_by_z,
        tables=tables,
        let_centers=let_centers,
    )
    output_kind = product_kind or _product_kind_from_energy_quantity(proton.total.quantity)

    return SpectrumProduct(
        kind=output_kind,
        spectrum=result.combined_let,
        label=label or f"GCR combined LET behind Al shield {proton.thickness_g_cm2:g} g/cm^2",
    )


__all__ = [
    "GCR_HZE_AL_SHIELDING_MODEL",
    "GCR_HZE_FRAGMENTATION_NOT_INCLUDED",
    "GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED",
    "GCR_LET_PRODUCTS_MODEL",
    "GCR_NO_HIDDEN_4PI",
    "GCR_PROTON_AL_SHIELDING_MODEL",
    "GcrHzeShieldingResult",
    "GcrLetProductsResult",
    "GcrProtonShieldingResult",
    "calculate_gcr_combined_let_product",
    "calculate_gcr_hze_shielding_product",
    "calculate_gcr_hze_shielding_spectrum",
    "calculate_gcr_let_products",
    "calculate_gcr_proton_shielding_product",
    "calculate_gcr_proton_shielding_spectrum",
    "calculate_gcr_shielding_products",
    "gcr_hze_charge_number_from_product",
]
