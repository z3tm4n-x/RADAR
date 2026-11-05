"""GCR pipeline assembly for shielding and LET products."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from radar.core.products import SpectrumProduct
from radar.core.spectra import ShieldingDescriptor, TransformationStep
from radar.core.spectrum_ops import derive_spectrum
from radar.core.types import (
    Particle,
    RadiationProductKind,
    SpectrumLocation,
    SpectrumQuantity,
)
from radar.gcr.shielding import (
    GcrHzeShieldingResult,
    GcrLetProductsResult,
    GcrProtonShieldingResult,
    calculate_gcr_hze_shielding_spectrum,
    calculate_gcr_let_products,
    calculate_gcr_proton_shielding_spectrum,
    gcr_hze_charge_number_from_product,
)
from radar.shielding.resources import ShieldingNormativeTables
from radar.shielding.hze_al import HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX

GCR_SHIELDING_LET_PRODUCT_KIND_ORDER: tuple[RadiationProductKind, ...] = (
    RadiationProductKind.MEAN_FLUX,
    RadiationProductKind.MAXIMUM_FLUX,
    RadiationProductKind.MISSION_FLUENCE,
)

GCR_SHIELDING_LET_DIFFERENTIAL_QUANTITIES: tuple[SpectrumQuantity, ...] = (
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
)

_GCR_LET_KIND_BY_ENERGY_PRODUCT_KIND: dict[
    RadiationProductKind,
    RadiationProductKind,
] = {
    RadiationProductKind.MEAN_FLUX: RadiationProductKind.MEAN_LET_FLUX,
    RadiationProductKind.MAXIMUM_FLUX: RadiationProductKind.MAXIMUM_LET_FLUX,
    RadiationProductKind.MISSION_FLUENCE: RadiationProductKind.MISSION_LET_FLUENCE,
}


@dataclass(frozen=True)
class GcrShieldingLetPipelineProducts:
    """GCR shielding and LET products for one aluminium thickness."""

    thickness_g_cm2: float
    proton_by_kind: dict[RadiationProductKind, GcrProtonShieldingResult]
    hze_by_kind: dict[RadiationProductKind, dict[int, GcrHzeShieldingResult]]
    shielded_products: tuple[SpectrumProduct, ...]
    let_products: tuple[SpectrumProduct, ...]
    diagnostic_let_products: tuple[SpectrumProduct, ...] = ()
    let_results_by_kind: dict[
        RadiationProductKind,
        GcrLetProductsResult,
    ] = field(default_factory=dict)
    incomplete_reasons: tuple[str, ...] = ()

    @property
    def products(self) -> tuple[SpectrumProduct, ...]:
        """Return shielded energy products followed by LET products."""

        return (*self.shielded_products, *self.let_products)

    def __post_init__(self) -> None:
        if not isfinite(self.thickness_g_cm2):
            msg = "GCR pipeline shielding thickness must be finite."
            raise ValueError(msg)

        if self.thickness_g_cm2 < 0.0:
            msg = "GCR pipeline shielding thickness must be non-negative."
            raise ValueError(msg)

        if not self.shielded_products:
            msg = "GCR pipeline shielding must produce at least one shielded product."
            raise ValueError(msg)

        if any(not reason.strip() for reason in self.incomplete_reasons):
            msg = "GCR shielding incomplete reasons must not contain empty values."
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
    return f"GCR {product_kind.value} HZE direct-ion LET behind Al shield {thickness_g_cm2:g} g/cm^2"


def _diagnostic_let_label(
    *,
    product_kind: RadiationProductKind,
    component: str,
    thickness_g_cm2: float,
) -> str:
    return (
        f"GCR {product_kind.value} {component} diagnostic LET "
        f"behind Al shield {thickness_g_cm2:g} g/cm^2"
    )


def _shielded_product(
    *,
    source_product: SpectrumProduct,
    spectrum_result: GcrProtonShieldingResult | GcrHzeShieldingResult,
) -> SpectrumProduct:
    if isinstance(spectrum_result, GcrProtonShieldingResult):
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


def _zero_aluminium_identity_product(
    source_product: SpectrumProduct,
) -> SpectrumProduct:
    """Represent a zero-thickness shield without requiring transport data."""

    spectrum = derive_spectrum(
        source_product.spectrum,
        model=f"{source_product.spectrum.model}+zero_aluminium_identity",
        location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor(
            material="aluminium",
            areal_density_g_cm2=0.0,
        ),
        transformation=TransformationStep(
            name="zero_aluminium_identity",
            version="1",
            parameters=(("thickness_g_cm2", "0"),),
        ),
    )
    return SpectrumProduct(
        kind=source_product.kind,
        spectrum=spectrum,
        label=_shielded_label(product=source_product, thickness_g_cm2=0.0),
    )


def _is_supported_energy_product(product: SpectrumProduct) -> bool:
    return (
        product.kind in GCR_SHIELDING_LET_PRODUCT_KIND_ORDER
        and product.spectrum.quantity in GCR_SHIELDING_LET_DIFFERENTIAL_QUANTITIES
    )


def calculate_gcr_shielding_let_products_for_thickness(
    *,
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
    include_mixed_let_diagnostic: bool = True,
) -> GcrShieldingLetPipelineProducts:
    """Calculate GCR shielding and combined LET products for one thickness."""

    if not products:
        msg = "GCR pipeline shielding requires at least one input product."
        raise ValueError(msg)

    proton_by_kind: dict[RadiationProductKind, GcrProtonShieldingResult] = {}
    hze_by_kind: dict[RadiationProductKind, dict[int, GcrHzeShieldingResult]] = {}
    shielded_products: list[SpectrumProduct] = []
    incomplete_reasons: list[str] = []

    for product in products:
        if not _is_supported_energy_product(product):
            continue

        if product.spectrum.particle is Particle.PROTON:
            proton_result = calculate_gcr_proton_shielding_spectrum(
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=thickness_g_cm2,
                include_nonelastic_survival=include_nonelastic_survival,
                include_secondary_protons=include_secondary_protons,
                survival_integration_steps=survival_integration_steps,
                secondary_depth_steps=secondary_depth_steps,
            )
            proton_by_kind[product.kind] = proton_result
            incomplete_reasons.extend(proton_result.total.incomplete_reasons)
            shielded_products.append(
                _shielded_product(
                    source_product=product,
                    spectrum_result=proton_result,
                )
            )
            continue

        if product.spectrum.particle is Particle.HZE:
            z = gcr_hze_charge_number_from_product(product)

            if z not in tables.hze_al_ranges_by_z:
                if thickness_g_cm2 == 0.0:
                    shielded_products.append(_zero_aluminium_identity_product(product))
                    continue
                incomplete_reasons.append(
                    f"no HZE aluminium range table is available for Z={z}"
                )
                continue

            hze_result = calculate_gcr_hze_shielding_spectrum(
                z=z,
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            hze_by_kind.setdefault(product.kind, {})[z] = hze_result
            incomplete_reasons.extend(
                reason
                for reason in hze_result.spectrum.incomplete_reasons
                if reason.startswith(HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
            )
            shielded_products.append(
                _shielded_product(
                    source_product=product,
                    spectrum_result=hze_result,
                )
            )
            continue

        msg = "GCR pipeline shielding supports only proton and HZE products."
        raise ValueError(msg)

    let_products: list[SpectrumProduct] = []
    diagnostic_let_products: list[SpectrumProduct] = []
    let_results_by_kind: dict[RadiationProductKind, GcrLetProductsResult] = {}

    for product_kind in GCR_SHIELDING_LET_PRODUCT_KIND_ORDER:
        proton_for_let = proton_by_kind.get(product_kind)
        if proton_for_let is None:
            continue

        hze_for_let = {
            z: result
            for z, result in hze_by_kind.get(product_kind, {}).items()
            if z in tables.hze_si_let_by_z
        }

        missing_let_z = tuple(
            sorted(set(hze_by_kind.get(product_kind, {})) - set(hze_for_let))
        )
        if missing_let_z:
            incomplete_reasons.append(
                "no HZE silicon LET table is available for Z="
                + ",".join(str(z) for z in missing_let_z)
                + f" ({product_kind.value})"
            )

        if not hze_for_let:
            continue

        let_result = calculate_gcr_let_products(
            proton=proton_for_let,
            hze_by_z=hze_for_let,
            tables=tables,
        )
        let_results_by_kind[product_kind] = let_result

        let_products.append(
            SpectrumProduct(
                kind=_GCR_LET_KIND_BY_ENERGY_PRODUCT_KIND[product_kind],
                spectrum=let_result.hze_aggregate_let,
                label=_let_label(
                    product_kind=product_kind,
                    thickness_g_cm2=thickness_g_cm2,
                ),
            )
        )
        if include_mixed_let_diagnostic:
            diagnostic_let_products.extend(
                (
                SpectrumProduct(
                    kind=_GCR_LET_KIND_BY_ENERGY_PRODUCT_KIND[product_kind],
                    spectrum=let_result.proton_let,
                    label=_diagnostic_let_label(
                        product_kind=product_kind,
                        component="proton",
                        thickness_g_cm2=thickness_g_cm2,
                    ),
                ),
                SpectrumProduct(
                    kind=_GCR_LET_KIND_BY_ENERGY_PRODUCT_KIND[product_kind],
                    spectrum=let_result.mixed_let_diagnostic,
                    label=_diagnostic_let_label(
                        product_kind=product_kind,
                        component="mixed proton+HZE",
                        thickness_g_cm2=thickness_g_cm2,
                    ),
                ),
                )
            )

    return GcrShieldingLetPipelineProducts(
        thickness_g_cm2=thickness_g_cm2,
        proton_by_kind=proton_by_kind,
        hze_by_kind=hze_by_kind,
        shielded_products=tuple(shielded_products),
        let_products=tuple(let_products),
        diagnostic_let_products=tuple(diagnostic_let_products),
        let_results_by_kind=let_results_by_kind,
        incomplete_reasons=tuple(dict.fromkeys(incomplete_reasons)),
    )


def calculate_gcr_shielding_let_products(
    *,
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
    thicknesses_g_cm2: tuple[float, ...],
    include_nonelastic_survival: bool = True,
    include_secondary_protons: bool = True,
    survival_integration_steps: int = 64,
    secondary_depth_steps: int = 24,
    include_mixed_let_diagnostic: bool = True,
) -> tuple[GcrShieldingLetPipelineProducts, ...]:
    """Calculate GCR shielding and LET products for all configured thicknesses."""

    if not thicknesses_g_cm2:
        msg = "GCR pipeline shielding requires at least one thickness."
        raise ValueError(msg)

    return tuple(
        calculate_gcr_shielding_let_products_for_thickness(
            products=products,
            tables=tables,
            thickness_g_cm2=thickness_g_cm2,
            include_nonelastic_survival=include_nonelastic_survival,
            include_secondary_protons=include_secondary_protons,
            survival_integration_steps=survival_integration_steps,
            secondary_depth_steps=secondary_depth_steps,
            include_mixed_let_diagnostic=include_mixed_let_diagnostic,
        )
        for thickness_g_cm2 in thicknesses_g_cm2
    )


__all__ = [
    "GCR_SHIELDING_LET_PRODUCT_KIND_ORDER",
    "GcrShieldingLetPipelineProducts",
    "calculate_gcr_shielding_let_products",
    "calculate_gcr_shielding_let_products_for_thickness",
]
