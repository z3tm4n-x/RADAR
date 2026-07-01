"""Dose pipeline assembly from calculated radiation products."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from radar.core.products import SpectrumProduct
from radar.core.project import CalculationConfig
from radar.core.types import RadiationSource
from radar.dose.electron_shieldose2 import Shieldose2ElectronDatabase
from radar.dose.erb_electron import (
    ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL,
    calculate_erb_electron_dose_with_shieldose2,
)
from radar.dose.inputs import (
    select_erb_electron_shieldose2_inputs,
    select_hze_let_dose_inputs,
    select_proton_dose_inputs,
)
from radar.dose.ion_let import ION_LET_DOSE_MODEL, calculate_ion_let_dose_from_fluence
from radar.dose.proton import (
    PROTON_SHIELDED_DOSE_MODEL,
    calculate_proton_dose_from_shielded_fluence,
)
from radar.pipelines.erb import ErbPipelineResult
from radar.pipelines.gcr import GcrPipelineResult
from radar.pipelines.sep import SepPipelineResult
from radar.shielding.proton_si_let import ProtonSiLetTable

DOSE_PIPELINE_MODEL = "radar_dose_pipeline"
DOSE_PIPELINE_MODEL_VERSION = "dose_components_v1"


@dataclass(frozen=True)
class DoseComponent:
    """One dose contribution at one shielding thickness."""

    source: RadiationSource
    component: str
    thickness_g_cm2: float
    dose_rad: float
    included_in_total: bool
    product: SpectrumProduct
    model: str


@dataclass(frozen=True)
class DoseTotalByThickness:
    """Dose sums for one shielding thickness."""

    thickness_g_cm2: float
    included_total_rad: float
    optional_total_rad: float
    component_count: int


@dataclass(frozen=True)
class DosePipelineResult:
    """Calculated dose components and totals."""

    components: tuple[DoseComponent, ...]
    totals_by_thickness: tuple[DoseTotalByThickness, ...]
    model: str = DOSE_PIPELINE_MODEL
    version: str = DOSE_PIPELINE_MODEL_VERSION

    @property
    def included_components(self) -> tuple[DoseComponent, ...]:
        """Return components included in the primary total dose."""

        return tuple(component for component in self.components if component.included_in_total)

    @property
    def optional_components(self) -> tuple[DoseComponent, ...]:
        """Return calculated components not included in the primary total dose."""

        return tuple(component for component in self.components if not component.included_in_total)


def _validate_thicknesses(thicknesses_g_cm2: tuple[float, ...]) -> tuple[float, ...]:
    if not thicknesses_g_cm2:
        msg = "Dose pipeline requires at least one shielding thickness."
        raise ValueError(msg)

    if any(not isfinite(value) for value in thicknesses_g_cm2):
        msg = "Dose pipeline shielding thicknesses must be finite."
        raise ValueError(msg)

    if any(value <= 0.0 for value in thicknesses_g_cm2):
        msg = "Dose pipeline shielding thicknesses must be positive."
        raise ValueError(msg)

    if len(set(thicknesses_g_cm2)) != len(thicknesses_g_cm2):
        msg = "Dose pipeline shielding thicknesses must not contain duplicates."
        raise ValueError(msg)

    return tuple(sorted(float(value) for value in thicknesses_g_cm2))


def _component_name_for_proton_source(source: RadiationSource) -> str:
    return f"{source.value}_proton"


def _component_name_for_let_source(source: RadiationSource) -> str:
    return f"{source.value}_combined_let_optional"


def _components_for_erb_electrons(
    *,
    erb_on_orbit_products: tuple[SpectrumProduct, ...],
    thicknesses_g_cm2: tuple[float, ...],
    electron_database: Shieldose2ElectronDatabase | None,
    electron_n_energy_points: int,
) -> tuple[DoseComponent, ...]:
    components: list[DoseComponent] = []

    for product in select_erb_electron_shieldose2_inputs(erb_on_orbit_products):
        result = calculate_erb_electron_dose_with_shieldose2(
            product,
            thicknesses_g_cm2=thicknesses_g_cm2,
            database=electron_database,
            n_energy_points=electron_n_energy_points,
        )

        for thickness, dose_rad in zip(
            result.thicknesses_g_cm2,
            result.total_rad,
            strict=True,
        ):
            components.append(
                DoseComponent(
                    source=RadiationSource.ERB,
                    component="erb_electron_shieldose2",
                    thickness_g_cm2=thickness,
                    dose_rad=dose_rad,
                    included_in_total=result.included_in_total,
                    product=product,
                    model=ERB_ELECTRON_SHIELDOSE2_DOSE_MODEL,
                )
            )

    return tuple(components)


def _components_for_shielded_protons(
    *,
    shielded_products_by_thickness: Mapping[float, tuple[SpectrumProduct, ...]],
    proton_si_let_table: ProtonSiLetTable | None,
) -> tuple[DoseComponent, ...]:
    components: list[DoseComponent] = []

    for thickness, products in shielded_products_by_thickness.items():
        for product in select_proton_dose_inputs(products):
            result = calculate_proton_dose_from_shielded_fluence(
                product,
                table=proton_si_let_table,
            )
            source = product.spectrum.source

            components.append(
                DoseComponent(
                    source=source,
                    component=_component_name_for_proton_source(source),
                    thickness_g_cm2=float(thickness),
                    dose_rad=result.dose_rad,
                    included_in_total=result.included_in_total,
                    product=product,
                    model=PROTON_SHIELDED_DOSE_MODEL,
                )
            )

    return tuple(components)


def _components_for_optional_let_dose(
    *,
    let_products_by_thickness: Mapping[float, tuple[SpectrumProduct, ...]],
) -> tuple[DoseComponent, ...]:
    components: list[DoseComponent] = []

    for thickness, products in let_products_by_thickness.items():
        for product in select_hze_let_dose_inputs(products):
            result = calculate_ion_let_dose_from_fluence(product)
            source = product.spectrum.source

            components.append(
                DoseComponent(
                    source=source,
                    component=_component_name_for_let_source(source),
                    thickness_g_cm2=float(thickness),
                    dose_rad=result.dose_rad,
                    included_in_total=result.included_in_total,
                    product=product,
                    model=ION_LET_DOSE_MODEL,
                )
            )

    return tuple(components)


def _totals_by_thickness(
    *,
    thicknesses_g_cm2: tuple[float, ...],
    components: tuple[DoseComponent, ...],
) -> tuple[DoseTotalByThickness, ...]:
    totals: list[DoseTotalByThickness] = []

    for thickness in thicknesses_g_cm2:
        thickness_components = tuple(
            component
            for component in components
            if component.thickness_g_cm2 == thickness
        )
        included_total = sum(
            component.dose_rad
            for component in thickness_components
            if component.included_in_total
        )
        optional_total = sum(
            component.dose_rad
            for component in thickness_components
            if not component.included_in_total
        )

        totals.append(
            DoseTotalByThickness(
                thickness_g_cm2=thickness,
                included_total_rad=included_total,
                optional_total_rad=optional_total,
                component_count=len(thickness_components),
            )
        )

    return tuple(totals)


def calculate_dose_pipeline_from_products(
    *,
    thicknesses_g_cm2: tuple[float, ...],
    erb_on_orbit_products: tuple[SpectrumProduct, ...],
    shielded_products_by_thickness: Mapping[float, tuple[SpectrumProduct, ...]],
    let_products_by_thickness: Mapping[float, tuple[SpectrumProduct, ...]],
    proton_si_let_table: ProtonSiLetTable | None = None,
    electron_database: Shieldose2ElectronDatabase | None = None,
    electron_n_energy_points: int = 1001,
) -> DosePipelineResult:
    """Calculate dose components from already calculated spectral products."""

    thicknesses = _validate_thicknesses(thicknesses_g_cm2)

    components = (
        *_components_for_erb_electrons(
            erb_on_orbit_products=erb_on_orbit_products,
            thicknesses_g_cm2=thicknesses,
            electron_database=electron_database,
            electron_n_energy_points=electron_n_energy_points,
        ),
        *_components_for_shielded_protons(
            shielded_products_by_thickness=shielded_products_by_thickness,
            proton_si_let_table=proton_si_let_table,
        ),
        *_components_for_optional_let_dose(
            let_products_by_thickness=let_products_by_thickness,
        ),
    )

    return DosePipelineResult(
        components=components,
        totals_by_thickness=_totals_by_thickness(
            thicknesses_g_cm2=thicknesses,
            components=components,
        ),
    )


def _append_products(
    grouped: dict[float, list[SpectrumProduct]],
    *,
    thickness_g_cm2: float,
    products: tuple[SpectrumProduct, ...],
) -> None:
    grouped.setdefault(float(thickness_g_cm2), []).extend(products)


def _freeze_product_mapping(
    grouped: dict[float, list[SpectrumProduct]],
) -> dict[float, tuple[SpectrumProduct, ...]]:
    return {
        thickness: tuple(products)
        for thickness, products in sorted(grouped.items())
    }


def shielded_products_by_thickness_from_pipeline_results(
    *,
    sep_result: SepPipelineResult | None = None,
    gcr_result: GcrPipelineResult | None = None,
    erb_result: ErbPipelineResult | None = None,
) -> dict[float, tuple[SpectrumProduct, ...]]:
    """Collect shielded energy products from source pipeline results."""

    grouped: dict[float, list[SpectrumProduct]] = {}

    if sep_result is not None:
        for sep_shielding_result in sep_result.shielding_let_by_thickness:
            _append_products(
                grouped,
                thickness_g_cm2=sep_shielding_result.thickness_g_cm2,
                products=sep_shielding_result.shielded_products,
            )

    if gcr_result is not None:
        for gcr_shielding_result in gcr_result.shielding_let_by_thickness:
            _append_products(
                grouped,
                thickness_g_cm2=gcr_shielding_result.thickness_g_cm2,
                products=gcr_shielding_result.shielded_products,
            )

    if erb_result is not None:
        for erb_shielding_result in erb_result.shielding_by_thickness:
            _append_products(
                grouped,
                thickness_g_cm2=erb_shielding_result.thickness_g_cm2,
                products=(erb_shielding_result.shielded_product,),
            )

    return _freeze_product_mapping(grouped)


def let_products_by_thickness_from_pipeline_results(
    *,
    sep_result: SepPipelineResult | None = None,
    gcr_result: GcrPipelineResult | None = None,
) -> dict[float, tuple[SpectrumProduct, ...]]:
    """Collect LET products from SEP/GCR pipeline results."""

    grouped: dict[float, list[SpectrumProduct]] = {}

    if sep_result is not None:
        for sep_shielding_result in sep_result.shielding_let_by_thickness:
            _append_products(
                grouped,
                thickness_g_cm2=sep_shielding_result.thickness_g_cm2,
                products=sep_shielding_result.let_products,
            )

    if gcr_result is not None:
        for gcr_shielding_result in gcr_result.shielding_let_by_thickness:
            _append_products(
                grouped,
                thickness_g_cm2=gcr_shielding_result.thickness_g_cm2,
                products=gcr_shielding_result.let_products,
            )

    return _freeze_product_mapping(grouped)


def calculate_dose_pipeline_from_pipeline_results(
    *,
    config: CalculationConfig,
    sep_result: SepPipelineResult | None = None,
    gcr_result: GcrPipelineResult | None = None,
    erb_result: ErbPipelineResult | None = None,
    proton_si_let_table: ProtonSiLetTable | None = None,
    electron_database: Shieldose2ElectronDatabase | None = None,
    electron_n_energy_points: int = 1001,
) -> DosePipelineResult:
    """Calculate dose from already executed SEP/GCR/ERB pipelines."""

    erb_on_orbit_products: tuple[SpectrumProduct, ...] = ()
    if erb_result is not None:
        erb_on_orbit_products = erb_result.on_orbit_products

    return calculate_dose_pipeline_from_products(
        thicknesses_g_cm2=config.shielding.thicknesses_g_cm2,
        erb_on_orbit_products=erb_on_orbit_products,
        shielded_products_by_thickness=shielded_products_by_thickness_from_pipeline_results(
            sep_result=sep_result,
            gcr_result=gcr_result,
            erb_result=erb_result,
        ),
        let_products_by_thickness=let_products_by_thickness_from_pipeline_results(
            sep_result=sep_result,
            gcr_result=gcr_result,
        ),
        proton_si_let_table=proton_si_let_table,
        electron_database=electron_database,
        electron_n_energy_points=electron_n_energy_points,
    )


__all__ = [
    "DOSE_PIPELINE_MODEL",
    "DOSE_PIPELINE_MODEL_VERSION",
    "DoseComponent",
    "DosePipelineResult",
    "DoseTotalByThickness",
    "calculate_dose_pipeline_from_pipeline_results",
    "calculate_dose_pipeline_from_products",
    "let_products_by_thickness_from_pipeline_results",
    "shielded_products_by_thickness_from_pipeline_results",
]
