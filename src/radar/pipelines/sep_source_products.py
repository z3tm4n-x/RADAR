"""SEP source-products pipeline with OST geomagnetic penetration."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    profile_uses_ost_134_1044_2007,
    validate_source_model_metadata_for_profile,
)
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.types import Particle, RadiationSource
from radar.geomagnetic.ost_penetration import build_ost_penetration_function_for_config
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid
from radar.geomagnetic.spectrum import apply_proton_penetration
from radar.physics.rigidity import proton_kinetic_energy_to_rigidity_gv
from radar.sep.model import (
    SepModelInput,
    SepModelProtocol,
    SepModelResult,
    validate_sep_energy_spectrum,
)

SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT = "sep_source_products_pipeline"
SEP_SOURCE_MODEL_COMPONENT = "sep_source_model"
SEP_GEOMAGNETIC_PENETRATION_COMPONENT = "sep_geomagnetic_penetration"
UNVERSIONED_MODEL = "unversioned"
SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION = "appendix_zh_mlt_average_v1"
SEP_GEOMAGNETIC_PENETRATION_DOCUMENT = "OST 134-1044-2007 Appendix Zh"


@dataclass(frozen=True)
class SepGeomagneticPenetrationOutput:
    """Output of applying geomagnetic penetration to SEP products."""

    products: tuple[SpectrumProduct, ...]
    penetration: PenetrationFunction
    rigidity_grid: RigidityGrid


def _same_spectrum_domain_and_semantics(
    product: SpectrumProduct,
    source_product: SpectrumProduct,
) -> bool:
    spectrum = product.spectrum
    source_spectrum = source_product.spectrum

    return (
        product.kind is source_product.kind
        and spectrum.x == source_spectrum.x
        and spectrum.x_unit is source_spectrum.x_unit
        and spectrum.y_unit is source_spectrum.y_unit
        and spectrum.quantity is source_spectrum.quantity
        and spectrum.particle is source_spectrum.particle
        and spectrum.source is source_spectrum.source
    )


def _validate_pipeline_products_derived_from_model_products(
    *,
    products: tuple[SpectrumProduct, ...],
    model_products: tuple[SpectrumProduct, ...],
) -> None:
    if products == model_products:
        return

    if len(products) != len(model_products):
        msg = "SEP pipeline products must match or derive from SEP model result products."
        raise ValueError(msg)

    for product, model_product in zip(products, model_products):
        if not _same_spectrum_domain_and_semantics(product, model_product):
            msg = (
                "SEP pipeline products must preserve product kind and spectrum "
                "domain from SEP model result products."
            )
            raise ValueError(msg)

        if product == model_product:
            continue

        expected_model_prefix = f"{model_product.spectrum.model}+"
        if not product.spectrum.model.startswith(expected_model_prefix):
            msg = (
                "SEP pipeline derived product spectra must identify their source "
                "model in the spectrum model string."
            )
            raise ValueError(msg)


def _proton_rigidity_grid_for_products(
    products: tuple[SpectrumProduct, ...],
) -> RigidityGrid:
    values_gv = sorted(
        {
            proton_kinetic_energy_to_rigidity_gv(energy_mev)
            for product in products
            if product.spectrum.particle is Particle.PROTON
            for energy_mev in product.spectrum.x
        }
    )

    if not values_gv:
        msg = "SEP geomagnetic penetration requires at least one proton product."
        raise ValueError(msg)

    return RigidityGrid(values_gv=tuple(values_gv))


def _apply_ost_geomagnetic_penetration_to_products(
    *,
    config: CalculationConfig,
    products: tuple[SpectrumProduct, ...],
) -> SepGeomagneticPenetrationOutput:
    rigidity_grid = _proton_rigidity_grid_for_products(products)
    penetration = build_ost_penetration_function_for_config(
        config,
        rigidity_grid=rigidity_grid,
    )

    transformed_products: list[SpectrumProduct] = []

    for product in products:
        if product.spectrum.particle is not Particle.PROTON:
            transformed_products.append(product)
            continue

        transformed_products.append(
            SpectrumProduct(
                kind=product.kind,
                spectrum=apply_proton_penetration(
                    spectrum=product.spectrum,
                    penetration=penetration,
                ),
                label=product.label,
            )
        )

    return SepGeomagneticPenetrationOutput(
        products=tuple(transformed_products),
        penetration=penetration,
        rigidity_grid=rigidity_grid,
    )


@dataclass(frozen=True)
class SepSourceProductsPipelineResult:
    """Result of SEP source-products calculation."""

    calculation_result: CalculationResult
    sep_model_result: SepModelResult
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        products = self.products or self.sep_model_result.products

        if not products:
            msg = "SEP source-products pipeline result must contain products."
            raise ValueError(msg)

        _validate_pipeline_products_derived_from_model_products(
            products=products,
            model_products=self.sep_model_result.products,
        )

        spectra = tuple(product.spectrum for product in products)

        for spectrum in spectra:
            validate_sep_energy_spectrum(spectrum)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=spectra,
            source=RadiationSource.SEP,
            mismatch_message="SEP source product spectra must match source spectra.",
        )

        object.__setattr__(self, "products", products)

    @property
    def source_products(self) -> tuple[SpectrumProduct, ...]:
        """Raw SEP source products before geomagnetic penetration."""

        return self.sep_model_result.products


def calculate_sep_source_products_pipeline(
    config: CalculationConfig,
    sep_model: SepModelProtocol,
) -> SepSourceProductsPipelineResult:
    """Calculate SEP source products and apply OST geomagnetic penetration."""

    calculation_result = CalculationResult(config=config)

    calculation_result = calculation_result.set_component_status(
        component=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        message="SEP source-products pipeline started.",
        details={
            "launch_year": str(config.mission.launch_year),
            "lifetime_years": str(config.mission.lifetime_years),
        },
    )

    metadata = sep_model.metadata
    validate_source_model_metadata_for_profile(
        metadata=metadata,
        profile=config.methodology.profile,
    )

    sep_model_result = sep_model.calculate(SepModelInput(mission=config.mission))
    products = sep_model_result.products

    calculation_result = calculation_result.set_component_status(
        component=SEP_SOURCE_MODEL_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )
    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=sep_model_result.model,
            version=metadata.version or UNVERSIONED_MODEL,
            status="calculated",
            source=sep_model_result.document,
        )
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_SOURCE_MODEL_COMPONENT,
        message="SEP source products calculated.",
        details={
            "model": sep_model_result.model,
            "document": sep_model_result.document,
            "version": metadata.version or UNVERSIONED_MODEL,
            "products": str(len(sep_model_result.products)),
        },
    )

    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        calculation_result = calculation_result.set_component_status(
            component=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            message="SEP geomagnetic penetration started.",
            details={
                "kp": str(config.kp),
                "products": str(len(products)),
            },
        )

        penetration_output = _apply_ost_geomagnetic_penetration_to_products(
            config=config,
            products=products,
        )
        products = penetration_output.products

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=penetration_output.penetration.model,
                version=SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                status="calculated",
                source=SEP_GEOMAGNETIC_PENETRATION_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=ComponentStatus.COMPLETED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            message="SEP geomagnetic penetration completed.",
            details={
                "products": str(len(products)),
                "penetration_model": penetration_output.penetration.model,
                "penetration_version": SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                "rigidity_grid_points": str(
                    len(penetration_output.rigidity_grid.values_gv)
                ),
            },
        )

    calculation_result = calculation_result.set_component_status(
        component=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        message="SEP source-products pipeline completed.",
    )

    return SepSourceProductsPipelineResult(
        calculation_result=calculation_result,
        sep_model_result=sep_model_result,
        products=products,
    )
