from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
)
from radar.core.profiles import MethodologyProfile
from radar.core.result import ComponentStatus
from radar.core.types import RadiationProductKind
from radar.erb.model import OstErbModel
from radar.gcr.model import OstGcrModel
from radar.pipelines.execution import (
    PipelineSourceModels,
    build_pipeline_source_models,
    calculate_sep_pipeline_for_config,
)
from radar.pipelines.sep import (
    SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_LET_COMPONENT,
    SEP_OUTPUT_TABLES_COMPONENT,
    SEP_PIPELINE_COMPONENT,
    SEP_SHIELDING_COMPONENT,
)
from radar.sep.model import GostSepModel, OstSepModel


def _config(
    profile: MethodologyProfile = MethodologyProfile.OST_134_1044_2007,
) -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=2),
        orbit=OrbitConfig.circular(altitude_km=2_000.0, inclination_deg=0.0),
        shielding=ShieldingConfig(thicknesses_g_cm2=(0.1,)),
        methodology=MethodologyConfig(profile=profile),
        kp=3,
    )


def test_build_pipeline_source_models_uses_configured_profile() -> None:
    models = build_pipeline_source_models(_config())

    assert isinstance(models, PipelineSourceModels)
    assert isinstance(models.sep_model, OstSepModel)
    assert isinstance(models.gcr_model, OstGcrModel)
    assert isinstance(models.erb_model, OstErbModel)


def test_build_pipeline_source_models_respects_gost_sep_profile() -> None:
    models = build_pipeline_source_models(
        _config(MethodologyProfile.OST_WITH_GOST_SEP),
    )

    assert isinstance(models.sep_model, GostSepModel)
    assert isinstance(models.gcr_model, OstGcrModel)
    assert isinstance(models.erb_model, OstErbModel)


def test_calculate_sep_pipeline_for_config_runs_full_sep_chain() -> None:
    pipeline_result = calculate_sep_pipeline_for_config(_config())

    result = pipeline_result.calculation_result

    assert result.has_errors() is False
    assert result.component_status(SEP_PIPELINE_COMPONENT) is ComponentStatus.COMPLETED
    assert (
        result.component_status(SEP_GEOMAGNETIC_PENETRATION_COMPONENT)
        is ComponentStatus.COMPLETED
    )
    assert result.component_status(SEP_SHIELDING_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_LET_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_OUTPUT_TABLES_COMPONENT) is ComponentStatus.COMPLETED

    assert pipeline_result.source_products
    assert pipeline_result.on_orbit_products
    assert pipeline_result.shielded_products
    assert pipeline_result.let_products
    assert result.output_tables
    assert len(pipeline_result.on_orbit_products) == len(pipeline_result.source_products)
    assert any(
        product.kind is RadiationProductKind.MISSION_LET_FLUENCE
        for product in pipeline_result.let_products
    )


def test_calculate_sep_pipeline_for_config_can_disable_hze_grid() -> None:
    pipeline_result = calculate_sep_pipeline_for_config(
        _config(),
        sep_hze_energy_grid_mev_per_nucleon=None,
    )

    assert pipeline_result.source_products
    assert all(
        product.spectrum.particle.value == "proton"
        for product in pipeline_result.source_products
    )
    assert pipeline_result.calculation_result.component_status(
        SEP_PIPELINE_COMPONENT
    ) is ComponentStatus.COMPLETED
