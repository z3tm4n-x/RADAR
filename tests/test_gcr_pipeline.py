import pytest

from radar.core.profiles import MethodologyProfile
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
)
from radar.core.result import ComponentStatus
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SolarActivityLevel, SpectrumQuantity
from radar.core.units import Unit
from radar.gcr.model import StaticGcrModel
from radar.pipelines.gcr import (
    GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
    GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    GCR_LET_COMPONENT,
    GCR_MODEL_COMPONENT,
    GCR_OUTPUT_TABLES_COMPONENT,
    GCR_PIPELINE_COMPONENT,
    GCR_SHIELDING_COMPONENT,
    GcrPipelineResult,
    calculate_gcr_pipeline,
)


def _config(lifetime_years: int = 7) -> CalculationConfig:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
        solar_activity_level=SolarActivityLevel.MEAN,
    )
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    return CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.CUSTOM),
    )


def _gcr_proton_spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        particle=Particle.PROTON,
        source=RadiationSource.GCR,
        model="gcr_proton_test",
    )


def _gcr_hze_spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(0.1, 1.0, 10.0),
        y=(4.0, 5.0, 6.0),
        x_unit=Unit.GEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        particle=Particle.HZE,
        source=RadiationSource.GCR,
        model="gcr_hze_test",
    )


def _gcr_model() -> StaticGcrModel:
    return StaticGcrModel(
        spectra=(
            _gcr_proton_spectrum(),
            _gcr_hze_spectrum(),
        ),
        model="static_gcr_test",
        document="test_document",
    )


def test_gcr_pipeline_returns_model_spectra() -> None:
    pipeline_result = calculate_gcr_pipeline(
        config=_config(lifetime_years=7),
        gcr_model=_gcr_model(),
    )

    assert len(pipeline_result.spectra) == 2
    assert pipeline_result.spectra[0].particle is Particle.PROTON
    assert pipeline_result.spectra[1].particle is Particle.HZE
    assert pipeline_result.gcr_model_result.lifetime_years == 7
    assert pipeline_result.gcr_model_result.model == "static_gcr_test"
    assert pipeline_result.gcr_model_result.document == "test_document"


def test_gcr_pipeline_records_component_statuses() -> None:
    pipeline_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    result = pipeline_result.calculation_result

    assert result.component_status(GCR_PIPELINE_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(GCR_MODEL_COMPONENT) is ComponentStatus.COMPLETED
    assert result.has_errors() is False


def test_gcr_pipeline_records_log_entries() -> None:
    pipeline_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    log_entries = pipeline_result.calculation_result.log.entries

    assert len(log_entries) == 3
    assert log_entries[0].stage == GCR_PIPELINE_COMPONENT
    assert log_entries[1].stage == GCR_MODEL_COMPONENT
    assert log_entries[2].stage == GCR_PIPELINE_COMPONENT
    assert log_entries[1].details == (
        ("document", "test_document"),
        ("lifetime_years", "7"),
        ("model", "static_gcr_test"),
        ("spectrum_count", "2"),
    )


def test_gcr_pipeline_records_model_info() -> None:
    pipeline_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    model_info = pipeline_result.calculation_result.model_info

    assert len(model_info) == 1
    assert model_info[0].name == "static_gcr_test"
    assert model_info[0].version == "unversioned"
    assert model_info[0].status == "calculated"
    assert model_info[0].source == "test_document"


def test_gcr_pipeline_result_validates_spectra() -> None:
    valid_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    bad_spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        particle=Particle.ELECTRON,
        source=RadiationSource.GCR,
        model="bad_gcr_electron",
    )

    with pytest.raises(ValueError, match="electrons"):
        GcrPipelineResult(
            calculation_result=valid_result.calculation_result,
            gcr_model_result=valid_result.gcr_model_result,
            spectra=(bad_spectrum,),
        )


def test_gcr_pipeline_result_rejects_empty_spectra() -> None:
    valid_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    with pytest.raises(ValueError, match="at least one spectrum"):
        GcrPipelineResult(
            calculation_result=valid_result.calculation_result,
            gcr_model_result=valid_result.gcr_model_result,
            spectra=(),
        )

def test_gcr_pipeline_returns_model_products() -> None:
    from radar.core.types import RadiationProductKind

    pipeline_result = calculate_gcr_pipeline(
        config=_config(lifetime_years=7),
        gcr_model=_gcr_model(),
    )

    assert len(pipeline_result.products) == 2
    assert pipeline_result.products[0].kind is RadiationProductKind.MEAN_FLUX
    assert pipeline_result.products[1].kind is RadiationProductKind.MEAN_FLUX
    assert pipeline_result.products[0].spectrum == pipeline_result.spectra[0]
    assert pipeline_result.products[1].spectrum == pipeline_result.spectra[1]


def test_gcr_pipeline_result_rejects_mismatched_products() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    valid_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    with pytest.raises(ValueError, match="product spectra"):
        GcrPipelineResult(
            calculation_result=valid_result.calculation_result,
            gcr_model_result=valid_result.gcr_model_result,
            spectra=valid_result.spectra,
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=valid_result.spectra[1],
                ),
                SpectrumProduct(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=valid_result.spectra[0],
                ),
            ),
        )

def test_gcr_pipeline_result_rejects_spectra_not_from_model_result() -> None:
    valid_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    spectrum = valid_result.spectra[0]
    other_spectrum = Spectrum1D(
        x=spectrum.x,
        y=(2.0, 4.0, 6.0),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=spectrum.model,
    )

    with pytest.raises(ValueError, match="product spectra"):
        GcrPipelineResult(
            calculation_result=valid_result.calculation_result,
            gcr_model_result=valid_result.gcr_model_result,
            spectra=(other_spectrum,),
        )


def test_gcr_pipeline_result_rejects_product_with_wrong_source() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    valid_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_gcr_model(),
    )

    spectrum = valid_result.spectra[0]
    wrong_source_spectrum = Spectrum1D(
        x=spectrum.x,
        y=spectrum.y,
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=RadiationSource.SEP,
        model=spectrum.model,
    )

    with pytest.raises(ValueError, match="spectrum source"):
        GcrPipelineResult(
            calculation_result=valid_result.calculation_result,
            gcr_model_result=valid_result.gcr_model_result,
            spectra=valid_result.spectra,
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=wrong_source_spectrum,
                ),
                valid_result.products[1],
            ),
        )

def test_gcr_pipeline_rejects_model_family_mismatched_profile() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(mission=mission, orbit=orbit)

    with pytest.raises(ValueError, match="model family"):
        calculate_gcr_pipeline(
            config=config,
            gcr_model=_gcr_model(),
        )

def test_gcr_pipeline_accepts_gost_gcr_model_for_gost_profile() -> None:
    from radar.core.types import RadiationProductKind
    from radar.gcr.model import GostGcrModel

    mission = MissionConfig(launch_year=2027, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        shielding=ShieldingConfig(thicknesses_g_cm2=(0.01,)),
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_GCR),
    )

    pipeline_result = calculate_gcr_pipeline(
        config=config,
        gcr_model=GostGcrModel(
            energy_grid_mev_per_nucleon=(10.0, 30.0, 100.0),
            symbols=("H", "Fe"),
        ),
    )

    assert pipeline_result.calculation_result.has_errors() is False
    assert pipeline_result.gcr_model_result.model == "gost_gcr_model"

    assert len(pipeline_result.source_products) == 6
    assert len(pipeline_result.on_orbit_products) == 6
    assert len(pipeline_result.shielded_products) == 6
    assert len(pipeline_result.let_products) == 3
    assert len(pipeline_result.products) == 15
    assert pipeline_result.products == (
        *pipeline_result.on_orbit_products,
        *pipeline_result.shielded_products,
        *pipeline_result.let_products,
    )

    assert tuple(product.kind for product in pipeline_result.let_products) == (
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
        RadiationProductKind.MISSION_LET_FLUENCE,
    )

    for source_product, penetrated_product in zip(
        pipeline_result.source_products,
        pipeline_result.on_orbit_products,
        strict=True,
    ):
        assert penetrated_product.kind is source_product.kind
        assert penetrated_product.spectrum.model.startswith(
            f"{source_product.spectrum.model}+"
        )
        assert penetrated_product.spectrum.x == source_product.spectrum.x
        assert all(
            penetrated_value <= source_value
            for source_value, penetrated_value in zip(
                source_product.spectrum.y,
                penetrated_product.spectrum.y,
                strict=True,
            )
        )

    assert pipeline_result.calculation_result.component_status(
        GCR_GEOMAGNETIC_PENETRATION_COMPONENT
    ) is ComponentStatus.COMPLETED
    assert pipeline_result.calculation_result.component_status(
        GCR_SHIELDING_COMPONENT
    ) is ComponentStatus.COMPLETED
    assert pipeline_result.calculation_result.component_status(
        GCR_LET_COMPONENT
    ) is ComponentStatus.COMPLETED
    assert pipeline_result.calculation_result.component_status(
        GCR_OUTPUT_TABLES_COMPONENT
    ) is ComponentStatus.COMPLETED

    assert pipeline_result.source_spectra is not None
    output_table_ids = {
        table.table_id
        for table in pipeline_result.calculation_result.output_tables
    }
    assert len(output_table_ids) == 23
    assert "gcr_gost_h_total_source_outside_magnetosphere" in output_table_ids
    assert "gcr_gost_fe_total_source_outside_magnetosphere" in output_table_ids
    assert "gcr_h_mean_flux_on_orbit" in output_table_ids
    assert "gcr_fe_mission_fluence_on_orbit" in output_table_ids
    assert (
        "gcr_t0_01_mean_flux_gcr_combined_let_behind_al"
        in output_table_ids
    )
    assert all(
        table.kind.value == "spectrum"
        for table in pipeline_result.calculation_result.output_tables
    )

    model_info_by_name = {
        info.name: info
        for info in pipeline_result.calculation_result.model_info
    }
    penetration_info = model_info_by_name[
        "ost_134_1044_2007_geomagnetic_penetration"
    ]
    assert penetration_info.version == GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION
    assert model_info_by_name["gcr_al_shielding"].version == (
        "al_spherical_csda_secondary_v1"
    )
    assert model_info_by_name["gcr_si_let"].version == "si_let_histogram_v1"

