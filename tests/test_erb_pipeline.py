import pytest

from radar.core.profiles import MethodologyProfile
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
)
from radar.core.result import ComponentStatus
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.erb.model import StaticErbModel
from radar.pipelines.erb import (
    ERB_MODEL_COMPONENT,
    ERB_OUTPUT_TABLES_COMPONENT,
    ERB_PIPELINE_COMPONENT,
    ErbPipelineResult,
    calculate_erb_pipeline,
)


def _config(lifetime_years: int = 7, kp: int = 3) -> CalculationConfig:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
    )
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    return CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.CUSTOM),
        kp=kp,
    )


def _erb_proton_spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(0.1, 1.0, 10.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        particle=Particle.PROTON,
        source=RadiationSource.ERB,
        model="erb_proton_test",
    )


def _erb_electron_spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(0.1, 1.0, 10.0),
        y=(4.0, 5.0, 6.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        particle=Particle.ELECTRON,
        source=RadiationSource.ERB,
        model="erb_electron_test",
    )


def _erb_model() -> StaticErbModel:
    return StaticErbModel(
        spectra=(
            _erb_proton_spectrum(),
            _erb_electron_spectrum(),
        ),
        model="static_erb_test",
        document="test_document",
    )


def test_erb_pipeline_returns_model_spectra() -> None:
    pipeline_result = calculate_erb_pipeline(
        config=_config(lifetime_years=7, kp=4),
        erb_model=_erb_model(),
    )

    assert len(pipeline_result.spectra) == 2
    assert pipeline_result.spectra[0].particle is Particle.PROTON
    assert pipeline_result.spectra[1].particle is Particle.ELECTRON
    assert pipeline_result.erb_model_result.lifetime_years == 7
    assert pipeline_result.erb_model_result.kp == 4
    assert pipeline_result.erb_model_result.model == "static_erb_test"
    assert pipeline_result.erb_model_result.document == "test_document"


def test_erb_pipeline_records_component_statuses() -> None:
    pipeline_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    result = pipeline_result.calculation_result

    assert result.component_status(ERB_PIPELINE_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(ERB_MODEL_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(ERB_OUTPUT_TABLES_COMPONENT) is ComponentStatus.COMPLETED
    assert result.has_errors() is False


def test_erb_pipeline_records_log_entries() -> None:
    pipeline_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    log_entries = pipeline_result.calculation_result.log.entries

    assert len(log_entries) == 5
    assert log_entries[0].stage == ERB_PIPELINE_COMPONENT
    assert log_entries[1].stage == ERB_MODEL_COMPONENT
    assert log_entries[2].stage == ERB_OUTPUT_TABLES_COMPONENT
    assert log_entries[3].stage == ERB_OUTPUT_TABLES_COMPONENT
    assert log_entries[4].stage == ERB_PIPELINE_COMPONENT
    assert log_entries[1].details == (
        ("document", "test_document"),
        ("kp", "3"),
        ("lifetime_years", "7"),
        ("model", "static_erb_test"),
        ("spectrum_count", "2"),
    )


def test_erb_pipeline_records_model_info() -> None:
    pipeline_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    model_info = pipeline_result.calculation_result.model_info

    assert len(model_info) == 1
    assert model_info[0].name == "static_erb_test"
    assert model_info[0].version == "unversioned"
    assert model_info[0].status == "calculated"
    assert model_info[0].source == "test_document"


def test_erb_pipeline_records_output_tables() -> None:
    pipeline_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    tables = pipeline_result.calculation_result.output_tables

    assert len(tables) == 2
    assert tuple(table.table_id for table in tables) == (
        "erb_proton_orbit_averaged_flux_differential_flux_on_orbit",
        "erb_electron_mean_flux_mean_differential_flux_on_orbit",
    )
    assert dict(tables[0].metadata)["source"] == "erb"
    assert dict(tables[0].metadata)["shielding"] == "not_applied"
    assert dict(tables[0].metadata)["dose_status"] == "not_calculated"
    assert dict(tables[1].metadata)["particle"] == "electron"


def test_erb_pipeline_result_validates_spectra() -> None:
    valid_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    bad_spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        particle=Particle.HZE,
        source=RadiationSource.ERB,
        model="bad_erb_hze",
    )

    with pytest.raises(ValueError, match="protons or electrons"):
        ErbPipelineResult(
            calculation_result=valid_result.calculation_result,
            erb_model_result=valid_result.erb_model_result,
            spectra=(bad_spectrum,),
        )


def test_erb_pipeline_result_rejects_empty_spectra() -> None:
    valid_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    with pytest.raises(ValueError, match="at least one spectrum"):
        ErbPipelineResult(
            calculation_result=valid_result.calculation_result,
            erb_model_result=valid_result.erb_model_result,
            spectra=(),
        )

def test_erb_pipeline_returns_model_products() -> None:
    from radar.core.types import RadiationProductKind

    pipeline_result = calculate_erb_pipeline(
        config=_config(lifetime_years=7, kp=4),
        erb_model=_erb_model(),
    )

    assert len(pipeline_result.products) == 2
    assert pipeline_result.products[0].kind is RadiationProductKind.ORBIT_AVERAGED_FLUX
    assert pipeline_result.products[1].kind is RadiationProductKind.MEAN_FLUX
    assert pipeline_result.products[0].spectrum == pipeline_result.spectra[0]
    assert pipeline_result.products[1].spectrum == pipeline_result.spectra[1]


def test_erb_pipeline_result_rejects_mismatched_products() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    valid_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    with pytest.raises(ValueError, match="product spectra"):
        ErbPipelineResult(
            calculation_result=valid_result.calculation_result,
            erb_model_result=valid_result.erb_model_result,
            spectra=valid_result.spectra,
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=valid_result.spectra[1],
                ),
                SpectrumProduct(
                    kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
                    spectrum=valid_result.spectra[0],
                ),
            ),
        )

def test_erb_pipeline_result_rejects_spectra_not_from_model_result() -> None:
    valid_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
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

    with pytest.raises(ValueError, match="model result spectra"):
        ErbPipelineResult(
            calculation_result=valid_result.calculation_result,
            erb_model_result=valid_result.erb_model_result,
            spectra=(other_spectrum,),
        )


def test_erb_pipeline_result_rejects_product_with_wrong_source() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    valid_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
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
        ErbPipelineResult(
            calculation_result=valid_result.calculation_result,
            erb_model_result=valid_result.erb_model_result,
            spectra=valid_result.spectra,
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
                    spectrum=wrong_source_spectrum,
                ),
                valid_result.products[1],
            ),
        )

def test_erb_pipeline_rejects_model_family_mismatched_profile() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(mission=mission, orbit=orbit)

    with pytest.raises(ValueError, match="model family"):
        calculate_erb_pipeline(
            config=config,
            erb_model=_erb_model(),
        )

def test_erb_pipeline_accepts_ost_erb_model() -> None:
    from radar.core.types import RadiationProductKind
    from radar.erb.igrf import IgrfCoefficients
    from radar.erb.model import OstErbModel

    mission = MissionConfig(launch_year=2027, lifetime_years=2)
    orbit = OrbitConfig.circular(altitude_km=2_000.0, inclination_deg=0.0)
    config = CalculationConfig(mission=mission, orbit=orbit)

    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=IgrfCoefficients(
            epoch=1985.0,
            g={(1, 0): -31_165.3},
            h={},
            nmax=1,
        ),
    )

    pipeline_result = calculate_erb_pipeline(
        config=config,
        erb_model=model,
    )

    assert len(pipeline_result.spectra) == 6
    assert tuple(product.kind for product in pipeline_result.products) == (
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
    )
    assert pipeline_result.calculation_result.component_status(ERB_MODEL_COMPONENT) is ComponentStatus.COMPLETED



def test_erb_pipeline_accepts_ae8_ap8_profile_model() -> None:
    from radar.core.types import RadiationProductKind
    from radar.erb.igrf import IgrfCoefficients
    from radar.erb.model import Ae8Ap8ErbModel

    model = Ae8Ap8ErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=IgrfCoefficients(
            epoch=1985.0,
            g={(1, 0): -31_165.3},
            h={},
            nmax=1,
        ),
    )

    pipeline_result = calculate_erb_pipeline(
        config=CalculationConfig(
            mission=MissionConfig(launch_year=2027, lifetime_years=2),
            orbit=OrbitConfig.circular(altitude_km=2_000.0, inclination_deg=0.0),
            methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_AE8_AP8_ERB),
        ),
        erb_model=model,
    )

    assert pipeline_result.erb_model_result.model == "ae8_ap8_radbelt_model"
    assert len(pipeline_result.spectra) == 6
    assert tuple(product.kind for product in pipeline_result.products) == (
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
    )
    assert pipeline_result.calculation_result.has_errors() is False
