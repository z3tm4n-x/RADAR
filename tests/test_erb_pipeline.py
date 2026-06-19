import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.result import ComponentStatus
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.erb.model import StaticErbModel
from radar.pipelines.erb import (
    ERB_MODEL_COMPONENT,
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

    return CalculationConfig(mission=mission, orbit=orbit, kp=kp)


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
    assert result.has_errors() is False


def test_erb_pipeline_records_log_entries() -> None:
    pipeline_result = calculate_erb_pipeline(
        config=_config(),
        erb_model=_erb_model(),
    )

    log_entries = pipeline_result.calculation_result.log.entries

    assert len(log_entries) == 3
    assert log_entries[0].stage == ERB_PIPELINE_COMPONENT
    assert log_entries[1].stage == ERB_MODEL_COMPONENT
    assert log_entries[2].stage == ERB_PIPELINE_COMPONENT
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
