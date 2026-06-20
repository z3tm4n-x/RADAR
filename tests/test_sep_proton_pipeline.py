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
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid
from radar.physics.rigidity import proton_rigidity_to_kinetic_energy_mev
from radar.pipelines.sep_protons import (
    GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_MODEL_COMPONENT,
    SEP_PROTON_PIPELINE_COMPONENT,
    SepProtonPipelineResult,
    calculate_sep_proton_pipeline,
)
from radar.sep.model import StaticSepModel


def _config(lifetime_years: int = 5) -> CalculationConfig:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
        sep_exceedance_probability=0.1,
    )
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    return CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.CUSTOM),
    )


def _annual_sep_spectrum() -> Spectrum1D:
    rigidities_gv = (1.0, 2.0, 3.0)

    return Spectrum1D(
        x=tuple(
            proton_rigidity_to_kinetic_energy_mev(rigidity)
            for rigidity in rigidities_gv
        ),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="annual_sep_test",
    )


def _sep_model() -> StaticSepModel:
    return StaticSepModel(
        annual_fluence_spectrum=_annual_sep_spectrum(),
        model="static_sep_test",
        document="test_document",
    )


def _penetration() -> PenetrationFunction:
    return PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
        values=(0.1, 0.5, 0.9),
        model="test_penetration",
    )


def test_sep_proton_pipeline_calculates_raw_and_penetrated_spectra() -> None:
    pipeline_result = calculate_sep_proton_pipeline(
        config=_config(lifetime_years=5),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    assert pipeline_result.raw_spectrum.y == pytest.approx((5.0, 10.0, 15.0))
    assert pipeline_result.penetrated_spectrum.y == pytest.approx((0.5, 5.0, 13.5))
    assert pipeline_result.sep_model_result.lifetime_years == 5
    assert pipeline_result.sep_model_result.exceedance_probability == pytest.approx(0.1)


def test_sep_proton_pipeline_records_component_statuses() -> None:
    pipeline_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    result = pipeline_result.calculation_result

    assert result.component_status(SEP_PROTON_PIPELINE_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_MODEL_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(GEOMAGNETIC_PENETRATION_COMPONENT) is ComponentStatus.COMPLETED
    assert result.has_errors() is False


def test_sep_proton_pipeline_records_log_entries() -> None:
    pipeline_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    log_entries = pipeline_result.calculation_result.log.entries

    assert len(log_entries) == 4
    assert log_entries[0].stage == SEP_PROTON_PIPELINE_COMPONENT
    assert log_entries[1].stage == SEP_MODEL_COMPONENT
    assert log_entries[2].stage == GEOMAGNETIC_PENETRATION_COMPONENT
    assert log_entries[3].stage == SEP_PROTON_PIPELINE_COMPONENT


def test_sep_proton_pipeline_records_model_info() -> None:
    pipeline_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    model_names = {
        model_info.name
        for model_info in pipeline_result.calculation_result.model_info
    }

    assert model_names == {"static_sep_test", "test_penetration"}


def test_sep_proton_pipeline_result_validates_spectra() -> None:
    valid_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    bad_spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.ELECTRON,
        source=RadiationSource.SEP,
        model="bad",
    )

    with pytest.raises(ValueError, match="protons"):
        SepProtonPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            raw_spectrum=bad_spectrum,
            penetrated_spectrum=valid_result.penetrated_spectrum,
        )

def test_sep_proton_pipeline_exposes_raw_and_penetrated_products() -> None:
    from radar.core.types import RadiationProductKind

    pipeline_result = calculate_sep_proton_pipeline(
        config=_config(lifetime_years=5),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    assert pipeline_result.raw_product is not None
    assert pipeline_result.penetrated_product is not None
    assert pipeline_result.raw_product.kind is RadiationProductKind.MISSION_FLUENCE
    assert pipeline_result.penetrated_product.kind is RadiationProductKind.MISSION_FLUENCE
    assert pipeline_result.raw_product.spectrum == pipeline_result.raw_spectrum
    assert (
        pipeline_result.penetrated_product.spectrum
        == pipeline_result.penetrated_spectrum
    )


def test_sep_proton_pipeline_result_rejects_mismatched_product() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    valid_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    with pytest.raises(ValueError, match="penetrated product spectrum"):
        SepProtonPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            raw_spectrum=valid_result.raw_spectrum,
            penetrated_spectrum=valid_result.penetrated_spectrum,
            raw_product=valid_result.raw_product,
            penetrated_product=SpectrumProduct(
                kind=RadiationProductKind.MISSION_FLUENCE,
                spectrum=valid_result.raw_spectrum,
            ),
        )

def test_sep_proton_pipeline_result_rejects_raw_spectrum_not_from_model_result() -> None:
    valid_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    raw_spectrum = valid_result.raw_spectrum
    other_raw_spectrum = Spectrum1D(
        x=raw_spectrum.x,
        y=(2.0, 4.0, 6.0),
        x_unit=raw_spectrum.x_unit,
        y_unit=raw_spectrum.y_unit,
        quantity=raw_spectrum.quantity,
        particle=raw_spectrum.particle,
        source=raw_spectrum.source,
        model=raw_spectrum.model,
    )

    with pytest.raises(ValueError, match="raw pipeline spectrum"):
        SepProtonPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            raw_spectrum=other_raw_spectrum,
            penetrated_spectrum=valid_result.penetrated_spectrum,
        )


def test_sep_proton_pipeline_result_rejects_product_with_wrong_source() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    valid_result = calculate_sep_proton_pipeline(
        config=_config(),
        sep_model=_sep_model(),
        penetration=_penetration(),
    )

    penetrated_spectrum = valid_result.penetrated_spectrum
    wrong_source_spectrum = Spectrum1D(
        x=penetrated_spectrum.x,
        y=penetrated_spectrum.y,
        x_unit=penetrated_spectrum.x_unit,
        y_unit=penetrated_spectrum.y_unit,
        quantity=penetrated_spectrum.quantity,
        particle=penetrated_spectrum.particle,
        source=RadiationSource.GCR,
        model=penetrated_spectrum.model,
    )

    with pytest.raises(ValueError, match="spectrum source"):
        SepProtonPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            raw_spectrum=valid_result.raw_spectrum,
            penetrated_spectrum=valid_result.penetrated_spectrum,
            raw_product=valid_result.raw_product,
            penetrated_product=SpectrumProduct(
                kind=RadiationProductKind.MISSION_FLUENCE,
                spectrum=wrong_source_spectrum,
            ),
        )

def test_sep_proton_pipeline_rejects_model_family_mismatched_profile() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=5)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(mission=mission, orbit=orbit)

    with pytest.raises(ValueError, match="model family"):
        calculate_sep_proton_pipeline(
            config=config,
            sep_model=_sep_model(),
            penetration=_penetration(),
        )

def test_sep_proton_pipeline_accepts_gost_sep_stub_metadata_before_calculation() -> None:
    from radar.core.profiles import MethodologyProfile
    from radar.core.project import MethodologyConfig
    from radar.sep.model import GostSepModel

    mission = MissionConfig(launch_year=2027, lifetime_years=5)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP),
    )

    with pytest.raises(NotImplementedError, match="GOST SEP"):
        calculate_sep_proton_pipeline(
            config=config,
            sep_model=GostSepModel(),
            penetration=_penetration(),
        )
