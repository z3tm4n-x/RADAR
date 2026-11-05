from dataclasses import replace

import pytest

from radar.core.log import LogLevel
from radar.core.profiles import MethodologyProfile
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
)
from radar.core.result import ComponentStatus
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumCompleteness, SpectrumQuantity
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

def test_sep_proton_pipeline_accepts_gost_sep_metadata_before_configuration() -> None:
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

    with pytest.raises(NotImplementedError, match="configured energy grid"):
        calculate_sep_proton_pipeline(
            config=config,
            sep_model=GostSepModel(),
            penetration=_penetration(),
        )


@pytest.mark.parametrize("source_incomplete", [False, True])
@pytest.mark.parametrize("operator_incomplete", [False, True])
def test_sep_proton_pipeline_preserves_source_and_operator_qualification(
    source_incomplete: bool, operator_incomplete: bool,
) -> None:
    source_reasons = ("Synthetic source support is incomplete.",) if source_incomplete else ()
    operator_reasons = ("Synthetic MLT qualification is unresolved.",) if operator_incomplete else ()
    spectrum = replace(
        _annual_sep_spectrum(),
        completeness=SpectrumCompleteness.INCOMPLETE if source_incomplete
        else SpectrumCompleteness.COMPLETE,
        incomplete_reasons=source_reasons,
    )
    model = StaticSepModel(spectrum, model="qualified_source", document="test_document")
    penetration = replace(
        _penetration(), model_version="qualified_access_v2", incomplete_reasons=operator_reasons,
    )
    output = calculate_sep_proton_pipeline(_config(), model, penetration)
    result = output.calculation_result
    reasons = (*source_reasons, *operator_reasons)
    expected = ComponentStatus.INCOMPLETE if reasons else ComponentStatus.COMPLETED
    assert result.component_status(SEP_PROTON_PIPELINE_COMPONENT) is expected
    assert result.component_status(GEOMAGNETIC_PENETRATION_COMPONENT) is expected
    assert result.component_status(SEP_MODEL_COMPONENT) is (
        ComponentStatus.INCOMPLETE if source_incomplete else ComponentStatus.COMPLETED
    )
    assert output.penetrated_spectrum.incomplete_reasons == reasons
    assert output.penetrated_spectrum.y == pytest.approx((0.5, 5.0, 13.5))
    assert output.raw_spectrum.y == pytest.approx((5.0, 10.0, 15.0))
    info = {entry.name: entry for entry in result.model_info}
    assert info[penetration.model].version == penetration.model_version
    assert info[penetration.model].status == ("diagnostic" if reasons else "applied")
    assert info[model.model].status == ("diagnostic" if source_reasons else "calculated")
    for entry in result.log.entries:
        if entry.stage not in (GEOMAGNETIC_PENETRATION_COMPONENT, SEP_PROTON_PIPELINE_COMPONENT):
            continue
        if entry.message.endswith("started."):
            continue
        assert entry.level is (LogLevel.WARNING if reasons else LogLevel.INFO)
        details = dict(entry.details)
        if reasons:
            assert details["qualification_reasons"] == "; ".join(reasons)
            assert "completed" not in entry.message.lower()
        else:
            assert "qualification_reasons" not in details


def test_sep_proton_pipeline_does_not_promote_incomplete_flag_without_reason() -> None:
    source = replace(_annual_sep_spectrum(), completeness=SpectrumCompleteness.INCOMPLETE)
    output = calculate_sep_proton_pipeline(
        _config(), StaticSepModel(source), _penetration(),
    )
    assert output.calculation_result.component_status(SEP_MODEL_COMPONENT) is ComponentStatus.INCOMPLETE
    assert (
        output.calculation_result.component_status(SEP_PROTON_PIPELINE_COMPONENT)
        is ComponentStatus.INCOMPLETE
    )
    final_details = dict(output.calculation_result.log.entries[-1].details)
    assert "without a supplied qualification reason" in final_details["qualification_reasons"]


def test_sep_proton_pipeline_merges_duplicate_source_operator_reasons() -> None:
    reason = "Shared unresolved qualification."
    source = replace(
        _annual_sep_spectrum(), completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=(reason,),
    )
    output = calculate_sep_proton_pipeline(
        _config(), StaticSepModel(source), replace(_penetration(), incomplete_reasons=(reason,)),
    )
    assert output.penetrated_spectrum.incomplete_reasons == (reason,)
    final_details = dict(output.calculation_result.log.entries[-1].details)
    assert final_details["qualification_reasons"] == reason
