from dataclasses import dataclass

import pytest

from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    OST_134_1044_2007_DOCUMENT,
    MethodologyProfile,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
)
from radar.core.result import ComponentStatus
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SolarActivityLevel,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.pipelines.sep import (
    SEP_LET_COMPONENT,
    SEP_LET_MODEL,
    SEP_LET_MODEL_VERSION,
    SEP_MODEL_COMPONENT,
    SEP_OUTPUT_TABLES_COMPONENT,
    SEP_PIPELINE_COMPONENT,
    SEP_SHIELDING_COMPONENT,
    SEP_SHIELDING_MODEL,
    SEP_SHIELDING_MODEL_VERSION,
    SepPipelineResult,
    calculate_sep_pipeline,
)
from radar.sep.model import SepModelInput, SepModelResult, StaticSepModel


def _config(
    *,
    profile: MethodologyProfile = MethodologyProfile.CUSTOM,
    lifetime_years: int = 2,
    thicknesses_g_cm2: tuple[float, ...] = (0.1,),
) -> CalculationConfig:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
        solar_activity_level=SolarActivityLevel.MEAN,
    )
    orbit = OrbitConfig.circular(altitude_km=2_000.0, inclination_deg=0.0)

    return CalculationConfig(
        mission=mission,
        orbit=orbit,
        shielding=ShieldingConfig(thicknesses_g_cm2=thicknesses_g_cm2),
        methodology=MethodologyConfig(profile=profile),
    )


def _sep_proton_spectrum(
    *,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    model: str = "static_sep_spectrum",
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model=model,
    )


def _sep_hze_spectrum(
    *,
    quantity: SpectrumQuantity,
    y_unit: Unit,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(4.0, 5.0, 6.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model=model,
    )


def _static_sep_model() -> StaticSepModel:
    return StaticSepModel(
        annual_fluence_spectrum=_sep_proton_spectrum(),
        model="static_sep_test",
        document="test_document",
    )


def _product(
    *,
    kind: RadiationProductKind,
    spectrum: Spectrum1D,
    label: str,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=spectrum,
        label=label,
    )


def _normative_products() -> tuple[SpectrumProduct, ...]:
    proton_mission = _sep_proton_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        model="ost_sep_test:proton:mission_fluence",
    )
    proton_peak = _sep_proton_spectrum(
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="ost_sep_test:proton:peak_flux",
    )
    proton_mean = _sep_proton_spectrum(
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="ost_sep_test:proton:mean_flux",
    )
    hze_mission = _sep_hze_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        model="ost_sep_test:hze:z26_fe:mission_fluence",
    )
    hze_peak = _sep_hze_spectrum(
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="ost_sep_test:hze:z26_fe:peak_flux",
    )
    hze_mean = _sep_hze_spectrum(
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="ost_sep_test:hze:z26_fe:mean_flux",
    )

    return (
        _product(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=proton_mission,
            label="SEP proton mission fluence",
        ),
        _product(
            kind=RadiationProductKind.PEAK_FLUX,
            spectrum=proton_peak,
            label="SEP proton peak flux",
        ),
        _product(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=proton_mean,
            label="SEP proton mean flux",
        ),
        _product(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=hze_mission,
            label="SEP Fe mission fluence",
        ),
        _product(
            kind=RadiationProductKind.PEAK_FLUX,
            spectrum=hze_peak,
            label="SEP Fe peak flux",
        ),
        _product(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=hze_mean,
            label="SEP Fe mean flux",
        ),
    )


@dataclass(frozen=True)
class _NormativeSepModel:
    model_family: SourceModelFamily = SourceModelFamily.OST_134_1044_2007

    @property
    def metadata(self) -> SourceModelMetadata:
        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=self.model_family,
            name="ost_sep_test",
            document=OST_134_1044_2007_DOCUMENT,
        )

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        products = _normative_products()
        mission_product = products[0]
        return SepModelResult(
            spectrum=mission_product.spectrum,
            lifetime_years=model_input.lifetime_years,
            exceedance_probability=model_input.exceedance_probability,
            model="ost_sep_test",
            document=OST_134_1044_2007_DOCUMENT,
            products=products,
        )


def test_sep_pipeline_returns_model_products_for_custom_profile() -> None:
    pipeline_result = calculate_sep_pipeline(
        config=_config(lifetime_years=2),
        sep_model=_static_sep_model(),
    )

    assert len(pipeline_result.spectra) == 1
    assert len(pipeline_result.products) == 1
    assert len(pipeline_result.on_orbit_products) == 1
    assert pipeline_result.shielded_products == ()
    assert pipeline_result.let_products == ()
    assert pipeline_result.sep_model_result.lifetime_years == 2
    assert pipeline_result.sep_model_result.model == "static_sep_test"
    assert pipeline_result.sep_model_result.document == "test_document"


def test_sep_pipeline_records_custom_component_statuses_and_tables() -> None:
    pipeline_result = calculate_sep_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    result = pipeline_result.calculation_result
    assert result.component_status(SEP_PIPELINE_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_MODEL_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_OUTPUT_TABLES_COMPONENT) is ComponentStatus.COMPLETED
    assert result.has_errors() is False

    assert len(result.output_tables) == 1
    assert result.output_tables[0].table_id == "sep_proton_mission_fluence_on_orbit"


def test_sep_pipeline_rejects_model_family_mismatched_profile() -> None:
    with pytest.raises(ValueError, match="model family"):
        calculate_sep_pipeline(
            config=_config(profile=MethodologyProfile.OST_134_1044_2007),
            sep_model=_static_sep_model(),
        )


def test_sep_pipeline_accepts_ost_sep_model_and_integrates_shielding_let() -> None:
    pipeline_result = calculate_sep_pipeline(
        config=_config(
            profile=MethodologyProfile.OST_134_1044_2007,
            thicknesses_g_cm2=(0.1,),
        ),
        sep_model=_NormativeSepModel(),
    )

    assert pipeline_result.calculation_result.has_errors() is False

    assert len(pipeline_result.source_products) == 6
    assert len(pipeline_result.on_orbit_products) == 6
    assert len(pipeline_result.shielded_products) == 6
    assert len(pipeline_result.let_products) == 3
    assert len(pipeline_result.products) == 15
    assert len(pipeline_result.spectra) == 15
    assert pipeline_result.products == (
        *pipeline_result.on_orbit_products,
        *pipeline_result.shielded_products,
        *pipeline_result.let_products,
    )

    assert tuple(product.kind for product in pipeline_result.let_products) == (
        RadiationProductKind.PEAK_LET_FLUX,
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MISSION_LET_FLUENCE,
    )

    result = pipeline_result.calculation_result
    assert result.component_status(SEP_SHIELDING_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_LET_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(SEP_OUTPUT_TABLES_COMPONENT) is ComponentStatus.COMPLETED

    output_table_ids = {
        table.table_id
        for table in result.output_tables
    }
    assert len(output_table_ids) == 21
    assert "sep_proton_mission_fluence_on_orbit" in output_table_ids
    assert "sep_hze_z26_peak_flux_on_orbit" in output_table_ids
    assert (
        "sep_t0_1_peak_flux_sep_combined_let_behind_al"
        in output_table_ids
    )
    assert (
        "sep_t0_1_mission_fluence_sep_proton_total_energy_behind_al"
        in output_table_ids
    )

    model_info_by_name = {
        info.name: info
        for info in result.model_info
    }
    assert model_info_by_name[SEP_SHIELDING_MODEL].version == SEP_SHIELDING_MODEL_VERSION
    assert model_info_by_name[SEP_LET_MODEL].version == SEP_LET_MODEL_VERSION


def test_sep_pipeline_result_rejects_empty_spectra() -> None:
    valid_result = calculate_sep_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    with pytest.raises(ValueError, match="at least one spectrum"):
        SepPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            spectra=(),
        )


def test_sep_pipeline_result_rejects_mismatched_products() -> None:
    valid_result = calculate_sep_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    with pytest.raises(ValueError, match="product spectra"):
        SepPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            spectra=valid_result.spectra,
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MISSION_FLUENCE,
                    spectrum=_sep_proton_spectrum(model="other"),
                ),
            ),
        )


def test_sep_pipeline_result_rejects_spectra_not_from_model_result() -> None:
    valid_result = calculate_sep_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    spectrum = valid_result.spectra[0]
    other_spectrum = Spectrum1D(
        x=spectrum.x,
        y=(20.0, 40.0, 60.0),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=spectrum.model,
    )

    with pytest.raises(ValueError, match="model result spectra"):
        SepPipelineResult(
            calculation_result=valid_result.calculation_result,
            sep_model_result=valid_result.sep_model_result,
            spectra=(other_spectrum,),
        )
