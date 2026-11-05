from types import SimpleNamespace

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
from radar.core.spectra import (
    ModelReference,
    ParticleSpecies,
    Spectrum1D,
    TransformationStep,
)
from radar.core.spectrum_ops import mark_spectrum_incomplete
from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    DoseQuantity,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SolarActivityLevel,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.gcr.model import OstGcrModel, StaticGcrModel
from radar.gcr.source_spectra import OST_NEK_HE_BC_INCOMPLETE_REASON
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


def _typed_mode_gcr_spectrum(
    *,
    kind: RadiationProductKind,
    incomplete: bool,
) -> Spectrum1D:
    cumulative = kind is RadiationProductKind.MISSION_FLUENCE
    spectrum = Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(3.0, 2.0, 1.0),
        x_unit=Unit.MEV,
        y_unit=(
            Unit.DIFFERENTIAL_FLUENCE
            if cumulative
            else Unit.DIFFERENTIAL_FLUX
        ),
        quantity=(
            SpectrumQuantity.DIFFERENTIAL_FLUENCE
            if cumulative
            else SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
        ),
        particle=Particle.PROTON,
        source=RadiationSource.GCR,
        model="mode_selective_gcr",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=(
            TemporalBasis.MISSION_TOTAL
            if cumulative
            else TemporalBasis.SOURCE_MAXIMUM
        ),
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies(symbol="p", atomic_number=1, mass_number=1),
        assessment_mode=(
            AssessmentMode.CUMULATIVE if cumulative else AssessmentMode.PEAK
        ),
        mission_duration_s=7.0 * 365.25 * 86400.0 if cumulative else None,
        model_reference=ModelReference(
            name="mode_selective_gcr",
            version="test-v1",
            document_refs=("test normative document",),
            dataset_refs=("test/mode_selective_gcr.csv@sha256:test",),
        ),
        transformation_chain=(
            TransformationStep(name="source_model", version="test-v1"),
        ),
        completeness=SpectrumCompleteness.COMPLETE,
    )
    if incomplete:
        return mark_spectrum_incomplete(
            spectrum,
            f"{kind.value}_test_incomplete",
        )
    return spectrum


def _mode_selective_gcr_model(
    *,
    incomplete_kind: RadiationProductKind,
) -> StaticGcrModel:
    return StaticGcrModel(
        spectra=tuple(
            _typed_mode_gcr_spectrum(
                kind=kind,
                incomplete=kind is incomplete_kind,
            )
            for kind in (
                RadiationProductKind.MISSION_FLUENCE,
                RadiationProductKind.MAXIMUM_FLUX,
            )
        ),
        model="mode_selective_gcr",
        document="test normative document",
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

    assert len(pipeline_result.source_products) == 12
    assert len(pipeline_result.on_orbit_products) == 12
    assert len(pipeline_result.shielded_products) == 2
    assert len(pipeline_result.let_products) == 1
    assert len(pipeline_result.products) == 15
    assert pipeline_result.products == (
        *pipeline_result.on_orbit_products,
        *pipeline_result.shielded_products,
        *pipeline_result.let_products,
    )

    assert tuple(product.kind for product in pipeline_result.let_products) == (
        RadiationProductKind.MISSION_LET_FLUENCE,
    )


    assert {
        product.kind for product in pipeline_result.shielded_products
    } == {RadiationProductKind.MISSION_FLUENCE}

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
    ) is ComponentStatus.INCOMPLETE
    assert pipeline_result.calculation_result.component_status(
        GCR_SHIELDING_COMPONENT
    ) is ComponentStatus.INCOMPLETE
    assert pipeline_result.calculation_result.component_status(
        GCR_LET_COMPONENT
    ) is ComponentStatus.INCOMPLETE
    assert pipeline_result.calculation_result.component_status(
        GCR_MODEL_COMPONENT
    ) is ComponentStatus.INCOMPLETE
    assert pipeline_result.calculation_result.component_status(
        GCR_PIPELINE_COMPONENT
    ) is ComponentStatus.INCOMPLETE
    assert pipeline_result.calculation_result.component_status(
        GCR_OUTPUT_TABLES_COMPONENT
    ) is ComponentStatus.COMPLETED

    assert pipeline_result.source_spectra is not None
    output_table_ids = {
        table.table_id
        for table in pipeline_result.calculation_result.output_tables
    }
    assert len(output_table_ids) == 8
    assert "gcr_gost_h_total_source_outside_magnetosphere" not in output_table_ids
    assert "gcr_gost_fe_total_source_outside_magnetosphere" not in output_table_ids
    assert "gcr_h_mean_flux_on_orbit" not in output_table_ids
    assert "gcr_h_mean_integral_flux_on_orbit" not in output_table_ids
    assert "gcr_fe_mission_fluence_on_orbit" in output_table_ids
    assert "gcr_fe_integral_fluence_on_orbit" not in output_table_ids
    assert "gcr_t0_01_mean_flux_gcr_combined_let_behind_al" not in output_table_ids
    assert "gcr_t0_01_mission_fluence_gcr_combined_let_behind_al" in output_table_ids
    assert (
        "gcr_t0_01_mission_fluence_gcr_hze_aggregate_let_behind_al"
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


@pytest.mark.parametrize("mode", list(AssessmentMode))
def test_gcr_pipeline_geometric_qualification_blocks_each_selected_mode(
    mode: AssessmentMode,
) -> None:
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2027, lifetime_years=1),
        orbit=OrbitConfig.circular(altitude_km=35_786.0, inclination_deg=0.0),
        shielding=ShieldingConfig(thicknesses_g_cm2=(0.01,)),
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_134_1044_2007),
        assessment_mode=mode,
        dose_quantity=(DoseQuantity.DOSE_RATE if mode is AssessmentMode.PEAK
                       else DoseQuantity.ACCUMULATED_DOSE),
        dose_unit=Unit.RAD_PER_SECOND if mode is AssessmentMode.PEAK else Unit.RAD,
    )
    result = calculate_gcr_pipeline(
        config=config, gcr_model=OstGcrModel(symbols=("H",)),
    )
    assert all(
        not any("shadow qualification unresolved" in reason
                for reason in product.spectrum.incomplete_reasons)
        for product in result.source_products
    )
    assert all(
        any("shadow qualification unresolved" in reason
            for reason in product.spectrum.incomplete_reasons)
        for product in result.on_orbit_products
    )
    assert all(
        any("representative of the mission" in reason
            for reason in product.spectrum.incomplete_reasons)
        for product in result.on_orbit_products
    )
    with pytest.raises(ValueError, match="shadow qualification unresolved"):
        result.production_release_products()
    with pytest.raises(ValueError, match="representative of the mission"):
        result.production_release_products()
    assert all(any("local-time convention" in reason
                   for reason in product.spectrum.incomplete_reasons)
               for product in result.on_orbit_products)
    with pytest.raises(ValueError, match="local-time convention"):
        result.production_release_products()


def test_gcr_pipeline_propagates_pending_helium_bc_to_all_stage_statuses() -> None:
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2027, lifetime_years=1),
        orbit=OrbitConfig.circular(altitude_km=35_786.0, inclination_deg=0.0),
        shielding=ShieldingConfig(thicknesses_g_cm2=(0.01,)),
        methodology=MethodologyConfig(
            profile=MethodologyProfile.OST_134_1044_2007
        ),
    )
    pipeline_result = calculate_gcr_pipeline(
        config=config,
        gcr_model=OstGcrModel(symbols=("He",)),
    )
    result = pipeline_result.calculation_result

    assert any(
        OST_NEK_HE_BC_INCOMPLETE_REASON in product.spectrum.incomplete_reasons
        for product in pipeline_result.source_products
    )
    for component in (
        GCR_MODEL_COMPONENT,
        GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
        GCR_SHIELDING_COMPONENT,
        GCR_LET_COMPONENT,
        GCR_PIPELINE_COMPONENT,
    ):
        assert result.component_status(component) is ComponentStatus.INCOMPLETE
    with pytest.raises(ValueError, match="GCR pipeline is incomplete"):
        pipeline_result.production_release_products()


def test_gcr_unselected_incomplete_mode_does_not_block_cumulative_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "radar.data.integrity.require_release_eligible_normative_resource",
        lambda _path: SimpleNamespace(sha256="test"),
    )
    pipeline_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_mode_selective_gcr_model(
            incomplete_kind=RadiationProductKind.MAXIMUM_FLUX,
        ),
    )

    result = pipeline_result.calculation_result
    assert result.component_status(GCR_MODEL_COMPONENT) is ComponentStatus.COMPLETED
    assert result.component_status(GCR_PIPELINE_COMPONENT) is ComponentStatus.COMPLETED
    released = pipeline_result.production_release_products()
    assert tuple(product.kind for product in released) == (
        RadiationProductKind.MISSION_FLUENCE,
    )


def test_gcr_selected_incomplete_mode_blocks_cumulative_release() -> None:
    pipeline_result = calculate_gcr_pipeline(
        config=_config(),
        gcr_model=_mode_selective_gcr_model(
            incomplete_kind=RadiationProductKind.MISSION_FLUENCE,
        ),
    )

    result = pipeline_result.calculation_result
    assert result.component_status(GCR_MODEL_COMPONENT) is ComponentStatus.INCOMPLETE
    assert result.component_status(GCR_PIPELINE_COMPONENT) is ComponentStatus.INCOMPLETE
    with pytest.raises(ValueError, match="mission_fluence_test_incomplete"):
        pipeline_result.production_release_products()

