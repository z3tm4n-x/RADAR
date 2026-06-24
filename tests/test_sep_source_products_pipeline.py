import pytest

from radar.core.products import SpectrumProduct
from radar.core.profiles import MethodologyProfile
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
)
from radar.core.result import ComponentStatus
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL,
    build_ost_penetration_function_for_config,
)
from radar.geomagnetic.rigidity import RigidityGrid
from radar.geomagnetic.spectrum import apply_proton_penetration
from radar.physics.rigidity import proton_kinetic_energy_to_rigidity_gv
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.sep.model import OstSepModel, SepModelInput, StaticSepModel
from radar.sep.proton_spectrum import (
    SepProtonCoefficientName,
    SepProtonCoefficientRecord,
    SepProtonSpectrumProduct,
)
from radar.pipelines.sep_source_products import (
    SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_GEOMAGNETIC_PENETRATION_DOCUMENT,
    SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    SEP_SOURCE_MODEL_COMPONENT,
    SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
    SepSourceProductsPipelineResult,
    calculate_sep_source_products_pipeline,
)


def _config(
    profile: MethodologyProfile = MethodologyProfile.CUSTOM,
    *,
    lifetime_years: int = 5,
    probability: float = 0.5,
) -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(
            launch_year=2027,
            lifetime_years=lifetime_years,
            sep_exceedance_probability=probability,
        ),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        methodology=MethodologyConfig(profile=profile),
    )


def _sep_proton_fluence_spectrum(
    *,
    model: str = "annual_sep_test",
    y: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 20.0, 30.0),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model=model,
    )


def _static_sep_model() -> StaticSepModel:
    return StaticSepModel(
        annual_fluence_spectrum=_sep_proton_fluence_spectrum(),
        model="static_sep_test",
        document="test_document",
    )


def _simple_ost_sep_proton_coefficient_records() -> tuple[SepProtonCoefficientRecord, ...]:
    records: list[SepProtonCoefficientRecord] = []

    values = {
        SepProtonSpectrumProduct.FLUENCE: {
            SepProtonCoefficientName.LOG10_C: 1.0,
            SepProtonCoefficientName.BREAK_ENERGY_MEV: 10.0,
            SepProtonCoefficientName.GAMMA1: 1.0,
            SepProtonCoefficientName.GAMMA2: 1.0,
        },
        SepProtonSpectrumProduct.PEAK_FLUX: {
            SepProtonCoefficientName.LOG10_C: 0.0,
            SepProtonCoefficientName.BREAK_ENERGY_MEV: 10.0,
            SepProtonCoefficientName.GAMMA1: 1.0,
            SepProtonCoefficientName.GAMMA2: 1.0,
        },
    }

    for product, parameters in values.items():
        for parameter, value in parameters.items():
            records.append(
                SepProtonCoefficientRecord(
                    model="ost_134_1044_2007",
                    product=product,
                    parameter=parameter,
                    event_count=2,
                    probability=0.5,
                    value=value,
                    source_table="test",
                )
            )

    return tuple(records)



def _simple_hze_ion_records():
    from radar.sep.hze_spectrum import OstSepHzeIonRecord

    return (
        OstSepHzeIonRecord(
            model="ost_134_1044_2007",
            z=2,
            symbol="He",
            mass_number=4.0,
            mass_to_charge=2.0,
            relative_abundance=1.23e-2,
            source_table="B.9",
        ),
    )


def _ost_sep_model() -> OstSepModel:
    return OstSepModel(
        energy_grid_mev=(10.0,),
        monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0135),) * 60,
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
        version="protons_only_v1",
    )


def test_sep_source_products_pipeline_calculates_static_model_products() -> None:
    pipeline_result = calculate_sep_source_products_pipeline(
        config=_config(lifetime_years=5),
        sep_model=_static_sep_model(),
    )

    assert pipeline_result.sep_model_result.spectrum.y == pytest.approx(
        (5.0, 10.0, 15.0)
    )
    assert pipeline_result.products == pipeline_result.sep_model_result.products
    assert len(pipeline_result.products) == 1
    assert pipeline_result.products[0].kind is RadiationProductKind.MISSION_FLUENCE


def test_sep_source_products_pipeline_records_component_statuses() -> None:
    pipeline_result = calculate_sep_source_products_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    result = pipeline_result.calculation_result

    assert result.component_status(SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT) is (
        ComponentStatus.COMPLETED
    )
    assert result.component_status(SEP_SOURCE_MODEL_COMPONENT) is ComponentStatus.COMPLETED
    assert result.has_errors() is False


def test_sep_source_products_pipeline_records_log_entries() -> None:
    pipeline_result = calculate_sep_source_products_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    log_entries = pipeline_result.calculation_result.log.entries

    assert len(log_entries) == 3
    assert log_entries[0].stage == SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT
    assert log_entries[1].stage == SEP_SOURCE_MODEL_COMPONENT
    assert log_entries[2].stage == SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT


def test_sep_source_products_pipeline_records_model_info() -> None:
    pipeline_result = calculate_sep_source_products_pipeline(
        config=_config(),
        sep_model=_static_sep_model(),
    )

    model_info = pipeline_result.calculation_result.model_info

    assert len(model_info) == 1
    assert model_info[0].name == "static_sep_test"
    assert model_info[0].version == "unversioned"
    assert model_info[0].source == "test_document"


def test_sep_source_products_pipeline_applies_ost_proton_penetration() -> None:
    pipeline_result = calculate_sep_source_products_pipeline(
        config=_config(
            profile=MethodologyProfile.OST_134_1044_2007,
            lifetime_years=5,
            probability=0.5,
        ),
        sep_model=_ost_sep_model(),
    )

    products_by_kind = {
        product.kind: product
        for product in pipeline_result.products
    }

    assert set(products_by_kind) == {
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.PEAK_FLUX,
        RadiationProductKind.MEAN_FLUX,
    }

    test_config = _config(
        profile=MethodologyProfile.OST_134_1044_2007,
        lifetime_years=5,
        probability=0.5,
    )
    rigidity_grid = RigidityGrid(
        values_gv=(proton_kinetic_energy_to_rigidity_gv(10.0),)
    )
    penetration = build_ost_penetration_function_for_config(
        test_config,
        rigidity_grid=rigidity_grid,
    )

    raw_products_by_kind = {
        product.kind: product
        for product in pipeline_result.sep_model_result.products
    }

    for kind, product in products_by_kind.items():
        expected_spectrum = apply_proton_penetration(
            spectrum=raw_products_by_kind[kind].spectrum,
            penetration=penetration,
        )
        assert product.spectrum.y == pytest.approx(expected_spectrum.y)
        assert product.spectrum.model == expected_spectrum.model

    assert (
        products_by_kind[RadiationProductKind.MEAN_FLUX].spectrum.quantity
        is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    )

    assert pipeline_result.calculation_result.component_status(
        SEP_GEOMAGNETIC_PENETRATION_COMPONENT
    ) is ComponentStatus.COMPLETED

    completed_entries = [
        entry
        for entry in pipeline_result.calculation_result.log.entries
        if entry.stage == SEP_GEOMAGNETIC_PENETRATION_COMPONENT
        and entry.message == "SEP geomagnetic penetration completed."
    ]
    assert len(completed_entries) == 1
    completed_details = dict(completed_entries[0].details)
    assert completed_details["penetration_model"] == OST_GEOMAGNETIC_PENETRATION_MODEL
    assert completed_details["penetration_version"] == (
        SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION
    )
    assert completed_details["rigidity_grid_points"] == "1"

    assert pipeline_result.source_products == pipeline_result.sep_model_result.products

    model_info_by_name = {
        info.name: info
        for info in pipeline_result.calculation_result.model_info
    }
    assert model_info_by_name["ost_sep_model"].version == "protons_only_v1"

    penetration_info = model_info_by_name[OST_GEOMAGNETIC_PENETRATION_MODEL]
    assert penetration_info.version == SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION
    assert penetration_info.source == SEP_GEOMAGNETIC_PENETRATION_DOCUMENT


def test_sep_source_products_pipeline_rejects_profile_model_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        calculate_sep_source_products_pipeline(
            config=_config(profile=MethodologyProfile.OST_134_1044_2007),
            sep_model=_static_sep_model(),
        )


def test_sep_source_products_pipeline_result_rejects_products_not_from_model_result() -> None:
    sep_model_result = _static_sep_model().calculate(
        SepModelInput(mission=_config().mission)
    )
    other_product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=_sep_proton_fluence_spectrum(model="other"),
    )

    with pytest.raises(ValueError, match="must identify their source model"):
        SepSourceProductsPipelineResult(
            calculation_result=calculate_sep_source_products_pipeline(
                config=_config(),
                sep_model=_static_sep_model(),
            ).calculation_result,
            sep_model_result=sep_model_result,
            products=(other_product,),
        )



def test_sep_source_products_pipeline_applies_ost_hze_penetration_when_configured() -> None:
    model = OstSepModel(
        energy_grid_mev=(10.0,),
        hze_energy_grid_mev_per_nucleon=(5.0,),
        monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0135),) * 60,
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
        hze_ion_records=_simple_hze_ion_records(),
        version="protons_hze_v1",
    )

    pipeline_result = calculate_sep_source_products_pipeline(
        config=_config(
            profile=MethodologyProfile.OST_134_1044_2007,
            lifetime_years=5,
            probability=0.5,
        ),
        sep_model=model,
    )

    raw_hze_products = tuple(
        product
        for product in pipeline_result.sep_model_result.products
        if product.spectrum.particle is Particle.HZE
    )
    penetrated_hze_products = tuple(
        product
        for product in pipeline_result.products
        if product.spectrum.particle is Particle.HZE
    )

    assert len(raw_hze_products) == 3
    assert len(penetrated_hze_products) == 3

    for raw_product, penetrated_product in zip(
        raw_hze_products,
        penetrated_hze_products,
        strict=True,
    ):
        assert penetrated_product.kind is raw_product.kind
        assert penetrated_product.spectrum.model.startswith(
            f"{raw_product.spectrum.model}+"
        )
        assert penetrated_product.spectrum.x == raw_product.spectrum.x
        assert penetrated_product.spectrum.particle is Particle.HZE
        assert all(
            penetrated_value <= raw_value
            for raw_value, penetrated_value in zip(
                raw_product.spectrum.y,
                penetrated_product.spectrum.y,
                strict=True,
            )
        )
