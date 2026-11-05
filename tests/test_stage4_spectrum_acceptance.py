from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import sqrt

import pytest

from radar.core.constants import (
    DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
    GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
)
from radar.core.products import SpectrumProduct, select_products_for_assessment_mode
from radar.core.project import (
    CalculationConfig,
    MissionConfig,
    ShieldingConfig,
)
from radar.core.spectrum_ops import (
    integrate_differential_spectrum_finite_upper_bound,
)
from radar.core.result import ComponentStatus
from radar.core.types import (
    AssessmentMode,
    DoseQuantity,
    LetSpectrumRole,
    Particle,
    RadiationProductKind,
    RadiationQuantityMode,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.erb.model import OstErbModel
from radar.gcr.model import GcrModelInput, OstGcrModel
from radar.gcr.pipeline_shielding import (
    calculate_gcr_shielding_let_products_for_thickness,
)
from radar.pipelines.erb import ErbPipelineResult, calculate_erb_pipeline
from radar.pipelines.execution import build_pipeline_source_models
from radar.pipelines.gcr import GcrPipelineResult, calculate_gcr_pipeline
from radar.pipelines.sep import SepPipelineResult, calculate_sep_pipeline
from radar.sep.model import OstSepModel, SepModelInput
from radar.sep.pipeline_shielding import (
    calculate_sep_shielding_let_products_for_thickness,
)
from radar.shielding.resources import load_normative_shielding_tables
from radar.solar_activity.model import build_calculation_solar_activity
from radar.validation.stage4_spectra import (
    STAGE4_REFERENCE_SCENARIOS as REFERENCE_SPECTRUM_SCENARIOS,
    STAGE4_REFERENCE_SHIELD_THICKNESSES_G_CM2 as REFERENCE_SHIELD_THICKNESSES_G_CM2,
    build_stage4_reference_config as _reference_config,
)


MISSION_DURATION_S = 5.0 * 365.25 * 24.0 * 60.0 * 60.0


@pytest.mark.parametrize("source", (RadiationSource.SEP, RadiationSource.GCR))
def test_stage4_real_peak_chain_preserves_rate_semantics_and_quarantine(
    source: RadiationSource,
) -> None:
    """Small real OST chain: a contract regression, never a numerical gold case.

    Three energy nodes and H/He deliberately leave the domain/roster incomplete.
    Only the selected PEAK products traverse one positive aluminium shield.
    """

    geo = next(item for item in REFERENCE_SPECTRUM_SCENARIOS if item.name == "GEO")
    config = replace(
        _reference_config(geo, AssessmentMode.PEAK),
        shielding=ShieldingConfig(thicknesses_g_cm2=(0.01,)),
    )
    grid = (10.0, 30.0, 100.0)
    result: SepPipelineResult | GcrPipelineResult
    if source is RadiationSource.SEP:
        # Keep the expected event count fixed at N=2 to isolate time units from
        # the legitimate mission-duration dependence of SEP event statistics.
        model = OstSepModel(
            energy_grid_mev=grid,
            hze_energy_grid_mev_per_nucleon=grid,
            monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0135),) * 60,
        )
        model = replace(model, hze_ion_records=tuple(
            ion for ion in model._hze_ion_records() if ion.z == 2
        ))
        result = calculate_sep_pipeline(config=config, sep_model=model)
        source_products = select_products_for_assessment_mode(
            result.sep_model_result.products, AssessmentMode.PEAK,
        )
        one_year_model = replace(
            model, monthly_smoothed_wolf_numbers=(2.0 / (12.0 * 0.0135),) * 12,
        )
        one_year_products = select_products_for_assessment_mode(
            one_year_model.calculate(SepModelInput(
                mission=replace(config.mission, lifetime_years=1),
            )).products,
            AssessmentMode.PEAK,
        )
        for five_year, one_year in zip(source_products, one_year_products, strict=True):
            assert five_year.spectrum.y == pytest.approx(one_year.spectrum.y)
        expected_kind = RadiationProductKind.PEAK_FLUX
        expected_let_kind = RadiationProductKind.PEAK_LET_FLUX
        temporal_basis = TemporalBasis.SEP_PEAK
    else:
        solar = build_calculation_solar_activity(config)
        gcr_model = OstGcrModel(
            energy_grid_mev_per_nucleon=grid, symbols=("H", "He"), solar_activity=solar,
        )
        result = calculate_gcr_pipeline(config=config, gcr_model=gcr_model)
        source_products = select_products_for_assessment_mode(
            result.gcr_model_result.products, AssessmentMode.PEAK,
        )
        # Independently assert a pointwise maximum of monthly rates, with no
        # monthly or mission-duration factor in the reference arithmetic.
        monthly = tuple(gcr_model._source_spectra_for_month(solar, month)
                        for month in range(solar.mission_month_count))
        for product in source_products:
            species = product.spectrum.species
            assert species is not None
            expected = tuple(max(item.ion(species.symbol).total.y[index] for item in monthly)
                             for index in range(len(grid)))
            assert product.spectrum.y == pytest.approx(expected)
        expected_kind = RadiationProductKind.MAXIMUM_FLUX
        expected_let_kind = RadiationProductKind.MAXIMUM_LET_FLUX
        temporal_basis = TemporalBasis.SOURCE_MAXIMUM

    on_orbit = select_products_for_assessment_mode(result.on_orbit_products, AssessmentMode.PEAK)
    assert len(source_products) == len(on_orbit) == len(result.shielded_products) == 2
    assert len(result.shielding_let_by_thickness) == 1
    assert len(result.let_products) == 1
    canonical = result.let_products[0]
    assert canonical.kind is expected_let_kind
    assert canonical.spectrum.let_role is LetSpectrumRole.HZE_DIRECT_ION
    assert canonical.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert any(value > 0.0 for value in canonical.spectrum.y)

    for products, location in (
        (source_products, SpectrumLocation.OUTSIDE_MAGNETOSPHERE),
        (on_orbit, SpectrumLocation.ON_ORBIT),
        (result.shielded_products, SpectrumLocation.BEHIND_SHIELD),
        (result.let_products, SpectrumLocation.BEHIND_SHIELD),
    ):
        for product in products:
            spectrum = product.spectrum
            product.validate_calculation_contract()
            assert product.kind in (expected_kind, expected_let_kind)
            assert product.radiation_quantity_mode is RadiationQuantityMode.FLUX
            assert spectrum.assessment_mode is AssessmentMode.PEAK
            assert spectrum.temporal_basis is temporal_basis
            assert spectrum.mission_duration_s is None
            assert spectrum.location is location
            assert spectrum.y_unit in (
                Unit.DIFFERENTIAL_FLUX, Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,
                Unit.DIFFERENTIAL_LET_FLUX,
            )
            assert spectrum.completeness is SpectrumCompleteness.INCOMPLETE
            assert any("ion roster incomplete" in reason for reason in spectrum.incomplete_reasons)

    # The pipeline must feed rates directly to transport. This public low-level
    # operation takes no mission duration and provides a numerical wiring check.
    transport = (
        calculate_sep_shielding_let_products_for_thickness
        if source is RadiationSource.SEP
        else calculate_gcr_shielding_let_products_for_thickness
    )
    direct = transport(products=on_orbit, tables=load_normative_shielding_tables(),
                       thickness_g_cm2=0.01)
    for actual, expected in zip(result.shielded_products, direct.shielded_products, strict=True):
        assert actual.spectrum.y == pytest.approx(expected.spectrum.y)
    assert canonical.spectrum.y == pytest.approx(direct.let_products[0].spectrum.y)
    assert result.calculation_result.component_status(f"{source.value}_pipeline") is ComponentStatus.INCOMPLETE
    with pytest.raises(ValueError, match="incomplete"):
        result.production_release_products()


@pytest.fixture(scope="module")
def reference_geo_results() -> tuple[
    CalculationConfig,
    SepPipelineResult,
    GcrPipelineResult,
    ErbPipelineResult,
]:
    """Run one bounded contract case without pretending it is a COSRAD gold case."""

    geo = next(scenario for scenario in REFERENCE_SPECTRUM_SCENARIOS if scenario.name == "GEO")
    config = _reference_config(geo, AssessmentMode.CUMULATIVE)
    models = build_pipeline_source_models(config)

    if not isinstance(models.erb_model, OstErbModel):
        msg = "The OST acceptance harness requires the OST ERB model."
        raise TypeError(msg)

    # Orbit sampling is reduced only in this contract harness.  Numerical ERB
    # gold values must be established separately against COSRAD/reference data.
    erb_contract_model = replace(
        models.erb_model,
        anomaly_samples=4,
        node_samples=3,
    )

    return (
        config,
        calculate_sep_pipeline(config=config, sep_model=models.sep_model),
        calculate_gcr_pipeline(config=config, gcr_model=models.gcr_model),
        calculate_erb_pipeline(config=config, erb_model=erb_contract_model),
    )


def _geometric_midpoint_refinement(grid: tuple[float, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for lower, upper in zip(grid, grid[1:]):
        values.extend((lower, sqrt(lower * upper)))
    values.append(grid[-1])
    return tuple(values)


def _integral_from_lowest_energy(
    *,
    x: tuple[float, ...],
    y: tuple[float, ...],
    upper_energy_bound: float | None = None,
) -> float:
    finite_bound = x[-1] if upper_energy_bound is None else upper_energy_bound
    return integrate_differential_spectrum_finite_upper_bound(
        energies=x,
        differential_values=y,
        upper_energy_bound=finite_bound,
        context="stage-4 finite-domain grid convergence",
    )[0]


def _assert_relative_change_below_one_percent(coarse: float, refined: float) -> None:
    assert refined > 0.0
    assert abs(refined - coarse) / refined < 0.01


def test_stage4_reference_scenario_matrix_locks_required_inputs() -> None:
    configs = tuple(
        _reference_config(scenario, mode)
        for scenario in REFERENCE_SPECTRUM_SCENARIOS
        for mode in AssessmentMode
    )

    assert len(configs) == 6
    assert {scenario.name for scenario in REFERENCE_SPECTRUM_SCENARIOS} == {
        "LEO",
        "GEO",
        "HEO",
    }
    assert all(config.mission.launch_year == 2026 for config in configs)
    assert all(config.mission.lifetime_years == 5 for config in configs)
    assert all(config.mission.sep_exceedance_probability == pytest.approx(0.1) for config in configs)
    assert all(
        config.shielding.thicknesses_g_cm2 == REFERENCE_SHIELD_THICKNESSES_G_CM2
        for config in configs
    )

    peak_configs = tuple(
        config for config in configs if config.assessment_mode is AssessmentMode.PEAK
    )
    cumulative_configs = tuple(
        config for config in configs if config.assessment_mode is AssessmentMode.CUMULATIVE
    )
    assert all(config.dose_quantity is DoseQuantity.DOSE_RATE for config in peak_configs)
    assert all(config.dose_unit is Unit.RAD_PER_SECOND for config in peak_configs)
    assert all(
        config.dose_quantity is DoseQuantity.ACCUMULATED_DOSE
        for config in cumulative_configs
    )
    assert all(config.dose_unit is Unit.RAD for config in cumulative_configs)


def test_stage4_reference_pipeline_routes_peak_and_cumulative_products_without_relabelling(
    reference_geo_results: tuple[
        CalculationConfig,
        SepPipelineResult,
        GcrPipelineResult,
        ErbPipelineResult,
    ],
) -> None:
    pipeline_results: tuple[
        SepPipelineResult,
        GcrPipelineResult,
        ErbPipelineResult,
    ] = (
        reference_geo_results[1],
        reference_geo_results[2],
        reference_geo_results[3],
    )
    fluence_units = {
        Unit.DIFFERENTIAL_FLUENCE,
        Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        Unit.DIFFERENTIAL_LET_FLUENCE,
    }
    flux_units = {
        Unit.DIFFERENTIAL_FLUX,
        Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,
        Unit.DIFFERENTIAL_LET_FLUX,
    }

    for result in pipeline_results:
        cumulative = result.assessment_products
        peak = select_products_for_assessment_mode(result.products, AssessmentMode.PEAK)

        assert cumulative
        assert peak
        assert cumulative == select_products_for_assessment_mode(
            result.products,
            AssessmentMode.CUMULATIVE,
        )

        for product in cumulative:
            product.validate_calculation_contract()
            assert product.radiation_quantity_mode is RadiationQuantityMode.FLUENCE
            assert product.spectrum.assessment_mode is AssessmentMode.CUMULATIVE
            assert product.spectrum.temporal_basis is TemporalBasis.MISSION_TOTAL
            assert product.spectrum.mission_duration_s == pytest.approx(MISSION_DURATION_S)
            assert product.spectrum.y_unit in fluence_units

        for product in peak:
            product.validate_calculation_contract()
            assert product.radiation_quantity_mode is RadiationQuantityMode.FLUX
            assert product.spectrum.assessment_mode is AssessmentMode.PEAK
            assert product.spectrum.temporal_basis in {
                TemporalBasis.SEP_PEAK,
                TemporalBasis.SOURCE_MAXIMUM,
            }
            assert product.spectrum.mission_duration_s is None
            assert product.spectrum.y_unit in flux_units


def _missing_range_charges(
    *,
    on_orbit_products: tuple[SpectrumProduct, ...],
    transported_by_kind: Mapping[
        RadiationProductKind,
        Mapping[int, object],
    ],
    product_kind: RadiationProductKind,
) -> set[int]:
    expected = {
        product.spectrum.species.atomic_number
        for product in on_orbit_products
        if product.kind is product_kind
        and product.spectrum.quantity
        in {
            SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        }
        and product.spectrum.particle is Particle.HZE
        and product.spectrum.species is not None
        and product.spectrum.species.atomic_number is not None
    }
    transported = set(transported_by_kind.get(product_kind, {}))
    return expected - transported


def test_stage4_reference_pipeline_builds_supported_products_at_every_shield_depth(
    reference_geo_results: tuple[
        CalculationConfig,
        SepPipelineResult,
        GcrPipelineResult,
        ErbPipelineResult,
    ],
) -> None:
    config, sep_result, gcr_result, erb_result = reference_geo_results

    selected_kind = RadiationProductKind.MISSION_FLUENCE
    for result in (sep_result, gcr_result):
        shielding_results = result.shielding_let_by_thickness
        assert tuple(item.thickness_g_cm2 for item in shielding_results) == (
            config.shielding.thicknesses_g_cm2
        )

        for item in shielding_results:
            assert set(item.proton_by_kind) == {selected_kind}
            assert set(item.let_results_by_kind) == {selected_kind}
            assert all(
                product.spectrum.shielding is not None
                and product.spectrum.shielding.areal_density_g_cm2
                == pytest.approx(item.thickness_g_cm2)
                for product in item.products
            )

            missing_charges = _missing_range_charges(
                on_orbit_products=result.on_orbit_products,
                transported_by_kind=item.hze_by_kind,
                product_kind=selected_kind,
            )
            for z in missing_charges:
                assert any(
                    f"no HZE aluminium range table is available for Z={z}" == reason
                    for reason in item.incomplete_reasons
                )

        missing_reasons = tuple(
            reason
            for item in shielding_results
            for reason in item.incomplete_reasons
        )
        if missing_reasons:
            with pytest.raises(ValueError, match="pipeline is incomplete"):
                result.production_release_products()

    for thickness in config.shielding.thicknesses_g_cm2:
        products = tuple(
            item.shielded_product
            for item in erb_result.shielding_by_thickness
            if item.thickness_g_cm2 == thickness
        )
        assert {
            (Particle.PROTON, RadiationProductKind.MISSION_FLUENCE),
            (Particle.ELECTRON, RadiationProductKind.MISSION_FLUENCE),
        } == {(product.spectrum.particle, product.kind) for product in products}
        assert all(
            product.spectrum.shielding is not None
            and product.spectrum.shielding.areal_density_g_cm2 == pytest.approx(thickness)
            for product in products
        )


def test_stage4_sep_peak_spectrum_is_not_a_mission_duration_scaling() -> None:
    geo = next(scenario for scenario in REFERENCE_SPECTRUM_SCENARIOS if scenario.name == "GEO")
    products_by_lifetime = {}

    for lifetime_years in (1, 5):
        config = replace(
            _reference_config(geo, AssessmentMode.CUMULATIVE),
            mission=MissionConfig(
                launch_year=2026,
                lifetime_years=lifetime_years,
                sep_exceedance_probability=0.1,
            ),
        )
        model = build_pipeline_source_models(
            config,
            sep_hze_energy_grid_mev_per_nucleon=None,
        ).sep_model
        products_by_lifetime[lifetime_years] = model.calculate(
            SepModelInput(mission=config.mission),
        ).products

    peak_by_lifetime = {
        lifetime: next(
            product
            for product in products
            if product.kind is RadiationProductKind.PEAK_FLUX
            and product.spectrum.particle is Particle.PROTON
        )
        for lifetime, products in products_by_lifetime.items()
    }

    assert peak_by_lifetime[5].spectrum.y != pytest.approx(
        tuple(5.0 * value for value in peak_by_lifetime[1].spectrum.y),
    )
    assert all(
        product.spectrum.mission_duration_s is None
        for product in peak_by_lifetime.values()
    )
    assert all(
        "mission_fluence_to_mean_flux"
        not in {step.name for step in product.spectrum.transformation_chain}
        for product in peak_by_lifetime.values()
    )


def test_stage4_default_sep_energy_grids_converge_below_one_percent() -> None:
    geo = next(scenario for scenario in REFERENCE_SPECTRUM_SCENARIOS if scenario.name == "GEO")
    config = _reference_config(geo, AssessmentMode.CUMULATIVE)
    coarse_model = build_pipeline_source_models(config).sep_model
    refined_model = build_pipeline_source_models(
        config,
        sep_energy_grid_mev=_geometric_midpoint_refinement(
            DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
        ),
        sep_hze_energy_grid_mev_per_nucleon=_geometric_midpoint_refinement(
            DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        ),
    ).sep_model
    coarse_products = coarse_model.calculate(SepModelInput(mission=config.mission)).products
    refined_products = refined_model.calculate(SepModelInput(mission=config.mission)).products

    for particle, symbol in (
        (Particle.PROTON, "p"),
        (Particle.HZE, "He"),
        (Particle.HZE, "Fe"),
    ):
        for kind in (
            RadiationProductKind.MISSION_FLUENCE,
            RadiationProductKind.PEAK_FLUX,
        ):
            def matches(product: SpectrumProduct) -> bool:
                spectrum = product.spectrum
                return (
                    product.kind is kind
                    and spectrum.particle is particle
                    and spectrum.species is not None
                    and spectrum.species.symbol == symbol
                )

            coarse = next(product.spectrum for product in coarse_products if matches(product))
            refined = next(product.spectrum for product in refined_products if matches(product))
            _assert_relative_change_below_one_percent(
                _integral_from_lowest_energy(x=coarse.x, y=coarse.y),
                _integral_from_lowest_energy(x=refined.x, y=refined.y),
            )


def test_stage4_default_gcr_energy_grid_converges_below_one_percent() -> None:
    geo = next(scenario for scenario in REFERENCE_SPECTRUM_SCENARIOS if scenario.name == "GEO")
    config = _reference_config(geo, AssessmentMode.CUMULATIVE)
    solar_activity = build_calculation_solar_activity(config)
    symbols = ("H", "He", "Fe", "U")
    coarse_model = OstGcrModel(
        energy_grid_mev_per_nucleon=DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        solar_activity=solar_activity,
        symbols=symbols,
    )
    refined_model = replace(
        coarse_model,
        energy_grid_mev_per_nucleon=_geometric_midpoint_refinement(
            DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        ),
    )
    model_input = GcrModelInput(
        mission=config.mission,
        solar_activity=solar_activity,
    )
    coarse_products = coarse_model.calculate(model_input).products
    refined_products = refined_model.calculate(model_input).products

    for symbol in symbols:
        for kind, quantity in (
            (
                RadiationProductKind.MISSION_FLUENCE,
                SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            ),
            (
                RadiationProductKind.MAXIMUM_FLUX,
                SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            ),
        ):
            coarse = next(
                product.spectrum
                for product in coarse_products
                if product.kind is kind
                and product.spectrum.quantity is quantity
                and product.spectrum.species is not None
                and product.spectrum.species.symbol == symbol
            )
            refined = next(
                product.spectrum
                for product in refined_products
                if product.kind is kind
                and product.spectrum.quantity is quantity
                and product.spectrum.species is not None
                and product.spectrum.species.symbol == symbol
            )
            _assert_relative_change_below_one_percent(
                _integral_from_lowest_energy(
                    x=coarse.x,
                    y=coarse.y,
                    upper_energy_bound=GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
                ),
                _integral_from_lowest_energy(
                    x=refined.x,
                    y=refined.y,
                    upper_energy_bound=GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
                ),
            )


def test_stage4_proton_transport_quadrature_converges_below_one_percent(
    reference_geo_results: tuple[
        CalculationConfig,
        SepPipelineResult,
        GcrPipelineResult,
        ErbPipelineResult,
    ],
) -> None:
    _, sep_result, gcr_result, _ = reference_geo_results
    tables = load_normative_shielding_tables()

    sep_proton = next(
        product
        for product in sep_result.on_orbit_products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.particle is Particle.PROTON
    )
    gcr_proton = next(
        product
        for product in gcr_result.on_orbit_products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
        and product.spectrum.particle is Particle.PROTON
    )

    sep_coarse = calculate_sep_shielding_let_products_for_thickness(
        products=(sep_proton,),
        tables=tables,
        thickness_g_cm2=3.0,
        survival_integration_steps=64,
        secondary_depth_steps=24,
    ).shielded_products[0].spectrum
    sep_refined = calculate_sep_shielding_let_products_for_thickness(
        products=(sep_proton,),
        tables=tables,
        thickness_g_cm2=3.0,
        survival_integration_steps=128,
        secondary_depth_steps=48,
    ).shielded_products[0].spectrum
    gcr_coarse = calculate_gcr_shielding_let_products_for_thickness(
        products=(gcr_proton,),
        tables=tables,
        thickness_g_cm2=3.0,
        survival_integration_steps=64,
        secondary_depth_steps=24,
    ).shielded_products[0].spectrum
    gcr_refined = calculate_gcr_shielding_let_products_for_thickness(
        products=(gcr_proton,),
        tables=tables,
        thickness_g_cm2=3.0,
        survival_integration_steps=128,
        secondary_depth_steps=48,
    ).shielded_products[0].spectrum

    _assert_relative_change_below_one_percent(
        sep_coarse.integral_conserved(),
        sep_refined.integral_conserved(),
    )
    _assert_relative_change_below_one_percent(
        gcr_coarse.integral_conserved(),
        gcr_refined.integral_conserved(),
    )
