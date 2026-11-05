from __future__ import annotations

from collections.abc import Mapping
from math import sqrt

import pytest

from radar.core.constants import (
    DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
    SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON,
    SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON,
)
from radar.core.products import SpectrumProduct
from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    SpectrumCompleteness,
)
from radar.sep.model import OstSepModel, SepModelInput
from radar.sep.pipeline_shielding import (
    SepShieldingLetPipelineProducts,
    calculate_sep_shielding_let_products_for_thickness,
)
from radar.shielding.resources import (
    ShieldingNormativeTables,
    load_normative_shielding_tables,
)
from radar.shielding.sep_hze import calculate_sep_hze_shielding_spectrum
from radar.shielding.sep_proton import calculate_sep_proton_shielding_spectrum


_MISSION = MissionConfig(launch_year=2026, lifetime_years=5)
_MONTHLY_WOLF_NUMBERS = (50.0,) * (12 * _MISSION.lifetime_years)
_STAGE4_THICKNESSES_G_CM2 = (0.01, 0.1, 1.0, 3.0, 10.0)
_ASSESSMENT_KINDS = (
    RadiationProductKind.MISSION_FLUENCE,
    RadiationProductKind.PEAK_FLUX,
)


def _geometric_midpoint_refinement(grid: tuple[float, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for lower, upper in zip(grid, grid[1:]):
        values.extend((lower, sqrt(lower * upper)))
    values.append(grid[-1])
    return tuple(values)


def _mission_products_by_symbol(
    *,
    proton_grid: tuple[float, ...],
    hze_grid: tuple[float, ...],
) -> dict[str, SpectrumProduct]:
    model = OstSepModel(
        energy_grid_mev=proton_grid,
        hze_energy_grid_mev_per_nucleon=hze_grid,
        monthly_smoothed_wolf_numbers=_MONTHLY_WOLF_NUMBERS,
    )
    result = model.calculate(SepModelInput(mission=_MISSION))
    return {
        product.spectrum.species.symbol: product
        for product in result.products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.species is not None
        and product.spectrum.species.symbol in {"p", "He", "Fe"}
    }


def _assessment_products(
    *,
    proton_grid: tuple[float, ...],
    hze_grid: tuple[float, ...],
    supported_hze_z: frozenset[int],
) -> tuple[SpectrumProduct, ...]:
    model = OstSepModel(
        energy_grid_mev=proton_grid,
        hze_energy_grid_mev_per_nucleon=hze_grid,
        monthly_smoothed_wolf_numbers=_MONTHLY_WOLF_NUMBERS,
    )
    result = model.calculate(SepModelInput(mission=_MISSION))
    selected: list[SpectrumProduct] = []
    for product in result.products:
        spectrum = product.spectrum
        species = spectrum.species
        if product.kind not in _ASSESSMENT_KINDS or species is None:
            continue
        if spectrum.particle is Particle.PROTON:
            selected.append(product)
        elif (
            spectrum.particle is Particle.HZE
            and species.atomic_number in supported_hze_z
        ):
            selected.append(product)
    return tuple(selected)


@pytest.fixture(scope="module")
def domain_products() -> tuple[
    Mapping[str, SpectrumProduct],
    Mapping[str, SpectrumProduct],
]:
    production = _mission_products_by_symbol(
        proton_grid=DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
        hze_grid=DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    )
    refined = _mission_products_by_symbol(
        proton_grid=_geometric_midpoint_refinement(
            DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
        ),
        hze_grid=_geometric_midpoint_refinement(
            DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        ),
    )
    return production, refined


@pytest.fixture(scope="module")
def shielding_tables() -> ShieldingNormativeTables:
    return load_normative_shielding_tables()


@pytest.fixture(scope="module")
def protected_grid_results(
    shielding_tables: ShieldingNormativeTables,
) -> Mapping[
    float,
    tuple[SepShieldingLetPipelineProducts, SepShieldingLetPipelineProducts],
]:
    available_hze_z = frozenset(
        shielding_tables.hze_al_ranges_by_z.keys()
        & shielding_tables.hze_si_let_by_z.keys()
    )
    supported_hze_z = frozenset({2, 26}) & available_hze_z
    assert supported_hze_z == frozenset({2, 26})
    production_products = _assessment_products(
        proton_grid=DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
        hze_grid=DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        supported_hze_z=supported_hze_z,
    )
    refined_products = _assessment_products(
        proton_grid=_geometric_midpoint_refinement(
            DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
        ),
        hze_grid=_geometric_midpoint_refinement(
            DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        ),
        supported_hze_z=supported_hze_z,
    )
    return {
        thickness: (
            calculate_sep_shielding_let_products_for_thickness(
                products=production_products,
                tables=shielding_tables,
                thickness_g_cm2=thickness,
                include_mixed_let_diagnostic=False,
            ),
            calculate_sep_shielding_let_products_for_thickness(
                products=refined_products,
                tables=shielding_tables,
                thickness_g_cm2=thickness,
                include_mixed_let_diagnostic=False,
            ),
        )
        for thickness in _STAGE4_THICKNESSES_G_CM2
    }


def _shielded_spectrum(
    product: SpectrumProduct,
    *,
    tables: ShieldingNormativeTables,
) -> Spectrum1D:
    spectrum = product.spectrum
    if spectrum.particle is Particle.PROTON:
        return calculate_sep_proton_shielding_spectrum(
            spectrum=spectrum,
            tables=tables,
            thickness_g_cm2=10.0,
        ).total

    species = spectrum.species
    assert species is not None
    assert species.atomic_number is not None
    return calculate_sep_hze_shielding_spectrum(
        z=species.atomic_number,
        spectrum=spectrum,
        tables=tables,
        thickness_g_cm2=10.0,
    ).spectrum


def test_production_sep_grids_cover_the_cosrad_source_domain() -> None:
    proton_grid = DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV
    hze_grid = DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON

    assert len(proton_grid) == 83
    assert len(hze_grid) == 165
    assert proton_grid[0] == SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON
    assert hze_grid[0] == SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON
    assert proton_grid[-1] == pytest.approx(
        SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON,
    )
    assert hze_grid[-1] == pytest.approx(
        SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON,
    )


def test_truncated_sep_source_grids_are_explicitly_incomplete() -> None:
    truncated = _mission_products_by_symbol(
        proton_grid=(1.0, 10.0, 100.0, 1_000.0),
        hze_grid=(1.0, 10.0, 100.0, 500.0),
    )

    for product in truncated.values():
        assert product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        assert any(
            reason.startswith("SEP source energy domain incomplete: last center")
            for reason in product.spectrum.incomplete_reasons
        )


@pytest.mark.parametrize("symbol", ("p", "He", "Fe"))
def test_sep_upper_domain_converges_after_ten_grams_below_one_percent(
    symbol: str,
    domain_products: tuple[
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
    ],
    shielding_tables: ShieldingNormativeTables,
) -> None:
    production, refined = domain_products
    production_spectrum = _shielded_spectrum(
        production[symbol],
        tables=shielding_tables,
    )
    refined_spectrum = _shielded_spectrum(
        refined[symbol],
        tables=shielding_tables,
    )
    production_integral = production_spectrum.integral_conserved()
    refined_integral = refined_spectrum.integral_conserved()

    assert refined_integral > 0.0
    assert abs(refined_integral - production_integral) / refined_integral < 0.01


def _energy_spectrum(
    result: SepShieldingLetPipelineProducts,
    *,
    kind: RadiationProductKind,
    symbol: str,
) -> Spectrum1D:
    return next(
        product.spectrum
        for product in result.shielded_products
        if product.kind is kind
        and product.spectrum.species is not None
        and product.spectrum.species.symbol == symbol
    )


def _canonical_hze_let_spectrum(
    result: SepShieldingLetPipelineProducts,
    *,
    energy_kind: RadiationProductKind,
) -> Spectrum1D:
    let_kind = {
        RadiationProductKind.MISSION_FLUENCE: (
            RadiationProductKind.MISSION_LET_FLUENCE
        ),
        RadiationProductKind.PEAK_FLUX: RadiationProductKind.PEAK_LET_FLUX,
    }[energy_kind]
    return next(product.spectrum for product in result.let_products if product.kind is let_kind)


def test_sep_protected_energy_and_canonical_hze_let_converge_below_one_percent(
    protected_grid_results: Mapping[
        float,
        tuple[SepShieldingLetPipelineProducts, SepShieldingLetPipelineProducts],
    ],
) -> None:
    """Check both user modes at every Stage-4 shield depth.

    The HZE LET tables remain quarantined, so this is a numerical convergence
    test of the explicitly diagnostic calculation, not a release-eligibility
    assertion.
    """

    for production, refined in protected_grid_results.values():
        for kind in _ASSESSMENT_KINDS:
            for symbol in ("p", "He", "Fe"):
                production_energy = _energy_spectrum(
                    production,
                    kind=kind,
                    symbol=symbol,
                )
                refined_energy = _energy_spectrum(
                    refined,
                    kind=kind,
                    symbol=symbol,
                )
                production_integral = production_energy.integral_conserved()
                refined_integral = refined_energy.integral_conserved()
                assert refined_integral > 0.0
                assert (
                    abs(refined_integral - production_integral) / refined_integral
                    < 0.01
                )

            production_let = _canonical_hze_let_spectrum(
                production,
                energy_kind=kind,
            )
            refined_let = _canonical_hze_let_spectrum(
                refined,
                energy_kind=kind,
            )
            production_let_integral = production_let.integral_conserved()
            refined_let_integral = refined_let.integral_conserved()
            assert refined_let_integral > 0.0
            assert (
                abs(refined_let_integral - production_let_integral)
                / refined_let_integral
                < 0.01
            )
