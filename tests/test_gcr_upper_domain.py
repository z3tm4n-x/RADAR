from __future__ import annotations

from collections.abc import Mapping
from math import sqrt

import pytest

from radar.core.constants import (
    DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON,
    GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON,
    GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
)
from radar.core.products import SpectrumProduct
from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import integrate_differential_spectrum_finite_upper_bound
from radar.core.types import (
    Particle,
    RadiationProductKind,
    SpectrumCompleteness,
    SpectrumQuantity,
)
from radar.gcr.products import GCR_FINITE_DOMAIN_INTEGRATION_STEP
from radar.gcr.model import (
    GCR_SOURCE_DOMAIN_REASON_PREFIX,
    OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX,
    GcrModelInput,
    OstGcrModel,
)
from radar.gcr.pipeline_shielding import (
    GcrShieldingLetPipelineProducts,
    calculate_gcr_shielding_let_products_for_thickness,
)
from radar.gcr.shielding import (
    calculate_gcr_hze_shielding_spectrum,
    calculate_gcr_proton_shielding_spectrum,
)
from radar.shielding.hze_al import (
    HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX,
    log_bin_edges_from_centers,
)
from radar.shielding.resources import (
    ShieldingNormativeTables,
    load_normative_shielding_tables,
)

_REPRESENTATIVE_SYMBOLS = ("H", "He", "Si", "Fe")
_GRID_DENSITY_SYMBOLS = ("H", "He", "Fe")
_THICKNESSES_G_CM2 = (0.01, 0.1, 1.0, 3.0, 10.0)
_ASSESSMENT_KINDS = (
    RadiationProductKind.MISSION_FLUENCE,
    RadiationProductKind.MAXIMUM_FLUX,
)
_LEGACY_300_MEV_GRID = tuple(
    energy
    for energy in DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON
    if energy <= 300.0
)
_EXTENDED_REFERENCE_GRID = (
    *DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
)
_LOWER_REFERENCE_GRID = (
    10.0**-0.1,
    *DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
)


def _geometric_midpoint_refinement(grid: tuple[float, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for lower, upper in zip(grid, grid[1:]):
        values.extend((lower, sqrt(lower * upper)))
    values.append(grid[-1])
    return tuple(values)


def _mission_products_by_symbol(
    energy_grid: tuple[float, ...],
) -> dict[str, SpectrumProduct]:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=energy_grid,
        symbols=_REPRESENTATIVE_SYMBOLS,
    )
    result = model.calculate(
        GcrModelInput(
            mission=MissionConfig(launch_year=2026, lifetime_years=1),
        )
    )
    return {
        product.spectrum.species.symbol: product
        for product in result.products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
        and product.spectrum.species is not None
    }


def _assessment_products(
    energy_grid: tuple[float, ...],
) -> tuple[SpectrumProduct, ...]:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=energy_grid,
        symbols=_GRID_DENSITY_SYMBOLS,
    )
    result = model.calculate(
        GcrModelInput(
            mission=MissionConfig(launch_year=2026, lifetime_years=5),
        )
    )
    return tuple(
        product
        for product in result.products
        if (
            product.kind is RadiationProductKind.MISSION_FLUENCE
            and product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
        )
        or (
            product.kind is RadiationProductKind.MAXIMUM_FLUX
            and product.spectrum.quantity
            is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
        )
    )


@pytest.fixture(scope="module")
def domain_products() -> tuple[
    Mapping[str, SpectrumProduct],
    Mapping[str, SpectrumProduct],
    Mapping[str, SpectrumProduct],
    Mapping[str, SpectrumProduct],
]:
    return (
        _mission_products_by_symbol(_LEGACY_300_MEV_GRID),
        _mission_products_by_symbol(DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON),
        _mission_products_by_symbol(_EXTENDED_REFERENCE_GRID),
        _mission_products_by_symbol(_LOWER_REFERENCE_GRID),
    )


@pytest.fixture(scope="module")
def shielding_tables() -> ShieldingNormativeTables:
    return load_normative_shielding_tables()


def _protected_results_for_grid(
    *,
    energy_grid: tuple[float, ...],
    tables: ShieldingNormativeTables,
    full_proton_transport: bool = True,
) -> Mapping[float, GcrShieldingLetPipelineProducts]:
    products = _assessment_products(energy_grid)
    return {
        thickness: calculate_gcr_shielding_let_products_for_thickness(
            products=products,
            tables=tables,
            thickness_g_cm2=thickness,
            include_nonelastic_survival=full_proton_transport,
            include_secondary_protons=full_proton_transport,
            include_mixed_let_diagnostic=False,
        )
        for thickness in _THICKNESSES_G_CM2
    }


@pytest.fixture(scope="module")
def protected_grid_results(
    shielding_tables: ShieldingNormativeTables,
) -> Mapping[
    float,
    tuple[GcrShieldingLetPipelineProducts, GcrShieldingLetPipelineProducts],
]:
    production = _protected_results_for_grid(
        energy_grid=DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        tables=shielding_tables,
    )
    refined = _protected_results_for_grid(
        energy_grid=_geometric_midpoint_refinement(
            DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        ),
        tables=shielding_tables,
    )
    return {
        thickness: (production[thickness], refined[thickness])
        for thickness in _THICKNESSES_G_CM2
    }


def _integral(product: SpectrumProduct) -> float:
    spectrum = product.spectrum
    return integrate_differential_spectrum_finite_upper_bound(
        energies=spectrum.x,
        differential_values=spectrum.y,
        upper_energy_bound=GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
        context="GCR upper-domain convergence",
    )[0]


def _shielded_spectrum(
    product: SpectrumProduct,
    *,
    tables: ShieldingNormativeTables,
    thickness_g_cm2: float,
) -> Spectrum1D:
    spectrum = product.spectrum
    if spectrum.particle is Particle.PROTON:
        return calculate_gcr_proton_shielding_spectrum(
            spectrum=spectrum,
            tables=tables,
            thickness_g_cm2=thickness_g_cm2,
        ).total

    species = spectrum.species
    assert species is not None
    assert species.atomic_number is not None
    return calculate_gcr_hze_shielding_spectrum(
        z=species.atomic_number,
        spectrum=spectrum,
        tables=tables,
        thickness_g_cm2=thickness_g_cm2,
    ).spectrum


def _energy_spectrum(
    result: GcrShieldingLetPipelineProducts,
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
    result: GcrShieldingLetPipelineProducts,
    *,
    energy_kind: RadiationProductKind,
) -> Spectrum1D:
    let_kind = {
        RadiationProductKind.MISSION_FLUENCE: (
            RadiationProductKind.MISSION_LET_FLUENCE
        ),
        RadiationProductKind.MAXIMUM_FLUX: RadiationProductKind.MAXIMUM_LET_FLUX,
    }[energy_kind]
    return next(product.spectrum for product in result.let_products if product.kind is let_kind)


def _bin_average_at_energy(spectrum: Spectrum1D, energy: float) -> float:
    assert spectrum.bin_edges is not None
    return next(
        value
        for value, lower, upper in zip(
            spectrum.y,
            spectrum.bin_edges,
            spectrum.bin_edges[1:],
        )
        if lower <= energy < upper
    )


def _protected_grid_integral_pairs(
    results: Mapping[
        float,
        tuple[GcrShieldingLetPipelineProducts, GcrShieldingLetPipelineProducts],
    ],
) -> dict[str, tuple[float, float]]:
    pairs: dict[str, tuple[float, float]] = {}
    for thickness, (production, refined) in results.items():
        for kind in _ASSESSMENT_KINDS:
            for symbol in _GRID_DENSITY_SYMBOLS:
                label = f"{kind.value}/{symbol}/{thickness:g}g_cm2"
                pairs[label] = (
                    _energy_spectrum(
                        production,
                        kind=kind,
                        symbol=symbol,
                    ).integral_conserved(),
                    _energy_spectrum(
                        refined,
                        kind=kind,
                        symbol=symbol,
                    ).integral_conserved(),
                )

            label = f"{kind.value}/canonical_hze_let/{thickness:g}g_cm2"
            pairs[label] = (
                _canonical_hze_let_spectrum(
                    production,
                    energy_kind=kind,
                ).integral_conserved(),
                _canonical_hze_let_spectrum(
                    refined,
                    energy_kind=kind,
                ).integral_conserved(),
            )
    return pairs


def test_production_gcr_grid_uses_cosrad_last_center_inside_reference_domain() -> None:
    grid = DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON
    edges = log_bin_edges_from_centers(grid)

    assert grid[0] == GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON
    assert 300.0 in grid
    assert grid[-1] == pytest.approx(GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON)
    assert edges[-1] < GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON


def test_truncated_custom_grid_is_incomplete_and_is_not_continued_to_100_gev() -> None:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=_LEGACY_300_MEV_GRID,
        symbols=("H",),
    )
    source = model.calculate_source_spectra().ion("H")
    result = model.calculate(
        GcrModelInput(
            mission=MissionConfig(launch_year=2026, lifetime_years=1),
        )
    )
    mission_integral = next(
        product.spectrum
        for product in result.products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.quantity is SpectrumQuantity.INTEGRAL_FLUENCE
    )

    assert all(
        spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        for spectrum in source.all_spectra
    )
    assert mission_integral.completeness is SpectrumCompleteness.INCOMPLETE
    assert mission_integral.y[-1] == 0.0
    finite_step = next(
        step
        for step in mission_integral.transformation_chain
        if step.name == GCR_FINITE_DOMAIN_INTEGRATION_STEP
    )
    assert dict(finite_step.parameters)["upper_energy_bound"] == "300"


def test_production_grid_closes_integral_at_finite_100_gev_boundary() -> None:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        symbols=("H",),
    )
    result = model.calculate(
        GcrModelInput(
            mission=MissionConfig(launch_year=2026, lifetime_years=1),
        )
    )
    mission_integral = next(
        product.spectrum
        for product in result.products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.quantity is SpectrumQuantity.INTEGRAL_FLUENCE
    )

    # The production grid closes the finite energy domain correctly, but an OST
    # calculation containing only H is intentionally not release-complete: the
    # normative H--U ion roster is enforced independently of domain closure.
    assert mission_integral.completeness is SpectrumCompleteness.INCOMPLETE
    assert any(
        reason.startswith(OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX)
        for reason in mission_integral.incomplete_reasons
    )
    assert not any(
        reason.startswith(GCR_SOURCE_DOMAIN_REASON_PREFIX)
        for reason in mission_integral.incomplete_reasons
    )
    assert mission_integral.y[-1] > 0.0
    finite_step = next(
        step
        for step in mission_integral.transformation_chain
        if step.name == GCR_FINITE_DOMAIN_INTEGRATION_STEP
    )
    assert dict(finite_step.parameters)["upper_energy_bound"] == "100000"


def test_production_gcr_source_upper_bound_converges_below_one_percent(
    domain_products: tuple[
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
    ],
) -> None:
    _, production, extended, _ = domain_products

    for symbol in _REPRESENTATIVE_SYMBOLS:
        production_integral = _integral(production[symbol])
        extended_integral = _integral(extended[symbol])

        assert extended_integral > 0.0
        assert abs(extended_integral - production_integral) / extended_integral < 0.01


def test_production_gcr_source_lower_bound_converges_below_one_percent(
    domain_products: tuple[
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
    ],
) -> None:
    _, production, _, lower_extended = domain_products

    for symbol in _REPRESENTATIVE_SYMBOLS:
        production_integral = _integral(production[symbol])
        lower_extended_integral = _integral(lower_extended[symbol])

        assert lower_extended_integral > 0.0
        assert (
            abs(lower_extended_integral - production_integral)
            / lower_extended_integral
            < 0.01
        )


def test_gcr_proton_source_reconstruction_converges_below_one_percent() -> None:
    """Verify the source measure actually consumed by the proton CSDA map.

    The source's finite-upper-bound integral uses a log-log reconstruction,
    whereas positive-thickness proton transport integrates point samples
    piecewise linearly.  Convergence of the former cannot certify the latter.
    This inexpensive check catches inadequate source sampling before running
    the considerably slower secondary-proton quadrature.
    """

    integrals_by_grid: list[dict[RadiationProductKind, float]] = []
    for grid in (
        DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        _geometric_midpoint_refinement(
            DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
        ),
    ):
        integrals_by_grid.append(
            {
                product.kind: product.spectrum.integral_conserved()
                for product in _assessment_products(grid)
                if product.spectrum.particle is Particle.PROTON
            }
        )

    production, refined = integrals_by_grid
    assert set(production) == set(refined) == set(_ASSESSMENT_KINDS)
    for kind in _ASSESSMENT_KINDS:
        assert refined[kind] > 0.0
        assert abs(production[kind] - refined[kind]) / refined[kind] < 0.01, kind


@pytest.mark.parametrize("symbol", ("Si", "Fe"))
def test_legacy_300_mev_domain_cannot_transport_last_bin_through_ten_grams(
    symbol: str,
    domain_products: tuple[
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
    ],
    shielding_tables: ShieldingNormativeTables,
) -> None:
    legacy, production, _, _ = domain_products
    legacy_spectrum = _shielded_spectrum(
        legacy[symbol],
        tables=shielding_tables,
        thickness_g_cm2=10.0,
    )
    production_spectrum = _shielded_spectrum(
        production[symbol],
        tables=shielding_tables,
        thickness_g_cm2=10.0,
    )

    assert _bin_average_at_energy(legacy_spectrum, 300.0) == 0.0
    assert _bin_average_at_energy(production_spectrum, 300.0) > 0.0
    assert any(
        "source upper support 300" in reason
        for reason in legacy_spectrum.incomplete_reasons
    )


def test_gcr_upper_bound_converges_at_every_stage4_shield_depth(
    domain_products: tuple[
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
    ],
    shielding_tables: ShieldingNormativeTables,
) -> None:
    _, production, extended, _ = domain_products

    for thickness_g_cm2 in _THICKNESSES_G_CM2:
        for symbol in _REPRESENTATIVE_SYMBOLS:
            production_spectrum = _shielded_spectrum(
                production[symbol],
                tables=shielding_tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            extended_spectrum = _shielded_spectrum(
                extended[symbol],
                tables=shielding_tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            production_integral = production_spectrum.integral_conserved()
            extended_integral = extended_spectrum.integral_conserved()

            assert extended_integral > 0.0
            assert (
                abs(extended_integral - production_integral) / extended_integral
                < 0.01
            )

            if symbol != "H":
                assert any(
                    reason.startswith(HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
                    for reason in production_spectrum.incomplete_reasons
                )


def test_gcr_lower_bound_converges_at_every_stage4_shield_depth(
    domain_products: tuple[
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
        Mapping[str, SpectrumProduct],
    ],
    shielding_tables: ShieldingNormativeTables,
) -> None:
    _, production, _, lower_extended = domain_products

    for thickness_g_cm2 in _THICKNESSES_G_CM2:
        for symbol in _REPRESENTATIVE_SYMBOLS:
            production_spectrum = _shielded_spectrum(
                production[symbol],
                tables=shielding_tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            lower_extended_spectrum = _shielded_spectrum(
                lower_extended[symbol],
                tables=shielding_tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            production_integral = production_spectrum.integral_conserved()
            lower_extended_integral = lower_extended_spectrum.integral_conserved()

            assert lower_extended_integral > 0.0
            assert (
                abs(lower_extended_integral - production_integral)
                / lower_extended_integral
                < 0.01
            )


def test_gcr_protected_energy_and_canonical_hze_let_converge_below_one_percent(
    protected_grid_results: Mapping[
        float,
        tuple[GcrShieldingLetPipelineProducts, GcrShieldingLetPipelineProducts],
    ],
) -> None:
    """Check both user modes at every Stage-4 shield depth.

    The HZE LET tables remain quarantined, so this is a numerical convergence
    test of the explicitly diagnostic calculation, not a release-eligibility
    assertion.
    """

    pairs = _protected_grid_integral_pairs(protected_grid_results)
    assert len(pairs) == len(_THICKNESSES_G_CM2) * len(_ASSESSMENT_KINDS) * 4

    relative_changes: dict[str, float] = {}
    for label, (production_integral, refined_integral) in pairs.items():
        assert refined_integral > 0.0, label
        relative_changes[label] = (
            abs(refined_integral - production_integral) / refined_integral
        )

    worst_label, worst_change = max(
        relative_changes.items(),
        key=lambda item: item[1],
    )
    largest_changes = sorted(
        relative_changes.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    assert worst_change < 0.01, (
        f"{worst_label}: relative change {worst_change:.12g}; "
        f"largest changes={largest_changes!r}"
    )
