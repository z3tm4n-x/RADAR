import pytest

from radar.core.types import Particle
from radar.erb.ost_tables import (
    OST_ERB_ELECTRON_ENERGIES_MEV,
    OST_ERB_PROTON_ENERGIES_MEV,
    OstErbInterpolationMode,
    load_ost_erb_appendix_a_tables,
)


def test_ost_erb_tables_load_all_appendix_a_tables() -> None:
    tables = load_ost_erb_appendix_a_tables()

    assert tuple(table.name for table in tables.tables) == (
        "elec_min",
        "elec_max",
        "prot_min",
        "prot_max",
    )
    assert tables.electron_min.particle is Particle.ELECTRON
    assert tables.electron_max.particle is Particle.ELECTRON
    assert tables.proton_min.particle is Particle.PROTON
    assert tables.proton_max.particle is Particle.PROTON


def test_ost_erb_energy_grids_match_appendix_a_headers() -> None:
    tables = load_ost_erb_appendix_a_tables()

    assert tables.electron_min.energies_mev == OST_ERB_ELECTRON_ENERGIES_MEV
    assert tables.electron_max.energies_mev == OST_ERB_ELECTRON_ENERGIES_MEV
    assert tables.proton_min.energies_mev == OST_ERB_PROTON_ENERGIES_MEV
    assert tables.proton_max.energies_mev == OST_ERB_PROTON_ENERGIES_MEV


def test_ost_erb_table_resource_row_counts_are_stable() -> None:
    tables = load_ost_erb_appendix_a_tables()

    assert sum(len(shell.b_over_b0) for shell in tables.electron_min.shells) == 254
    assert sum(len(shell.b_over_b0) for shell in tables.electron_max.shells) == 276
    assert sum(len(shell.b_over_b0) for shell in tables.proton_min.shells) == 436
    assert sum(len(shell.b_over_b0) for shell in tables.proton_max.shells) == 434


def test_ost_erb_vector_returns_exact_first_electron_min_row() -> None:
    table = load_ost_erb_appendix_a_tables().electron_min

    values = table.vector(
        l_shell=1.2,
        b_over_b0=1.0,
        mode=OstErbInterpolationMode.STRICT,
    )

    assert values[:3] == pytest.approx((1_700_000.0, 518_000.0, 49_000.0))


def test_ost_erb_vector_returns_exact_first_proton_min_row() -> None:
    table = load_ost_erb_appendix_a_tables().proton_min

    values = table.vector(
        l_shell=1.14,
        b_over_b0=1.0,
        mode=OstErbInterpolationMode.STRICT,
    )

    assert values[:4] == pytest.approx((2730.0, 424.0, 61.6, 7.24))


def test_ost_erb_vector_returns_zero_outside_l_grid() -> None:
    table = load_ost_erb_appendix_a_tables().proton_min

    values = table.vector(
        l_shell=0.5,
        b_over_b0=1.0,
    )

    assert values == tuple(0.0 for _ in OST_ERB_PROTON_ENERGIES_MEV)


def test_ost_erb_vector_returns_zero_above_b_over_b0_coverage() -> None:
    table = load_ost_erb_appendix_a_tables().electron_min

    values = table.vector(
        l_shell=1.2,
        b_over_b0=1.0e9,
    )

    assert values == tuple(0.0 for _ in OST_ERB_ELECTRON_ENERGIES_MEV)


def test_ost_erb_strict_and_engineering_modes_are_available() -> None:
    table = load_ost_erb_appendix_a_tables().electron_max

    strict = table.vector(
        l_shell=2.0,
        b_over_b0=1.2,
        mode=OstErbInterpolationMode.STRICT,
    )
    engineering = table.vector(
        l_shell=2.0,
        b_over_b0=1.2,
        mode=OstErbInterpolationMode.ENGINEERING,
    )

    assert len(strict) == len(OST_ERB_ELECTRON_ENERGIES_MEV)
    assert len(engineering) == len(OST_ERB_ELECTRON_ENERGIES_MEV)
    assert all(value >= 0.0 for value in strict)
    assert all(value >= 0.0 for value in engineering)


def test_ost_erb_tables_by_name() -> None:
    tables = load_ost_erb_appendix_a_tables()

    assert tables.by_name("elec_min") is tables.electron_min
    assert tables.by_name("elec_max") is tables.electron_max
    assert tables.by_name("prot_min") is tables.proton_min
    assert tables.by_name("prot_max") is tables.proton_max
