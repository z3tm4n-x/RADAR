import pytest

from radar.core.types import Particle
from radar.erb.ost_tables import (
    OST_ERB_ELECTRON_ENERGIES_MEV,
    OST_ERB_PROTON_ENERGIES_MEV,
    OstErbInterpolationMode,
    OstErbFluxTable,
    OstErbShell,
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
    assert sum(len(shell.b_over_b0) for shell in tables.electron_max.shells) == 270
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


def test_ost_erb_e2_recovers_a_log_quadratic_surface_in_two_coordinates() -> None:
    def flux(l_shell: float, bb0: float) -> float:
        return 10.0 ** (0.1 * l_shell**2 + 0.2 * bb0**2 + 0.05 * l_shell * bb0)

    table = OstErbFluxTable(
        name="prot_min", particle=Particle.PROTON, solar_state="min",
        energies_mev=(1.0,),
        shells=tuple(
            OstErbShell(
                l_shell=l_shell, b_over_b0=(1.0, 2.0, 3.0),
                flux=tuple((flux(l_shell, bb0),) for bb0 in (1.0, 2.0, 3.0)),
            )
            for l_shell in (1.0, 2.0, 3.0)
        ),
    )
    assert table.vector(l_shell=1.4, b_over_b0=1.7) == pytest.approx(
        (flux(1.4, 1.7),), rel=1.0e-13
    )


def test_ost_erb_zero_node_fallback_clamps_negative_lagrange_value() -> None:
    table = OstErbFluxTable(
        name="prot_min", particle=Particle.PROTON, solar_state="min",
        energies_mev=(1.0,),
        shells=tuple(
            OstErbShell(
                l_shell=l_shell, b_over_b0=(1.0, 2.0, 3.0),
                flux=((0.0,), (0.0,), (1.0,)),
            )
            for l_shell in (1.0, 2.0, 3.0)
        ),
    )
    # Literal E.2 gives -0.125 here. This separately documented implementation
    # rule keeps a diagnostic flux non-negative, including in STRICT mode.
    assert table.vector(l_shell=2.0, b_over_b0=1.5) == (0.0,)


@pytest.mark.parametrize(
    ("b_over_b0", "expected_e06_flux"),
    (
        (1.000, 3.18e6),
        (2.053, 2.50e6),
        (4.107, 1.63e6),
        (12.320, 6.83e5),
        (20.534, 4.38e5),
        (41.068, 2.01e5),
        (61.602, 9.77e4),
        (82.136, 3.58e4),
    ),
)
def test_ost_erb_a_1_1_l4_e06_control_values(
    b_over_b0: float,
    expected_e06_flux: float,
) -> None:
    values = load_ost_erb_appendix_a_tables().electron_min.vector(
        l_shell=4.0,
        b_over_b0=b_over_b0,
        mode=OstErbInterpolationMode.STRICT,
    )

    assert values[2] == pytest.approx(expected_e06_flux)


def test_ost_erb_a_1_2_l11_contains_only_two_normative_rows() -> None:
    shell = next(
        shell
        for shell in load_ost_erb_appendix_a_tables().electron_max.shells
        if shell.l_shell == 11.0
    )

    assert shell.b_over_b0 == pytest.approx((1.0, 43.478))
    assert shell.flux[0] == pytest.approx(
        (1.36e9, 3.82e8, 3.81e7, 3.33e6, 2.47e5, 2.00e4, 995.0, 0.0, 0.0, 0.0)
    )
    assert shell.flux[1] == pytest.approx((0.0,) * 10)


def test_ost_erb_a_2_1_l2_b6417_e02_control_value() -> None:
    values = load_ost_erb_appendix_a_tables().proton_min.vector(
        l_shell=2.0,
        b_over_b0=6.417,
        mode=OstErbInterpolationMode.STRICT,
    )

    assert values[0] == pytest.approx(4.24e3)


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


def test_ost_erb_support_uses_the_actual_three_shell_interpolation_stencil() -> None:
    table = load_ost_erb_appendix_a_tables().electron_max

    support = table.interpolation_support(l_shell=1.3, b_over_b0=1.5)

    assert support.selected_shell_indices == (0, 1, 2)
    assert support.b_over_b0_above_shell_indices == (0,)
    assert support.fully_supported is False
    nearest_shell = table.shells[1]
    assert 1.5 <= nearest_shell.b_over_b0[-1]


def test_ost_erb_support_reports_l_domain_without_a_stencil() -> None:
    table = load_ost_erb_appendix_a_tables().proton_min

    support = table.interpolation_support(l_shell=0.5, b_over_b0=1.0)

    assert support.l_below_table is True
    assert support.l_above_table is False
    assert support.selected_shell_indices == ()
    assert support.fully_supported is False


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
