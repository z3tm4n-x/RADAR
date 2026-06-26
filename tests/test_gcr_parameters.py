from math import exp, sqrt

import pytest

from radar.gcr.parameters import (
    AklParameters,
    GCR_ELEMENT_SYMBOLS,
    GcrIonParameters,
    NekParameters,
    OST_NEK_HE_BC_CORRECTION_NOTE,
    OST_NEK_HE_BC_PRINTED,
    OST_NEK_HE_BC_USED,
    get_gost_gcr_ion_parameters,
    get_ost_akl_parameters,
    get_ost_gcr_ion_parameters,
    get_ost_nek_parameter,
    get_ost_nek_parameters,
)


def test_gcr_element_symbol_table_covers_h_to_u() -> None:
    assert GCR_ELEMENT_SYMBOLS[0] == "H"
    assert GCR_ELEMENT_SYMBOLS[1] == "He"
    assert GCR_ELEMENT_SYMBOLS[25] == "Fe"
    assert GCR_ELEMENT_SYMBOLS[27] == "Ni"
    assert GCR_ELEMENT_SYMBOLS[-1] == "U"
    assert len(GCR_ELEMENT_SYMBOLS) == 92


@pytest.mark.parametrize(
    ("symbol", "z", "mass_number", "c_value", "gamma", "alpha"),
    [
        ("H", 1, 1.0, 1.98e4, 2.80, 3.60),
        ("He", 2, 4.0, 3.69e3, 2.77, 3.12),
        ("O", 8, 16.0, 82.7, 2.70, 3.60),
        ("Fe", 26, 55.8, 11.2, 2.70, 3.60),
        ("Ni", 28, 58.7, 0.49, 2.70, 3.60),
        ("U", 92, 238.0, 4.57e-7, 2.70, 3.60),
    ],
)
def test_gost_gcr_ion_parameters_exact_values(
    symbol: str,
    z: int,
    mass_number: float,
    c_value: float,
    gamma: float,
    alpha: float,
) -> None:
    parameters = get_gost_gcr_ion_parameters()[symbol]

    assert parameters.symbol == symbol
    assert parameters.z == z
    assert parameters.mass_number == pytest.approx(mass_number)
    assert parameters.c == pytest.approx(c_value)
    assert parameters.gamma == pytest.approx(gamma)
    assert parameters.alpha == pytest.approx(alpha)
    assert parameters.mass_to_charge == pytest.approx(mass_number / z)


@pytest.mark.parametrize(
    ("symbol", "z", "mass_number", "c_value", "gamma", "alpha"),
    [
        ("H", 1, 1.0, 1.32e4, 2.93, 2.85),
        ("He", 2, 4.0, 1.72e3, 2.64, 3.12),
        ("O", 8, 16.0, 55.0, 2.64, 3.11),
        ("Fe", 26, 55.8, 8.40, 2.64, 3.01),
        ("Ni", 28, 58.7, 0.40, 2.64, 3.52),
        ("U", 92, 238.0, 4.57e-7, 2.63, 3.01),
    ],
)
def test_ost_gcr_ion_parameters_exact_values(
    symbol: str,
    z: int,
    mass_number: float,
    c_value: float,
    gamma: float,
    alpha: float,
) -> None:
    parameters = get_ost_gcr_ion_parameters()[symbol]

    assert parameters.symbol == symbol
    assert parameters.z == z
    assert parameters.mass_number == pytest.approx(mass_number)
    assert parameters.c == pytest.approx(c_value)
    assert parameters.gamma == pytest.approx(gamma)
    assert parameters.alpha == pytest.approx(alpha)
    assert parameters.mass_to_charge == pytest.approx(mass_number / z)


def test_ost_gcr_parameters_do_not_swap_gamma_and_alpha() -> None:
    parameters = get_ost_gcr_ion_parameters()

    assert parameters["H"].gamma == pytest.approx(2.93)
    assert parameters["H"].alpha == pytest.approx(2.85)
    assert parameters["Fe"].gamma == pytest.approx(2.64)
    assert parameters["Fe"].alpha == pytest.approx(3.01)


def test_ost_nek_parameters_use_corrected_he_bc_with_traceability() -> None:
    parameters = get_ost_nek_parameters()

    assert parameters[1].ac == pytest.approx(4.1e-2)
    assert parameters[1].bc == pytest.approx(2.57e-2)
    assert parameters[1].ag == pytest.approx(3.50)
    assert parameters[1].bg == pytest.approx(0.0)

    assert OST_NEK_HE_BC_PRINTED == pytest.approx(6.6e-2)
    assert OST_NEK_HE_BC_USED == pytest.approx(6.6e-3)
    assert "probable misprint correction" in OST_NEK_HE_BC_CORRECTION_NOTE

    assert parameters[2].ac == pytest.approx(3.6e-2)
    assert parameters[2].bc == pytest.approx(OST_NEK_HE_BC_USED)
    assert parameters[2].ag == pytest.approx(2.45)
    assert parameters[2].bg == pytest.approx(0.0)


def test_ost_nek_parameters_include_zero_group_for_z3_to_z5() -> None:
    parameters = get_ost_nek_parameters()

    for z in (3, 4, 5):
        assert parameters[z].ac == pytest.approx(0.0)
        assert parameters[z].bc == pytest.approx(0.0)
        assert parameters[z].ag == pytest.approx(0.0)
        assert parameters[z].bg == pytest.approx(0.0)


@pytest.mark.parametrize("z", [6, 8, 26, 28])
def test_ost_nek_parameters_for_z6_to_z28_follow_group_formula(z: int) -> None:
    parameters = get_ost_nek_parameter(z)

    assert parameters is not None
    assert parameters.ac == pytest.approx(2.3e-4 * exp(-0.18 * z))
    assert parameters.bc == pytest.approx(1.64e-2 * exp(0.05 * z))
    assert parameters.ag == pytest.approx(2.47 + 0.17 * sqrt(z))
    assert parameters.bg == pytest.approx(0.016 - 0.03 / sqrt(z))


def test_ost_nek_parameters_are_absent_above_z28() -> None:
    assert get_ost_nek_parameter(29) is None
    assert get_ost_nek_parameter(92) is None


def test_ost_nek_parameter_rejects_non_positive_z() -> None:
    with pytest.raises(ValueError, match="positive"):
        get_ost_nek_parameter(0)


def test_ost_akl_parameters_exact_values() -> None:
    parameters = get_ost_akl_parameters()

    assert set(parameters) == {"He", "C", "N", "O", "Ne", "Ar"}
    assert parameters["He"].z == 2
    assert parameters["He"].c == pytest.approx(1.2e-3)
    assert parameters["C"].z == 6
    assert parameters["C"].c == pytest.approx(4.6e-6)
    assert parameters["N"].z == 7
    assert parameters["N"].c == pytest.approx(4.6e-5)
    assert parameters["O"].z == 8
    assert parameters["O"].c == pytest.approx(2.4e-4)
    assert parameters["Ne"].z == 10
    assert parameters["Ne"].c == pytest.approx(2.4e-5)
    assert parameters["Ar"].z == 18
    assert parameters["Ar"].c == pytest.approx(4.6e-7)


def test_parameter_dataclasses_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="symbol"):
        GcrIonParameters(
            symbol="",
            z=1,
            mass_number=1.0,
            c=1.0,
            gamma=1.0,
            alpha=1.0,
        )

    with pytest.raises(ValueError, match="charge"):
        NekParameters(z=0, ac=1.0, bc=1.0, ag=1.0, bg=1.0)

    with pytest.raises(ValueError, match="non-negative"):
        AklParameters(symbol="He", z=2, c=-1.0)
