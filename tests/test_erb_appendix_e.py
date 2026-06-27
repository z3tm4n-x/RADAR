import pytest

from radar.erb.appendix_e import (
    OstErbElectronApproximationCoefficients,
    OstErbElectronIntegralApproximation,
    OstErbProtonApproximationCoefficients,
    OstErbProtonIntegralApproximation,
    integrate_differential_flux_tail_power_law,
)


def test_appendix_e_integrates_constant_differential_flux_without_tail() -> None:
    integral_flux = integrate_differential_flux_tail_power_law(
        energies_mev=(1.0, 2.0, 3.0),
        differential_flux=(10.0, 10.0, 10.0),
    )

    assert integral_flux == pytest.approx((20.0, 10.0, 0.0))


def test_appendix_e_integrates_power_law_with_finite_tail() -> None:
    integral_flux = integrate_differential_flux_tail_power_law(
        energies_mev=(1.0, 2.0, 4.0),
        differential_flux=(8.0, 2.0, 0.5),
    )

    assert integral_flux == pytest.approx((8.0, 4.0, 2.0))


@pytest.mark.parametrize(
    ("energies_mev", "differential_flux", "message"),
    (
        ((), (), "must not be empty"),
        ((1.0, 2.0), (1.0,), "same length"),
        ((1.0, 0.0), (1.0, 1.0), "positive"),
        ((2.0, 1.0), (1.0, 1.0), "sorted"),
        ((1.0, 1.0), (1.0, 1.0), "unique"),
        ((1.0, 2.0), (1.0, -1.0), "non-negative"),
    ),
)
def test_appendix_e_integral_flux_validates_inputs(
    energies_mev: tuple[float, ...],
    differential_flux: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        integrate_differential_flux_tail_power_law(
            energies_mev=energies_mev,
            differential_flux=differential_flux,
        )

def test_appendix_e_proton_approximation_uses_ost_coefficient_relations() -> None:
    approximation = OstErbProtonIntegralApproximation(
        coefficients=OstErbProtonApproximationCoefficients(
            a1=2.0,
            a2=3.0,
            a3=4.0,
            b1=5.0,
            b2=6.0,
        )
    )

    differential = approximation.differential_coefficients

    assert differential.a1 == pytest.approx(6.0)
    assert differential.a2 == pytest.approx(4.0)
    assert differential.a3 == pytest.approx(4.0)
    assert differential.b1 == pytest.approx(30.0)
    assert differential.b2 == pytest.approx(6.0)
    assert approximation.integral_flux_gt_e(2.0) == pytest.approx(
        2.0 / 6.0**3.0 + 5.0 * __import__("math").exp(-12.0)
    )
    assert approximation.differential_flux(2.0) == pytest.approx(
        6.0 / 6.0**4.0 + 30.0 * __import__("math").exp(-12.0)
    )


def test_appendix_e_electron_approximation_uses_ost_coefficient_relations() -> None:
    approximation = OstErbElectronIntegralApproximation(
        coefficients=OstErbElectronApproximationCoefficients(
            a1=2.0,
            a2=3.0,
            b1=5.0,
            b2=6.0,
            c1=7.0,
            c2=8.0,
        )
    )

    differential = approximation.differential_coefficients

    assert differential.a1 == pytest.approx(6.0)
    assert differential.a2 == pytest.approx(3.0)
    assert differential.b1 == pytest.approx(30.0)
    assert differential.b2 == pytest.approx(6.0)
    assert differential.c1 == pytest.approx(56.0)
    assert differential.c2 == pytest.approx(8.0)
    assert approximation.differential_flux(2.0) == pytest.approx(
        6.0 * __import__("math").exp(-6.0)
        + 30.0 * __import__("math").exp(-24.0)
        + 56.0 * __import__("math").exp(-16.0)
    )


@pytest.mark.parametrize(
    "coefficients",
    (
        {"a1": -1.0, "a2": 2.0, "a3": 0.0, "b1": 1.0, "b2": 1.0},
        {"a1": 1.0, "a2": 0.0, "a3": 0.0, "b1": 1.0, "b2": 1.0},
        {"a1": 1.0, "a2": 2.0, "a3": -1.0, "b1": 1.0, "b2": 1.0},
        {"a1": 1.0, "a2": 2.0, "a3": 0.0, "b1": -1.0, "b2": 1.0},
        {"a1": 1.0, "a2": 2.0, "a3": 0.0, "b1": 1.0, "b2": 0.0},
    ),
)
def test_appendix_e_proton_approximation_validates_coefficients(
    coefficients: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        OstErbProtonApproximationCoefficients(**coefficients)


@pytest.mark.parametrize(
    "coefficients",
    (
        {"a1": -1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0, "c1": 1.0, "c2": 1.0},
        {"a1": 1.0, "a2": 0.0, "b1": 1.0, "b2": 1.0, "c1": 1.0, "c2": 1.0},
        {"a1": 1.0, "a2": 1.0, "b1": -1.0, "b2": 1.0, "c1": 1.0, "c2": 1.0},
        {"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 0.0, "c1": 1.0, "c2": 1.0},
        {"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0, "c1": -1.0, "c2": 1.0},
        {"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0, "c1": 1.0, "c2": 0.0},
    ),
)
def test_appendix_e_electron_approximation_validates_coefficients(
    coefficients: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        OstErbElectronApproximationCoefficients(**coefficients)


def test_appendix_e_approximation_validates_evaluation_energy() -> None:
    proton = OstErbProtonIntegralApproximation(
        coefficients=OstErbProtonApproximationCoefficients(
            a1=1.0,
            a2=2.0,
            a3=0.0,
            b1=1.0,
            b2=1.0,
        )
    )
    electron = OstErbElectronIntegralApproximation(
        coefficients=OstErbElectronApproximationCoefficients(
            a1=1.0,
            a2=1.0,
            b1=1.0,
            b2=1.0,
            c1=1.0,
            c2=1.0,
        )
    )

    with pytest.raises(ValueError, match="energy"):
        proton.integral_flux_gt_e(0.0)

    with pytest.raises(ValueError, match="energy"):
        electron.differential_flux(0.0)
