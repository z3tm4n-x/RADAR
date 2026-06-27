import math

import pytest

from radar.erb.appendix_e import (
    OstErbApproximationFitQuality,
    OstErbElectronApproximationCoefficients,
    OstErbElectronIntegralApproximation,
    OstErbProtonApproximationCoefficients,
    OstErbProtonIntegralApproximation,
    fit_ost_erb_electron_integral_approximation,
    fit_ost_erb_proton_integral_approximation,
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
        2.0 / 6.0**3.0 + 5.0 * math.exp(-12.0)
    )
    assert approximation.differential_flux(2.0) == pytest.approx(
        6.0 / 6.0**4.0 + 30.0 * math.exp(-12.0)
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
        6.0 * math.exp(-6.0)
        + 30.0 * math.exp(-24.0)
        + 56.0 * math.exp(-16.0)
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



def test_appendix_e_fit_quality_validates_values() -> None:
    with pytest.raises(ValueError, match="objective"):
        OstErbApproximationFitQuality(
            objective=-1.0,
            rms_relative_error=0.0,
            max_abs_relative_error=0.0,
            sample_count=1,
            success=True,
        )

    with pytest.raises(ValueError, match="sample count"):
        OstErbApproximationFitQuality(
            objective=0.0,
            rms_relative_error=0.0,
            max_abs_relative_error=0.0,
            sample_count=0,
            success=True,
        )


def test_appendix_e_fits_proton_integral_approximation_relative_objective() -> None:
    reference = OstErbProtonIntegralApproximation(
        coefficients=OstErbProtonApproximationCoefficients(
            a1=1200.0,
            a2=2.0,
            a3=1.0,
            b1=40.0,
            b2=0.5,
        )
    )
    energies = (0.2, 0.6, 1.25, 3.0, 5.0, 12.5, 30.0, 50.0)
    target = tuple(reference.integral_flux_gt_e(energy) for energy in energies)

    fit = fit_ost_erb_proton_integral_approximation(
        energies_mev=energies,
        integral_flux_gt_e=target,
    )

    assert fit.quality.success
    assert fit.quality.sample_count == len(energies)
    assert fit.quality.rms_relative_error < 1.0e-3
    assert fit.quality.max_abs_relative_error < 5.0e-3

    fitted = tuple(fit.approximation.integral_flux_gt_e(energy) for energy in energies)
    assert fitted == pytest.approx(target, rel=5.0e-3)


def test_appendix_e_fits_electron_integral_approximation_relative_objective() -> None:
    reference = OstErbElectronIntegralApproximation(
        coefficients=OstErbElectronApproximationCoefficients(
            a1=800.0,
            a2=0.7,
            b1=300.0,
            b2=0.08,
            c1=50.0,
            c2=2.0,
        )
    )
    energies = (0.16, 0.35, 0.6, 0.9, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5)
    target = tuple(reference.integral_flux_gt_e(energy) for energy in energies)

    fit = fit_ost_erb_electron_integral_approximation(
        energies_mev=energies,
        integral_flux_gt_e=target,
    )

    assert fit.quality.success
    assert fit.quality.sample_count == len(energies)
    assert fit.quality.rms_relative_error < 1.0e-3
    assert fit.quality.max_abs_relative_error < 5.0e-3

    fitted = tuple(fit.approximation.integral_flux_gt_e(energy) for energy in energies)
    assert fitted == pytest.approx(target, rel=5.0e-3)


@pytest.mark.parametrize(
    ("energies_mev", "integral_flux_gt_e", "message"),
    (
        ((1.0,), (1.0,), "at least two"),
        ((1.0, 2.0), (1.0,), "same length"),
        ((1.0, 0.0), (1.0, 1.0), "positive"),
        ((2.0, 1.0), (1.0, 1.0), "sorted"),
        ((1.0, 1.0), (1.0, 1.0), "unique"),
        ((1.0, 2.0), (1.0, 0.0), "positive"),
    ),
)
def test_appendix_e_fit_validates_integral_flux_grid(
    energies_mev: tuple[float, ...],
    integral_flux_gt_e: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_ost_erb_proton_integral_approximation(
            energies_mev=energies_mev,
            integral_flux_gt_e=integral_flux_gt_e,
        )
