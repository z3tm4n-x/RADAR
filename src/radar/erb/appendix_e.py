"""OST 134-1044-2007 Appendix E ERB spectrum helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Any


def _validate_energy_flux_grid(
    *,
    energies_mev: Sequence[float],
    differential_flux: Sequence[float],
) -> None:
    if not energies_mev:
        msg = "Appendix E energy grid must not be empty."
        raise ValueError(msg)

    if len(energies_mev) != len(differential_flux):
        msg = "Appendix E energy and flux grids must have the same length."
        raise ValueError(msg)

    if any(not isfinite(energy) or energy <= 0.0 for energy in energies_mev):
        msg = "Appendix E energy grid values must be finite and positive."
        raise ValueError(msg)

    if tuple(sorted(energies_mev)) != tuple(energies_mev):
        msg = "Appendix E energy grid must be sorted."
        raise ValueError(msg)

    if len(set(energies_mev)) != len(energies_mev):
        msg = "Appendix E energy grid values must be unique."
        raise ValueError(msg)

    if any(not isfinite(value) or value < 0.0 for value in differential_flux):
        msg = "Appendix E differential flux values must be finite and non-negative."
        raise ValueError(msg)


def _power_law_interval_integral(
    *,
    lower_energy_mev: float,
    upper_energy_mev: float,
    lower_flux: float,
    upper_flux: float,
) -> float:
    if lower_flux <= 0.0 and upper_flux <= 0.0:
        return 0.0

    if lower_flux > 0.0 and upper_flux > 0.0:
        exponent = -log(upper_flux / lower_flux) / log(upper_energy_mev / lower_energy_mev)

        if abs(exponent - 1.0) < 1.0e-8:
            return lower_flux * lower_energy_mev * log(upper_energy_mev / lower_energy_mev)

        return float(
            lower_flux
            * lower_energy_mev**exponent
            * (
                upper_energy_mev ** (1.0 - exponent)
                - lower_energy_mev ** (1.0 - exponent)
            )
            / (1.0 - exponent)
        )

    return 0.5 * (lower_flux + upper_flux) * (upper_energy_mev - lower_energy_mev)


def _power_law_tail_integral(
    *,
    energies_mev: Sequence[float],
    differential_flux: Sequence[float],
) -> float:
    positive_indices = [
        index
        for index, value in enumerate(differential_flux)
        if value > 0.0
    ]

    if len(positive_indices) < 2:
        return 0.0

    upper_index = positive_indices[-1]
    lower_index = positive_indices[-2]
    lower_flux = differential_flux[lower_index]
    upper_flux = differential_flux[upper_index]
    lower_energy = energies_mev[lower_index]
    upper_energy = energies_mev[upper_index]

    exponent = -log(upper_flux / lower_flux) / log(upper_energy / lower_energy)

    if exponent <= 1.05:
        return 0.0

    return upper_flux * upper_energy / (exponent - 1.0)


def integrate_differential_flux_tail_power_law(
    *,
    energies_mev: Sequence[float],
    differential_flux: Sequence[float],
) -> tuple[float, ...]:
    """Return integral flux Φ(>E) from a tabulated differential flux φ(E).

    This implements the Appendix E.4 integral spectrum numerically for a finite
    tabulated energy grid. Positive intervals use a power-law interpolation in
    energy; mixed zero/non-zero intervals use trapezoidal integration. The
    high-energy tail is extrapolated from the last two positive points only when
    the inferred power-law exponent gives a finite integral.
    """

    _validate_energy_flux_grid(
        energies_mev=energies_mev,
        differential_flux=differential_flux,
    )

    integral_flux = [0.0 for _ in differential_flux]

    accumulated = _power_law_tail_integral(
        energies_mev=energies_mev,
        differential_flux=differential_flux,
    )
    integral_flux[-1] = accumulated

    for index in range(len(energies_mev) - 2, -1, -1):
        accumulated += _power_law_interval_integral(
            lower_energy_mev=energies_mev[index],
            upper_energy_mev=energies_mev[index + 1],
            lower_flux=differential_flux[index],
            upper_flux=differential_flux[index + 1],
        )
        integral_flux[index] = accumulated

    return tuple(integral_flux)


def _validate_positive_energy(energy_mev: float) -> None:
    if not isfinite(energy_mev) or energy_mev <= 0.0:
        msg = "Appendix E approximation energy must be finite and positive."
        raise ValueError(msg)


def _validate_non_negative_coefficient(
    *,
    name: str,
    value: float,
) -> None:
    if not isfinite(value) or value < 0.0:
        msg = f"Appendix E approximation coefficient {name} must be finite and non-negative."
        raise ValueError(msg)


def _validate_positive_coefficient(
    *,
    name: str,
    value: float,
) -> None:
    if not isfinite(value) or value <= 0.0:
        msg = f"Appendix E approximation coefficient {name} must be finite and positive."
        raise ValueError(msg)


@dataclass(frozen=True)
class OstErbProtonApproximationCoefficients:
    """OST Appendix E.7 proton integral approximation coefficients."""

    a1: float
    a2: float
    a3: float
    b1: float
    b2: float

    def __post_init__(self) -> None:
        _validate_non_negative_coefficient(name="a1", value=self.a1)
        _validate_positive_coefficient(name="a2", value=self.a2)
        _validate_non_negative_coefficient(name="a3", value=self.a3)
        _validate_non_negative_coefficient(name="b1", value=self.b1)
        _validate_positive_coefficient(name="b2", value=self.b2)


@dataclass(frozen=True)
class OstErbProtonDifferentialApproximationCoefficients:
    """OST Appendix E.9 proton differential approximation coefficients."""

    a1: float
    a2: float
    a3: float
    b1: float
    b2: float

    def __post_init__(self) -> None:
        _validate_non_negative_coefficient(name="a1", value=self.a1)
        _validate_positive_coefficient(name="a2", value=self.a2)
        _validate_non_negative_coefficient(name="a3", value=self.a3)
        _validate_non_negative_coefficient(name="b1", value=self.b1)
        _validate_positive_coefficient(name="b2", value=self.b2)


@dataclass(frozen=True)
class OstErbProtonIntegralApproximation:
    """OST Appendix E.7/E.9/E.10 proton ERB approximation."""

    coefficients: OstErbProtonApproximationCoefficients

    @property
    def differential_coefficients(self) -> OstErbProtonDifferentialApproximationCoefficients:
        coefficients = self.coefficients

        return OstErbProtonDifferentialApproximationCoefficients(
            a1=coefficients.a1 * coefficients.a2,
            a2=coefficients.a2 + 1.0,
            a3=coefficients.a3,
            b1=coefficients.b1 * coefficients.b2,
            b2=coefficients.b2,
        )

    def integral_flux_gt_e(self, energy_mev: float) -> float:
        _validate_positive_energy(energy_mev)
        coefficients = self.coefficients

        return float(
            coefficients.a1 / (coefficients.a3 + energy_mev) ** coefficients.a2
            + coefficients.b1 * exp(-coefficients.b2 * energy_mev)
        )

    def differential_flux(self, energy_mev: float) -> float:
        _validate_positive_energy(energy_mev)
        coefficients = self.differential_coefficients

        return float(
            coefficients.a1 / (coefficients.a3 + energy_mev) ** coefficients.a2
            + coefficients.b1 * exp(-coefficients.b2 * energy_mev)
        )


@dataclass(frozen=True)
class OstErbElectronApproximationCoefficients:
    """OST Appendix E.11 electron integral approximation coefficients."""

    a1: float
    a2: float
    b1: float
    b2: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        _validate_non_negative_coefficient(name="a1", value=self.a1)
        _validate_positive_coefficient(name="a2", value=self.a2)
        _validate_non_negative_coefficient(name="b1", value=self.b1)
        _validate_positive_coefficient(name="b2", value=self.b2)
        _validate_non_negative_coefficient(name="c1", value=self.c1)
        _validate_positive_coefficient(name="c2", value=self.c2)


@dataclass(frozen=True)
class OstErbElectronDifferentialApproximationCoefficients:
    """OST Appendix E.12 electron differential approximation coefficients."""

    a1: float
    a2: float
    b1: float
    b2: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        _validate_non_negative_coefficient(name="a1", value=self.a1)
        _validate_positive_coefficient(name="a2", value=self.a2)
        _validate_non_negative_coefficient(name="b1", value=self.b1)
        _validate_positive_coefficient(name="b2", value=self.b2)
        _validate_non_negative_coefficient(name="c1", value=self.c1)
        _validate_positive_coefficient(name="c2", value=self.c2)


@dataclass(frozen=True)
class OstErbElectronIntegralApproximation:
    """OST Appendix E.11/E.12/E.13 electron ERB approximation."""

    coefficients: OstErbElectronApproximationCoefficients

    @property
    def differential_coefficients(self) -> OstErbElectronDifferentialApproximationCoefficients:
        coefficients = self.coefficients

        return OstErbElectronDifferentialApproximationCoefficients(
            a1=coefficients.a1 * coefficients.a2,
            a2=coefficients.a2,
            b1=coefficients.b1 * coefficients.b2,
            b2=coefficients.b2,
            c1=coefficients.c1 * coefficients.c2,
            c2=coefficients.c2,
        )

    def integral_flux_gt_e(self, energy_mev: float) -> float:
        _validate_positive_energy(energy_mev)
        coefficients = self.coefficients

        return float(
            coefficients.a1 * exp(-coefficients.a2 * energy_mev)
            + coefficients.b1 * exp(-coefficients.b2 * energy_mev**2)
            + coefficients.c1 * exp(-coefficients.c2 * energy_mev)
        )

    def differential_flux(self, energy_mev: float) -> float:
        _validate_positive_energy(energy_mev)
        coefficients = self.differential_coefficients

        return float(
            coefficients.a1 * exp(-coefficients.a2 * energy_mev)
            + coefficients.b1 * exp(-coefficients.b2 * energy_mev**2)
            + coefficients.c1 * exp(-coefficients.c2 * energy_mev)
        )


FitResidualFunction = Callable[[tuple[float, ...]], tuple[float, ...]]


@dataclass(frozen=True)
class OstErbApproximationFitQuality:
    """Quality metrics for the Appendix E.8 relative-error fit."""

    objective: float
    rms_relative_error: float
    max_abs_relative_error: float
    sample_count: int
    success: bool

    def __post_init__(self) -> None:
        if not isfinite(self.objective) or self.objective < 0.0:
            msg = "Appendix E fit objective must be finite and non-negative."
            raise ValueError(msg)

        if not isfinite(self.rms_relative_error) or self.rms_relative_error < 0.0:
            msg = "Appendix E fit RMS relative error must be finite and non-negative."
            raise ValueError(msg)

        if not isfinite(self.max_abs_relative_error) or self.max_abs_relative_error < 0.0:
            msg = "Appendix E fit max relative error must be finite and non-negative."
            raise ValueError(msg)

        if self.sample_count < 1:
            msg = "Appendix E fit sample count must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class OstErbProtonIntegralApproximationFit:
    """Result of an Appendix E.8 proton integral approximation fit."""

    approximation: OstErbProtonIntegralApproximation
    quality: OstErbApproximationFitQuality


@dataclass(frozen=True)
class OstErbElectronIntegralApproximationFit:
    """Result of an Appendix E.8 electron integral approximation fit."""

    approximation: OstErbElectronIntegralApproximation
    quality: OstErbApproximationFitQuality


def _validate_integral_fit_grid(
    *,
    energies_mev: Sequence[float],
    integral_flux_gt_e: Sequence[float],
) -> None:
    if len(energies_mev) < 2:
        msg = "Appendix E fit requires at least two energy points."
        raise ValueError(msg)

    if len(energies_mev) != len(integral_flux_gt_e):
        msg = "Appendix E fit energy and integral flux grids must have the same length."
        raise ValueError(msg)

    if any(not isfinite(energy) or energy <= 0.0 for energy in energies_mev):
        msg = "Appendix E fit energy grid values must be finite and positive."
        raise ValueError(msg)

    if tuple(sorted(energies_mev)) != tuple(energies_mev):
        msg = "Appendix E fit energy grid must be sorted."
        raise ValueError(msg)

    if len(set(energies_mev)) != len(energies_mev):
        msg = "Appendix E fit energy grid values must be unique."
        raise ValueError(msg)

    if any(not isfinite(value) or value <= 0.0 for value in integral_flux_gt_e):
        msg = "Appendix E fit integral flux values must be finite and positive."
        raise ValueError(msg)


def _relative_residuals(
    *,
    approximation: Callable[[float], float],
    energies_mev: tuple[float, ...],
    integral_flux_gt_e: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        1.0 - approximation(energy) / target
        for energy, target in zip(energies_mev, integral_flux_gt_e, strict=True)
    )


def _fit_quality(
    *,
    residuals: tuple[float, ...],
    success: bool,
) -> OstErbApproximationFitQuality:
    objective = sum(residual * residual for residual in residuals)
    sample_count = len(residuals)

    return OstErbApproximationFitQuality(
        objective=objective,
        rms_relative_error=sqrt(objective / sample_count),
        max_abs_relative_error=max(abs(residual) for residual in residuals),
        sample_count=sample_count,
        success=success,
    )


def _least_squares(*args: Any, **kwargs: Any) -> Any:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    return least_squares(*args, **kwargs)


def _run_bounded_relative_least_squares(
    *,
    initial_guesses: tuple[tuple[float, ...], ...],
    lower_bounds: tuple[float, ...],
    upper_bounds: tuple[float, ...],
    residual_function: FitResidualFunction,
) -> tuple[tuple[float, ...], bool]:
    best_parameters = initial_guesses[0]
    best_objective = float("inf")
    best_success = False

    for initial_guess in initial_guesses:
        def scipy_residuals(parameters: Any) -> tuple[float, ...]:
            return residual_function(tuple(float(value) for value in parameters))

        result = _least_squares(
            scipy_residuals,
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=20_000,
        )
        parameters = tuple(float(value) for value in result.x)
        residuals = residual_function(parameters)
        objective = sum(residual * residual for residual in residuals)

        if objective < best_objective:
            best_parameters = parameters
            best_objective = objective
            best_success = bool(result.success)

    return best_parameters, best_success


def fit_ost_erb_proton_integral_approximation(
    *,
    energies_mev: Sequence[float],
    integral_flux_gt_e: Sequence[float],
) -> OstErbProtonIntegralApproximationFit:
    """Fit the OST Appendix E.7 proton integral approximation by E.8."""

    _validate_integral_fit_grid(
        energies_mev=energies_mev,
        integral_flux_gt_e=integral_flux_gt_e,
    )
    energies = tuple(float(energy) for energy in energies_mev)
    targets = tuple(float(value) for value in integral_flux_gt_e)
    scale = max(targets)
    energy_min = min(energies)
    energy_max = max(energies)
    energy_span = max(energy_max - energy_min, 1.0)

    def residual_function(parameters: tuple[float, ...]) -> tuple[float, ...]:
        approximation = OstErbProtonIntegralApproximation(
            coefficients=OstErbProtonApproximationCoefficients(
                a1=parameters[0],
                a2=parameters[1],
                a3=parameters[2],
                b1=parameters[3],
                b2=parameters[4],
            )
        )

        return _relative_residuals(
            approximation=approximation.integral_flux_gt_e,
            energies_mev=energies,
            integral_flux_gt_e=targets,
        )

    initial_guesses = (
        (scale * (energy_min + 1.0) ** 2.0, 2.0, 1.0, scale * 0.1, 1.0 / energy_span),
        (scale * (energy_min + 1.0), 1.0, energy_min, scale * 0.5, 1.0 / energy_max),
        (scale * (energy_min + 1.0) ** 3.0, 3.0, 0.0, scale * 0.01, 2.0 / energy_span),
    )
    parameters, success = _run_bounded_relative_least_squares(
        initial_guesses=initial_guesses,
        lower_bounds=(0.0, 1.0e-12, 0.0, 0.0, 1.0e-12),
        upper_bounds=(float("inf"), 20.0, float("inf"), float("inf"), 100.0),
        residual_function=residual_function,
    )
    approximation = OstErbProtonIntegralApproximation(
        coefficients=OstErbProtonApproximationCoefficients(
            a1=parameters[0],
            a2=parameters[1],
            a3=parameters[2],
            b1=parameters[3],
            b2=parameters[4],
        )
    )
    residuals = _relative_residuals(
        approximation=approximation.integral_flux_gt_e,
        energies_mev=energies,
        integral_flux_gt_e=targets,
    )

    return OstErbProtonIntegralApproximationFit(
        approximation=approximation,
        quality=_fit_quality(residuals=residuals, success=success),
    )


def fit_ost_erb_electron_integral_approximation(
    *,
    energies_mev: Sequence[float],
    integral_flux_gt_e: Sequence[float],
) -> OstErbElectronIntegralApproximationFit:
    """Fit the OST Appendix E.11 electron integral approximation by E.8."""

    _validate_integral_fit_grid(
        energies_mev=energies_mev,
        integral_flux_gt_e=integral_flux_gt_e,
    )
    energies = tuple(float(energy) for energy in energies_mev)
    targets = tuple(float(value) for value in integral_flux_gt_e)
    scale = max(targets)
    energy_max = max(energies)
    energy_span = max(energy_max - min(energies), 1.0)

    def residual_function(parameters: tuple[float, ...]) -> tuple[float, ...]:
        approximation = OstErbElectronIntegralApproximation(
            coefficients=OstErbElectronApproximationCoefficients(
                a1=parameters[0],
                a2=parameters[1],
                b1=parameters[2],
                b2=parameters[3],
                c1=parameters[4],
                c2=parameters[5],
            )
        )

        return _relative_residuals(
            approximation=approximation.integral_flux_gt_e,
            energies_mev=energies,
            integral_flux_gt_e=targets,
        )

    initial_guesses = (
        (
            scale * 0.34,
            1.0 / energy_span,
            scale * 0.33,
            1.0 / max(energy_max * energy_max, 1.0),
            scale * 0.33,
            2.0 / energy_span,
        ),
        (scale, 1.0 / energy_span, 0.0, 1.0, 0.0, 1.0 / energy_span),
        (scale * 0.5, 2.0 / energy_span, scale * 0.5, 1.0 / energy_span**2, 0.0, 1.0),
    )
    parameters, success = _run_bounded_relative_least_squares(
        initial_guesses=initial_guesses,
        lower_bounds=(0.0, 1.0e-12, 0.0, 1.0e-12, 0.0, 1.0e-12),
        upper_bounds=(float("inf"), 100.0, float("inf"), 100.0, float("inf"), 100.0),
        residual_function=residual_function,
    )
    approximation = OstErbElectronIntegralApproximation(
        coefficients=OstErbElectronApproximationCoefficients(
            a1=parameters[0],
            a2=parameters[1],
            b1=parameters[2],
            b2=parameters[3],
            c1=parameters[4],
            c2=parameters[5],
        )
    )
    residuals = _relative_residuals(
        approximation=approximation.integral_flux_gt_e,
        energies_mev=energies,
        integral_flux_gt_e=targets,
    )

    return OstErbElectronIntegralApproximationFit(
        approximation=approximation,
        quality=_fit_quality(residuals=residuals, success=success),
    )


__all__ = [
    "OstErbApproximationFitQuality",
    "OstErbElectronApproximationCoefficients",
    "OstErbElectronDifferentialApproximationCoefficients",
    "OstErbElectronIntegralApproximation",
    "OstErbElectronIntegralApproximationFit",
    "OstErbProtonApproximationCoefficients",
    "OstErbProtonDifferentialApproximationCoefficients",
    "OstErbProtonIntegralApproximation",
    "OstErbProtonIntegralApproximationFit",
    "fit_ost_erb_electron_integral_approximation",
    "fit_ost_erb_proton_integral_approximation",
    "integrate_differential_flux_tail_power_law",
]
