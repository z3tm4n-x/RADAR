"""OST 134-1044-2007 Appendix E ERB spectrum helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log


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


__all__ = [
    "OstErbElectronApproximationCoefficients",
    "OstErbElectronDifferentialApproximationCoefficients",
    "OstErbElectronIntegralApproximation",
    "OstErbProtonApproximationCoefficients",
    "OstErbProtonDifferentialApproximationCoefficients",
    "OstErbProtonIntegralApproximation",
    "integrate_differential_flux_tail_power_law",
]
