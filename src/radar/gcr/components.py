"""Scalar GCR, NEK and AKL component formulas."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Final, Literal, cast

from radar.gcr.kinematics import (
    MEV_PER_GEV,
    gcr_beta_from_mev_per_nucleon,
    gcr_rigidity_gv_from_mev_per_nucleon,
)
from radar.gcr.parameters import AklParameters, GcrIonParameters, NekParameters

GostWolfVersion = Literal["1.0", "2.0"]

GOST_MODULATION_BASE_GV: Final[float] = 0.379
GOST_WOLF_1_MODULATION_SLOPE_GV: Final[float] = 6.2e-3
GOST_WOLF_2_MODULATION_SLOPE_GV: Final[float] = 4.4e-3
GOST_MODULATION_EXPONENT: Final[float] = 4.5

OST_MAIN_GCR_UNITS_NOTE: Final[str] = (
    "OST Appendix V.1 is evaluated in the same numerical normalization as GOST: "
    "m^-2 sr^-1 s^-1 (GeV/nucleon)^-1. The printed cm^-2/MeV unit label is "
    "treated as a dimensional misprint; no 1e-3 normalization factor is applied "
    "inside the main OST GCR core."
)

OST_R0_WOLF_LOW_EXPONENT: Final[float] = 0.09
OST_R0_WOLF_HIGH_COEFFICIENT: Final[float] = 4.0e-4
OST_R0_WOLF_HIGH_EXPONENT: Final[float] = 1.45
OST_MAIN_MODULATION_RIGIDITY_EXPONENT: Final[float] = 1.15
OST_DELTA_BASE: Final[float] = 5.5
OST_DELTA_DRIFT_COEFFICIENT: Final[float] = 1.13
OST_DRIFT_WOLF_EXPONENT: Final[float] = 2.7

OST_DELAY_T_PLUS_COEFFICIENT_MONTHS: Final[float] = 16.5
OST_DELAY_T_PLUS_EXPONENT: Final[float] = -0.10
OST_DELAY_T_MINUS_COEFFICIENT_MONTHS: Final[float] = 7.5
OST_DELAY_T_MINUS_EXPONENT: Final[float] = -0.45
OST_DELAY_TAU_EXPONENT: Final[float] = 0.2

OST_AKL_TIME_ATTENUATION_COEFFICIENT: Final[float] = 0.08
OST_AKL_TIME_ATTENUATION_EXPONENT: Final[float] = 2.5
OST_AKL_GAMMA_BASE: Final[float] = 0.15
OST_AKL_GAMMA_TIME_COEFFICIENT: Final[float] = 4.5e-5
OST_AKL_GAMMA_TIME_EXPONENT: Final[float] = 4.6


@dataclass(frozen=True)
class GostGcrSolarState:
    """Solar state for the GOST main GCR formula."""

    wolf_number: float
    wolf_version: GostWolfVersion = "1.0"

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            value=self.wolf_number,
            name="GOST Wolf number",
        )

        if self.wolf_version not in ("1.0", "2.0"):
            msg = "GOST Wolf number version must be '1.0' or '2.0'."
            raise ValueError(msg)


@dataclass(frozen=True)
class OstGcrSolarState:
    """Solar state for the OST main GCR formula.

    ``wolf_lagged`` is W(t - Delta t), already evaluated outside the scalar
    flux formula. The delay itself is available as ``ost_gcr_delay_months``.
    """

    wolf_current: float
    wolf_lagged: float
    wolf_min: float
    wolf_max: float
    cycle_number: int
    after_polarity_reversal: bool

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            value=self.wolf_current,
            name="OST current Wolf number",
        )
        _validate_positive_finite(
            value=self.wolf_lagged,
            name="OST lagged Wolf number",
        )
        _validate_non_negative_finite(
            value=self.wolf_min,
            name="OST cycle minimum Wolf number",
        )
        _validate_positive_finite(
            value=self.wolf_max,
            name="OST cycle maximum Wolf number",
        )

        if self.wolf_max <= self.wolf_min:
            msg = "OST cycle maximum Wolf number must be greater than minimum."
            raise ValueError(msg)

        if not isinstance(self.cycle_number, int):
            msg = "OST solar cycle number must be an integer."
            raise ValueError(msg)

        if self.cycle_number < 1:
            msg = "OST solar cycle number must be positive."
            raise ValueError(msg)


def _validate_positive_finite(
    *,
    value: float,
    name: str,
) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value <= 0.0:
        msg = f"{name} must be positive."
        raise ValueError(msg)


def _validate_non_negative_finite(
    *,
    value: float,
    name: str,
) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _energy_gev_to_mev(kinetic_energy_gev_per_nucleon: float) -> float:
    _validate_positive_finite(
        value=kinetic_energy_gev_per_nucleon,
        name="GCR kinetic energy per nucleon",
    )

    return kinetic_energy_gev_per_nucleon * MEV_PER_GEV


def _rigidity_beta_from_gev(
    *,
    kinetic_energy_gev_per_nucleon: float,
    mass_to_charge: float,
) -> tuple[float, float]:
    kinetic_energy_mev_per_nucleon = _energy_gev_to_mev(
        kinetic_energy_gev_per_nucleon,
    )

    rigidity_gv = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=kinetic_energy_mev_per_nucleon,
        mass_to_charge=mass_to_charge,
    )
    beta = gcr_beta_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon,
    )

    return rigidity_gv, beta


def gost_modulation_potential_gv(state: GostGcrSolarState) -> float:
    """Return GOST modulation potential rho in GV."""

    if state.wolf_version == "1.0":
        slope = GOST_WOLF_1_MODULATION_SLOPE_GV
    else:
        slope = GOST_WOLF_2_MODULATION_SLOPE_GV

    return GOST_MODULATION_BASE_GV + slope * state.wolf_number


def gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
    *,
    kinetic_energy_gev_per_nucleon: float,
    parameters: GcrIonParameters,
    solar_state: GostGcrSolarState,
) -> float:
    """Return GOST main GCR directional differential flux.

    Units:
        m^-2 s^-1 sr^-1 (GeV/nucleon)^-1
    """

    if parameters.c == 0.0:
        return 0.0

    rigidity_gv, beta = _rigidity_beta_from_gev(
        kinetic_energy_gev_per_nucleon=kinetic_energy_gev_per_nucleon,
        mass_to_charge=parameters.mass_to_charge,
    )
    rho_gv = gost_modulation_potential_gv(solar_state)

    phi0 = parameters.c * beta**parameters.alpha * rigidity_gv ** (-parameters.gamma)
    modulation = (rigidity_gv / (rigidity_gv + rho_gv)) ** GOST_MODULATION_EXPONENT

    return cast(float, phi0 * modulation * parameters.mass_to_charge / beta)


def ost_gcr_delay_months(
    *,
    rigidity_gv: float,
    wolf_for_tau: float,
    wolf_min: float,
    wolf_max: float,
    cycle_number: int,
) -> float:
    """Return OST GCR solar-activity delay Delta t in months."""

    _validate_positive_finite(value=rigidity_gv, name="OST GCR rigidity")
    _validate_non_negative_finite(
        value=wolf_for_tau,
        name="OST Wolf number for tau",
    )
    _validate_non_negative_finite(
        value=wolf_min,
        name="OST cycle minimum Wolf number",
    )
    _validate_positive_finite(
        value=wolf_max,
        name="OST cycle maximum Wolf number",
    )

    if not isinstance(cycle_number, int):
        msg = "OST solar cycle number must be an integer."
        raise ValueError(msg)

    if cycle_number < 1:
        msg = "OST solar cycle number must be positive."
        raise ValueError(msg)

    t_plus = (
        OST_DELAY_T_PLUS_COEFFICIENT_MONTHS
        * rigidity_gv**OST_DELAY_T_PLUS_EXPONENT
    )
    t_minus = (
        OST_DELAY_T_MINUS_COEFFICIENT_MONTHS
        * rigidity_gv**OST_DELAY_T_MINUS_EXPONENT
    )
    tau_ratio = max((wolf_for_tau - wolf_min) / wolf_max, 0.0)
    tau_sign = 1.0 if cycle_number % 2 == 0 else -1.0
    tau = tau_sign * tau_ratio**OST_DELAY_TAU_EXPONENT

    return cast(float, 0.5 * (t_plus + t_minus) + 0.5 * (t_plus - t_minus) * tau)


def ost_gcr_drift_m(state: OstGcrSolarState) -> float:
    """Return OST drift parameter M(W,n)."""

    ratio = (state.wolf_current - state.wolf_min) / (
        state.wolf_max - state.wolf_min
    )
    ratio = _clamp(ratio, 0.0, 1.0)

    parity_factor = 1.0 if (state.cycle_number - 1) % 2 == 0 else -1.0
    polarity_factor = 1.0 if state.after_polarity_reversal else -1.0

    return cast(
        float,
        parity_factor
        * polarity_factor
        * (1.0 - ratio**OST_DRIFT_WOLF_EXPONENT),
    )


def ost_modulation_potential_r0_gv(wolf_lagged: float) -> float:
    """Return OST modulation potential R0 in GV."""

    _validate_positive_finite(
        value=wolf_lagged,
        name="OST lagged Wolf number",
    )

    return cast(
        float,
        0.28 * wolf_lagged**OST_R0_WOLF_LOW_EXPONENT
        + OST_R0_WOLF_HIGH_COEFFICIENT
        * wolf_lagged**OST_R0_WOLF_HIGH_EXPONENT,
    )


def ost_delta_exponent(
    *,
    beta: float,
    rigidity_gv: float,
    modulation_potential_r0_gv: float,
    drift_m: float,
) -> float:
    """Return OST Delta(R,t) exponent."""

    _validate_positive_finite(value=beta, name="OST GCR beta")
    _validate_positive_finite(value=rigidity_gv, name="OST GCR rigidity")
    _validate_positive_finite(
        value=modulation_potential_r0_gv,
        name="OST modulation potential R0",
    )

    if not isfinite(drift_m):
        msg = "OST drift M must be finite."
        raise ValueError(msg)

    argument = beta * rigidity_gv / modulation_potential_r0_gv

    return (
        OST_DELTA_BASE
        + OST_DELTA_DRIFT_COEFFICIENT
        * drift_m
        * argument
        * exp(-argument)
    )


def ost_main_gcr_differential_flux_directional_m2_s_sr_gev(
    *,
    kinetic_energy_gev_per_nucleon: float,
    parameters: GcrIonParameters,
    solar_state: OstGcrSolarState,
) -> float:
    """Return OST main GCR directional differential flux.

    Units:
        m^-2 s^-1 sr^-1 (GeV/nucleon)^-1

    RADAR intentionally does not apply the printed OST 1e-3 factor here; see
    ``OST_MAIN_GCR_UNITS_NOTE``.
    """

    if parameters.c == 0.0:
        return 0.0

    rigidity_gv, beta = _rigidity_beta_from_gev(
        kinetic_energy_gev_per_nucleon=kinetic_energy_gev_per_nucleon,
        mass_to_charge=parameters.mass_to_charge,
    )
    r0_gv = ost_modulation_potential_r0_gv(solar_state.wolf_lagged)
    drift_m = ost_gcr_drift_m(solar_state)
    delta = ost_delta_exponent(
        beta=beta,
        rigidity_gv=rigidity_gv,
        modulation_potential_r0_gv=r0_gv,
        drift_m=drift_m,
    )

    rigidity_power = rigidity_gv**OST_MAIN_MODULATION_RIGIDITY_EXPONENT
    modulation = (rigidity_power / (rigidity_power + r0_gv)) ** delta

    return cast(
        float,
        parameters.c
        * parameters.mass_to_charge
        * beta ** (parameters.alpha - 1.0)
        * rigidity_gv ** (-parameters.gamma)
        * modulation,
    )


def ost_nek_differential_flux_directional_cm2_s_sr_mev(
    *,
    kinetic_energy_mev_per_nucleon: float,
    parameters: NekParameters,
    wolf_number: float,
) -> float:
    """Return OST NEK directional differential flux.

    Units:
        cm^-2 s^-1 sr^-1 (MeV/nucleon)^-1
    """

    _validate_positive_finite(
        value=kinetic_energy_mev_per_nucleon,
        name="NEK kinetic energy per nucleon",
    )
    _validate_non_negative_finite(
        value=wolf_number,
        name="NEK Wolf number",
    )

    if parameters.ac == 0.0:
        return 0.0

    gamma = parameters.ag - parameters.bg * wolf_number

    return cast(
        float,
        parameters.ac
        * exp(parameters.bc * wolf_number)
        * kinetic_energy_mev_per_nucleon ** (-gamma),
    )


def ost_akl_gamma(years_from_cycle_start: float) -> float:
    """Return OST AKL gamma coefficient."""

    _validate_non_negative_finite(
        value=years_from_cycle_start,
        name="AKL time from solar-cycle start",
    )

    return cast(
        float,
        OST_AKL_GAMMA_BASE
        - OST_AKL_GAMMA_TIME_COEFFICIENT
        * years_from_cycle_start**OST_AKL_GAMMA_TIME_EXPONENT,
    )


def ost_akl_differential_flux_directional_cm2_s_sr_mev(
    *,
    kinetic_energy_mev_per_nucleon: float,
    parameters: AklParameters,
    years_from_cycle_start: float,
) -> float:
    """Return OST AKL directional differential flux.

    Units:
        cm^-2 s^-1 sr^-1 (MeV/nucleon)^-1
    """

    _validate_positive_finite(
        value=kinetic_energy_mev_per_nucleon,
        name="AKL kinetic energy per nucleon",
    )
    _validate_non_negative_finite(
        value=years_from_cycle_start,
        name="AKL time from solar-cycle start",
    )

    if parameters.c == 0.0:
        return 0.0

    gamma = ost_akl_gamma(years_from_cycle_start)

    if gamma <= 0.0:
        return 0.0

    time_attenuation = exp(
        -OST_AKL_TIME_ATTENUATION_COEFFICIENT
        * years_from_cycle_start**OST_AKL_TIME_ATTENUATION_EXPONENT,
    )

    return (
        parameters.c
        * time_attenuation
        * exp(-gamma * kinetic_energy_mev_per_nucleon)
    )


__all__ = [
    "GOST_MODULATION_BASE_GV",
    "GOST_MODULATION_EXPONENT",
    "GOST_WOLF_1_MODULATION_SLOPE_GV",
    "GOST_WOLF_2_MODULATION_SLOPE_GV",
    "OST_MAIN_GCR_UNITS_NOTE",
    "GostGcrSolarState",
    "OstGcrSolarState",
    "gost_main_gcr_differential_flux_directional_m2_s_sr_gev",
    "gost_modulation_potential_gv",
    "ost_akl_differential_flux_directional_cm2_s_sr_mev",
    "ost_akl_gamma",
    "ost_delta_exponent",
    "ost_gcr_delay_months",
    "ost_gcr_drift_m",
    "ost_main_gcr_differential_flux_directional_m2_s_sr_gev",
    "ost_modulation_potential_r0_gv",
    "ost_nek_differential_flux_directional_cm2_s_sr_mev",
]
