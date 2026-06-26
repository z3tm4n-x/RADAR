from math import exp, inf, nan

import pytest

from radar.gcr.components import (
    GOST_MODULATION_BASE_GV,
    GOST_MODULATION_EXPONENT,
    GOST_WOLF_1_MODULATION_SLOPE_GV,
    GOST_WOLF_2_MODULATION_SLOPE_GV,
    OST_MAIN_GCR_UNITS_NOTE,
    GostGcrSolarState,
    OstGcrSolarState,
    gost_main_gcr_differential_flux_directional_m2_s_sr_gev,
    gost_modulation_potential_gv,
    ost_akl_differential_flux_directional_cm2_s_sr_mev,
    ost_akl_gamma,
    ost_delta_exponent,
    ost_gcr_delay_months,
    ost_gcr_drift_m,
    ost_main_gcr_differential_flux_directional_m2_s_sr_gev,
    ost_modulation_potential_r0_gv,
    ost_nek_differential_flux_directional_cm2_s_sr_mev,
)
from radar.gcr.kinematics import (
    gcr_beta_from_mev_per_nucleon,
    gcr_rigidity_gv_from_mev_per_nucleon,
)
from radar.gcr.parameters import (
    GcrIonParameters,
    get_gost_gcr_ion_parameters,
    get_ost_akl_parameters,
    get_ost_gcr_ion_parameters,
    get_ost_nek_parameter,
)


def test_gost_modulation_potential_uses_wolf_version() -> None:
    assert gost_modulation_potential_gv(
        GostGcrSolarState(wolf_number=50.0, wolf_version="1.0"),
    ) == pytest.approx(GOST_MODULATION_BASE_GV + 50.0 * GOST_WOLF_1_MODULATION_SLOPE_GV)

    assert gost_modulation_potential_gv(
        GostGcrSolarState(wolf_number=50.0, wolf_version="2.0"),
    ) == pytest.approx(GOST_MODULATION_BASE_GV + 50.0 * GOST_WOLF_2_MODULATION_SLOPE_GV)


def test_gost_main_gcr_formula_matches_manual_calculation() -> None:
    parameters = get_gost_gcr_ion_parameters()["H"]
    state = GostGcrSolarState(wolf_number=70.0, wolf_version="1.0")
    energy_gev = 1.0
    energy_mev = 1000.0

    rigidity = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=energy_mev,
        mass_to_charge=parameters.mass_to_charge,
    )
    beta = gcr_beta_from_mev_per_nucleon(energy_mev)
    rho = GOST_MODULATION_BASE_GV + 70.0 * GOST_WOLF_1_MODULATION_SLOPE_GV
    phi0 = parameters.c * beta**parameters.alpha * rigidity ** (-parameters.gamma)
    modulation = (rigidity / (rigidity + rho)) ** GOST_MODULATION_EXPONENT
    expected = phi0 * modulation * parameters.mass_to_charge / beta

    assert gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
        kinetic_energy_gev_per_nucleon=energy_gev,
        parameters=parameters,
        solar_state=state,
    ) == pytest.approx(expected)


def test_gost_main_gcr_returns_zero_for_zero_normalization() -> None:
    parameters = GcrIonParameters(
        symbol="Po",
        z=84,
        mass_number=209.0,
        c=0.0,
        gamma=2.70,
        alpha=3.60,
    )

    assert gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
        kinetic_energy_gev_per_nucleon=1.0,
        parameters=parameters,
        solar_state=GostGcrSolarState(wolf_number=50.0),
    ) == pytest.approx(0.0)


def test_ost_gcr_delay_months_matches_manual_calculation_for_even_cycle() -> None:
    rigidity = 2.0
    wolf_for_tau = 60.0
    wolf_min = 10.0
    wolf_max = 100.0

    t_plus = 16.5 * rigidity ** (-0.10)
    t_minus = 7.5 * rigidity ** (-0.45)
    tau = ((wolf_for_tau - wolf_min) / wolf_max) ** 0.2
    expected = 0.5 * (t_plus + t_minus) + 0.5 * (t_plus - t_minus) * tau

    assert ost_gcr_delay_months(
        rigidity_gv=rigidity,
        wolf_for_tau=wolf_for_tau,
        wolf_min=wolf_min,
        wolf_max=wolf_max,
        cycle_number=24,
    ) == pytest.approx(expected)


def test_ost_gcr_delay_months_uses_opposite_tau_sign_for_odd_cycle() -> None:
    even = ost_gcr_delay_months(
        rigidity_gv=2.0,
        wolf_for_tau=60.0,
        wolf_min=10.0,
        wolf_max=100.0,
        cycle_number=24,
    )
    odd = ost_gcr_delay_months(
        rigidity_gv=2.0,
        wolf_for_tau=60.0,
        wolf_min=10.0,
        wolf_max=100.0,
        cycle_number=25,
    )

    assert even != pytest.approx(odd)


def test_ost_gcr_drift_m_matches_manual_calculation() -> None:
    state = OstGcrSolarState(
        wolf_current=40.0,
        wolf_lagged=35.0,
        wolf_min=10.0,
        wolf_max=100.0,
        cycle_number=24,
        after_polarity_reversal=True,
    )
    ratio = (state.wolf_current - state.wolf_min) / (state.wolf_max - state.wolf_min)
    expected = -1.0 * 1.0 * (1.0 - ratio**2.7)

    assert ost_gcr_drift_m(state) == pytest.approx(expected)


def test_ost_modulation_potential_matches_formula() -> None:
    wolf_lagged = 50.0
    expected = 0.28 * wolf_lagged**0.09 + 4.0e-4 * wolf_lagged**1.45

    assert ost_modulation_potential_r0_gv(wolf_lagged) == pytest.approx(expected)


def test_ost_delta_exponent_matches_formula() -> None:
    beta = 0.8
    rigidity = 2.0
    r0 = 0.7
    drift_m = -0.2
    argument = beta * rigidity / r0
    expected = 5.5 + 1.13 * drift_m * argument * exp(-argument)

    assert ost_delta_exponent(
        beta=beta,
        rigidity_gv=rigidity,
        modulation_potential_r0_gv=r0,
        drift_m=drift_m,
    ) == pytest.approx(expected)


def test_ost_main_gcr_formula_matches_manual_calculation() -> None:
    parameters = get_ost_gcr_ion_parameters()["Fe"]
    state = OstGcrSolarState(
        wolf_current=60.0,
        wolf_lagged=50.0,
        wolf_min=10.0,
        wolf_max=100.0,
        cycle_number=24,
        after_polarity_reversal=True,
    )
    energy_gev = 1.0
    energy_mev = 1000.0

    rigidity = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=energy_mev,
        mass_to_charge=parameters.mass_to_charge,
    )
    beta = gcr_beta_from_mev_per_nucleon(energy_mev)
    r0 = 0.28 * state.wolf_lagged**0.09 + 4.0e-4 * state.wolf_lagged**1.45
    ratio = (state.wolf_current - state.wolf_min) / (state.wolf_max - state.wolf_min)
    drift_m = -1.0 * (1.0 - ratio**2.7)
    argument = beta * rigidity / r0
    delta = 5.5 + 1.13 * drift_m * argument * exp(-argument)
    rigidity_power = rigidity**1.15
    modulation = (rigidity_power / (rigidity_power + r0)) ** delta
    expected = (
        parameters.c
        * parameters.mass_to_charge
        * beta ** (parameters.alpha - 1.0)
        * rigidity ** (-parameters.gamma)
        * modulation
    )

    assert ost_main_gcr_differential_flux_directional_m2_s_sr_gev(
        kinetic_energy_gev_per_nucleon=energy_gev,
        parameters=parameters,
        solar_state=state,
    ) == pytest.approx(expected)


def test_ost_main_gcr_units_note_documents_dimensional_correction() -> None:
    assert "printed cm^-2/MeV unit label" in OST_MAIN_GCR_UNITS_NOTE
    assert "no 1e-3 normalization factor" in OST_MAIN_GCR_UNITS_NOTE


def test_ost_nek_formula_uses_corrected_he_parameter() -> None:
    parameters = get_ost_nek_parameter(2)
    assert parameters is not None

    energy = 10.0
    wolf = 70.0
    gamma = parameters.ag - parameters.bg * wolf
    expected = parameters.ac * exp(parameters.bc * wolf) * energy ** (-gamma)

    assert ost_nek_differential_flux_directional_cm2_s_sr_mev(
        kinetic_energy_mev_per_nucleon=energy,
        parameters=parameters,
        wolf_number=wolf,
    ) == pytest.approx(expected)


def test_ost_nek_formula_returns_zero_for_zero_component() -> None:
    parameters = get_ost_nek_parameter(3)
    assert parameters is not None

    assert ost_nek_differential_flux_directional_cm2_s_sr_mev(
        kinetic_energy_mev_per_nucleon=10.0,
        parameters=parameters,
        wolf_number=70.0,
    ) == pytest.approx(0.0)


def test_ost_akl_gamma_matches_formula() -> None:
    time = 5.0
    expected = 0.15 - 4.5e-5 * time**4.6

    assert ost_akl_gamma(time) == pytest.approx(expected)


def test_ost_akl_formula_matches_manual_calculation() -> None:
    parameters = get_ost_akl_parameters()["O"]
    energy = 5.0
    time = 2.0
    gamma = 0.15 - 4.5e-5 * time**4.6
    expected = parameters.c * exp(-0.08 * time**2.5) * exp(-gamma * energy)

    assert ost_akl_differential_flux_directional_cm2_s_sr_mev(
        kinetic_energy_mev_per_nucleon=energy,
        parameters=parameters,
        years_from_cycle_start=time,
    ) == pytest.approx(expected)


def test_ost_akl_formula_returns_zero_when_gamma_is_not_positive() -> None:
    parameters = get_ost_akl_parameters()["He"]

    assert ost_akl_differential_flux_directional_cm2_s_sr_mev(
        kinetic_energy_mev_per_nucleon=1.0,
        parameters=parameters,
        years_from_cycle_start=6.0,
    ) == pytest.approx(0.0)


@pytest.mark.parametrize("bad_value", [-1.0, inf, nan])
def test_solar_state_validation_rejects_bad_wolf_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        GostGcrSolarState(wolf_number=bad_value)


@pytest.mark.parametrize("bad_energy", [0.0, -1.0, inf, nan])
def test_component_fluxes_reject_bad_energy(bad_energy: float) -> None:
    gost_parameters = get_gost_gcr_ion_parameters()["H"]
    ost_parameters = get_ost_gcr_ion_parameters()["H"]
    nek_parameters = get_ost_nek_parameter(1)
    assert nek_parameters is not None
    akl_parameters = get_ost_akl_parameters()["He"]

    with pytest.raises(ValueError, match="finite|positive"):
        gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
            kinetic_energy_gev_per_nucleon=bad_energy,
            parameters=gost_parameters,
            solar_state=GostGcrSolarState(wolf_number=50.0),
        )

    with pytest.raises(ValueError, match="finite|positive"):
        ost_main_gcr_differential_flux_directional_m2_s_sr_gev(
            kinetic_energy_gev_per_nucleon=bad_energy,
            parameters=ost_parameters,
            solar_state=OstGcrSolarState(
                wolf_current=50.0,
                wolf_lagged=40.0,
                wolf_min=10.0,
                wolf_max=100.0,
                cycle_number=24,
                after_polarity_reversal=True,
            ),
        )

    with pytest.raises(ValueError, match="finite|positive"):
        ost_nek_differential_flux_directional_cm2_s_sr_mev(
            kinetic_energy_mev_per_nucleon=bad_energy,
            parameters=nek_parameters,
            wolf_number=50.0,
        )

    with pytest.raises(ValueError, match="finite|positive"):
        ost_akl_differential_flux_directional_cm2_s_sr_mev(
            kinetic_energy_mev_per_nucleon=bad_energy,
            parameters=akl_parameters,
            years_from_cycle_start=1.0,
        )
