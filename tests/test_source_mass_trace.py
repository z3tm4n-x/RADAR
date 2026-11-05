"""The integer species label must not erase the source's mass convention."""

import pytest

from radar.core.project import MissionConfig
from radar.core.types import Particle
from radar.gcr.components import OstGcrSolarState
from radar.gcr.parameters import get_ost_gcr_ion_parameters
from radar.gcr.source_spectra import calculate_ost_gcr_source_spectra_outside_magnetosphere
from radar.sep.model import OstSepModel, SepModelInput


@pytest.mark.parametrize("symbol", ["He", "Ne", "Fe", "Ni", "Sm", "Pb", "Bi"])
def test_gcr_all_components_retain_tabulated_mass_without_isotope_claim(symbol: str) -> None:
    state = OstGcrSolarState(
        wolf_current=60.0, wolf_lagged=50.0, wolf_min=10.0, wolf_max=100.0,
        cycle_number=24, after_polarity_reversal=True,
    )
    result = calculate_ost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=(10.0, 100.0), solar_state=state,
        nek_wolf_number=60.0, akl_years_from_cycle_start=2.0, symbols=(symbol,),
    )
    ion = result.ion(symbol)
    original_mass = get_ost_gcr_ion_parameters()[symbol].mass_number
    for spectrum in (ion.main, ion.nek, ion.akl, ion.total):
        step = next(s for s in spectrum.transformation_chain if s.name == "gcr_source_spectrum")
        metadata = dict(step.parameters)
        assert float(metadata["source_tabulated_mass_parameter"]) == original_mass
        assert metadata["species_mass_label_policy"] == "legacy_round_not_isotope_identification"
        assert metadata["transport_mass_policy"] == "not_established"
        assert spectrum.species is not None
        assert spectrum.species.mass_number == int(round(original_mass))


def test_sep_mass_trace_preserves_separate_charge_ratio_and_gamma_correction() -> None:
    model = OstSepModel(
        energy_grid_mev=(1.0, 10.0**4.1),
        hze_energy_grid_mev_per_nucleon=(1.0, 10.0**4.1),
        monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0135),) * 60,
    )
    result = model.calculate(SepModelInput(mission=MissionConfig(
        launch_year=2026, lifetime_years=5, sep_exceedance_probability=0.1,
    )))
    iron_products = 0
    for product in result.products:
        spectrum = product.spectrum
        if spectrum.particle is not Particle.HZE:
            continue
        step = next(s for s in spectrum.transformation_chain if s.name == "ost_sep_hze_scaling")
        metadata = dict(step.parameters)
        assert metadata["source_tabulated_mass_parameter"] == metadata["mass_number"]
        assert metadata["species_mass_label_policy"] == "legacy_round_not_isotope_identification"
        assert metadata["transport_mass_policy"] == "not_established"
        assert float(metadata["gamma2_multiplier"]) == 1.68
        if metadata["z"] == "26":
            iron_products += 1
            assert float(metadata["source_tabulated_mass_parameter"]) == 55.8
            assert float(metadata["mass_to_charge"]) == 4.12
    assert iron_products == 3
