from math import inf, nan

import pytest

from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.gcr.components import (
    GostGcrSolarState,
    OstGcrSolarState,
    gost_main_gcr_differential_flux_directional_m2_s_sr_gev,
    ost_akl_differential_flux_directional_cm2_s_sr_mev,
    ost_main_gcr_differential_flux_directional_m2_s_sr_gev,
    ost_nek_differential_flux_directional_cm2_s_sr_mev,
)
from radar.gcr.kinematics import (
    low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev,
    main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev,
    mev_per_nucleon_to_gev_per_nucleon,
)
from radar.gcr.parameters import (
    get_gost_gcr_ion_parameters,
    get_ost_akl_parameters,
    get_ost_gcr_ion_parameters,
    get_ost_nek_parameter,
)
from radar.gcr.source_spectra import (
    GCR_SOURCE_SPECTRA_UNITS_NOTE,
    GOST_GCR_SOURCE_SPECTRA_MODEL,
    OST_GCR_SOURCE_SPECTRA_MODEL,
    calculate_gost_gcr_source_spectra_outside_magnetosphere,
    calculate_ost_gcr_source_spectra_outside_magnetosphere,
)


def _gost_state() -> GostGcrSolarState:
    return GostGcrSolarState(wolf_number=70.0, wolf_version="1.0")


def _ost_state() -> OstGcrSolarState:
    return OstGcrSolarState(
        wolf_current=60.0,
        wolf_lagged=50.0,
        wolf_min=10.0,
        wolf_max=100.0,
        cycle_number=24,
        after_polarity_reversal=True,
    )


def test_gost_gcr_source_spectra_build_total_proton_from_main_and_nek() -> None:
    energy_grid = (10.0, 100.0)
    nek_wolf = 70.0
    result = calculate_gost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=energy_grid,
        solar_state=_gost_state(),
        nek_wolf_number=nek_wolf,
        akl_years_from_cycle_start=2.0,
        symbols=("H",),
    )

    ion = result.ion("H")
    parameters = get_gost_gcr_ion_parameters()["H"]
    nek_parameters = get_ost_nek_parameter(1)
    assert nek_parameters is not None

    expected_main: list[float] = []
    expected_nek: list[float] = []

    for energy_mev in energy_grid:
        main_directional = gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
            kinetic_energy_gev_per_nucleon=mev_per_nucleon_to_gev_per_nucleon(
                energy_mev,
            ),
            parameters=parameters,
            solar_state=_gost_state(),
        )
        expected_main.append(
            main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(
                main_directional,
            )
        )

        nek_directional = ost_nek_differential_flux_directional_cm2_s_sr_mev(
            kinetic_energy_mev_per_nucleon=energy_mev,
            parameters=nek_parameters,
            wolf_number=nek_wolf,
        )
        expected_nek.append(
            low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
                nek_directional,
            )
        )

    assert ion.main.y == pytest.approx(tuple(expected_main))
    assert ion.nek.y == pytest.approx(tuple(expected_nek))
    assert ion.akl.y == pytest.approx((0.0, 0.0))
    assert ion.total.y == pytest.approx(
        tuple(main + nek for main, nek in zip(expected_main, expected_nek, strict=True))
    )


def test_gost_gcr_source_spectra_include_akl_for_supported_hze_ions() -> None:
    energy_grid = (5.0, 20.0)
    nek_wolf = 70.0
    akl_time = 2.0
    result = calculate_gost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=energy_grid,
        solar_state=_gost_state(),
        nek_wolf_number=nek_wolf,
        akl_years_from_cycle_start=akl_time,
        symbols=("He",),
    )

    ion = result.ion("He")
    main_parameters = get_gost_gcr_ion_parameters()["He"]
    nek_parameters = get_ost_nek_parameter(2)
    assert nek_parameters is not None
    akl_parameters = get_ost_akl_parameters()["He"]

    expected_total: list[float] = []
    expected_akl: list[float] = []

    for energy_mev in energy_grid:
        main = main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(
            gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
                kinetic_energy_gev_per_nucleon=mev_per_nucleon_to_gev_per_nucleon(
                    energy_mev,
                ),
                parameters=main_parameters,
                solar_state=_gost_state(),
            )
        )
        nek = low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
            ost_nek_differential_flux_directional_cm2_s_sr_mev(
                kinetic_energy_mev_per_nucleon=energy_mev,
                parameters=nek_parameters,
                wolf_number=nek_wolf,
            )
        )
        akl = low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
            ost_akl_differential_flux_directional_cm2_s_sr_mev(
                kinetic_energy_mev_per_nucleon=energy_mev,
                parameters=akl_parameters,
                years_from_cycle_start=akl_time,
            )
        )
        expected_akl.append(akl)
        expected_total.append(main + nek + akl)

    assert ion.akl.y == pytest.approx(tuple(expected_akl))
    assert ion.total.y == pytest.approx(tuple(expected_total))


def test_ost_gcr_source_spectra_build_total_from_ost_main_and_low_energy_components() -> None:
    energy_grid = (10.0, 100.0)
    nek_wolf = 60.0
    akl_time = 2.0
    result = calculate_ost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=energy_grid,
        solar_state=_ost_state(),
        nek_wolf_number=nek_wolf,
        akl_years_from_cycle_start=akl_time,
        symbols=("O",),
    )

    ion = result.ion("O")
    main_parameters = get_ost_gcr_ion_parameters()["O"]
    nek_parameters = get_ost_nek_parameter(8)
    assert nek_parameters is not None
    akl_parameters = get_ost_akl_parameters()["O"]

    expected_total: list[float] = []

    for energy_mev in energy_grid:
        main = main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(
            ost_main_gcr_differential_flux_directional_m2_s_sr_gev(
                kinetic_energy_gev_per_nucleon=mev_per_nucleon_to_gev_per_nucleon(
                    energy_mev,
                ),
                parameters=main_parameters,
                solar_state=_ost_state(),
            )
        )
        nek = low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
            ost_nek_differential_flux_directional_cm2_s_sr_mev(
                kinetic_energy_mev_per_nucleon=energy_mev,
                parameters=nek_parameters,
                wolf_number=nek_wolf,
            )
        )
        akl = low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
            ost_akl_differential_flux_directional_cm2_s_sr_mev(
                kinetic_energy_mev_per_nucleon=energy_mev,
                parameters=akl_parameters,
                years_from_cycle_start=akl_time,
            )
        )
        expected_total.append(main + nek + akl)

    assert ion.total.y == pytest.approx(tuple(expected_total))


def test_source_spectra_use_radar_gcr_differential_flux_contract() -> None:
    result = calculate_gost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=(10.0, 100.0),
        solar_state=_gost_state(),
        nek_wolf_number=70.0,
        akl_years_from_cycle_start=2.0,
        symbols=("H", "Fe"),
    )

    proton = result.ion("H")
    hze = result.ion("Fe")

    assert result.model == GOST_GCR_SOURCE_SPECTRA_MODEL
    assert proton.particle is Particle.PROTON
    assert hze.particle is Particle.HZE

    for spectrum in result.all_component_spectra:
        assert spectrum.x_unit is Unit.MEV
        assert spectrum.y_unit is Unit.DIFFERENTIAL_FLUX
        assert spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUX
        assert spectrum.source is RadiationSource.GCR

    assert len(result.total_spectra) == 2
    assert result.total_spectra == (proton.total, hze.total)


def test_source_spectra_return_zero_low_energy_components_when_not_applicable() -> None:
    result = calculate_ost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=(10.0, 100.0),
        solar_state=_ost_state(),
        nek_wolf_number=60.0,
        akl_years_from_cycle_start=2.0,
        symbols=("U",),
    )

    ion = result.ion("U")

    assert ion.nek.y == pytest.approx((0.0, 0.0))
    assert ion.akl.y == pytest.approx((0.0, 0.0))
    assert ion.total.y == pytest.approx(ion.main.y)


def test_source_spectra_model_names_identify_components() -> None:
    result = calculate_ost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=(10.0,),
        solar_state=_ost_state(),
        nek_wolf_number=60.0,
        akl_years_from_cycle_start=2.0,
        symbols=("He",),
    )

    ion = result.ion("He")

    assert result.model == OST_GCR_SOURCE_SPECTRA_MODEL
    assert ion.main.model == "ost_gcr_source_outside_magnetosphere:He:main"
    assert ion.nek.model == "ost_gcr_source_outside_magnetosphere:He:nek"
    assert ion.akl.model == "ost_gcr_source_outside_magnetosphere:He:akl"
    assert ion.total.model == "ost_gcr_source_outside_magnetosphere:He:total"


def test_gcr_source_spectra_units_note_documents_both_conversion_paths() -> None:
    assert "4*pi*1e-7" in GCR_SOURCE_SPECTRA_UNITS_NOTE
    assert "4*pi" in GCR_SOURCE_SPECTRA_UNITS_NOTE
    assert "cm^-2 s^-1 MeV^-1" in GCR_SOURCE_SPECTRA_UNITS_NOTE


@pytest.mark.parametrize(
    "energy_grid",
    [
        (),
        (0.0,),
        (-1.0,),
        (10.0, 10.0),
        (20.0, 10.0),
        (nan,),
        (inf,),
    ],
)
def test_source_spectra_reject_bad_energy_grid(
    energy_grid: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="energy grid"):
        calculate_gost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=energy_grid,
            solar_state=_gost_state(),
            nek_wolf_number=70.0,
            akl_years_from_cycle_start=2.0,
            symbols=("H",),
        )


def test_source_spectra_reject_unknown_symbol() -> None:
    with pytest.raises(ValueError, match="Unknown GCR ion symbol"):
        calculate_gost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=(10.0,),
            solar_state=_gost_state(),
            nek_wolf_number=70.0,
            akl_years_from_cycle_start=2.0,
            symbols=("Xx",),
        )


@pytest.mark.parametrize("bad_value", [-1.0, nan, inf])
def test_source_spectra_reject_bad_low_energy_state(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        calculate_gost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=(10.0,),
            solar_state=_gost_state(),
            nek_wolf_number=bad_value,
            akl_years_from_cycle_start=2.0,
            symbols=("H",),
        )

    with pytest.raises(ValueError, match="finite|non-negative"):
        calculate_gost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=(10.0,),
            solar_state=_gost_state(),
            nek_wolf_number=70.0,
            akl_years_from_cycle_start=bad_value,
            symbols=("H",),
        )


def test_source_spectra_ion_lookup_rejects_missing_symbol() -> None:
    result = calculate_gost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=(10.0,),
        solar_state=_gost_state(),
        nek_wolf_number=70.0,
        akl_years_from_cycle_start=2.0,
        symbols=("H",),
    )

    with pytest.raises(KeyError, match="do not contain"):
        result.ion("Fe")


def test_default_source_spectra_cover_all_gcr_elements() -> None:
    result = calculate_gost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=(10.0,),
        solar_state=_gost_state(),
        nek_wolf_number=70.0,
        akl_years_from_cycle_start=2.0,
    )

    assert len(result.ions) == 92
    assert result.ions[0].symbol == "H"
    assert result.ions[-1].symbol == "U"
