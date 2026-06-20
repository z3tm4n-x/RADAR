import json

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
    SourceModelSelectionConfig,
)
from radar.core.project_snapshot import (
    calculation_config_snapshot,
    calculation_config_snapshot_json,
    methodology_config_snapshot,
    mission_config_snapshot,
    orbit_config_snapshot,
    shielding_config_snapshot,
    source_model_selection_snapshot,
)
from radar.core.types import ShieldGeometry, SolarActivityLevel


def test_mission_config_snapshot() -> None:
    mission = MissionConfig(
        launch_year=2028,
        lifetime_years=7,
        solar_activity_level=SolarActivityLevel.MAXIMUM,
        sep_exceedance_probability=0.95,
    )

    assert mission_config_snapshot(mission) == {
        "launch_year": 2028,
        "lifetime_years": 7,
        "solar_activity_level": "maximum",
        "sep_exceedance_probability": 0.95,
    }


def test_orbit_config_snapshot() -> None:
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    assert orbit_config_snapshot(orbit) == {
        "orbit_type": "circular",
        "perigee_altitude_km": 35786.0,
        "apogee_altitude_km": 35786.0,
        "inclination_deg": 0.0,
        "argument_of_perigee_deg": 0.0,
    }


def test_shielding_config_snapshot() -> None:
    shielding = ShieldingConfig(
        thicknesses_g_cm2=(10.0, 1.0),
        geometry=ShieldGeometry.SPHERE,
    )

    assert shielding_config_snapshot(shielding) == {
        "thicknesses_g_cm2": [1.0, 10.0],
        "geometry": "sphere",
    }


def test_source_model_selection_snapshot() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    assert source_model_selection_snapshot(selection) == {
        "profile": "ost_with_gost_sep_gcr",
        "sep_model_family": SourceModelFamily.GOST_SEP.value,
        "gcr_model_family": SourceModelFamily.GOST_GCR.value,
        "erb_model_family": SourceModelFamily.OST_134_1044_2007.value,
    }


def test_methodology_config_snapshot_contains_profile_and_selected_models() -> None:
    methodology = MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP)

    snapshot = methodology_config_snapshot(methodology)

    assert snapshot["profile"] == "ost_with_gost_sep"
    assert snapshot["uses_ost_134_1044_2007"] is True
    assert snapshot["uses_gost_sep"] is True
    assert snapshot["uses_gost_gcr"] is False
    assert snapshot["calculate_gcr_electrons"] is False
    assert snapshot["include_bremsstrahlung"] is True
    assert snapshot["include_protons_in_let"] is True
    assert snapshot["source_model_selection"] == {
        "profile": "ost_with_gost_sep",
        "sep_model_family": SourceModelFamily.GOST_SEP.value,
        "gcr_model_family": SourceModelFamily.OST_134_1044_2007.value,
        "erb_model_family": SourceModelFamily.OST_134_1044_2007.value,
    }


def test_calculation_config_snapshot_contains_project_sections() -> None:
    mission = MissionConfig(launch_year=2028, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_GCR),
        kp=4,
    )

    snapshot = calculation_config_snapshot(config)

    assert set(snapshot) == {
        "mission",
        "orbit",
        "shielding",
        "methodology",
        "source_model_selection",
        "kp",
        "dose",
    }
    assert snapshot["kp"] == 4
    assert snapshot["source_model_selection"] == {
        "profile": "ost_with_gost_gcr",
        "sep_model_family": SourceModelFamily.OST_134_1044_2007.value,
        "gcr_model_family": SourceModelFamily.GOST_GCR.value,
        "erb_model_family": SourceModelFamily.OST_134_1044_2007.value,
    }


def test_calculation_config_snapshot_json_is_json_compatible() -> None:
    mission = MissionConfig(launch_year=2028, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(mission=mission, orbit=orbit)

    encoded = calculation_config_snapshot_json(config)
    decoded = json.loads(encoded)

    assert decoded == calculation_config_snapshot(config)
