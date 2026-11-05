import json

import pytest

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
    source_model_selection_from_snapshot,
    shielding_config_from_snapshot,
    orbit_config_from_snapshot,
    mission_config_from_snapshot,
    methodology_config_from_snapshot,
    calculation_config_from_snapshot_json,
    calculation_config_from_snapshot,
    calculation_config_snapshot,
    calculation_config_snapshot_json,
    methodology_config_snapshot,
    mission_config_snapshot,
    orbit_config_snapshot,
    shielding_config_snapshot,
    source_model_selection_snapshot,
)
from radar.core.types import (
    AssessmentMode,
    DoseQuantity,
    ShieldGeometry,
    SolarActivityLevel,
)
from radar.core.units import Unit


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
        "assessment_mode",
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


def test_mission_config_from_snapshot_round_trip() -> None:
    mission = MissionConfig(
        launch_year=2028,
        lifetime_years=7,
        solar_activity_level=SolarActivityLevel.MAXIMUM,
        sep_exceedance_probability=0.95,
    )

    assert mission_config_from_snapshot(mission_config_snapshot(mission)) == mission


def test_orbit_config_from_snapshot_round_trip() -> None:
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    assert orbit_config_from_snapshot(orbit_config_snapshot(orbit)) == orbit


def test_shielding_config_from_snapshot_round_trip() -> None:
    shielding = ShieldingConfig(
        thicknesses_g_cm2=(10.0, 1.0),
        geometry=ShieldGeometry.SPHERE,
    )

    assert shielding_config_from_snapshot(shielding_config_snapshot(shielding)) == shielding


def test_source_model_selection_from_snapshot_round_trip() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    assert (
        source_model_selection_from_snapshot(source_model_selection_snapshot(selection))
        == selection
    )


def test_methodology_config_from_snapshot_round_trip() -> None:
    methodology = MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_GCR)

    assert methodology_config_from_snapshot(methodology_config_snapshot(methodology)) == methodology


def test_calculation_config_from_snapshot_round_trip() -> None:
    mission = MissionConfig(launch_year=2028, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP),
        kp=4,
    )

    assert calculation_config_from_snapshot(calculation_config_snapshot(config)) == config


def test_calculation_config_from_snapshot_json_round_trip() -> None:
    mission = MissionConfig(launch_year=2028, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR),
        kp=5,
    )

    encoded = calculation_config_snapshot_json(config)

    assert calculation_config_from_snapshot_json(encoded) == config


def test_calculation_config_from_snapshot_rejects_missing_required_field() -> None:
    mission = MissionConfig(launch_year=2028, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(mission=mission, orbit=orbit)
    snapshot = calculation_config_snapshot(config)
    del snapshot["mission"]

    with pytest.raises(ValueError, match="Required project field is missing"):
        calculation_config_from_snapshot(snapshot)


def test_calculation_config_from_snapshot_rejects_model_selection_mismatch() -> None:
    mission = MissionConfig(launch_year=2028, lifetime_years=7)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP),
    )
    snapshot = calculation_config_snapshot(config)
    source_selection = snapshot["source_model_selection"]
    assert isinstance(source_selection, dict)
    source_selection["sep_model_family"] = SourceModelFamily.OST_134_1044_2007.value

    with pytest.raises(ValueError, match="does not match methodology profile"):
        calculation_config_from_snapshot(snapshot)


def test_calculation_config_from_snapshot_json_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="Project JSON root must be an object"):
        calculation_config_from_snapshot_json("[]")


def test_solar_cycle_anchor_and_assessment_mode_round_trip() -> None:
    methodology = MethodologyConfig(
        solar_cycle_reference_start_year=2031,
        solar_cycle_reference_number=26,
    )
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2034, lifetime_years=3),
        orbit=OrbitConfig.circular(altitude_km=500.0, inclination_deg=51.6),
        methodology=methodology,
        assessment_mode=AssessmentMode.PEAK,
        dose_quantity=DoseQuantity.DOSE_RATE,
        dose_unit=Unit.RAD_PER_SECOND,
    )

    restored = calculation_config_from_snapshot(calculation_config_snapshot(config))

    assert restored == config
    assert restored.assessment_mode is AssessmentMode.PEAK
    assert restored.methodology.solar_cycle_reference_start_year == 2031
    assert restored.methodology.solar_cycle_reference_number == 26


def test_old_snapshot_without_solar_anchor_uses_explicit_application_defaults() -> None:
    methodology = MethodologyConfig()
    snapshot = methodology_config_snapshot(methodology)
    del snapshot["solar_cycle_reference_start_year"]
    del snapshot["solar_cycle_reference_number"]
    del snapshot["solar_activity_wolf_version"]
    del snapshot["solar_activity_monthly_policy"]

    restored = methodology_config_from_snapshot(snapshot)

    assert restored.solar_cycle_reference_start_year == 2020
    assert restored.solar_cycle_reference_number == 25
    assert restored.solar_activity_wolf_version == "1.0"
    assert restored.solar_activity_monthly_policy == "ost_g1_annual_step_12_months"


def test_old_calculation_snapshot_without_assessment_mode_defaults_to_cumulative() -> None:
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=2),
        orbit=OrbitConfig.circular(altitude_km=500.0, inclination_deg=51.6),
    )
    snapshot = calculation_config_snapshot(config)
    del snapshot["assessment_mode"]

    restored = calculation_config_from_snapshot(snapshot)

    assert restored.assessment_mode is AssessmentMode.CUMULATIVE
