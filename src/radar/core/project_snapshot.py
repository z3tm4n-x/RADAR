"""JSON-compatible snapshots for project configuration objects."""

from __future__ import annotations

import json

from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
    SourceModelSelectionConfig,
)


def mission_config_snapshot(mission: MissionConfig) -> dict[str, object]:
    """Return JSON-compatible mission configuration snapshot."""

    return {
        "launch_year": mission.launch_year,
        "lifetime_years": mission.lifetime_years,
        "solar_activity_level": mission.solar_activity_level.value,
        "sep_exceedance_probability": mission.sep_exceedance_probability,
    }


def orbit_config_snapshot(orbit: OrbitConfig) -> dict[str, object]:
    """Return JSON-compatible orbit configuration snapshot."""

    return {
        "orbit_type": orbit.orbit_type.value,
        "perigee_altitude_km": orbit.perigee_altitude_km,
        "apogee_altitude_km": orbit.apogee_altitude_km,
        "inclination_deg": orbit.inclination_deg,
        "argument_of_perigee_deg": orbit.argument_of_perigee_deg,
    }


def shielding_config_snapshot(shielding: ShieldingConfig) -> dict[str, object]:
    """Return JSON-compatible shielding configuration snapshot."""

    return {
        "thicknesses_g_cm2": list(shielding.thicknesses_g_cm2),
        "geometry": shielding.geometry.value,
    }


def source_model_selection_snapshot(
    selection: SourceModelSelectionConfig,
) -> dict[str, object]:
    """Return JSON-compatible source model selection snapshot."""

    return {
        "profile": selection.profile.value,
        "sep_model_family": selection.sep_model_family.value,
        "gcr_model_family": selection.gcr_model_family.value,
        "erb_model_family": selection.erb_model_family.value,
    }


def methodology_config_snapshot(methodology: MethodologyConfig) -> dict[str, object]:
    """Return JSON-compatible methodology configuration snapshot."""

    profile_spec = methodology.profile_spec

    return {
        "profile": methodology.profile.value,
        "documents": list(profile_spec.documents),
        "uses_ost_134_1044_2007": profile_spec.uses_ost_134_1044_2007,
        "uses_gost_sep": profile_spec.uses_gost_sep,
        "uses_gost_gcr": profile_spec.uses_gost_gcr,
        "is_custom": profile_spec.is_custom,
        "calculate_gcr_electrons": methodology.calculate_gcr_electrons,
        "include_bremsstrahlung": methodology.include_bremsstrahlung,
        "include_protons_in_let": methodology.include_protons_in_let,
        "source_model_selection": source_model_selection_snapshot(
            methodology.source_model_selection,
        ),
    }


def calculation_config_snapshot(config: CalculationConfig) -> dict[str, object]:
    """Return JSON-compatible calculation configuration snapshot."""

    return {
        "mission": mission_config_snapshot(config.mission),
        "orbit": orbit_config_snapshot(config.orbit),
        "shielding": shielding_config_snapshot(config.shielding),
        "methodology": methodology_config_snapshot(config.methodology),
        "source_model_selection": source_model_selection_snapshot(
            config.source_model_selection,
        ),
        "kp": config.kp,
        "dose": {
            "quantity": config.dose_quantity.value,
            "unit": config.dose_unit.value,
        },
    }


def calculation_config_snapshot_json(config: CalculationConfig) -> str:
    """Return stable JSON representation of calculation configuration snapshot."""

    return json.dumps(
        calculation_config_snapshot(config),
        ensure_ascii=False,
        sort_keys=True,
    )
