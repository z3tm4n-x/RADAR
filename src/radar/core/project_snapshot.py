"""JSON-compatible snapshots for project configuration objects."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import cast

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
    SourceModelSelectionConfig,
)
from radar.core.types import (
    DoseQuantity,
    OrbitType,
    ShieldGeometry,
    SolarActivityLevel,
)
from radar.core.units import Unit


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

def _required_value(data: Mapping[str, object], key: str) -> object:
    """Return required value from mapping."""

    try:
        return data[key]
    except KeyError as exc:
        msg = f"Required project field is missing: {key}"
        raise ValueError(msg) from exc


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return required nested mapping."""

    value = _required_value(data, key)
    if not isinstance(value, Mapping):
        msg = f"Project field must be an object: {key}"
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _optional_mapping(
    data: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    """Return optional nested mapping."""

    if key not in data:
        return None
    return _required_mapping(data, key)


def _required_str(data: Mapping[str, object], key: str) -> str:
    """Return required string value."""

    value = _required_value(data, key)
    if not isinstance(value, str):
        msg = f"Project field must be a string: {key}"
        raise ValueError(msg)
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    """Return required boolean value."""

    value = _required_value(data, key)
    if not isinstance(value, bool):
        msg = f"Project field must be a boolean: {key}"
        raise ValueError(msg)
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    """Return required integer value."""

    value = _required_value(data, key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Project field must be an integer: {key}"
        raise ValueError(msg)
    return value


def _required_float(data: Mapping[str, object], key: str) -> float:
    """Return required floating-point value."""

    value = _required_value(data, key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"Project field must be a number: {key}"
        raise ValueError(msg)
    return float(value)


def _required_float_tuple(data: Mapping[str, object], key: str) -> tuple[float, ...]:
    """Return required tuple of floating-point values."""

    value = _required_value(data, key)
    if not isinstance(value, list):
        msg = f"Project field must be a list: {key}"
        raise ValueError(msg)

    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            msg = f"Project field must contain only numbers: {key}"
            raise ValueError(msg)
        result.append(float(item))

    return tuple(result)


def _methodology_profile(value: str) -> MethodologyProfile:
    """Return methodology profile parsed from saved value."""

    try:
        return MethodologyProfile(value)
    except ValueError as exc:
        msg = f"Unknown methodology profile: {value}"
        raise ValueError(msg) from exc


def _source_model_family(value: str) -> SourceModelFamily:
    """Return source model family parsed from saved value."""

    try:
        return SourceModelFamily(value)
    except ValueError as exc:
        msg = f"Unknown source model family: {value}"
        raise ValueError(msg) from exc


def _solar_activity_level(value: str) -> SolarActivityLevel:
    """Return solar activity level parsed from saved value."""

    try:
        return SolarActivityLevel(value)
    except ValueError as exc:
        msg = f"Unknown solar activity level: {value}"
        raise ValueError(msg) from exc


def _orbit_type(value: str) -> OrbitType:
    """Return orbit type parsed from saved value."""

    try:
        return OrbitType(value)
    except ValueError as exc:
        msg = f"Unknown orbit type: {value}"
        raise ValueError(msg) from exc


def _shield_geometry(value: str) -> ShieldGeometry:
    """Return shield geometry parsed from saved value."""

    try:
        return ShieldGeometry(value)
    except ValueError as exc:
        msg = f"Unknown shield geometry: {value}"
        raise ValueError(msg) from exc


def _dose_quantity(value: str) -> DoseQuantity:
    """Return dose quantity parsed from saved value."""

    try:
        return DoseQuantity(value)
    except ValueError as exc:
        msg = f"Unknown dose quantity: {value}"
        raise ValueError(msg) from exc


def _unit(value: str) -> Unit:
    """Return unit parsed from saved value."""

    try:
        return Unit(value)
    except ValueError as exc:
        msg = f"Unknown unit: {value}"
        raise ValueError(msg) from exc


def mission_config_from_snapshot(data: Mapping[str, object]) -> MissionConfig:
    """Restore mission configuration from saved representation."""

    return MissionConfig(
        launch_year=_required_int(data, "launch_year"),
        lifetime_years=_required_int(data, "lifetime_years"),
        solar_activity_level=_solar_activity_level(
            _required_str(data, "solar_activity_level"),
        ),
        sep_exceedance_probability=_required_float(
            data,
            "sep_exceedance_probability",
        ),
    )


def orbit_config_from_snapshot(data: Mapping[str, object]) -> OrbitConfig:
    """Restore orbit configuration from saved representation."""

    return OrbitConfig(
        orbit_type=_orbit_type(_required_str(data, "orbit_type")),
        perigee_altitude_km=_required_float(data, "perigee_altitude_km"),
        apogee_altitude_km=_required_float(data, "apogee_altitude_km"),
        inclination_deg=_required_float(data, "inclination_deg"),
        argument_of_perigee_deg=_required_float(data, "argument_of_perigee_deg"),
    )


def shielding_config_from_snapshot(data: Mapping[str, object]) -> ShieldingConfig:
    """Restore shielding configuration from saved representation."""

    return ShieldingConfig(
        thicknesses_g_cm2=_required_float_tuple(data, "thicknesses_g_cm2"),
        geometry=_shield_geometry(_required_str(data, "geometry")),
    )


def source_model_selection_from_snapshot(
    data: Mapping[str, object],
) -> SourceModelSelectionConfig:
    """Restore source model selection from saved representation."""

    return SourceModelSelectionConfig(
        profile=_methodology_profile(_required_str(data, "profile")),
        sep_model_family=_source_model_family(_required_str(data, "sep_model_family")),
        gcr_model_family=_source_model_family(_required_str(data, "gcr_model_family")),
        erb_model_family=_source_model_family(_required_str(data, "erb_model_family")),
    )


def methodology_config_from_snapshot(data: Mapping[str, object]) -> MethodologyConfig:
    """Restore methodology configuration from saved representation."""

    methodology = MethodologyConfig(
        profile=_methodology_profile(_required_str(data, "profile")),
        calculate_gcr_electrons=_required_bool(data, "calculate_gcr_electrons"),
        include_bremsstrahlung=_required_bool(data, "include_bremsstrahlung"),
        include_protons_in_let=_required_bool(data, "include_protons_in_let"),
    )

    saved_selection_data = _optional_mapping(data, "source_model_selection")
    if saved_selection_data is not None:
        saved_selection = source_model_selection_from_snapshot(saved_selection_data)
        if saved_selection != methodology.source_model_selection:
            msg = "Source model selection does not match methodology profile."
            raise ValueError(msg)

    return methodology


def calculation_config_from_snapshot(data: Mapping[str, object]) -> CalculationConfig:
    """Restore calculation configuration from saved representation."""

    config = CalculationConfig(
        mission=mission_config_from_snapshot(_required_mapping(data, "mission")),
        orbit=orbit_config_from_snapshot(_required_mapping(data, "orbit")),
        shielding=shielding_config_from_snapshot(_required_mapping(data, "shielding")),
        methodology=methodology_config_from_snapshot(
            _required_mapping(data, "methodology"),
        ),
        kp=_required_int(data, "kp"),
        dose_quantity=_dose_quantity(
            _required_str(_required_mapping(data, "dose"), "quantity"),
        ),
        dose_unit=_unit(_required_str(_required_mapping(data, "dose"), "unit")),
    )

    saved_selection_data = _optional_mapping(data, "source_model_selection")
    if saved_selection_data is not None:
        saved_selection = source_model_selection_from_snapshot(saved_selection_data)
        if saved_selection != config.source_model_selection:
            msg = "Source model selection does not match calculation methodology."
            raise ValueError(msg)

    return config


def calculation_config_from_snapshot_json(data: str) -> CalculationConfig:
    """Restore calculation configuration from saved JSON representation."""

    loaded: object = json.loads(data)
    if not isinstance(loaded, Mapping):
        msg = "Project JSON root must be an object."
        raise ValueError(msg)

    return calculation_config_from_snapshot(cast("Mapping[str, object]", loaded))
