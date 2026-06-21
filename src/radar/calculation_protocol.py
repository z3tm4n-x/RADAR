"""Calculation input protocol."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import CalculationConfig
from radar.project_metadata import PROJECT_PROGRAM_NAME, PROJECT_SCHEMA_VERSION
from radar.version import __version__


@dataclass(frozen=True)
class CalculationProtocolEntry:
    """Calculation protocol entry."""

    section: str
    parameter: str
    value: str


_METHODOLOGY_PROFILE_TITLES: dict[MethodologyProfile, str] = {
    MethodologyProfile.OST_134_1044_2007: "ОСТ 134-1044-2007",
    MethodologyProfile.OST_WITH_GOST_SEP: "ОСТ 134-1044-2007 + ГОСТ СКЛ",
    MethodologyProfile.OST_WITH_GOST_GCR: "ОСТ 134-1044-2007 + ГОСТ ГКЛ",
    MethodologyProfile.OST_WITH_GOST_SEP_GCR: (
        "ОСТ 134-1044-2007 + ГОСТ СКЛ + ГОСТ ГКЛ"
    ),
    MethodologyProfile.CUSTOM: "Пользовательская методика",
}

_SOURCE_MODEL_FAMILY_TITLES: dict[SourceModelFamily, str] = {
    SourceModelFamily.OST_134_1044_2007: "ОСТ 134-1044-2007",
    SourceModelFamily.GOST_SEP: "ГОСТ СКЛ",
    SourceModelFamily.GOST_GCR: "ГОСТ ГКЛ",
    SourceModelFamily.CUSTOM: "Пользовательская модель",
}

_SOLAR_ACTIVITY_TITLES: dict[str, str] = {
    "minimum": "минимальный уровень цикла",
    "mean": "средний уровень цикла",
    "maximum": "максимальный уровень цикла",
}

_ORBIT_TYPE_TITLES: dict[str, str] = {
    "circular": "круговая",
    "elliptical": "эллиптическая",
}

_SHIELD_GEOMETRY_TITLES: dict[str, str] = {
    "sphere": "сферическая",
}

_DOSE_QUANTITY_TITLES: dict[str, str] = {
    "absorbed_dose": "поглощённая доза",
    "dose_rate": "мощность поглощённой дозы",
}


def methodology_profile_title(profile: MethodologyProfile) -> str:
    """Return Russian title for methodology profile."""

    return _METHODOLOGY_PROFILE_TITLES.get(profile, profile.value)


def source_model_family_title(model_family: SourceModelFamily) -> str:
    """Return Russian title for source model family."""

    return _SOURCE_MODEL_FAMILY_TITLES.get(model_family, model_family.value)


def _title_from_value(titles: dict[str, str], value: str) -> str:
    """Return Russian title for saved value."""

    return titles.get(value, value)


def _bool_title(value: bool) -> str:
    """Return Russian boolean title."""

    return "да" if value else "нет"


def _float_title(value: float) -> str:
    """Return compact floating-point title."""

    return f"{value:g}"


def _float_sequence_title(values: tuple[float, ...]) -> str:
    """Return compact floating-point sequence title."""

    return ", ".join(_float_title(value) for value in values)


def _entry(section: str, parameter: str, value: object) -> CalculationProtocolEntry:
    """Create calculation protocol entry."""

    return CalculationProtocolEntry(
        section=section,
        parameter=parameter,
        value=str(value),
    )


def calculation_protocol_entries(
    config: CalculationConfig,
) -> tuple[CalculationProtocolEntry, ...]:
    """Return calculation input protocol entries."""

    methodology = config.methodology
    profile_spec = methodology.profile_spec
    source_selection = config.source_model_selection

    return (
        _entry("Программа", "Название программы", PROJECT_PROGRAM_NAME),
        _entry("Программа", "Версия программы", __version__),
        _entry("Программа", "Версия схемы файла проекта", PROJECT_SCHEMA_VERSION),
        _entry(
            "Методика расчёта",
            "Профиль методики",
            methodology_profile_title(methodology.profile),
        ),
        _entry(
            "Методика расчёта",
            "Использованные документы",
            "; ".join(profile_spec.documents),
        ),
        _entry(
            "Методика расчёта",
            "Расчёт электронов ГКЛ",
            _bool_title(methodology.calculate_gcr_electrons),
        ),
        _entry(
            "Методика расчёта",
            "Учёт тормозного излучения",
            _bool_title(methodology.include_bremsstrahlung),
        ),
        _entry(
            "Методика расчёта",
            "Учёт протонов при расчёте ЛПЭ",
            _bool_title(methodology.include_protons_in_let),
        ),
        _entry(
            "Модели источников излучения",
            "СКЛ",
            source_model_family_title(source_selection.sep_model_family),
        ),
        _entry(
            "Модели источников излучения",
            "ГКЛ",
            source_model_family_title(source_selection.gcr_model_family),
        ),
        _entry(
            "Модели источников излучения",
            "ЕРПЗ",
            source_model_family_title(source_selection.erb_model_family),
        ),
        _entry("Миссия", "Год запуска", config.mission.launch_year),
        _entry("Миссия", "Срок активного существования, лет", config.mission.lifetime_years),
        _entry(
            "Миссия",
            "Уровень цикла СА",
            _title_from_value(
                _SOLAR_ACTIVITY_TITLES,
                config.mission.solar_activity_level.value,
            ),
        ),
        _entry(
            "Миссия",
            "Вероятность превышения потока СКЛ",
            _float_title(config.mission.sep_exceedance_probability),
        ),
        _entry(
            "Орбита",
            "Тип орбиты",
            _title_from_value(_ORBIT_TYPE_TITLES, config.orbit.orbit_type.value),
        ),
        _entry(
            "Орбита",
            "Высота перигея, км",
            _float_title(config.orbit.perigee_altitude_km),
        ),
        _entry(
            "Орбита",
            "Высота апогея, км",
            _float_title(config.orbit.apogee_altitude_km),
        ),
        _entry(
            "Орбита",
            "Наклонение, град",
            _float_title(config.orbit.inclination_deg),
        ),
        _entry(
            "Орбита",
            "Аргумент перигея, град",
            _float_title(config.orbit.argument_of_perigee_deg),
        ),
        _entry(
            "Защита",
            "Геометрия защиты",
            _title_from_value(_SHIELD_GEOMETRY_TITLES, config.shielding.geometry.value),
        ),
        _entry(
            "Защита",
            "Толщины защиты, г/см²",
            _float_sequence_title(config.shielding.thicknesses_g_cm2),
        ),
        _entry("Параметры расчёта", "Kp", config.kp),
        _entry(
            "Параметры расчёта",
            "Дозовая величина",
            _title_from_value(_DOSE_QUANTITY_TITLES, config.dose_quantity.value),
        ),
        _entry("Параметры расчёта", "Единица измерения дозы", config.dose_unit.value),
    )


def calculation_protocol_snapshot(config: CalculationConfig) -> tuple[dict[str, str], ...]:
    """Return JSON-compatible calculation protocol representation."""

    return tuple(
        {
            "section": entry.section,
            "parameter": entry.parameter,
            "value": entry.value,
        }
        for entry in calculation_protocol_entries(config)
    )


def calculation_protocol_text(config: CalculationConfig) -> str:
    """Return human-readable calculation protocol."""

    lines: list[str] = []
    current_section = ""

    for entry in calculation_protocol_entries(config):
        if entry.section != current_section:
            current_section = entry.section
            if lines:
                lines.append("")
            lines.append(current_section)
        lines.append(f"{entry.parameter}: {entry.value}")

    return "\n".join(lines)
