from radar.calculation_protocol import (
    CalculationProtocolEntry,
    calculation_protocol_entries,
    calculation_protocol_snapshot,
    calculation_protocol_text,
    methodology_profile_title,
    source_model_family_title,
)
from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import CalculationConfig, MethodologyConfig, MissionConfig, OrbitConfig
from radar.project_file import PROJECT_PROGRAM_NAME, PROJECT_SCHEMA_VERSION
from radar.version import __version__


def _calculation_config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR),
        kp=4,
    )


def _value(
    entries: tuple[CalculationProtocolEntry, ...],
    section: str,
    parameter: str,
) -> str:
    for entry in entries:
        if entry.section == section and entry.parameter == parameter:
            return entry.value

    raise AssertionError(f"Protocol entry not found: {section} / {parameter}")


def test_methodology_profile_title() -> None:
    assert (
        methodology_profile_title(MethodologyProfile.OST_WITH_GOST_SEP_GCR)
        == "ОСТ 134-1044-2007 + ГОСТ СКЛ + ГОСТ ГКЛ"
    )


def test_source_model_family_title() -> None:
    assert source_model_family_title(SourceModelFamily.GOST_SEP) == "ГОСТ СКЛ"
    assert source_model_family_title(SourceModelFamily.GOST_GCR) == "ГОСТ ГКЛ"
    assert (
        source_model_family_title(SourceModelFamily.OST_134_1044_2007)
        == "ОСТ 134-1044-2007"
    )


def test_calculation_protocol_entries_include_program_information() -> None:
    entries = calculation_protocol_entries(_calculation_config())

    assert _value(entries, "Программа", "Название программы") == PROJECT_PROGRAM_NAME
    assert _value(entries, "Программа", "Версия программы") == __version__
    assert (
        _value(entries, "Программа", "Версия схемы файла проекта")
        == PROJECT_SCHEMA_VERSION
    )


def test_calculation_protocol_entries_include_methodology_information() -> None:
    entries = calculation_protocol_entries(_calculation_config())

    assert (
        _value(entries, "Методика расчёта", "Профиль методики")
        == "ОСТ 134-1044-2007 + ГОСТ СКЛ + ГОСТ ГКЛ"
    )
    assert _value(entries, "Методика расчёта", "Расчёт электронов ГКЛ") == "нет"
    assert _value(entries, "Методика расчёта", "Учёт тормозного излучения") == "да"
    assert (
        _value(entries, "Методика расчёта", "Учёт протонов при расчёте ЛПЭ")
        == "да"
    )


def test_calculation_protocol_entries_include_selected_source_models() -> None:
    entries = calculation_protocol_entries(_calculation_config())

    assert _value(entries, "Модели источников излучения", "СКЛ") == "ГОСТ СКЛ"
    assert _value(entries, "Модели источников излучения", "ГКЛ") == "ГОСТ ГКЛ"
    assert (
        _value(entries, "Модели источников излучения", "ЕРПЗ")
        == "ОСТ 134-1044-2007"
    )


def test_calculation_protocol_entries_include_mission_orbit_and_shielding() -> None:
    entries = calculation_protocol_entries(_calculation_config())

    assert _value(entries, "Миссия", "Год запуска") == "2028"
    assert _value(entries, "Миссия", "Срок активного существования, лет") == "7"
    assert _value(entries, "Миссия", "Уровень солнечной активности") == "средняя"
    assert _value(entries, "Орбита", "Тип орбиты") == "круговая"
    assert _value(entries, "Орбита", "Высота перигея, км") == "35786"
    assert _value(entries, "Орбита", "Наклонение, град") == "0"
    assert _value(entries, "Защита", "Геометрия защиты") == "сферическая"


def test_calculation_protocol_entries_include_calculation_parameters() -> None:
    entries = calculation_protocol_entries(_calculation_config())

    assert _value(entries, "Параметры расчёта", "Kp") == "4"
    assert _value(entries, "Параметры расчёта", "Единица измерения дозы")


def test_calculation_protocol_snapshot_is_json_compatible() -> None:
    snapshot = calculation_protocol_snapshot(_calculation_config())

    assert isinstance(snapshot, tuple)
    assert snapshot[0] == {
        "section": "Программа",
        "parameter": "Название программы",
        "value": PROJECT_PROGRAM_NAME,
    }


def test_calculation_protocol_text_is_human_readable() -> None:
    text = calculation_protocol_text(_calculation_config())

    assert "Программа" in text
    assert "Методика расчёта" in text
    assert "Модели источников излучения" in text
    assert "СКЛ: ГОСТ СКЛ" in text
    assert "ГКЛ: ГОСТ ГКЛ" in text
