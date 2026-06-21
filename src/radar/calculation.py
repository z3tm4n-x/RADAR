"""Calculation execution entry points."""

from __future__ import annotations

from radar.core.log import LogLevel
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.types import RadiationSource
from radar.model_registry import SourceModelRegistration, source_model_bundle_for_selection
from radar.solar_activity.model import build_mission_solar_activity
from radar.solar_activity.ost import ost_wolf_number_cycle_table
from radar.output_tables import OutputTable
from radar.standard_output_tables import (
    DoseByThicknessPoint,
    SourceContribution,
    dose_by_thickness_output_table,
    source_contribution_output_table,
)

_SOURCE_COMPONENT_TITLES: dict[RadiationSource, str] = {
    RadiationSource.SEP: "СКЛ",
    RadiationSource.GCR: "ГКЛ",
    RadiationSource.ERB: "ЕРПЗ",
}


def _source_component_title(source: RadiationSource) -> str:
    """Return Russian calculation component title."""

    return _SOURCE_COMPONENT_TITLES[source]


def _model_info_from_registration(
    registration: SourceModelRegistration,
) -> ModelInfo:
    """Return model information for a registered source model."""

    model = registration.model_class()
    metadata = model.metadata

    return ModelInfo(
        name=metadata.name,
        version=metadata.version,
        status="численная часть не реализована",
        source=metadata.document,
    )


def _set_placeholder_source_state(
    result: CalculationResult,
    registration: SourceModelRegistration,
) -> CalculationResult:
    """Add placeholder state for one source model."""

    component = _source_component_title(registration.source)
    model_info = _model_info_from_registration(registration)

    result = result.set_model_info(model_info)
    result = result.set_component_status(component, ComponentStatus.SKIPPED)
    return result.add_log_entry(
        LogLevel.WARNING,
        component,
        "Численная модель пока не реализована; добавлены только сведения о модели.",
        {
            "model": model_info.name,
            "document": model_info.source,
            "version": model_info.version,
        },
    )


def _set_solar_activity_state(
    result: CalculationResult,
    config: CalculationConfig,
) -> CalculationResult:
    """Add OST solar activity data used by this calculation."""

    solar_cycle_table = ost_wolf_number_cycle_table(config.mission.solar_activity_level)
    solar_activity = build_mission_solar_activity(
        mission=config.mission,
        cycle_table=solar_cycle_table,
        reference_start_year=config.mission.launch_year,
    )

    result = result.set_model_info(
        ModelInfo(
            name="solar_activity",
            version=solar_activity.table_id,
            status="использовано",
            source=solar_activity.source,
        )
    )

    return result.add_log_entry(
        LogLevel.INFO,
        "СА",
        (
            "Числа Вольфа по таблице Г.1 ОСТ: "
            f"{', '.join(f'{value:g}' for value in solar_activity.wolf_numbers)}"
        ),
        {
            "cycle_years": ", ".join(str(year) for year in solar_activity.cycle_years),
            "level": config.mission.solar_activity_level.value,
            "table": solar_activity.table_id,
        },
    )


def _placeholder_dose_table(config: CalculationConfig) -> OutputTable:
    """Return zero dose-by-thickness table for placeholder execution."""

    return dose_by_thickness_output_table(
        table_id="dose_by_thickness",
        title="Доза по толщине защиты",
        points=tuple(
            DoseByThicknessPoint(
                thickness_g_cm2=thickness,
                value=0.0,
            )
            for thickness in config.shielding.thicknesses_g_cm2
        ),
        dose_quantity=config.dose_quantity,
        value_unit=config.dose_unit,
        metadata={
            "status": "placeholder",
            "shield_geometry": config.shielding.geometry.value,
        },
    )


def _placeholder_source_contribution_table(config: CalculationConfig) -> OutputTable:
    """Return zero source-contribution table for placeholder execution."""

    return source_contribution_output_table(
        table_id="source_contributions",
        title="Вклады источников излучения",
        contributions=(
            SourceContribution(source=RadiationSource.SEP, value=0.0),
            SourceContribution(source=RadiationSource.GCR, value=0.0),
            SourceContribution(source=RadiationSource.ERB, value=0.0),
        ),
        value_title="Значение",
        value_unit=config.dose_unit,
        metadata={
            "status": "placeholder",
            "dose_quantity": config.dose_quantity.value,
        },
    )


def execute_calculation(config: CalculationConfig) -> CalculationResult:
    """Execute temporary RADAR calculation plumbing.

    This function intentionally does not call normative source-model equations yet.
    It creates a reproducible calculation result containing selected model metadata,
    component states, log entries and placeholder output tables.
    """

    result = CalculationResult(config=config)
    result = result.add_log_entry(
        LogLevel.INFO,
        "запуск расчёта",
        "Начат временный запуск расчёта.",
        {
            "profile": config.methodology.profile.value,
            "kp": str(config.kp),
        },
    )

    result = _set_solar_activity_state(result, config)

    bundle = source_model_bundle_for_selection(config.source_model_selection)

    for registration in bundle.registrations:
        result = _set_placeholder_source_state(result, registration)

    result = result.set_output_table(_placeholder_dose_table(config))
    result = result.set_output_table(_placeholder_source_contribution_table(config))

    return result.add_log_entry(
        LogLevel.WARNING,
        "выходные таблицы",
        "Сформированы временные нулевые таблицы без физического расчёта.",
        {
            "tables": str(len(result.output_tables)),
        },
    )
