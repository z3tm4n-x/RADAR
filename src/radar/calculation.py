"""Calculation execution entry points."""

from __future__ import annotations

from radar.core.log import LogLevel
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, InputDataInfo, ModelInfo
from radar.core.types import Particle, RadiationSource
from radar.model_registry import SourceModelRegistration, source_model_bundle_for_selection
from radar.pipelines.execution import (
    calculate_erb_pipeline_for_config,
    calculate_gcr_pipeline_for_config,
    calculate_sep_pipeline_for_config,
)
from radar.solar_activity.model import build_mission_solar_activity
from radar.solar_activity.ost import ost_wolf_number_cycle_table
from radar.output_tables import OutputTable
from radar.single_event_effects import (
    SingleEventEffectContribution,
    SingleEventEffectMechanism,
    SingleEventEffectPoint,
)
from radar.standard_output_tables import (
    DoseByThicknessPoint,
    SourceContribution,
    dose_by_thickness_output_table,
    single_event_effects_output_table,
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



def _merge_calculation_result(
    result: CalculationResult,
    additional: CalculationResult,
) -> CalculationResult:
    """Merge one calculation result into another."""

    for log_entry in additional.log.entries:
        result = result.add_log_entry(
            level=log_entry.level,
            stage=log_entry.stage,
            message=log_entry.message,
            details=dict(log_entry.details),
        )

    for status_entry in additional.component_statuses:
        result = result.set_component_status(
            component=status_entry.component,
            status=status_entry.status,
        )

    for model in additional.model_info:
        result = result.set_model_info(model)

    for info in additional.input_data_info:
        result = result.set_input_data_info(info)

    for table in additional.output_tables:
        result = result.set_output_table(table)

    return result


def _set_sep_pipeline_source_state(
    result: CalculationResult,
    config: CalculationConfig,
) -> CalculationResult:
    """Run the configured SEP pipeline and register it as the SEP source state."""

    pipeline_result = calculate_sep_pipeline_for_config(config)

    result = _merge_calculation_result(
        result=result,
        additional=pipeline_result.calculation_result,
    )
    result = result.set_component_status(
        component=_source_component_title(RadiationSource.SEP),
        status=ComponentStatus.COMPLETED,
    )

    return result.add_log_entry(
        LogLevel.INFO,
        _source_component_title(RadiationSource.SEP),
        "\u0420\u0430\u0441\u0447\u0451\u0442 \u0421\u041a\u041b \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d \u0447\u0435\u0440\u0435\u0437 \u043f\u043e\u043b\u043d\u044b\u0439 pipeline.",
        {
            "source_products": str(len(pipeline_result.source_products)),
            "on_orbit_products": str(len(pipeline_result.on_orbit_products)),
            "shielded_products": str(len(pipeline_result.shielded_products)),
            "let_products": str(len(pipeline_result.let_products)),
        },
    )


def _set_gcr_pipeline_source_state(
    result: CalculationResult,
    config: CalculationConfig,
) -> CalculationResult:
    """Run the configured GCR pipeline and register it as the GCR source state."""

    pipeline_result = calculate_gcr_pipeline_for_config(config)

    result = _merge_calculation_result(
        result=result,
        additional=pipeline_result.calculation_result,
    )
    result = result.set_component_status(
        component=_source_component_title(RadiationSource.GCR),
        status=ComponentStatus.COMPLETED,
    )

    return result.add_log_entry(
        LogLevel.INFO,
        _source_component_title(RadiationSource.GCR),
        "\u0420\u0430\u0441\u0447\u0451\u0442 \u0413\u041a\u041b \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d \u0447\u0435\u0440\u0435\u0437 \u043f\u043e\u043b\u043d\u044b\u0439 pipeline.",
        {
            "source_products": str(len(pipeline_result.source_products)),
            "on_orbit_products": str(len(pipeline_result.on_orbit_products)),
            "shielded_products": str(len(pipeline_result.shielded_products)),
            "let_products": str(len(pipeline_result.let_products)),
        },
    )


def _set_erb_pipeline_source_state(
    result: CalculationResult,
    config: CalculationConfig,
) -> CalculationResult:
    """Run the configured ERB pipeline and register it as the ERB source state."""

    pipeline_result = calculate_erb_pipeline_for_config(config)

    electron_on_orbit_products = tuple(
        product
        for product in pipeline_result.on_orbit_products
        if product.spectrum.particle is Particle.ELECTRON
    )
    electron_shielded_products = tuple(
        product
        for product in pipeline_result.shielded_products
        if product.spectrum.particle is Particle.ELECTRON
    )

    result = _merge_calculation_result(
        result=result,
        additional=pipeline_result.calculation_result,
    )
    result = result.set_component_status(
        component=_source_component_title(RadiationSource.ERB),
        status=ComponentStatus.COMPLETED,
    )
    result = result.add_log_entry(
        LogLevel.INFO,
        _source_component_title(RadiationSource.ERB),
        "\u0420\u0430\u0441\u0447\u0451\u0442 \u0415\u0420\u041f\u0417 \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d \u0447\u0435\u0440\u0435\u0437 ERB pipeline.",
        {
            "on_orbit_products": str(len(pipeline_result.on_orbit_products)),
            "shielded_products": str(len(pipeline_result.shielded_products)),
            "electron_on_orbit_products": str(len(electron_on_orbit_products)),
            "electron_shielded_products": str(len(electron_shielded_products)),
            "output_tables": str(len(pipeline_result.calculation_result.output_tables)),
        },
    )

    if electron_on_orbit_products and not electron_shielded_products:
        result = result.add_log_entry(
            LogLevel.WARNING,
            _source_component_title(RadiationSource.ERB),
            (
                "\u042d\u043b\u0435\u043a\u0442\u0440\u043e\u043d\u044b \u0437\u0430 \u0437\u0430\u0449\u0438\u0442\u043e\u0439, "
                "\u0442\u043e\u0440\u043c\u043e\u0437\u043d\u043e\u0435 \u0438\u0437\u043b\u0443\u0447\u0435\u043d\u0438\u0435, "
                "\u0434\u043e\u0437\u0430 \u0438 \u043e\u0434\u0438\u043d\u043e\u0447\u043d\u044b\u0435 \u044d\u0444\u0444\u0435\u043a\u0442\u044b "
                "\u0434\u043b\u044f \u0415\u0420\u041f\u0417 \u043f\u043e\u043a\u0430 \u043d\u0435 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u043d\u044b."
            ),
            {
                "electron_shielding_status": "not_calculated",
                "bremsstrahlung_status": "not_calculated",
                "dose_status": "not_calculated",
                "single_event_effects_status": "not_calculated",
            },
        )

    return result


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

    result = result.set_input_data_info(
        InputDataInfo(
            name="СА",
            source=solar_activity.source,
            table_id=solar_activity.table_id,
            values=(
                ("level", config.mission.solar_activity_level.value),
                (
                    "cycle_years",
                    ", ".join(str(year) for year in solar_activity.cycle_years),
                ),
                (
                    "wolf_numbers",
                    ", ".join(f"{value:g}" for value in solar_activity.wolf_numbers),
                ),
            ),
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


def _placeholder_single_event_effects_table(config: CalculationConfig) -> OutputTable:
    """Return zero single event effect table for placeholder execution."""

    mechanisms = tuple(SingleEventEffectMechanism)

    return single_event_effects_output_table(
        table_id="single_event_effects",
        title="Одиночные эффекты по толщине защиты",
        points=tuple(
            SingleEventEffectPoint(
                thickness_g_cm2=thickness,
                contributions=tuple(
                    SingleEventEffectContribution(
                        mechanism=mechanism,
                        event_rate_per_day=0.0,
                        expected_events=0.0,
                    )
                    for mechanism in mechanisms
                ),
            )
            for thickness in config.shielding.thicknesses_g_cm2
        ),
        metadata={
            "status": "placeholder",
        },
    )


def execute_calculation(config: CalculationConfig) -> CalculationResult:
    """Execute RADAR calculation plumbing.

    SEP, GCR and ERB source pipelines are calculated through configured
    pipeline execution. ERB electron shielding, bremsstrahlung, dose and SEE
    remain explicitly deferred.
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
        if registration.source is RadiationSource.SEP:
            result = _set_sep_pipeline_source_state(result, config)
            continue

        if registration.source is RadiationSource.GCR:
            result = _set_gcr_pipeline_source_state(result, config)
            continue

        if registration.source is RadiationSource.ERB:
            result = _set_erb_pipeline_source_state(result, config)
            continue

        result = _set_placeholder_source_state(result, registration)

    result = result.set_output_table(_placeholder_dose_table(config))
    result = result.set_output_table(_placeholder_source_contribution_table(config))
    result = result.set_output_table(_placeholder_single_event_effects_table(config))

    return result.add_log_entry(
        LogLevel.WARNING,
        "выходные таблицы",
        "Сформированы временные нулевые итоговые таблицы дозы и эффектов.",
        {
            "tables": str(len(result.output_tables)),
        },
    )
