"""Calculation execution entry points."""

from __future__ import annotations

from radar.core.log import LogLevel
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, InputDataInfo, ModelInfo
from radar.core.types import Particle, RadiationSource
from radar.model_registry import SourceModelRegistration, source_model_bundle_for_selection
from radar.pipelines.dose import (
    DosePipelineResult,
    calculate_dose_pipeline_from_pipeline_results,
)
from radar.pipelines.erb import ErbPipelineResult
from radar.pipelines.execution import (
    calculate_erb_pipeline_for_config,
    calculate_gcr_pipeline_for_config,
    calculate_sep_pipeline_for_config,
)
from radar.pipelines.gcr import GcrPipelineResult
from radar.pipelines.sep import SepPipelineResult
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
) -> tuple[CalculationResult, SepPipelineResult]:
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

    result = result.add_log_entry(
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

    return result, pipeline_result


def _set_gcr_pipeline_source_state(
    result: CalculationResult,
    config: CalculationConfig,
) -> tuple[CalculationResult, GcrPipelineResult]:
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

    result = result.add_log_entry(
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

    return result, pipeline_result


def _set_erb_pipeline_source_state(
    result: CalculationResult,
    config: CalculationConfig,
) -> tuple[CalculationResult, ErbPipelineResult]:
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

    electron_shielding_status = (
        "calculated_primary_csda"
        if electron_shielded_products
        else "not_calculated"
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

    if electron_on_orbit_products:
        result = result.add_log_entry(
            LogLevel.WARNING,
            _source_component_title(RadiationSource.ERB),
            (
                "\u0421\u043f\u0435\u043a\u0442\u0440\u044b \u044d\u043b\u0435\u043a\u0442\u0440\u043e\u043d\u043e\u0432 \u0437\u0430 \u0437\u0430\u0449\u0438\u0442\u043e\u0439 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u043d\u044b CSDA \u0431\u0435\u0437 "
                "\u043f\u0435\u0440\u0435\u043d\u043e\u0441\u0430 \u0442\u043e\u0440\u043c\u043e\u0437\u043d\u043e\u0433\u043e \u0438\u0437\u043b\u0443\u0447\u0435\u043d\u0438\u044f \u0438 \u0432\u0442\u043e\u0440\u0438\u0447\u043d\u044b\u0445 \u0447\u0430\u0441\u0442\u0438\u0446. "
                "\u0414\u043e\u0437\u0430 \u044d\u043b\u0435\u043a\u0442\u0440\u043e\u043d\u043e\u0432 \u0415\u0420\u041f\u0417 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u043e \u0447\u0435\u0440\u0435\u0437 "
                "SHIELDOSE-2 \u043f\u043e on-orbit \u0441\u043f\u0435\u043a\u0442\u0440\u0443."
            ),
            {
                "electron_shielding_status": electron_shielding_status,
                "bremsstrahlung_transport_status": (
                    "not_included_in_spectrum_transport"
                ),
                "electron_dose_status": (
                    "calculated_by_shieldose2_in_dose_pipeline"
                ),
                "single_event_effects_status": "not_calculated",
            },
        )

    return result, pipeline_result


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


def _dose_by_thickness_table(
    *,
    config: CalculationConfig,
    dose_result: DosePipelineResult,
) -> OutputTable:
    """Return calculated dose-by-thickness output table."""

    return dose_by_thickness_output_table(
        table_id="dose_by_thickness",
        title="\u0414\u043e\u0437\u0430 \u043f\u043e \u0442\u043e\u043b\u0449\u0438\u043d\u0435 \u0437\u0430\u0449\u0438\u0442\u044b",
        points=tuple(
            DoseByThicknessPoint(
                thickness_g_cm2=total.thickness_g_cm2,
                value=total.included_total_rad,
            )
            for total in dose_result.totals_by_thickness
        ),
        dose_quantity=config.dose_quantity,
        value_unit=config.dose_unit,
        metadata={
            "status": "calculated",
            "shield_geometry": config.shielding.geometry.value,
            "model": dose_result.model,
            "version": dose_result.version,
            "included_components": ",".join(
                sorted(
                    {
                        component.component
                        for component in dose_result.included_components
                    }
                )
            ),
            "optional_let_dose_calculated": str(
                bool(dose_result.optional_components)
            ).lower(),
            "optional_let_dose_included_in_total": "false",
        },
    )


def _source_contribution_table(
    *,
    config: CalculationConfig,
    dose_result: DosePipelineResult,
) -> OutputTable:
    """Return calculated source contribution output table at maximum thickness."""

    if dose_result.totals_by_thickness:
        reference_thickness = max(
            total.thickness_g_cm2 for total in dose_result.totals_by_thickness
        )
    else:
        reference_thickness = max(config.shielding.thicknesses_g_cm2)

    values_by_source = {
        RadiationSource.SEP: 0.0,
        RadiationSource.GCR: 0.0,
        RadiationSource.ERB: 0.0,
    }

    for component in dose_result.included_components:
        if component.thickness_g_cm2 == reference_thickness:
            values_by_source[component.source] += component.dose_rad

    total = sum(values_by_source.values())

    contributions = tuple(
        SourceContribution(
            source=source,
            value=value,
            share_percent=(100.0 * value / total) if total > 0.0 else 0.0,
        )
        for source, value in values_by_source.items()
    )

    return source_contribution_output_table(
        table_id="source_contributions",
        title="\u0412\u043a\u043b\u0430\u0434\u044b \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u043e\u0432 \u0438\u0437\u043b\u0443\u0447\u0435\u043d\u0438\u044f \u0432 \u0434\u043e\u0437\u0443",
        contributions=contributions,
        value_title="\u041d\u0430\u043a\u043e\u043f\u043b\u0435\u043d\u043d\u0430\u044f \u0434\u043e\u0437\u0430",
        value_unit=config.dose_unit,
        metadata={
            "status": "calculated",
            "dose_quantity": config.dose_quantity.value,
            "reference_thickness_g_cm2": f"{reference_thickness:g}",
            "aggregation": "included_components_at_reference_thickness",
            "model": dose_result.model,
            "version": dose_result.version,
            "optional_let_dose_included_in_total": "false",
        },
    )


def _set_dose_pipeline_state(
    *,
    result: CalculationResult,
    config: CalculationConfig,
    sep_pipeline_result: SepPipelineResult | None,
    gcr_pipeline_result: GcrPipelineResult | None,
    erb_pipeline_result: ErbPipelineResult | None,
) -> CalculationResult:
    """Calculate dose pipeline and attach final dose output tables."""

    component = "dose_pipeline"
    result = result.set_component_status(component, ComponentStatus.NOT_STARTED)

    dose_result = calculate_dose_pipeline_from_pipeline_results(
        config=config,
        sep_result=sep_pipeline_result,
        gcr_result=gcr_pipeline_result,
        erb_result=erb_pipeline_result,
    )

    result = result.set_model_info(
        ModelInfo(
            name=dose_result.model,
            version=dose_result.version,
            status="calculated",
            source="RADAR dose backends",
        )
    )
    result = result.set_output_table(
        _dose_by_thickness_table(
            config=config,
            dose_result=dose_result,
        )
    )
    result = result.set_output_table(
        _source_contribution_table(
            config=config,
            dose_result=dose_result,
        )
    )
    result = result.set_component_status(component, ComponentStatus.COMPLETED)

    return result.add_log_entry(
        LogLevel.INFO,
        component,
        "\u0420\u0430\u0441\u0447\u0451\u0442 \u0434\u043e\u0437\u044b \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d \u0447\u0435\u0440\u0435\u0437 dose pipeline.",
        {
            "components": str(len(dose_result.components)),
            "included_components": str(len(dose_result.included_components)),
            "optional_components": str(len(dose_result.optional_components)),
            "thickness_count": str(len(dose_result.totals_by_thickness)),
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
    pipeline execution. Dose output tables are calculated from the source
    pipeline products. SEE remains explicitly deferred.
    """

    result = CalculationResult(config=config)
    result = result.add_log_entry(
        LogLevel.INFO,
        "\u0437\u0430\u043f\u0443\u0441\u043a \u0440\u0430\u0441\u0447\u0451\u0442\u0430",
        "\u041d\u0430\u0447\u0430\u0442 \u0432\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0439 \u0437\u0430\u043f\u0443\u0441\u043a \u0440\u0430\u0441\u0447\u0451\u0442\u0430.",
        {
            "profile": config.methodology.profile.value,
            "kp": str(config.kp),
        },
    )

    result = _set_solar_activity_state(result, config)

    bundle = source_model_bundle_for_selection(config.source_model_selection)

    sep_pipeline_result: SepPipelineResult | None = None
    gcr_pipeline_result: GcrPipelineResult | None = None
    erb_pipeline_result: ErbPipelineResult | None = None

    for registration in bundle.registrations:
        if registration.source is RadiationSource.SEP:
            result, sep_pipeline_result = _set_sep_pipeline_source_state(
                result,
                config,
            )
            continue

        if registration.source is RadiationSource.GCR:
            result, gcr_pipeline_result = _set_gcr_pipeline_source_state(
                result,
                config,
            )
            continue

        if registration.source is RadiationSource.ERB:
            result, erb_pipeline_result = _set_erb_pipeline_source_state(
                result,
                config,
            )
            continue

        result = _set_placeholder_source_state(result, registration)

    result = _set_dose_pipeline_state(
        result=result,
        config=config,
        sep_pipeline_result=sep_pipeline_result,
        gcr_pipeline_result=gcr_pipeline_result,
        erb_pipeline_result=erb_pipeline_result,
    )
    result = result.set_output_table(_placeholder_single_event_effects_table(config))

    return result.add_log_entry(
        LogLevel.WARNING,
        "\u0432\u044b\u0445\u043e\u0434\u043d\u044b\u0435 \u0442\u0430\u0431\u043b\u0438\u0446\u044b",
        "\u0421\u0444\u043e\u0440\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u044b \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u043d\u043d\u044b\u0435 \u0442\u0430\u0431\u043b\u0438\u0446\u044b \u0434\u043e\u0437\u044b \u0438 \u0432\u0440\u0435\u043c\u0435\u043d\u043d\u0430\u044f \u0442\u0430\u0431\u043b\u0438\u0446\u0430 \u043e\u0434\u0438\u043d\u043e\u0447\u043d\u044b\u0445 \u044d\u0444\u0444\u0435\u043a\u0442\u043e\u0432.",
        {
            "tables": str(len(result.output_tables)),
            "single_event_effects_status": "placeholder",
        },
    )


