"""Calculation result snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from radar.core.log import CalculationLog, CalculationLogEntry, LogLevel
from radar.core.project_snapshot import (
    calculation_config_from_snapshot,
    calculation_config_snapshot,
)
from radar.core.result import (
    CalculationResult,
    ComponentStatus,
    ComponentStatusEntry,
    InputDataInfo,
    ModelInfo,
)
from radar.output_tables import (
    OutputTable,
    output_table_from_snapshot,
    output_table_snapshot,
)


def _required_value(data: Mapping[str, object], key: str) -> object:
    """Return required result field value."""

    try:
        return data[key]
    except KeyError as exc:
        msg = f"Required calculation result field is missing: {key}"
        raise ValueError(msg) from exc


def _required_str(data: Mapping[str, object], key: str) -> str:
    """Return required string result field value."""

    value = _required_value(data, key)
    if not isinstance(value, str):
        msg = f"Calculation result field must be a string: {key}"
        raise ValueError(msg)
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    """Return required boolean result field value."""

    value = _required_value(data, key)
    if not isinstance(value, bool):
        msg = f"Calculation result field must be a boolean: {key}"
        raise ValueError(msg)
    return value


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return required object result field value."""

    value = _required_value(data, key)
    if not isinstance(value, Mapping):
        msg = f"Calculation result field must be an object: {key}"
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _required_array(data: Mapping[str, object], key: str) -> tuple[object, ...]:
    """Return required array result field value."""

    value = _required_value(data, key)
    if not isinstance(value, (list, tuple)):
        msg = f"Calculation result field must be an array: {key}"
        raise ValueError(msg)
    return tuple(value)


def _log_level(value: str) -> LogLevel:
    """Return log level from saved value."""

    try:
        return LogLevel(value)
    except ValueError as exc:
        msg = f"Unsupported calculation log level: {value}"
        raise ValueError(msg) from exc


def _component_status(value: str) -> ComponentStatus:
    """Return component status from saved value."""

    try:
        return ComponentStatus(value)
    except ValueError as exc:
        msg = f"Unsupported calculation component status: {value}"
        raise ValueError(msg) from exc


def _details_snapshot(details: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    """Return JSON-compatible log details."""

    return [
        {
            "key": key,
            "value": value,
        }
        for key, value in details
    ]


def _details_from_snapshot(value: object) -> tuple[tuple[str, str], ...]:
    """Restore log details from saved value."""

    if not isinstance(value, (list, tuple)):
        msg = "Calculation log details field must be an array."
        raise ValueError(msg)

    details: list[tuple[str, str]] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"Calculation log detail entry must be an object: {index}"
            raise ValueError(msg)

        detail = cast("Mapping[str, object]", item)
        details.append(
            (
                _required_str(detail, "key"),
                _required_str(detail, "value"),
            ),
        )

    return tuple(details)


def calculation_log_entry_snapshot(entry: CalculationLogEntry) -> dict[str, object]:
    """Return JSON-compatible calculation log entry."""

    return {
        "level": entry.level.value,
        "stage": entry.stage,
        "message": entry.message,
        "details": _details_snapshot(entry.details),
    }


def calculation_log_entry_from_snapshot(
    data: Mapping[str, object],
) -> CalculationLogEntry:
    """Restore calculation log entry from saved representation."""

    return CalculationLogEntry(
        level=_log_level(_required_str(data, "level")),
        stage=_required_str(data, "stage"),
        message=_required_str(data, "message"),
        details=_details_from_snapshot(_required_value(data, "details")),
    )


def calculation_log_snapshot(log: CalculationLog) -> dict[str, object]:
    """Return JSON-compatible calculation log."""

    return {
        "entries": [
            calculation_log_entry_snapshot(entry)
            for entry in log.entries
        ],
    }


def calculation_log_from_snapshot(data: Mapping[str, object]) -> CalculationLog:
    """Restore calculation log from saved representation."""

    raw_entries = _required_array(data, "entries")
    entries: list[CalculationLogEntry] = []

    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            msg = f"Calculation log entry must be an object: {index}"
            raise ValueError(msg)

        entries.append(
            calculation_log_entry_from_snapshot(
                cast("Mapping[str, object]", entry),
            ),
        )

    return CalculationLog(entries=tuple(entries))


def component_status_entry_snapshot(entry: ComponentStatusEntry) -> dict[str, str]:
    """Return JSON-compatible component status entry."""

    return {
        "component": entry.component,
        "status": entry.status.value,
    }


def component_status_entry_from_snapshot(
    data: Mapping[str, object],
) -> ComponentStatusEntry:
    """Restore component status entry from saved representation."""

    return ComponentStatusEntry(
        component=_required_str(data, "component"),
        status=_component_status(_required_str(data, "status")),
    )


def model_info_snapshot(model: ModelInfo) -> dict[str, str]:
    """Return JSON-compatible model information."""

    return {
        "name": model.name,
        "version": model.version,
        "status": model.status,
        "source": model.source,
    }


def model_info_from_snapshot(data: Mapping[str, object]) -> ModelInfo:
    """Restore model information from saved representation."""

    return ModelInfo(
        name=_required_str(data, "name"),
        version=_required_str(data, "version"),
        status=_required_str(data, "status"),
        source=_required_str(data, "source"),
    )


def input_data_info_snapshot(info: InputDataInfo) -> dict[str, object]:
    """Return JSON-compatible input data information."""

    return {
        "name": info.name,
        "source": info.source,
        "table_id": info.table_id,
        "values": _details_snapshot(info.values),
    }


def input_data_info_from_snapshot(data: Mapping[str, object]) -> InputDataInfo:
    """Restore input data information from saved representation."""

    return InputDataInfo(
        name=_required_str(data, "name"),
        source=_required_str(data, "source"),
        table_id=_required_str(data, "table_id"),
        values=_details_from_snapshot(_required_value(data, "values")),
    )


def calculation_result_snapshot(result: CalculationResult) -> dict[str, object]:
    """Return JSON-compatible calculation result representation."""

    return {
        "calculation_config": calculation_config_snapshot(result.config),
        "log": calculation_log_snapshot(result.log),
        "component_statuses": [
            component_status_entry_snapshot(entry)
            for entry in result.component_statuses
        ],
        "model_info": [
            model_info_snapshot(model)
            for model in result.model_info
        ],
        "input_data_info": [
            input_data_info_snapshot(info)
            for info in result.input_data_info
        ],
        "output_tables": [
            output_table_snapshot(table)
            for table in result.output_tables
        ],
        "has_errors": result.has_errors(),
    }


def _component_statuses_from_snapshot(value: object) -> tuple[ComponentStatusEntry, ...]:
    """Restore component statuses from saved value."""

    if not isinstance(value, (list, tuple)):
        msg = "Calculation result component_statuses field must be an array."
        raise ValueError(msg)

    entries: list[ComponentStatusEntry] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"Calculation component status entry must be an object: {index}"
            raise ValueError(msg)

        entries.append(
            component_status_entry_from_snapshot(
                cast("Mapping[str, object]", item),
            ),
        )

    return tuple(entries)


def _model_info_from_snapshot(value: object) -> tuple[ModelInfo, ...]:
    """Restore model information from saved value."""

    if not isinstance(value, (list, tuple)):
        msg = "Calculation result model_info field must be an array."
        raise ValueError(msg)

    models: list[ModelInfo] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"Calculation model information entry must be an object: {index}"
            raise ValueError(msg)

        models.append(
            model_info_from_snapshot(
                cast("Mapping[str, object]", item),
            ),
        )

    return tuple(models)


def _input_data_info_from_snapshot(value: object) -> tuple[InputDataInfo, ...]:
    """Restore input data information from saved value."""

    if not isinstance(value, (list, tuple)):
        msg = "Calculation result input_data_info field must be an array."
        raise ValueError(msg)

    entries: list[InputDataInfo] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"Calculation input data information entry must be an object: {index}"
            raise ValueError(msg)

        entries.append(
            input_data_info_from_snapshot(
                cast("Mapping[str, object]", item),
            ),
        )

    return tuple(entries)


def _output_tables_from_snapshot(value: object) -> tuple[OutputTable, ...]:
    """Restore output tables from saved value."""

    if not isinstance(value, (list, tuple)):
        msg = "Calculation result output_tables field must be an array."
        raise ValueError(msg)

    tables: list[OutputTable] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"Calculation output table entry must be an object: {index}"
            raise ValueError(msg)

        tables.append(
            output_table_from_snapshot(
                cast("Mapping[str, object]", item),
            ),
        )

    return tuple(tables)


def calculation_result_from_snapshot(
    data: Mapping[str, object],
) -> CalculationResult:
    """Restore calculation result from saved representation."""

    result = CalculationResult(
        config=calculation_config_from_snapshot(
            _required_mapping(data, "calculation_config"),
        ),
        log=calculation_log_from_snapshot(_required_mapping(data, "log")),
        component_statuses=_component_statuses_from_snapshot(
            _required_value(data, "component_statuses"),
        ),
        model_info=_model_info_from_snapshot(_required_value(data, "model_info")),
        input_data_info=_input_data_info_from_snapshot(data.get("input_data_info", ())),
        output_tables=_output_tables_from_snapshot(
            _required_value(data, "output_tables"),
        ),
    )

    saved_has_errors = _required_bool(data, "has_errors")
    if result.has_errors() != saved_has_errors:
        msg = "Calculation result error flag does not match restored result."
        raise ValueError(msg)

    return result
