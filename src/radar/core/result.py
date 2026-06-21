"""Calculation result containers for RADAR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Mapping

from radar.core.log import CalculationLog, LogLevel
from radar.core.project import CalculationConfig

if TYPE_CHECKING:
    from radar.output_tables import OutputTable


class ComponentStatus(StrEnum):
    """Status of one calculation component."""

    NOT_STARTED = "not_started"
    SKIPPED = "skipped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ComponentStatusEntry:
    """Status record for one named calculation component."""

    component: str
    status: ComponentStatus

    def __post_init__(self) -> None:
        if not self.component:
            msg = "Component name must not be empty."
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelInfo:
    """Model implementation information included in calculation results."""

    name: str
    version: str
    status: str
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            msg = "Model name must not be empty."
            raise ValueError(msg)

        if not self.version:
            msg = "Model version must not be empty."
            raise ValueError(msg)

        if not self.status:
            msg = "Model status must not be empty."
            raise ValueError(msg)


@dataclass(frozen=True)
class InputDataInfo:
    """Input data reference included in calculation results."""

    name: str
    source: str
    table_id: str
    values: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            msg = "Input data name must not be empty."
            raise ValueError(msg)

        if not self.source:
            msg = "Input data source must not be empty."
            raise ValueError(msg)

        if not self.table_id:
            msg = "Input data table_id must not be empty."
            raise ValueError(msg)

        for key, value in self.values:
            if not key:
                msg = "Input data value key must not be empty."
                raise ValueError(msg)

            if not value:
                msg = "Input data value must not be empty."
                raise ValueError(msg)


@dataclass(frozen=True)
class CalculationResult:
    """Top-level immutable calculation result container."""

    config: CalculationConfig
    log: CalculationLog = CalculationLog()
    component_statuses: tuple[ComponentStatusEntry, ...] = ()
    model_info: tuple[ModelInfo, ...] = ()
    input_data_info: tuple[InputDataInfo, ...] = ()
    output_tables: tuple[OutputTable, ...] = ()

    def add_log_entry(
        self,
        level: LogLevel,
        stage: str,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> CalculationResult:
        """Return a new result with an appended log entry."""

        return replace(
            self,
            log=self.log.add(
                level=level,
                stage=stage,
                message=message,
                details=details,
            ),
        )

    def set_component_status(
        self,
        component: str,
        status: ComponentStatus,
    ) -> CalculationResult:
        """Return a new result with updated component status."""

        entry = ComponentStatusEntry(component=component, status=status)
        other_entries = tuple(
            existing_entry
            for existing_entry in self.component_statuses
            if existing_entry.component != component
        )

        return replace(self, component_statuses=(*other_entries, entry))

    def component_status(self, component: str) -> ComponentStatus:
        """Return status of a component or NOT_STARTED if it was never registered."""

        if not component:
            msg = "Component name must not be empty."
            raise ValueError(msg)

        for entry in self.component_statuses:
            if entry.component == component:
                return entry.status

        return ComponentStatus.NOT_STARTED

    def set_model_info(self, model: ModelInfo) -> CalculationResult:
        """Return a new result with inserted or replaced model information."""

        other_models = tuple(
            existing_model
            for existing_model in self.model_info
            if existing_model.name != model.name
        )

        return replace(self, model_info=(*other_models, model))

    def set_input_data_info(self, info: InputDataInfo) -> CalculationResult:
        """Return a new result with inserted or replaced input data information."""

        other_entries = tuple(
            existing_info
            for existing_info in self.input_data_info
            if existing_info.name != info.name
        )

        return replace(self, input_data_info=(*other_entries, info))

    def set_output_table(self, table: OutputTable) -> CalculationResult:
        """Return a new result with inserted or replaced output table."""

        other_tables = tuple(
            existing_table
            for existing_table in self.output_tables
            if existing_table.table_id != table.table_id
        )

        return replace(self, output_tables=(*other_tables, table))

    def has_errors(self) -> bool:
        """Return True if result log contains errors or any component failed."""

        if self.log.has_errors():
            return True

        return any(
            entry.status is ComponentStatus.FAILED
            for entry in self.component_statuses
        )
