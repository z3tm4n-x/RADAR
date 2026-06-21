"""Output table structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TypeAlias, cast

from radar.core.spectra import Spectrum1D
from radar.core.types import SpectrumQuantity

TableCell: TypeAlias = str | int | float | bool | None


class OutputTableKind(StrEnum):
    """Output table kind."""

    SPECTRUM = "spectrum"
    DOSE = "dose"
    SINGLE_EVENT = "single_event"
    SUMMARY = "summary"


@dataclass(frozen=True)
class OutputTableColumn:
    """Output table column."""

    key: str
    title: str
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            msg = "Output table column key must not be empty."
            raise ValueError(msg)

        if not self.title:
            msg = "Output table column title must not be empty."
            raise ValueError(msg)


@dataclass(frozen=True)
class OutputTableRow:
    """Output table row."""

    cells: tuple[TableCell, ...]


@dataclass(frozen=True)
class OutputTable:
    """Rectangular output table."""

    table_id: str
    title: str
    kind: OutputTableKind
    columns: tuple[OutputTableColumn, ...]
    rows: tuple[OutputTableRow, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.table_id:
            msg = "Output table id must not be empty."
            raise ValueError(msg)

        if not self.title:
            msg = "Output table title must not be empty."
            raise ValueError(msg)

        if not self.columns:
            msg = "Output table must contain at least one column."
            raise ValueError(msg)

        keys = tuple(column.key for column in self.columns)
        if len(set(keys)) != len(keys):
            msg = "Output table column keys must be unique."
            raise ValueError(msg)

        for row in self.rows:
            if len(row.cells) != len(self.columns):
                msg = "Output table row must have the same length as columns."
                raise ValueError(msg)

            for cell in row.cells:
                _validate_cell(cell)

        for key, value in self.metadata:
            if not key:
                msg = "Output table metadata key must not be empty."
                raise ValueError(msg)

            if not value:
                msg = "Output table metadata value must not be empty."
                raise ValueError(msg)


_SPECTRUM_VALUE_TITLES: dict[SpectrumQuantity, str] = {
    SpectrumQuantity.DIFFERENTIAL_FLUENCE: "Дифференциальный флюенс",
    SpectrumQuantity.DIFFERENTIAL_FLUX: "Дифференциальный поток",
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX: "Пиковый дифференциальный поток",
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX: "Максимальный дифференциальный поток",
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX: "Средний дифференциальный поток",
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE: "Дифференциальный ЛПЭ-флюенс",
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX: "Дифференциальный ЛПЭ-поток",
}

_LET_QUANTITIES = {
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
}


def _validate_cell(value: TableCell) -> None:
    """Validate output table cell value."""

    if value is None:
        return

    if isinstance(value, bool | str | int):
        return

    if isinstance(value, float):
        if not isfinite(value):
            msg = "Output table float cell must be finite."
            raise ValueError(msg)
        return

    msg = "Output table cell has unsupported value type."
    raise ValueError(msg)


def _cell_from_value(value: object, field: str) -> TableCell:
    """Return table cell from saved value."""

    if value is None:
        return None

    if isinstance(value, bool | str | int):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            msg = f"Output table field must contain only finite floats: {field}"
            raise ValueError(msg)
        return value

    msg = f"Output table field has unsupported value type: {field}"
    raise ValueError(msg)


def output_table_from_rows(
    *,
    table_id: str,
    title: str,
    kind: OutputTableKind,
    columns: Sequence[OutputTableColumn],
    rows: Sequence[Mapping[str, TableCell]],
    metadata: Mapping[str, str] | None = None,
) -> OutputTable:
    """Create rectangular output table from row mappings."""

    column_tuple = tuple(columns)
    column_keys = tuple(column.key for column in column_tuple)
    column_key_set = set(column_keys)
    table_rows: list[OutputTableRow] = []

    for row_index, row in enumerate(rows):
        row_keys = set(row)
        if row_keys != column_key_set:
            msg = (
                f"Output table row keys do not match table columns: "
                f"row {row_index}"
            )
            raise ValueError(msg)

        table_rows.append(
            OutputTableRow(
                cells=tuple(row[key] for key in column_keys),
            ),
        )

    metadata_items: tuple[tuple[str, str], ...] = ()
    if metadata is not None:
        metadata_items = tuple(sorted(metadata.items()))

    return OutputTable(
        table_id=table_id,
        title=title,
        kind=kind,
        columns=column_tuple,
        rows=tuple(table_rows),
        metadata=metadata_items,
    )


def output_table_snapshot(table: OutputTable) -> dict[str, object]:
    """Return JSON-compatible output table representation."""

    return {
        "table_id": table.table_id,
        "title": table.title,
        "kind": table.kind.value,
        "columns": [
            {
                "key": column.key,
                "title": column.title,
                "unit": column.unit,
                "description": column.description,
            }
            for column in table.columns
        ],
        "rows": [
            {
                column.key: cell
                for column, cell in zip(table.columns, row.cells)
            }
            for row in table.rows
        ],
        "metadata": [
            {
                "key": key,
                "value": value,
            }
            for key, value in table.metadata
        ],
    }


def output_tables_snapshot(tables: Sequence[OutputTable]) -> tuple[dict[str, object], ...]:
    """Return JSON-compatible output tables representation."""

    return tuple(output_table_snapshot(table) for table in tables)


def _required_value(data: Mapping[str, object], key: str) -> object:
    """Return required output table field value."""

    try:
        return data[key]
    except KeyError as exc:
        msg = f"Required output table field is missing: {key}"
        raise ValueError(msg) from exc


def _required_str(data: Mapping[str, object], key: str) -> str:
    """Return required string output table field value."""

    value = _required_value(data, key)
    if not isinstance(value, str):
        msg = f"Output table field must be a string: {key}"
        raise ValueError(msg)
    return value


def _required_array(data: Mapping[str, object], key: str) -> tuple[object, ...]:
    """Return required array output table field value."""

    value = _required_value(data, key)
    if not isinstance(value, (list, tuple)):
        msg = f"Output table field must be an array: {key}"
        raise ValueError(msg)
    return tuple(value)


def _output_table_kind(value: str) -> OutputTableKind:
    """Return output table kind from saved value."""

    try:
        return OutputTableKind(value)
    except ValueError as exc:
        msg = f"Unsupported output table kind: {value}"
        raise ValueError(msg) from exc


def _column_from_value(value: object, index: int) -> OutputTableColumn:
    """Restore output table column from saved value."""

    if not isinstance(value, Mapping):
        msg = f"Output table column must be an object: {index}"
        raise ValueError(msg)

    data = cast("Mapping[str, object]", value)

    return OutputTableColumn(
        key=_required_str(data, "key"),
        title=_required_str(data, "title"),
        unit=_required_str(data, "unit"),
        description=_required_str(data, "description"),
    )


def _row_from_value(
    value: object,
    columns: tuple[OutputTableColumn, ...],
    index: int,
) -> OutputTableRow:
    """Restore output table row from saved value."""

    if not isinstance(value, Mapping):
        msg = f"Output table row must be an object: {index}"
        raise ValueError(msg)

    data = cast("Mapping[str, object]", value)
    column_keys = tuple(column.key for column in columns)

    if set(data) != set(column_keys):
        msg = f"Output table row keys do not match table columns: row {index}"
        raise ValueError(msg)

    return OutputTableRow(
        cells=tuple(_cell_from_value(data[key], key) for key in column_keys),
    )


def _metadata_from_value(value: object) -> tuple[tuple[str, str], ...]:
    """Restore output table metadata from saved value."""

    if not isinstance(value, (list, tuple)):
        msg = "Output table metadata field must be an array."
        raise ValueError(msg)

    items: list[tuple[str, str]] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"Output table metadata entry must be an object: {index}"
            raise ValueError(msg)

        data = cast("Mapping[str, object]", item)
        items.append((_required_str(data, "key"), _required_str(data, "value")))

    return tuple(items)


def output_table_from_snapshot(data: Mapping[str, object]) -> OutputTable:
    """Restore output table from saved representation."""

    columns = tuple(
        _column_from_value(column, index)
        for index, column in enumerate(_required_array(data, "columns"))
    )

    return OutputTable(
        table_id=_required_str(data, "table_id"),
        title=_required_str(data, "title"),
        kind=_output_table_kind(_required_str(data, "kind")),
        columns=columns,
        rows=tuple(
            _row_from_value(row, columns, index)
            for index, row in enumerate(_required_array(data, "rows"))
        ),
        metadata=_metadata_from_value(_required_value(data, "metadata")),
    )


def output_tables_from_snapshot(
    data: Sequence[Mapping[str, object]],
) -> tuple[OutputTable, ...]:
    """Restore output tables from saved representation."""

    return tuple(output_table_from_snapshot(table) for table in data)


def _spectrum_x_title(spectrum: Spectrum1D) -> str:
    """Return title for spectrum grid column."""

    if spectrum.quantity in _LET_QUANTITIES:
        return "ЛПЭ"

    return "Энергия"


def _spectrum_value_title(quantity: SpectrumQuantity) -> str:
    """Return title for spectrum value column."""

    return _SPECTRUM_VALUE_TITLES.get(quantity, quantity.value)


def spectrum_output_table(
    *,
    table_id: str,
    title: str,
    spectrum: Spectrum1D,
) -> OutputTable:
    """Create output table from one-dimensional spectrum."""

    return output_table_from_rows(
        table_id=table_id,
        title=title,
        kind=OutputTableKind.SPECTRUM,
        columns=(
            OutputTableColumn(
                key="x",
                title=_spectrum_x_title(spectrum),
                unit=spectrum.x_unit.value,
            ),
            OutputTableColumn(
                key="value",
                title=_spectrum_value_title(spectrum.quantity),
                unit=spectrum.y_unit.value,
            ),
        ),
        rows=tuple(
            {
                "x": x,
                "value": y,
            }
            for x, y in zip(spectrum.x, spectrum.y)
        ),
        metadata={
            "quantity": spectrum.quantity.value,
            "particle": spectrum.particle.value,
            "source": spectrum.source.value,
            "model": spectrum.model,
        },
    )
