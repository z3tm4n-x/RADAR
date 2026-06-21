"""OST solar activity tables."""

from __future__ import annotations

from radar.core.types import SolarActivityLevel
from radar.data_tables import (
    get_normative_table_ref,
    load_normative_table_rows,
    validate_normative_table_control_values,
)
from radar.solar_activity.cycle import SOLAR_CYCLE_YEARS, SolarCycleTable

OST_134_1044_2007_COLLECTION_ID = "ost_134_1044_2007"
OST_WOLF_NUMBER_TABLE_ID = "ost_134_1044_2007_table_g_1_wolf_numbers"

_WOLF_NUMBER_COLUMN_BY_LEVEL: dict[SolarActivityLevel, str] = {
    SolarActivityLevel.MINIMUM: "minimum_cycle_wolf_number",
    SolarActivityLevel.MEAN: "mean_cycle_wolf_number",
    SolarActivityLevel.MAXIMUM: "maximum_cycle_wolf_number",
}


def _cycle_point(row: dict[str, str]) -> int:
    """Return cycle point from transferred OST table row."""

    try:
        return int(row["cycle_point"])
    except KeyError as exc:
        msg = "OST Wolf number table row has no cycle_point column."
        raise ValueError(msg) from exc
    except ValueError as exc:
        msg = f"OST Wolf number table cycle_point is not an integer: {row['cycle_point']}"
        raise ValueError(msg) from exc


def _wolf_number(row: dict[str, str], column: str) -> float:
    """Return Wolf number from transferred OST table row."""

    try:
        value = float(row[column])
    except KeyError as exc:
        msg = f"OST Wolf number table row has no {column} column."
        raise ValueError(msg) from exc
    except ValueError as exc:
        msg = f"OST Wolf number value is not a number: {row[column]}"
        raise ValueError(msg) from exc

    if value < 0.0:
        msg = f"OST Wolf number value must be non-negative: {value}"
        raise ValueError(msg)

    return value


def ost_wolf_number_cycle_table(level: SolarActivityLevel) -> SolarCycleTable:
    """Return OST table G.1 Wolf numbers for one solar cycle level."""

    table_ref = get_normative_table_ref(
        OST_134_1044_2007_COLLECTION_ID,
        OST_WOLF_NUMBER_TABLE_ID,
    )
    validate_normative_table_control_values(
        OST_134_1044_2007_COLLECTION_ID,
        OST_WOLF_NUMBER_TABLE_ID,
    )

    if table_ref.cycle_period_years != str(SOLAR_CYCLE_YEARS):
        msg = (
            "OST Wolf number table cycle period does not match "
            f"{SOLAR_CYCLE_YEARS} years."
        )
        raise ValueError(msg)

    rows = load_normative_table_rows(
        OST_134_1044_2007_COLLECTION_ID,
        OST_WOLF_NUMBER_TABLE_ID,
    )

    row_by_cycle_point = {_cycle_point(row): row for row in rows}
    expected_points = set(range(1, SOLAR_CYCLE_YEARS + 2))
    if set(row_by_cycle_point) != expected_points:
        msg = "OST Wolf number table must contain cycle points 1..12."
        raise ValueError(msg)

    if row_by_cycle_point[1] != row_by_cycle_point[12] | {"cycle_point": "1"}:
        msg = "OST Wolf number table closing point 12 must match point 1."
        raise ValueError(msg)

    try:
        column = _WOLF_NUMBER_COLUMN_BY_LEVEL[level]
    except KeyError as exc:
        msg = f"Unsupported solar cycle level: {level}"
        raise ValueError(msg) from exc

    wolf_numbers = tuple(
        _wolf_number(row_by_cycle_point[cycle_point], column)
        for cycle_point in range(1, SOLAR_CYCLE_YEARS + 1)
    )

    return SolarCycleTable(
        level=level,
        wolf_numbers=wolf_numbers,
        source=table_ref.document,
        table_id=table_ref.table_id,
    )


def ost_wolf_number_for_cycle_year(
    level: SolarActivityLevel,
    cycle_year: int,
) -> float:
    """Return OST Wolf number for a 1-based 11-year solar cycle year."""

    table = ost_wolf_number_cycle_table(level)
    return table.wolf_number_for_cycle_year(cycle_year)
