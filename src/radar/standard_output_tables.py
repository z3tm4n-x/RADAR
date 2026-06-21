"""Standard RADAR output table builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from radar.core.types import DoseQuantity, RadiationSource
from radar.core.units import Unit
from radar.output_tables import (
    OutputTable,
    OutputTableColumn,
    OutputTableKind,
    output_table_from_rows,
)

_SOURCE_TITLES: dict[RadiationSource, str] = {
    RadiationSource.SEP: "СКЛ",
    RadiationSource.GCR: "ГКЛ",
    RadiationSource.ERB: "ЕРПЗ",
}

_DOSE_QUANTITY_TITLES: dict[DoseQuantity, str] = {
    DoseQuantity.ACCUMULATED_DOSE: "Накопленная доза",
    DoseQuantity.DOSE_RATE: "Мощность дозы",
}


@dataclass(frozen=True)
class DoseByThicknessPoint:
    """Dose value at one shielding thickness."""

    thickness_g_cm2: float
    value: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            self.thickness_g_cm2,
            "Shielding thickness",
        )
        _validate_non_negative_finite(self.value, "Dose value")


@dataclass(frozen=True)
class SourceContribution:
    """Contribution of one radiation source to a result value."""

    source: RadiationSource
    value: float
    share_percent: float | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.value, "Contribution value")

        if self.share_percent is not None:
            _validate_share_percent(self.share_percent)


def _validate_non_negative_finite(value: float, name: str) -> None:
    """Validate finite non-negative value."""

    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def _validate_share_percent(value: float) -> None:
    """Validate percent share value."""

    if not isfinite(value):
        msg = "Contribution share must be finite."
        raise ValueError(msg)

    if value < 0.0 or value > 100.0:
        msg = "Contribution share must be between 0 and 100 percent."
        raise ValueError(msg)


def _metadata_with(
    base: Mapping[str, str],
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return merged output table metadata."""

    metadata = dict(base)

    if extra is not None:
        metadata.update(extra)

    return metadata


def _source_title(source: RadiationSource) -> str:
    """Return Russian source title."""

    return _SOURCE_TITLES[source]


def _dose_value_title(dose_quantity: DoseQuantity, value_title: str | None) -> str:
    """Return dose value column title."""

    if value_title is not None:
        return value_title

    return _DOSE_QUANTITY_TITLES[dose_quantity]


def dose_by_thickness_output_table(
    *,
    table_id: str,
    title: str,
    points: Sequence[DoseByThicknessPoint],
    dose_quantity: DoseQuantity,
    value_unit: Unit,
    value_title: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> OutputTable:
    """Create standard dose-by-shielding-thickness table."""

    return output_table_from_rows(
        table_id=table_id,
        title=title,
        kind=OutputTableKind.DOSE,
        columns=(
            OutputTableColumn(
                key="thickness_g_cm2",
                title="Толщина защиты",
                unit=Unit.THICKNESS.value,
            ),
            OutputTableColumn(
                key="value",
                title=_dose_value_title(dose_quantity, value_title),
                unit=value_unit.value,
            ),
        ),
        rows=tuple(
            {
                "thickness_g_cm2": point.thickness_g_cm2,
                "value": point.value,
            }
            for point in points
        ),
        metadata=_metadata_with(
            {
                "dose_quantity": dose_quantity.value,
            },
            metadata,
        ),
    )


def source_contribution_output_table(
    *,
    table_id: str,
    title: str,
    contributions: Sequence[SourceContribution],
    value_title: str,
    value_unit: Unit,
    metadata: Mapping[str, str] | None = None,
) -> OutputTable:
    """Create standard radiation-source contribution table."""

    return output_table_from_rows(
        table_id=table_id,
        title=title,
        kind=OutputTableKind.SUMMARY,
        columns=(
            OutputTableColumn(
                key="source",
                title="Источник",
            ),
            OutputTableColumn(
                key="value",
                title=value_title,
                unit=value_unit.value,
            ),
            OutputTableColumn(
                key="share_percent",
                title="Доля",
                unit="%",
            ),
        ),
        rows=tuple(
            {
                "source": _source_title(contribution.source),
                "value": contribution.value,
                "share_percent": contribution.share_percent,
            }
            for contribution in contributions
        ),
        metadata=_metadata_with({}, metadata),
    )
