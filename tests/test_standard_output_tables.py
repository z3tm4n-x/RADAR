import pytest

from radar.core.types import DoseQuantity, RadiationSource
from radar.core.units import Unit
from radar.output_tables import OutputTableKind
from radar.standard_output_tables import (
    DoseByThicknessPoint,
    SourceContribution,
    dose_by_thickness_output_table,
    source_contribution_output_table,
)


def test_dose_by_thickness_output_table() -> None:
    table = dose_by_thickness_output_table(
        table_id="dose_by_thickness",
        title="Накопленная доза по толщине защиты",
        points=(
            DoseByThicknessPoint(thickness_g_cm2=1.0, value=10.0),
            DoseByThicknessPoint(thickness_g_cm2=2.0, value=20.0),
        ),
        dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
        value_unit=Unit.RAD,
        metadata={"source": "total"},
    )

    assert table.kind is OutputTableKind.DOSE
    assert table.columns[0].key == "thickness_g_cm2"
    assert table.columns[0].title == "Толщина защиты"
    assert table.columns[0].unit == Unit.THICKNESS.value
    assert table.columns[1].key == "value"
    assert table.columns[1].title == "Накопленная доза"
    assert table.columns[1].unit == Unit.RAD.value
    assert table.rows[0].cells == (1.0, 10.0)
    assert ("dose_quantity", DoseQuantity.ACCUMULATED_DOSE.value) in table.metadata
    assert ("source", "total") in table.metadata


def test_dose_by_thickness_output_table_supports_dose_rate() -> None:
    table = dose_by_thickness_output_table(
        table_id="dose_rate_by_thickness",
        title="Мощность дозы по толщине защиты",
        points=(DoseByThicknessPoint(thickness_g_cm2=1.0, value=0.5),),
        dose_quantity=DoseQuantity.DOSE_RATE,
        value_unit=Unit.RAD_PER_SECOND,
    )

    assert table.columns[1].title == "Мощность дозы"
    assert table.columns[1].unit == Unit.RAD_PER_SECOND.value
    assert ("dose_quantity", DoseQuantity.DOSE_RATE.value) in table.metadata


def test_dose_by_thickness_output_table_supports_custom_value_title() -> None:
    table = dose_by_thickness_output_table(
        table_id="daily_dose_rate_by_thickness",
        title="Суточная мощность дозы по толщине защиты",
        points=(DoseByThicknessPoint(thickness_g_cm2=1.0, value=5.0),),
        dose_quantity=DoseQuantity.DOSE_RATE,
        value_unit=Unit.RAD_PER_DAY,
        value_title="Суточная мощность дозы",
    )

    assert table.columns[1].title == "Суточная мощность дозы"
    assert table.columns[1].unit == Unit.RAD_PER_DAY.value


def test_dose_by_thickness_point_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="Shielding thickness must be non-negative"):
        DoseByThicknessPoint(thickness_g_cm2=-1.0, value=10.0)


def test_dose_by_thickness_point_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="Dose value must be non-negative"):
        DoseByThicknessPoint(thickness_g_cm2=1.0, value=-10.0)


def test_source_contribution_output_table() -> None:
    table = source_contribution_output_table(
        table_id="dose_contributions",
        title="Вклады источников в накопленную дозу",
        contributions=(
            SourceContribution(
                source=RadiationSource.SEP,
                value=10.0,
                share_percent=25.0,
            ),
            SourceContribution(
                source=RadiationSource.GCR,
                value=20.0,
                share_percent=50.0,
            ),
            SourceContribution(
                source=RadiationSource.ERB,
                value=10.0,
                share_percent=25.0,
            ),
        ),
        value_title="Накопленная доза",
        value_unit=Unit.RAD,
        metadata={"quantity": "accumulated_dose"},
    )

    assert table.kind is OutputTableKind.SUMMARY
    assert table.columns[0].title == "Источник"
    assert table.columns[1].title == "Накопленная доза"
    assert table.columns[1].unit == Unit.RAD.value
    assert table.columns[2].title == "Доля"
    assert table.columns[2].unit == "%"
    assert table.rows[0].cells == ("СКЛ", 10.0, 25.0)
    assert table.rows[1].cells == ("ГКЛ", 20.0, 50.0)
    assert table.rows[2].cells == ("ЕРПЗ", 10.0, 25.0)
    assert table.metadata == (("quantity", "accumulated_dose"),)


def test_source_contribution_output_table_allows_missing_share() -> None:
    table = source_contribution_output_table(
        table_id="dose_contributions",
        title="Вклады источников в накопленную дозу",
        contributions=(
            SourceContribution(source=RadiationSource.SEP, value=10.0),
        ),
        value_title="Накопленная доза",
        value_unit=Unit.RAD,
    )

    assert table.rows[0].cells == ("СКЛ", 10.0, None)


def test_source_contribution_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="Contribution value must be non-negative"):
        SourceContribution(source=RadiationSource.SEP, value=-1.0)


def test_source_contribution_rejects_invalid_share() -> None:
    with pytest.raises(
        ValueError,
        match="Contribution share must be between 0 and 100 percent",
    ):
        SourceContribution(
            source=RadiationSource.SEP,
            value=10.0,
            share_percent=101.0,
        )
