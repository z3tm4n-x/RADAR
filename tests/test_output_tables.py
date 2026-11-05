import pytest

import json

from radar.core.spectra import (
    ModelReference,
    ParticleSpecies,
    ShieldingDescriptor,
    Spectrum1D,
    TransformationStep,
)
from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    Particle,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.output_tables import (
    OutputTableColumn,
    OutputTableKind,
    output_table_from_rows,
    output_table_from_snapshot,
    output_table_snapshot,
    output_tables_from_snapshot,
    output_tables_snapshot,
    spectrum_output_table,
)


def _spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(10.0, 20.0, 30.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="ГОСТ СКЛ",
    )


def test_output_table_from_rows() -> None:
    table = output_table_from_rows(
        table_id="dose",
        title="Накопленная доза",
        kind=OutputTableKind.DOSE,
        columns=(
            OutputTableColumn(
                key="thickness",
                title="Толщина защиты",
                unit=Unit.THICKNESS.value,
            ),
            OutputTableColumn(
                key="dose",
                title="Накопленная доза",
                unit=Unit.RAD.value,
            ),
        ),
        rows=(
            {
                "thickness": 1.0,
                "dose": 10.0,
            },
            {
                "thickness": 2.0,
                "dose": 20.0,
            },
        ),
        metadata={"source": "total"},
    )

    assert table.table_id == "dose"
    assert table.kind is OutputTableKind.DOSE
    assert table.rows[0].cells == (1.0, 10.0)
    assert table.metadata == (("source", "total"),)


def test_output_table_snapshot_round_trip() -> None:
    table = spectrum_output_table(
        table_id="sep_proton_mission_fluence",
        title="СКЛ. Протоны. Флюенс за миссию",
        spectrum=_spectrum(),
    )

    restored = output_table_from_snapshot(output_table_snapshot(table))

    assert restored == table


def test_output_tables_snapshot_round_trip() -> None:
    table = spectrum_output_table(
        table_id="sep_proton_mission_fluence",
        title="СКЛ. Протоны. Флюенс за миссию",
        spectrum=_spectrum(),
    )

    restored = output_tables_from_snapshot(output_tables_snapshot((table,)))

    assert restored == (table,)


def test_spectrum_output_table_uses_spectrum_axes_and_metadata() -> None:
    table = spectrum_output_table(
        table_id="sep_proton_mission_fluence",
        title="СКЛ. Протоны. Флюенс за миссию",
        spectrum=_spectrum(),
    )

    assert table.kind is OutputTableKind.SPECTRUM
    assert table.columns[0].title == "Энергия"
    assert table.columns[0].unit == Unit.MEV.value
    assert table.columns[1].title == "Дифференциальный флюенс"
    assert table.columns[1].unit == Unit.DIFFERENTIAL_FLUENCE.value
    assert table.rows[0].cells == (1.0, 10.0)
    assert ("quantity", SpectrumQuantity.DIFFERENTIAL_FLUENCE.value) in table.metadata
    assert ("particle", Particle.PROTON.value) in table.metadata
    assert ("source", RadiationSource.SEP.value) in table.metadata
    assert ("model", "ГОСТ СКЛ") in table.metadata


def test_spectrum_output_table_accepts_extra_metadata() -> None:
    table = spectrum_output_table(
        table_id="sep_proton_behind_shielding",
        title="СКЛ. Протоны за защитой",
        spectrum=_spectrum(),
        metadata={
            "stage": "shielding",
            "location": "behind_shielding",
            "shield_material": "Al",
        },
    )

    assert ("quantity", SpectrumQuantity.DIFFERENTIAL_FLUENCE.value) in table.metadata
    assert ("stage", "shielding") in table.metadata
    assert ("location", "behind_shielding") in table.metadata
    assert ("shield_material", "Al") in table.metadata


def test_spectrum_output_table_rejects_canonical_metadata_override() -> None:
    with pytest.raises(ValueError, match="cannot override canonical keys"):
        spectrum_output_table(
            table_id="bad",
            title="Bad",
            spectrum=_spectrum(),
            metadata={"quantity": "wrong"},
        )


def test_spectrum_output_table_serializes_full_production_contract() -> None:
    duration_s = 365.25 * 86400.0
    spectrum = Spectrum1D(
        x=(10.0, 20.0),
        y=(2.0, 1.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="ost_sep+shield",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.MISSION_TOTAL,
        location=SpectrumLocation.BEHIND_SHIELD,
        species=ParticleSpecies(symbol="p", atomic_number=1, mass_number=1),
        shielding=ShieldingDescriptor("aluminium", 1.5),
        assessment_mode=AssessmentMode.CUMULATIVE,
        mission_duration_s=duration_s,
        model_reference=ModelReference(
            name="ost_sep+shield",
            version="protons_v1",
            document_refs=("OST 134-1044-2007",),
            dataset_refs=("sep_protons/ost_134_1044_2007_coefficients.csv",),
        ),
        transformation_chain=(
            TransformationStep(
                name="aluminium_transport",
                version="1",
                parameters=(("thickness_g_cm2", "1.5"),),
            ),
        ),
        completeness=SpectrumCompleteness.COMPLETE,
    )

    metadata = dict(
        spectrum_output_table(
            table_id="production",
            title="Production",
            spectrum=spectrum,
        ).metadata
    )

    assert metadata["axis_kind"] == SpectrumAxisKind.KINETIC_ENERGY.value
    assert metadata["angular_basis"] == AngularBasis.OMNIDIRECTIONAL_4PI.value
    assert metadata["temporal_basis"] == TemporalBasis.MISSION_TOTAL.value
    assert metadata["assessment_mode"] == AssessmentMode.CUMULATIVE.value
    assert metadata["location"] == SpectrumLocation.BEHIND_SHIELD.value
    assert float(metadata["mission_duration_s"]) == pytest.approx(duration_s)
    assert metadata["species_z"] == "1"
    assert metadata["species_a"] == "1"
    assert metadata["shielding_material"] == "aluminium"
    assert metadata["model_version"] == "protons_v1"
    assert json.loads(metadata["model_document_refs"]) == ["OST 134-1044-2007"]
    assert json.loads(metadata["transformation_chain"])[0]["name"] == (
        "aluminium_transport"
    )
    assert metadata["completeness"] == SpectrumCompleteness.COMPLETE.value


def test_let_spectrum_is_regular_spectrum_with_let_axis() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 2.0),
        y=(3.0, 4.0),
        x_unit=Unit.LET,
        y_unit=Unit.DIFFERENTIAL_LET_FLUENCE,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.GCR,
        model="ГОСТ ГКЛ",
    )

    table = spectrum_output_table(
        table_id="gcr_let",
        title="ГКЛ. ЛПЭ-спектр",
        spectrum=spectrum,
    )

    assert table.kind is OutputTableKind.SPECTRUM
    assert table.columns[0].title == "ЛПЭ"
    assert table.columns[0].unit == Unit.LET.value
    assert table.columns[1].title == "Дифференциальный ЛПЭ-флюенс"
    assert table.columns[1].unit == Unit.DIFFERENTIAL_LET_FLUENCE.value


def test_output_table_rejects_row_key_mismatch() -> None:
    with pytest.raises(ValueError, match="row keys do not match table columns"):
        output_table_from_rows(
            table_id="bad",
            title="Ошибочная таблица",
            kind=OutputTableKind.SUMMARY,
            columns=(OutputTableColumn(key="a", title="A"),),
            rows=({"b": 1.0},),
        )


def test_output_table_rejects_duplicate_column_keys() -> None:
    with pytest.raises(ValueError, match="column keys must be unique"):
        output_table_from_rows(
            table_id="bad",
            title="Ошибочная таблица",
            kind=OutputTableKind.SUMMARY,
            columns=(
                OutputTableColumn(key="a", title="A"),
                OutputTableColumn(key="a", title="B"),
            ),
            rows=(),
        )


def test_output_table_from_snapshot_rejects_unsupported_kind() -> None:
    table = spectrum_output_table(
        table_id="sep_proton_mission_fluence",
        title="СКЛ. Протоны. Флюенс за миссию",
        spectrum=_spectrum(),
    )
    snapshot = output_table_snapshot(table)
    snapshot["kind"] = "unsupported"

    with pytest.raises(ValueError, match="Unsupported output table kind"):
        output_table_from_snapshot(snapshot)


def test_output_table_from_snapshot_rejects_non_finite_cell() -> None:
    table = spectrum_output_table(
        table_id="sep_proton_mission_fluence",
        title="СКЛ. Протоны. Флюенс за миссию",
        spectrum=_spectrum(),
    )
    snapshot = output_table_snapshot(table)
    rows = snapshot["rows"]
    assert isinstance(rows, list)
    first_row = rows[0]
    assert isinstance(first_row, dict)
    first_row["value"] = float("inf")

    with pytest.raises(ValueError, match="finite floats"):
        output_table_from_snapshot(snapshot)
