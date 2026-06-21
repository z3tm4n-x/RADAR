import pytest

from radar.data_tables import (
    NormativeCollectionManifest,
    NormativeCollectionRef,
    NormativeTableRef,
    get_normative_table_ref,
    list_normative_collections,
    load_normative_collection_manifest,
    load_normative_table_rows,
    validate_normative_table_control_values,
)

_WOLF_TABLE_ID = "ost_134_1044_2007_table_g_1_wolf_numbers"


def test_list_normative_collections_contains_ost_134_1044_2007() -> None:
    collections = list_normative_collections()

    assert collections == (
        NormativeCollectionRef(
            collection_id="ost_134_1044_2007",
            title="ОСТ 134-1044-2007",
            manifest="ost_134_1044_2007/manifest.json",
        ),
    )


def test_load_normative_collection_manifest() -> None:
    manifest = load_normative_collection_manifest("ost_134_1044_2007")

    assert isinstance(manifest, NormativeCollectionManifest)
    assert manifest.schema_version == "1.0"
    assert manifest.collection_id == "ost_134_1044_2007"
    assert manifest.document == "ОСТ 134-1044-2007"
    assert manifest.status == "partially_transferred"
    assert len(manifest.tables) == 1


def test_get_normative_table_ref() -> None:
    table = get_normative_table_ref("ost_134_1044_2007", _WOLF_TABLE_ID)

    assert table == NormativeTableRef(
        table_id=_WOLF_TABLE_ID,
        title=(
            "Таблица Г.1 – Среднегодовые значения чисел Вольфа для среднего, "
            "максимального и минимального циклов СА"
        ),
        document="ОСТ 134-1044-2007",
        appendix="Г",
        table_number="Г.1",
        model="Методика прогнозирования СА",
        quantity="Среднегодовое число Вольфа в цикле СА",
        x_unit="год от начала солнечного цикла",
        y_unit="число Вольфа",
        data_file="ost_134_1044_2007/table_g_1_wolf_numbers.csv",
        verification_status="transferred_from_ost",
        control_values=(
            {
                "year_from_cycle_start": "1",
                "mean_cycle_wolf_number": "7.1",
                "maximum_cycle_wolf_number": "11.5",
                "minimum_cycle_wolf_number": "2.6",
            },
            {
                "year_from_cycle_start": "5",
                "mean_cycle_wolf_number": "107.4",
                "maximum_cycle_wolf_number": "147.4",
                "minimum_cycle_wolf_number": "67.4",
            },
            {
                "year_from_cycle_start": "12",
                "mean_cycle_wolf_number": "7.1",
                "maximum_cycle_wolf_number": "11.5",
                "minimum_cycle_wolf_number": "2.6",
            },
        ),
    )


def test_load_normative_table_rows_reads_wolf_number_table() -> None:
    rows = load_normative_table_rows("ost_134_1044_2007", _WOLF_TABLE_ID)

    assert len(rows) == 12
    assert rows[0] == {
        "year_from_cycle_start": "1",
        "mean_cycle_wolf_number": "7.1",
        "maximum_cycle_wolf_number": "11.5",
        "minimum_cycle_wolf_number": "2.6",
    }
    assert rows[4] == {
        "year_from_cycle_start": "5",
        "mean_cycle_wolf_number": "107.4",
        "maximum_cycle_wolf_number": "147.4",
        "minimum_cycle_wolf_number": "67.4",
    }
    assert rows[-1] == {
        "year_from_cycle_start": "12",
        "mean_cycle_wolf_number": "7.1",
        "maximum_cycle_wolf_number": "11.5",
        "minimum_cycle_wolf_number": "2.6",
    }


def test_validate_normative_table_control_values_accepts_wolf_number_table() -> None:
    validate_normative_table_control_values("ost_134_1044_2007", _WOLF_TABLE_ID)


def test_load_normative_collection_manifest_rejects_unknown_collection() -> None:
    with pytest.raises(ValueError, match="Unknown normative collection"):
        load_normative_collection_manifest("missing")


def test_get_normative_table_ref_rejects_unknown_table() -> None:
    with pytest.raises(ValueError, match="Unknown normative table"):
        get_normative_table_ref("ost_134_1044_2007", "missing")


def test_load_normative_table_rows_rejects_table_without_data_file() -> None:
    table = get_normative_table_ref("ost_134_1044_2007", _WOLF_TABLE_ID)

    assert table.data_file
