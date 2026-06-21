import pytest

from radar.data_tables import (
    NormativeCollectionManifest,
    NormativeCollectionRef,
    NormativeTableRef,
    get_normative_table_ref,
    list_normative_collections,
    load_normative_collection_manifest,
)


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
    assert manifest.status == "skeleton"
    assert len(manifest.tables) == 1


def test_get_normative_table_ref() -> None:
    table = get_normative_table_ref(
        "ost_134_1044_2007",
        "ost_134_1044_2007_placeholder",
    )

    assert table == NormativeTableRef(
        table_id="ost_134_1044_2007_placeholder",
        title="Паспорт набора нормативных таблиц ОСТ 134-1044-2007",
        document="ОСТ 134-1044-2007",
        appendix="",
        table_number="",
        model="общий паспорт",
        quantity="",
        x_unit="",
        y_unit="",
        data_file="",
        verification_status="not_transferred",
        control_values=(),
    )


def test_load_normative_collection_manifest_rejects_unknown_collection() -> None:
    with pytest.raises(ValueError, match="Unknown normative collection"):
        load_normative_collection_manifest("missing")


def test_get_normative_table_ref_rejects_unknown_table() -> None:
    with pytest.raises(ValueError, match="Unknown normative table"):
        get_normative_table_ref("ost_134_1044_2007", "missing")
