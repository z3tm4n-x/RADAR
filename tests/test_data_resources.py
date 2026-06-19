import pytest

from radar.data.metadata import DataFormat, TableMetadata, ValidationStatus
from radar.data.resources import ResourceManager


def test_table_metadata_requires_required_fields() -> None:
    with pytest.raises(ValueError, match="model"):
        TableMetadata(
            model="",
            document="OST 134-1044-2007",
            table_id="table",
            filename="table.csv",
            data_format=DataFormat.CSV,
            units={"energy": "MeV"},
        )


def test_table_metadata_requires_units() -> None:
    with pytest.raises(ValueError, match="units"):
        TableMetadata(
            model="SEP",
            document="OST 134-1044-2007",
            table_id="table",
            filename="table.csv",
            data_format=DataFormat.CSV,
            units={},
        )


def test_table_metadata_checks_file_suffix() -> None:
    with pytest.raises(ValueError, match="suffix"):
        TableMetadata(
            model="SEP",
            document="OST 134-1044-2007",
            table_id="table",
            filename="table.json",
            data_format=DataFormat.CSV,
            units={"energy": "MeV"},
        )


def test_resource_manager_loads_csv(tmp_path) -> None:
    csv_file = tmp_path / "sep_table.csv"
    csv_file.write_text("energy,fluence\n1,10\n2,20\n", encoding="utf-8")

    metadata = TableMetadata(
        model="SEP",
        document="OST 134-1044-2007",
        table_id="test_csv",
        filename="sep_table.csv",
        data_format=DataFormat.CSV,
        units={"energy": "MeV", "fluence": "cm^-2 MeV^-1"},
        validation_status=ValidationStatus.CHECKED,
    )

    manager = ResourceManager(tmp_path)

    rows = manager.load_csv_dicts(metadata)

    assert rows == [
        {"energy": "1", "fluence": "10"},
        {"energy": "2", "fluence": "20"},
    ]


def test_resource_manager_loads_json(tmp_path) -> None:
    json_file = tmp_path / "metadata.json"
    json_file.write_text('{"model": "SEP", "version": 1}', encoding="utf-8")

    metadata = TableMetadata(
        model="SEP",
        document="OST 134-1044-2007",
        table_id="test_json",
        filename="metadata.json",
        data_format=DataFormat.JSON,
        units={"version": "1"},
    )

    manager = ResourceManager(tmp_path)

    data = manager.load_json(metadata)

    assert isinstance(data, dict)
    assert data["model"] == "SEP"
    assert data["version"] == 1


def test_resource_manager_loads_text(tmp_path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("RADAR data note", encoding="utf-8")

    metadata = TableMetadata(
        model="ERB",
        document="OST 134-1044-2007",
        table_id="test_text",
        filename="notes.txt",
        data_format=DataFormat.TEXT,
        units={"text": "1"},
    )

    manager = ResourceManager(tmp_path)

    assert manager.load_text(metadata) == "RADAR data note"


def test_resource_manager_rejects_wrong_loader(tmp_path) -> None:
    csv_file = tmp_path / "table.csv"
    csv_file.write_text("x,y\n1,2\n", encoding="utf-8")

    metadata = TableMetadata(
        model="SEP",
        document="OST 134-1044-2007",
        table_id="test_csv",
        filename="table.csv",
        data_format=DataFormat.CSV,
        units={"x": "MeV", "y": "1"},
    )

    manager = ResourceManager(tmp_path)

    with pytest.raises(ValueError, match="JSON"):
        manager.load_json(metadata)


def test_resource_manager_reports_missing_file(tmp_path) -> None:
    metadata = TableMetadata(
        model="SEP",
        document="OST 134-1044-2007",
        table_id="missing",
        filename="missing.csv",
        data_format=DataFormat.CSV,
        units={"energy": "MeV"},
    )

    manager = ResourceManager(tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.resolve(metadata)
