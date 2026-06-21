from datetime import UTC, datetime

from radar.__main__ import main
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.project_io import read_project_file, save_project_file


def _calculation_config() -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(launch_year=2028, lifetime_years=7),
        orbit=OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0),
        kp=4,
    )


def test_main_run_overwrites_input_project_file(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.radar.json"
    created_at = datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC)

    save_project_file(
        input_path,
        _calculation_config(),
        created_at=created_at,
    )

    exit_code = main(["run", str(input_path)])
    restored = read_project_file(input_path)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert restored.created_at == created_at.isoformat()
    assert restored.calculation_result is not None
    assert "RADAR project calculation saved" in captured.out


def test_main_run_writes_output_project_file(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.radar.json"
    output_path = tmp_path / "output.radar.json"

    save_project_file(
        input_path,
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    exit_code = main(["run", str(input_path), "--output", str(output_path)])
    original = read_project_file(input_path)
    restored = read_project_file(output_path)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert original.calculation_result is None
    assert restored.calculation_result is not None
    assert str(output_path) in captured.out


def test_main_rejects_missing_command() -> None:
    try:
        main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit for missing command.")
