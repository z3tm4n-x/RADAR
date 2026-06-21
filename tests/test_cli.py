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


def test_main_init_creates_project_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "input.radar.json"

    exit_code = main(
        [
            "init",
            str(output_path),
            "--launch-year",
            "2028",
            "--lifetime-years",
            "7",
            "--orbit",
            "geo",
        ],
    )
    restored = read_project_file(output_path)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert restored.calculation_result is None
    assert restored.calculation_config.mission.launch_year == 2028
    assert restored.calculation_config.mission.lifetime_years == 7
    assert restored.calculation_config.orbit.perigee_altitude_km == 35786.0
    assert "Файл проекта RADAR создан" in captured.out


def test_main_init_creates_circular_project_file(tmp_path) -> None:
    output_path = tmp_path / "input.radar.json"

    exit_code = main(
        [
            "init",
            str(output_path),
            "--launch-year",
            "2028",
            "--lifetime-years",
            "7",
            "--orbit",
            "circular",
            "--altitude-km",
            "550",
            "--inclination-deg",
            "97.6",
            "--kp",
            "5",
        ],
    )
    restored = read_project_file(output_path)

    assert exit_code == 0
    assert restored.calculation_config.orbit.perigee_altitude_km == 550.0
    assert restored.calculation_config.orbit.inclination_deg == 97.6
    assert restored.calculation_config.kp == 5


def test_main_init_rejects_circular_orbit_without_altitude(tmp_path, capsys) -> None:
    output_path = tmp_path / "input.radar.json"

    exit_code = main(
        [
            "init",
            str(output_path),
            "--launch-year",
            "2028",
            "--lifetime-years",
            "7",
            "--orbit",
            "circular",
        ],
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Для круговой орбиты укажите --altitude-km" in captured.err


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
    assert "Расчёт проекта RADAR сохранён" in captured.out


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


def test_main_run_reports_missing_input_file(tmp_path, capsys) -> None:
    input_path = tmp_path / "missing.radar.json"

    exit_code = main(["run", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Ошибка RADAR: файл не найден" in captured.err
    assert str(input_path) in captured.err


def test_main_rejects_missing_command() -> None:
    try:
        main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit for missing command.")
