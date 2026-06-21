from datetime import UTC, datetime

from radar.__main__ import main
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.project_io import calculate_and_save_project_file, read_project_file, save_project_file


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


def test_main_init_accepts_shield_thicknesses(tmp_path) -> None:
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
            "--shield-thickness",
            "2.0",
            "--shield-thickness",
            "0.5",
            "--shield-thickness",
            "1.0",
        ],
    )
    restored = read_project_file(output_path)

    assert exit_code == 0
    assert restored.calculation_config.shielding.thicknesses_g_cm2 == (
        0.5,
        1.0,
        2.0,
    )


def test_main_init_accepts_solar_activity_level(tmp_path) -> None:
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
            "--solar-activity",
            "maximum",
        ],
    )
    restored = read_project_file(output_path)

    assert exit_code == 0
    assert restored.calculation_config.mission.solar_activity_level.value == "maximum"


def test_main_init_rejects_invalid_shield_thickness(tmp_path, capsys) -> None:
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
            "--shield-thickness",
            "0",
        ],
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Shield thicknesses must be positive" in captured.err


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


def test_main_show_project_without_result(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.radar.json"

    save_project_file(
        input_path,
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    exit_code = main(["show", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Год запуска: 2028" in captured.out
    assert "Срок миссии, лет: 7" in captured.out
    assert "Солнечная активность: mean" in captured.out
    assert "Вероятность превышения СКЛ: 0.1" in captured.out
    assert "Kp: 4" in captured.out
    assert "Геометрия защиты: sphere" in captured.out
    assert "Толщины защиты, г/см²:" in captured.out
    assert "0.01" in captured.out
    assert "10.0" in captured.out
    assert "Результат расчёта: отсутствует" in captured.out
    assert "Выходные таблицы: отсутствуют" in captured.out


def test_main_show_project_with_result(tmp_path, capsys) -> None:
    input_path = tmp_path / "output.radar.json"

    calculate_and_save_project_file(
        input_path,
        _calculation_config(),
        created_at=datetime(2028, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    exit_code = main(["show", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Результат расчёта: есть" in captured.out
    assert "dose_by_thickness" in captured.out
    assert "source_contributions" in captured.out


def test_main_run_preserves_solar_activity_in_protocol(tmp_path) -> None:
    input_path = tmp_path / "input.radar.json"
    output_path = tmp_path / "output.radar.json"

    exit_code = main(
        [
            "init",
            str(input_path),
            "--launch-year",
            "2028",
            "--lifetime-years",
            "7",
            "--orbit",
            "geo",
            "--solar-activity",
            "minimum",
        ],
    )
    assert exit_code == 0

    exit_code = main(["run", str(input_path), "--output", str(output_path)])
    restored = read_project_file(output_path)

    assert exit_code == 0
    assert restored.calculation_config.mission.solar_activity_level.value == "minimum"
    assert {
        (entry["section"], entry["parameter"], entry["value"])
        for entry in restored.calculation_protocol
    } >= {
        ("Миссия", "Уровень солнечной активности", "минимальная"),
    }


def test_main_run_uses_configured_shield_thicknesses(tmp_path) -> None:
    input_path = tmp_path / "input.radar.json"
    output_path = tmp_path / "output.radar.json"

    exit_code = main(
        [
            "init",
            str(input_path),
            "--launch-year",
            "2028",
            "--lifetime-years",
            "7",
            "--orbit",
            "geo",
            "--shield-thickness",
            "0.25",
            "--shield-thickness",
            "0.75",
        ],
    )
    assert exit_code == 0

    exit_code = main(["run", str(input_path), "--output", str(output_path)])
    restored = read_project_file(output_path)

    assert exit_code == 0
    assert restored.calculation_result is not None

    dose_table = next(
        table
        for table in restored.calculation_result.output_tables
        if table.table_id == "dose_by_thickness"
    )
    thickness_column_index = tuple(
        column.key for column in dose_table.columns
    ).index("thickness_g_cm2")
    assert [row.cells[thickness_column_index] for row in dose_table.rows] == [
        0.25,
        0.75,
    ]


def test_main_show_reports_missing_input_file(tmp_path, capsys) -> None:
    input_path = tmp_path / "missing.radar.json"

    exit_code = main(["show", str(input_path)])
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
