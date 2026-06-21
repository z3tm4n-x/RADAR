"""Command line entry point for RADAR."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
from radar.core.types import SolarActivityLevel
from radar.project_file import ProjectFile
from radar.project_io import calculate_project_file, read_project_file, save_project_file

_GEO_ALTITUDE_KM = 35786.0


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create RADAR command line argument parser."""

    parser = argparse.ArgumentParser(
        prog="python -m radar",
        description="Командный запуск RADAR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="создать входной файл проекта RADAR",
    )
    init_parser.add_argument(
        "output",
        type=Path,
        help="создаваемый файл проекта RADAR",
    )
    init_parser.add_argument(
        "--launch-year",
        type=int,
        required=True,
        help="год начала миссии",
    )
    init_parser.add_argument(
        "--lifetime-years",
        type=int,
        required=True,
        help="длительность миссии, лет",
    )

    init_parser.add_argument(
        "--solar-activity",
        choices=tuple(level.value for level in SolarActivityLevel),
        default=SolarActivityLevel.MEAN.value,
        help="уровень солнечной активности: minimum, mean или maximum",
    )
    init_parser.add_argument(
        "--orbit",
        choices=("geo", "circular"),
        default="geo",
        help="тип орбиты: geo или circular",
    )
    init_parser.add_argument(
        "--altitude-km",
        type=float,
        default=None,
        help="высота круговой орбиты, км; для geo по умолчанию 35786 км",
    )
    init_parser.add_argument(
        "--inclination-deg",
        type=float,
        default=0.0,
        help="наклонение орбиты, град",
    )
    init_parser.add_argument(
        "--kp",
        type=int,
        default=4,
        help="индекс Kp",
    )

    init_parser.add_argument(
        "--shield-thickness",
        dest="shield_thicknesses",
        action="append",
        type=float,
        default=None,
        help=(
            "толщина сферической алюминиевой защиты, г/см²; "
            "можно указать несколько раз"
        ),
    )

    run_parser = subparsers.add_parser(
        "run",
        help="прочитать файл проекта RADAR, выполнить расчёт и сохранить результат",
    )
    run_parser.add_argument(
        "input",
        type=Path,
        help="входной файл проекта RADAR",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="выходной файл проекта RADAR; по умолчанию перезаписывается входной файл",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="показать краткое описание файла проекта RADAR",
    )
    show_parser.add_argument(
        "input",
        type=Path,
        help="файл проекта RADAR",
    )

    return parser


def _calculation_config_from_init_args(args: argparse.Namespace) -> CalculationConfig:
    """Create calculation configuration from init command arguments."""

    altitude_km = args.altitude_km
    if args.orbit == "geo":
        altitude_km = _GEO_ALTITUDE_KM if altitude_km is None else altitude_km

    if altitude_km is None:
        msg = "Для круговой орбиты укажите --altitude-km."
        raise ValueError(msg)

    shielding = (
        ShieldingConfig(thicknesses_g_cm2=tuple(args.shield_thicknesses))
        if args.shield_thicknesses is not None
        else ShieldingConfig()
    )

    return CalculationConfig(
        mission=MissionConfig(
            launch_year=args.launch_year,
            lifetime_years=args.lifetime_years,
            solar_activity_level=SolarActivityLevel(args.solar_activity),
        ),
        orbit=OrbitConfig.circular(
            altitude_km=altitude_km,
            inclination_deg=args.inclination_deg,
        ),
        shielding=shielding,
        kp=args.kp,
    )


def _run_init_command(args: argparse.Namespace) -> int:
    """Run init command."""

    config = _calculation_config_from_init_args(args)
    project_file = save_project_file(args.output, config)

    print(f"Файл проекта RADAR создан: {args.output}")
    print(f"Версия схемы: {project_file.schema_version}")
    print(f"Дата создания: {project_file.created_at}")
    return 0


def _run_calculation_command(args: argparse.Namespace) -> int:
    """Run calculation command."""

    project_file = calculate_project_file(
        args.input,
        output_path=args.output,
    )
    output_path = args.output or args.input

    print(f"Расчёт проекта RADAR сохранён: {output_path}")
    print(f"Версия схемы: {project_file.schema_version}")
    print(f"Дата создания: {project_file.created_at}")
    return 0


def _print_project_summary(project_file: ProjectFile) -> None:
    """Print project file summary."""

    config = project_file.calculation_config
    result = project_file.calculation_result

    print(f"Программа: {project_file.program_name}")
    print(f"Версия схемы: {project_file.schema_version}")
    print(f"Версия программы: {project_file.program_version}")
    print(f"Дата создания: {project_file.created_at}")
    print(f"Год запуска: {config.mission.launch_year}")
    print(f"Срок миссии, лет: {config.mission.lifetime_years}")
    print(f"Солнечная активность: {config.mission.solar_activity_level.value}")
    print(f"Вероятность превышения СКЛ: {config.mission.sep_exceedance_probability}")
    print(f"Тип орбиты: {config.orbit.orbit_type.value}")
    print(f"Перигей, км: {config.orbit.perigee_altitude_km}")
    print(f"Апогей, км: {config.orbit.apogee_altitude_km}")
    print(f"Наклонение, град: {config.orbit.inclination_deg}")
    print(f"Kp: {config.kp}")
    print(f"Геометрия защиты: {config.shielding.geometry.value}")
    thicknesses = ", ".join(
        str(thickness) for thickness in config.shielding.thicknesses_g_cm2
    )
    print(f"Толщины защиты, г/см²: {thicknesses}")
    print(f"Методика: {config.methodology.profile.value}")
    print(f"Дозовая величина: {config.dose_quantity.value}")
    print(f"Единица дозы: {config.dose_unit.value}")

    if result is None:
        print("Результат расчёта: отсутствует")
        print("Выходные таблицы: отсутствуют")
        return

    print("Результат расчёта: есть")
    if result.output_tables:
        table_ids = ", ".join(table.table_id for table in result.output_tables)
        print(f"Выходные таблицы: {table_ids}")
    else:
        print("Выходные таблицы: отсутствуют")


def _run_show_command(args: argparse.Namespace) -> int:
    """Run show command."""

    project_file = read_project_file(args.input)
    _print_project_summary(project_file)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run RADAR command line interface."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return _run_init_command(args)

        if args.command == "run":
            return _run_calculation_command(args)

        if args.command == "show":
            return _run_show_command(args)

        parser.error(f"Неподдерживаемая команда: {args.command}")
        return 2
    except FileNotFoundError as exc:
        filename = exc.filename or "неизвестный файл"
        print(f"Ошибка RADAR: файл не найден: {filename}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Ошибка RADAR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
