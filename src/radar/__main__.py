"""Command line entry point for RADAR."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.project_io import calculate_project_file, save_project_file

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

    return parser


def _calculation_config_from_init_args(args: argparse.Namespace) -> CalculationConfig:
    """Create calculation configuration from init command arguments."""

    altitude_km = args.altitude_km
    if args.orbit == "geo":
        altitude_km = _GEO_ALTITUDE_KM if altitude_km is None else altitude_km

    if altitude_km is None:
        msg = "Для круговой орбиты укажите --altitude-km."
        raise ValueError(msg)

    return CalculationConfig(
        mission=MissionConfig(
            launch_year=args.launch_year,
            lifetime_years=args.lifetime_years,
        ),
        orbit=OrbitConfig.circular(
            altitude_km=altitude_km,
            inclination_deg=args.inclination_deg,
        ),
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run RADAR command line interface."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return _run_init_command(args)

        if args.command == "run":
            return _run_calculation_command(args)

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
