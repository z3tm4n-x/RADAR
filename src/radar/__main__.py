"""Command line entry point for RADAR."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from radar.project_io import calculate_project_file


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create RADAR command line argument parser."""

    parser = argparse.ArgumentParser(
        prog="python -m radar",
        description="RADAR calculation command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="read a RADAR project file, execute calculation and save the result",
    )
    run_parser.add_argument(
        "input",
        type=Path,
        help="input RADAR project file",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="output RADAR project file; defaults to overwriting the input file",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run RADAR command line interface."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        project_file = calculate_project_file(
            args.input,
            output_path=args.output,
        )
        output_path = args.output or args.input
        print(f"RADAR project calculation saved: {output_path}")
        print(f"Schema version: {project_file.schema_version}")
        print(f"Created at: {project_file.created_at}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
