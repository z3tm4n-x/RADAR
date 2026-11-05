"""Compare candidate table quadrature grids, not spectral or physical convergence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import fsum
from pathlib import Path
from typing import Any

from validate_ion_stopping_candidate import run_validator, write_report_exclusively


def _read_verified(path: Path, validation: dict[str, Any]) -> dict[str, Any]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != validation["input_sha256_after"]:
        raise ValueError("Candidate changed after individual validation.")
    return dict(json.loads(data))


def _totals(validation: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for interval in validation["validation"]["intervals"]:
        groups[interval["z"], interval["mass_number"], interval["material_z"]].append(
            interval["integrated_delta_range_g_cm2"]
        )
    return {key: fsum(values) for key, values in groups.items()}


def compare_grids(
    coarse_path: Path, fine_path: Path, *, repository_root: Path | None = None,
) -> dict[str, Any]:
    """Replay both datasets before checking a strictly nested, matched grid pair."""
    reports = [run_validator(path, repository_root=repository_root)
               for path in (coarse_path, fine_path)]
    coarse, fine = [_read_verified(path, report) for path, report in
                    zip((coarse_path, fine_path), reports, strict=True)]
    for key in ("source_files_sha256_after", "production_resources_sha256_after"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Different {key} cannot establish a grid-only comparison.")
    for key in ("ions", "material_atomic_numbers", "materials", "backend_configuration",
                "nonradiative_configuration", "backend_info_literal", "source_provenance"):
        if coarse.get(key) != fine.get(key):
            raise ValueError(f"Different {key} cannot establish a grid-only comparison.")
    a, b = coarse["energies_mev_per_nucleon"], fine["energies_mev_per_nucleon"]
    if len(b) <= len(a) or a[0] != b[0] or a[-1] != b[-1] or not set(a) <= set(b):
        raise ValueError("Fine grid must strictly refine the coarse grid with identical endpoints.")
    fine_rows = {(r["z"], r["mass_number"], r["material_z"], r["energy_mev_per_nucleon"]): r
                 for r in fine["rows"]}
    for row in coarse["rows"]:
        node = (row["z"], row["mass_number"], row["material_z"], row["energy_mev_per_nucleon"])
        if row != fine_rows[node]:
            raise ValueError(f"Shared-node values differ for {node}.")
    coarse_totals, fine_totals = (_totals(report) for report in reports)
    comparisons = [
        {"z": key[0], "mass_number": key[1], "material_z": key[2],
         "coarse_integrated_range_increment_g_cm2": coarse_totals[key],
         "fine_integrated_range_increment_g_cm2": fine_totals[key],
         "relative_change": abs(fine_totals[key] - coarse_totals[key]) / fine_totals[key]}
        for key in sorted(coarse_totals)
    ]
    maximum = max(row["relative_change"] for row in comparisons)
    # An input mutation during comparison must not receive a completed report.
    for path, report in zip((coarse_path, fine_path), reports, strict=True):
        _read_verified(path, report)
    return {
        "schema_version": "radar.ion_stopping_candidate.grid_comparison.v1",
        "coarse_input_sha256": reports[0]["input_sha256_after"],
        "fine_input_sha256": reports[1]["input_sha256_after"],
        "source_files_sha256": reports[0]["source_files_sha256_after"],
        "production_resources_sha256": reports[0]["production_resources_sha256_after"],
        "coarse_energy_node_count": len(a), "fine_energy_node_count": len(b),
        "shared_rows_exactly_equal": True, "shared_row_count": len(coarse["rows"]),
        "comparisons": comparisons, "max_relative_change": maximum,
        "table_quadrature_relative_tolerance": 0.01,
        "table_quadrature_grid_check_passed": maximum < 0.01,
        "coarse_local_consistency_passed": reports[0]["validation"]["numerical_consistency_passed"],
        "fine_local_consistency_passed": reports[1]["validation"]["numerical_consistency_passed"],
        "coarse_max_local_discrepancy": reports[0]["validation"]["max_relative_discrepancy"],
        "fine_max_local_discrepancy": reports[1]["validation"]["max_relative_discrepancy"],
        "normative_release_eligible": False, "spectral_grid_convergence_established": False,
        "scope": "total_stopping_table_quadrature_over_declared_nonzero_energy_interval_only",
        "limitations": [
            "Small whole-interval changes do not override failed local range/stopping checks.",
            "This is not an independent physics comparison or a protected-spectrum calculation.",
            "Common numerical node equality does not establish absolute range from zero energy.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = compare_grids(args.coarse, args.fine)
        write_report_exclusively(report, args.output)
        print(json.dumps({"output": str(args.output), "max_relative_change": report[
            "max_relative_change"], "normative_release_eligible": False}))
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
