"""Independent NIST ASTAR He/Al,Si checks, not physical or normative acceptance.

Derived comparisons use the redistributable NIST STAR 1.2 native tabulations.
No Fortran executable, ASTAR spline, extrapolation or production replacement is
performed. Raw absolute ranges and origin-independent range increments are
reported separately. Current online responses are a separate version check.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from decimal import Decimal
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

from compare_parameterized_stopping_grids import _read_verified
import generate_catima_candidate as native
import generate_astar_helium_controls as controls_generator
from generate_parameterized_stopping import reject_constant, unique_object
from validate_ion_stopping_candidate import _resource_hashes
from validate_parameterized_stopping import run_validator, write_report_exclusively

from radar.validation.nist_astar import AstarMaterial, AstarRow, load_astar_archive
from radar.validation.nist_astar_live import half_printed_unit_decimal, parse_astar_live
from radar.validation.parameterized_stopping import ParameterizedStoppingRow

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/nist_star_20261031"
POLICY = "source_tabulated_parameter_diagnostic"
EVIDENCE_PINS = {
    "download.html": "bf6b8cb9af8cbfd1a7769f6df0a8642ee52a830ca37bf2e3b6df15a5761abd76",
    "programs.html": "d942e2eeb5eafd5ac49010eb4d30daec35e899f747897a4eacfc195c706df2fe",
    "readme.html": "a11290bb6920acd07b2fe01505fe8d57cab26ec1adc09151b87fec532cb916c7",
    "composition_al.html": "4d4e95e62fcb8ec12021c87e096dba3c46baa5ce5be96821516b8d8fac36fb9e",
    "composition_si.html": "5f9e80c21d709cd4f59667eb4f5611f251dbdecf60f7ec3ec545c04bdd3d9453",
    "live_astar_013.html": "20807cc7e660f5a4169b461d80b7acdbd5dec9d6b2272ea2f8be7d44869def71",
    "live_astar_014.html": "d8b7c744889f85a492373f629a20d899f05acabe852fa8b988e91cbcbe2e5156",
}
IMPLEMENTATION = (
    "scripts/compare_astar_helium.py", "scripts/generate_astar_helium_controls.py",
    "scripts/validate_parameterized_stopping.py", "scripts/validate_ion_stopping_candidate.py",
    "scripts/compare_parameterized_stopping_grids.py", "scripts/build_stopping_mass_manifest.py",
    "src/radar/validation/nist_astar.py", "src/radar/validation/nist_astar_live.py",
    "src/radar/validation/parameterized_stopping.py",
)
FIELDS = ("stopping_electronic_mev_cm2_g", "stopping_nuclear_mev_cm2_g",
          "stopping_total_mev_cm2_g", "range_g_cm2")
CONFIG = {"z_effective": 1, "corrections": 0, "calculation": 1, "low_energy": 0,
          "scattering": 255, "scattering_factor": 1.0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, *, zero_allowed: bool = False) -> float:
    try:
        invalid = (isinstance(value, bool) or not isinstance(value, (int, float))
                   or not isfinite(value) or (value < 0 if zero_allowed else value <= 0))
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError("Expected a finite positive value (or explicitly allowed zero).")
    return float(value)


def _configuration(config: Any, expected: dict[str, Any]) -> None:
    if not isinstance(config, dict) or config != expected:
        raise ValueError("Native/candidate configuration differs.")
    if any(type(v) is not int for k, v in config.items() if k != "scattering_factor"):
        raise ValueError("Configuration integer flags must not be bool or float.")
    _number(config["scattering_factor"])


def _native_controls(
    path: Path, expected_sha256: str, archive_sha256: str,
    energies: tuple[float, ...],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("An explicit lowercase native control SHA256 is required.")
    if _sha(path) != expected_sha256:
        raise ValueError("Native control file SHA256 mismatch.")
    raw = json.loads(path.read_bytes(), object_pairs_hook=unique_object, parse_constant=reject_constant)
    json.dumps(raw, allow_nan=False)
    if not isinstance(raw, dict) or raw.get("schema_version") != "radar.astar_helium_native_controls.v1":
        raise ValueError("Unexpected native control schema.")
    if (raw.get("archive_sha256") != archive_sha256 or raw.get("archive_version") != "1.2"
            or raw.get("projectile") != {"z": 2, "mass_parameter_u": 4.0, "charge_input": 2}
            or any(isinstance(v, bool) for v in raw["projectile"].values())
            or raw.get("energy_conversion")
            != "backend_energy_mev_per_u=ASTAR_whole_alpha_energy_mev/4.0"
            or raw.get("range_origin_energy_mev_per_nucleon") != 0.001
            or raw.get("energy_coordinate_compatibility")
            != "source_MeV_n_numerically_equal_backend_MeV_u_assumed"
            or raw.get("material_atomic_numbers") != [13, 14]
            or raw.get("energies_mev_total") != list(energies)):
        raise ValueError("Native control physical identity, units, range origin or grid mismatch.")
    if any(type(raw["projectile"][key]) is not int for key in ("z", "charge_input")):
        raise ValueError("Projectile identity fields must be integers.")
    for flag in ("normative_export_allowed", "physical_accuracy_verified", "physical_mass_policy_qualified",
                 "production_tables_replaced", "fortran_executed", "absolute_range_origin_verified"):
        if raw.get(flag) is not False:
            raise ValueError("Native control qualification flags must remain false.")
    _configuration(raw.get("backend_configuration"), CONFIG)
    _configuration(raw.get("nonradiative_configuration"), {**CONFIG, "corrections": 8})
    if raw.get("native_source_pins_before") != native.PINS:
        raise ValueError("Native source/package pins differ from the retained backend.")
    if set(raw.get("implementation_sha256_before", {})) != set(controls_generator._IMPLEMENTATION_PATHS):
        raise ValueError("Native generator implementation pin coverage differs.")
    # A generation-time snapshot must cover the current physical/resource set;
    # the per-file checks below also verify bytes, not just path cardinality.
    if set(raw.get("production_resources_sha256_before", {})) != set(native.production_hashes()):
        raise ValueError("Native production/resource snapshot coverage differs.")
    for prefix, base in (("native_source_pins", native.SOURCE),
                         ("production_resources_sha256", ROOT), ("implementation_sha256", ROOT)):
        if raw.get(prefix + "_before") != raw.get(prefix + "_after"):
            raise ValueError("Native control inputs changed during its generation.")
        _resource_hashes(raw[prefix + "_after"], base)
    materials = raw.get("materials")
    if not isinstance(materials, list) or len(materials) != 2:
        raise ValueError("Exactly Al and Si material metadata must be retained.")
    material_keys = {"z", "atomic_mass_u", "density_g_cm3", "ionization_potential_override_ev",
                     "elemental_ionization_potential_ev", "ionization_potential_policy"}
    for material, z, potential in zip(materials, (13, 14), (166., 173.), strict=True):
        if (not isinstance(material, dict) or set(material) != material_keys
                or type(material["z"]) is not int or material["z"] != z
                or material["ionization_potential_policy"] != "unchanged_backend_internal_elemental_table"
                or _number(material["elemental_ionization_potential_ev"]) != potential
                or _number(material["ionization_potential_override_ev"], zero_allowed=True) != 0):
            raise ValueError("Native material identity or unmodified ionization policy differs.")
        for field in ("atomic_mass_u", "density_g_cm3"):
            _number(material[field])
    row_fields = {"material_z", "energy_mev_total", "energy_mev_per_nucleon",
                  "stopping_electronic_mev_cm2_g", "stopping_nuclear_mev_cm2_g",
                  "stopping_radiative_mev_cm2_g", "range_g_cm2"}
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 2 * len(energies):
        raise ValueError("Native controls must cover both materials on the full overlap grid.")
    expected_keys = [(z, e) for z in (13, 14) for e in energies]
    for row, key in zip(rows, expected_keys, strict=True):
        if (not isinstance(row, dict) or set(row) != row_fields
                or type(row["material_z"]) is not int
                or (row["material_z"], row["energy_mev_total"]) != key
                or row["energy_mev_per_nucleon"] != key[1] / 4):
            raise ValueError("Native row identity, field coverage or total/per-nucleon energy mismatch.")
        for field in row_fields - {"material_z"}:
            _number(row[field], zero_allowed=field == "stopping_radiative_mev_cm2_g")
        if row["stopping_radiative_mev_cm2_g"] != 0:
            raise ValueError("Radiative stopping is not active in this native overlap domain.")
    for z in (13, 14):
        ranges = [r["range_g_cm2"] for r in rows if r["material_z"] == z]
        if any(a >= b for a, b in zip(ranges, ranges[1:])):
            raise ValueError("Native ranges must increase strictly in each material.")
    return raw


def _reference(row: AstarRow) -> dict[str, float]:
    return {"stopping_electronic_mev_cm2_g": row.stopping_electronic_mev_cm2_g,
            "stopping_nuclear_mev_cm2_g": row.stopping_nuclear_mev_cm2_g,
            "stopping_total_mev_cm2_g": row.stopping_total_mev_cm2_g,
            "range_g_cm2": row.range_csda_g_cm2}


def _sample_candidate(rows: tuple[ParameterizedStoppingRow, ...], energy: float) -> dict[str, float]:
    _number(energy)
    if len(rows) < 2 or any(not isinstance(r, ParameterizedStoppingRow) for r in rows):
        raise ValueError("At least two parameterized stopping rows are required.")
    if len({(r.z, r.mass_parameter_u, r.material_z) for r in rows}) != 1:
        raise ValueError("A candidate interpolation group must have one exact projectile/material.")
    nodes = tuple(row.energy_mev_per_nucleon for row in rows)
    if any(a >= b for a, b in zip(nodes, nodes[1:])):
        raise ValueError("Candidate interpolation energies must increase strictly.")
    if energy < nodes[0] or energy > nodes[-1]:
        raise ValueError("Comparison requires supported candidate energy; no extrapolation.")
    i = min(bisect_right(nodes, energy) - 1, len(nodes) - 2)
    if i < 0:
        raise ValueError("At least two candidate nodes are required.")
    fraction = (energy - nodes[i]) / (nodes[i + 1] - nodes[i])
    return {field: ((1 - fraction) * getattr(rows[i], field)
                   + fraction * getattr(rows[i + 1], field)) for field in FIELDS}


def _relative_metrics(
    candidate: dict[str, float], reference: dict[str, float],
    candidate_anchor: float, reference_anchor: float,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {field: candidate[field] / reference[field] - 1 for field in FIELDS}
    delta = reference["range_g_cm2"] - reference_anchor
    metrics["range_increment_from_0.04_mev"] = (
        (candidate["range_g_cm2"] - candidate_anchor) / delta - 1 if delta > 0 else None)
    return metrics


def _extrema(rows: list[dict[str, Any]], quantity: str) -> dict[str, Any]:
    metrics = (*FIELDS, "range_increment_from_0.04_mev")
    return {metric: max((
        {"signed_relative_difference": r[quantity][metric], "energy_mev_total": r["energy_mev_total"]}
        for r in rows if r[quantity][metric] is not None),
        key=lambda item: abs(item["signed_relative_difference"]), default=None) for metric in metrics}


def _live_check(material: AstarMaterial, text: str) -> dict[str, Any]:
    live = parse_astar_live(text, expected_material_id=material.id)
    archive = {row.energy_mev_total: row for row in material.rows}
    online = {row.energy_mev_total: row for row in live.rows}
    fields = ("energy_mev_total", "stopping_electronic_mev_cm2_g", "stopping_nuclear_mev_cm2_g",
              "stopping_total_mev_cm2_g", "range_csda_g_cm2", "range_projected_g_cm2", "detour_factor")
    checks = []
    for energy in sorted(set(archive) & set(online)):
        left, right = archive[energy], online[energy]
        for field, token in zip(fields, right.printed_tokens, strict=True):
            value, printed = getattr(left, field), getattr(right, field)
            # Retain exact decimal ties. Binary float subtraction can otherwise
            # falsely classify a printed rounding boundary as a version change.
            if field == "stopping_total_mev_cm2_g":
                decimal_value = (Decimal(str(left.stopping_electronic_mev_cm2_g))
                                 + Decimal(str(left.stopping_nuclear_mev_cm2_g)))
            elif field == "range_projected_g_cm2":
                decimal_value = Decimal(str(left.range_csda_g_cm2)) * Decimal(str(left.detour_factor))
            else:
                decimal_value = Decimal(str(value))
            half_unit = half_printed_unit_decimal(token)
            difference = abs(Decimal(token) - decimal_value)
            checks.append({"energy_mev_total": energy, "field": field,
                           "archive_value": value, "online_printed_value": printed,
                           "online_printed_token": token, "printed_half_unit": float(half_unit),
                           "absolute_difference": float(difference),
                           "outside_printed_cell": difference > half_unit,
                           "difference_over_printed_half_unit": float(difference / half_unit)})
    return {
        "archive_node_count": len(archive), "live_node_count": len(online),
        "shared_node_count": len(set(archive) & set(online)),
        "archive_nodes_absent_from_live": sorted(set(archive) - set(online)),
        "live_nodes_absent_from_archive": sorted(set(online) - set(archive)),
        "canonical_online_tokens_sha256": hashlib.sha256(json.dumps(
            [row.printed_tokens for row in live.rows], separators=(",", ":")).encode()).hexdigest(),
        "maximum_difference_over_printed_half_unit": max(checks, key=lambda c: c["difference_over_printed_half_unit"]),
        "outside_printed_rounding_cell": [c for c in checks if c["outside_printed_cell"]],
        "comparison_count": len(checks), "online_version_equivalence_verified": False,
        "limitations": ["Four-significant-digit output is compared with native archive decimal nodes.",
                        "Original Fortran REAL32 log/spline/exp and final formatting are not replayed.",
                        "A missing online energy is not padded, interpolated or called zero."],
    }


def compare(
    coarse_path: Path, dense_path: Path, controls_path: Path, *, controls_sha256: str,
    evidence_directory: Path = EVIDENCE,
) -> dict[str, Any]:
    implementation_before = {p: _sha(ROOT / p) for p in IMPLEMENTATION}
    evidence_before = _resource_hashes(EVIDENCE_PINS, evidence_directory)
    archive = load_astar_archive(evidence_directory / "STAR.tar.gz")
    energies = tuple(r.energy_mev_total for r in archive.material(13).rows if r.energy_mev_total >= .04)
    controls = _native_controls(controls_path, controls_sha256, archive.archive_sha256, energies)
    if controls["archive_member_sha256"] != dict(archive.member_sha256):
        raise ValueError("Native control ASTAR member pins differ from the parsed archive.")
    validations = [run_validator(p) for p in (coarse_path, dense_path)]
    payloads = [_read_verified(p, v["input_sha256_after"])
                for p, v in zip((coarse_path, dense_path), validations, strict=True)]
    if (payloads[0]["manifest"] != payloads[1]["manifest"]
            or payloads[0]["manifest_sha256"] != payloads[1]["manifest_sha256"]
            or payloads[0]["points_per_decade"] != 100 or payloads[1]["points_per_decade"] != 200):
        raise ValueError("Expected matched source-mass candidates on 100 and 200 nodes/decade grids.")
    for payload in payloads:
        dataset = payload["datasets"][POLICY]
        _configuration(dataset["backend_configuration"], controls["backend_configuration"])
        _configuration(dataset["nonradiative_configuration"], controls["nonradiative_configuration"])
        for material, native_material in zip(dataset["materials"], controls["materials"], strict=True):
            if any(material[k] != native_material[k] for k in ("z", "atomic_mass_u", "density_g_cm3")):
                raise ValueError("Material constants differ between candidate and exact controls.")
    comparisons = []
    for z in (13, 14):
        reference_material = archive.material(z)
        reference_rows = tuple(r for r in reference_material.rows if r.energy_mev_total >= .04)
        groups = [tuple(ParameterizedStoppingRow(**row) for row in p["datasets"][POLICY]["rows"]
                        if row["z"] == 2 and row["mass_parameter_u"] == 4 and row["material_z"] == z)
                  for p in payloads]
        values = [[_sample_candidate(group, e / 4) for e in energies] for group in groups]
        direct = [{**{f: r[f] for f in FIELDS if f != "stopping_total_mev_cm2_g"},
                   "stopping_total_mev_cm2_g": (r["stopping_electronic_mev_cm2_g"]
                                                 + r["stopping_nuclear_mev_cm2_g"])}
                  for r in controls["rows"] if r["material_z"] == z]
        rows = []
        for i, reference_row in enumerate(reference_rows):
            ref = _reference(reference_row)
            result: dict[str, Any] = {"energy_mev_total": reference_row.energy_mev_total,
                                      "energy_mev_per_nucleon": reference_row.energy_mev_total / 4,
                                      "reference_astar_archive": ref}
            for name, series in zip(("coarse", "dense", "native_exact"), (*values, direct), strict=True):
                result[name] = series[i]
                result[name + "_over_astar_minus_one"] = _relative_metrics(
                    series[i], ref, series[0]["range_g_cm2"], reference_rows[0].range_csda_g_cm2)
            result["dense_over_native_minus_one"] = _relative_metrics(
                values[1][i], direct[i], values[1][0]["range_g_cm2"], direct[0]["range_g_cm2"])
            rows.append(result)
        bands = []
        for lo, hi in ((.04, .1), (.1, 1.), (1., 10.), (10., 100.), (100., 1000.)):
            selected = [r for r in rows if lo <= r["energy_mev_total"] <= hi]
            bands.append({"inclusive_total_energy_bounds_mev": [lo, hi], "node_count": len(selected),
                          "native_exact_over_astar_extrema": _extrema(selected, "native_exact_over_astar_minus_one")})
        comparisons.append({
            "material_z": z, "material_name": reference_material.name, "node_count": len(rows),
            "reference_range_anchor_mev_total": .04, "rows": rows, "energy_band_summaries": bands,
            "extrema": {name: _extrema(rows, name + "_over_astar_minus_one")
                        for name in ("coarse", "dense", "native_exact")},
            "dense_interpolation_vs_native_extrema": _extrema(rows, "dense_over_native_minus_one"),
            "online_archive_comparison": _live_check(reference_material,
                (evidence_directory / f"live_astar_{z:03d}.html").read_text(encoding="utf-8")),
        })
    evidence_after = _resource_hashes(EVIDENCE_PINS, evidence_directory)
    implementation_after = {p: _sha(ROOT / p) for p in IMPLEMENTATION}
    if (evidence_before != evidence_after or implementation_before != implementation_after
            or _sha(evidence_directory / "STAR.tar.gz") != archive.archive_sha256
            or _sha(controls_path) != controls_sha256):
        raise ValueError("Comparison inputs changed during replay.")
    for p, v in zip((coarse_path, dense_path), validations, strict=True):
        if _sha(p) != v["input_sha256_after"]:
            raise ValueError("Candidate changed during comparison.")
        _resource_hashes(v["production_resources_sha256_after"], ROOT)
    for payload in payloads:
        if _sha(Path(payload["manifest_path"])) != payload["manifest_sha256"]:
            raise ValueError("Candidate manifest changed during comparison.")
    for prefix, base in (("native_source_pins", native.SOURCE),
                         ("production_resources_sha256", ROOT), ("implementation_sha256", ROOT)):
        _resource_hashes(controls[prefix + "_after"], base)
    return {
        "schema_version": "radar.astar_helium_comparison.v1",
        "reference": {"url": "https://physics.nist.gov/PhysRefData/Star/STAR.tar.gz",
            "version": archive.version, "archive_sha256": archive.archive_sha256,
            "member_sha256": dict(archive.member_sha256), "notice": archive.derivative_notice,
            "live_query": {"url": "https://physics.nist.gov/cgi-bin/Star/ap_table-t.pl",
                "method": "POST", "prog": "ASTAR", "matno": ["013", "014"],
                "ShowDefault": "on", "Energies": "", "retrieval_date": "2026-10-31"}},
        "evidence_sha256_before": evidence_before, "evidence_sha256_after": evidence_after,
        "implementation_sha256_before": implementation_before, "implementation_sha256_after": implementation_after,
        "candidate_sha256": [v["input_sha256_after"] for v in validations],
        "candidate_paths": [str(p.resolve()) for p in (coarse_path, dense_path)],
        "native_controls_sha256": controls_sha256, "native_controls_path": str(controls_path.resolve()),
        "native_materials": controls["materials"],
        "production_resources_sha256": controls["production_resources_sha256_after"],
        "energy_conversion": controls["energy_conversion"], "mass_policy": POLICY,
        "range_origin_mev_per_nucleon": .001, "raw_absolute_range_origin_confounded": True,
        "interpolation": "candidate_piecewise_linear; native_exact_controls; no_ASTAr_interpolation",
        "signed_relative_difference_definition": "candidate/ASTAR-1; dense/native-1 for interpolation",
        "comparisons": comparisons, "physical_accuracy_verified": False,
        "physical_mass_policy_qualified": False, "normative_export_allowed": False,
        "production_tables_replaced": False, "fortran_execution_replayed": False,
        "physical_acceptance_tolerance_established": False,
        "limitations": [
            "ASTAR is an independent evaluated model/database, not an error-free physical oracle.",
            "He4 Al/Si overlap only: .04..1000 MeV whole alpha = .01..250 MeV/u under the stated mass policy.",
            "This does not qualify other HZE species, unsupported energies, straggling or fragmentation.",
            "The numerical grid-refinement 1% criterion is not a physical model-accuracy requirement.",
            "Anchored range differences remove additive origin, not differences in stopping models.",
            "No ASTAR offset is added to CATIMA and no table is fitted to the reference.",
            "Published NIST uncertainty guidance depends on energy; it is not assigned as an acceptance band.",
            "Archive and online results are kept separate; neither compiled Fortran nor online version equivalence is claimed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("native_controls", type=Path)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.coarse, args.dense, args.native_controls, controls_sha256=args.native_sha256)
    write_report_exclusively(report, args.output)
    print(json.dumps({"output_sha256": _sha(args.output), "materials": len(report["comparisons"]),
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
