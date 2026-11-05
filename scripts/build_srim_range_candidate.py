"""Build a separate anchored SRIM stopping candidate and bounded transport checks.

Read-only replay of frozen native outputs. No SRIM process, production adapter,
absolute stopping threshold, source-mass choice or release promotion is created.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import audit_srim_range_stopping as replay  # noqa: E402
from scripts import run_srim_coverage as native_controls  # noqa: E402
from radar.validation.srim_anchored_range import (  # noqa: E402
    AnchoredSrimRangeTable, StoppingInterpolation,
)
from radar.validation.srim_native_coverage import audit_srim_native_coverage  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput  # noqa: E402

PARENT_REPORT = "outputs/srim_range_stopping_final_20261101.json"
PARENT_SHA256 = "8a2deb5b889db0bbff18a0c4732bf0f4a1c228bb3f81737dedd82ea318e65cfd"
MASS_MANIFEST = "outputs/stopping_mass_manifest_20261031.json"
MASS_MANIFEST_SHA256 = "f367a1b5f047451e40ea37d4cb03139ff6cbdb7c7eb9333888b44c11d46250fc"
IMPLEMENTATION = (
    "scripts/build_srim_range_candidate.py",
    "src/radar/validation/srim_anchored_range.py",
    "src/radar/validation/srim_native_coverage.py",
)
SELECTION = tuple(
    ("baseline" if z == 2 else "upper", f"{symbol.lower()}{mass}_{material}")
    for z, symbol, mass in native_controls.LEGACY_IONS for material in ("al", "si")
) + (("boundary", "h_al_4000"), ("boundary", "h_si_4000"))
SHIELDS = tuple(Decimal(value) for value in ("0", ".01", ".1", "1", "3", "10"))
INTERPOLATIONS: tuple[StoppingInterpolation, ...] = ("linear_stopping", "loglog_stopping")
_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_NUMERICAL_TOLERANCE = Decimal("1e-60")


def _pins() -> dict[str, str]:
    """Validate the frozen parent/mass contracts before interpreting native data."""
    if replay._sha(ROOT / PARENT_REPORT) != PARENT_SHA256:
        raise ValueError("Pinned parent range/stopping report changed.")
    parent = json.loads((ROOT / PARENT_REPORT).read_text(encoding="utf-8"))
    if parent["report_sha256"] != replay.REPORT_PINS:
        raise ValueError("Native catalog pins differ from the parent report.")
    if replay._sha(ROOT / MASS_MANIFEST) != MASS_MANIFEST_SHA256:
        raise ValueError("Pinned source-mass manifest changed.")
    mass = json.loads((ROOT / MASS_MANIFEST).read_text(encoding="utf-8"))
    if mass["source_files_sha256_before"] != mass["source_files_sha256_after"]:
        raise ValueError("Mass manifest recorded changed source files.")
    pins = {PARENT_REPORT: PARENT_SHA256, MASS_MANIFEST: MASS_MANIFEST_SHA256,
            **parent["implementation_sha256"], **mass["source_files_sha256_after"]}
    for relative, expected in pins.items():
        if replay._sha(ROOT / relative) != expected:
            raise ValueError(f"Source or parent implementation changed: {relative}")
    return {**pins, **{p: replay._sha(ROOT / p) for p in IMPLEMENTATION}}


def _inverse_linear(value: Decimal, x: tuple[Decimal, ...], y: tuple[Decimal, ...]) -> Decimal:
    """Explicit projected-table comparison operator; no extrapolation."""
    if value < x[0] or value > x[-1]:
        raise ValueError("Projected inverse outside observed support.")
    i = min(bisect_right(x, value) - 1, len(x) - 2)
    if value == x[i]:
        return y[i]
    if value == x[i + 1]:
        return y[i + 1]
    return y[i] + (value - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i])


def _transport_checks(table: AnchoredSrimRangeTable, native: SrimRangeOutput) -> dict[str, Any]:
    """Known incident nodes only, spherical-centre material paths, no source convolution."""
    if (native.raw_sha256 != table.native_raw_sha256
            or native.semantic_sha256 != table.native_semantic_sha256):
        raise ValueError("Candidate and projected comparison require the same native identity.")
    with localcontext(_CONTEXT):
        energy = table.energy_total_mev
        projected = tuple((row.projected_range_cm - native.rows[0].projected_range_cm)
                          * native.density_g_cm3 for row in native.rows)
        midpoint_checks = []
        for e0, e1 in zip(energy, energy[1:]):
            e = (e0 * e1).sqrt()
            r = table.range_at_energy(e)
            recovered = table.energy_at_range(r)
            error = abs(recovered - e) / max(Decimal(1), e)
            if error > _NUMERICAL_TOLERANCE:
                raise ValueError("Candidate off-node inverse fails numerical roundtrip.")
            midpoint_checks.append({"energy_total_mev": e, "anchored_range_g_cm2": r,
                                    "inverse_energy_total_mev": recovered, "scaled_error": error})
        shields = []
        for thickness in SHIELDS:
            rows = []
            for incident, projected_range in zip(energy, projected, strict=True):
                out = table.residual_energy(incident, thickness)
                projected_out = (None if thickness > projected_range else _inverse_linear(
                    projected_range - thickness, projected, energy))
                residual = None
                if out is not None:
                    if not energy[0] <= out <= incident:
                        raise ValueError("Candidate returns an unsupported or increased exit energy.")
                    residual = table.range_at_energy(incident) - table.range_at_energy(out) - thickness
                    if abs(residual) / max(Decimal(1), table.anchored_range_g_cm2[-1]) > (
                            _NUMERICAL_TOLERANCE):
                        raise ValueError("Candidate path difference fails numerical closure.")
                if thickness == 0 and (out != incident or projected_out != incident):
                    raise ValueError("Zero shielding must preserve the exact input energy.")
                rows.append({
                    "incident_total_mev": incident, "candidate_exit_total_mev": out,
                    "candidate_status": "SUPPORTED_EXIT" if out is not None else "UNRESOLVED_BELOW_SUPPORT",
                    "projected_comparison_exit_total_mev": projected_out,
                    "projected_comparison_status": "SUPPORTED_EXIT" if projected_out is not None
                    else "UNRESOLVED_BELOW_SUPPORT",
                    "candidate_path_closure_residual_g_cm2": residual,
                    "candidate_minus_projected_exit_mev": None if out is None or projected_out is None
                    else out - projected_out,
                })
            shields.append({"thickness_g_cm2": thickness, "rows": rows,
                            "candidate_supported": sum(r["candidate_exit_total_mev"] is not None
                                                       for r in rows),
                            "candidate_unresolved": sum(r["candidate_exit_total_mev"] is None
                                                        for r in rows)})
        return {"off_node_roundtrips": midpoint_checks, "shields": shields,
                "projected_comparison_operator": "linear_R_vs_total_energy_after_subtracting_R_at_Emin",
                "candidate_operator": "analytic_integrated_stopping_and_inverse",
                "scaled_numerical_tolerance": _NUMERICAL_TOLERANCE,
                "roundtrip_scale": "max(1_MeV, incident_total_energy)",
                "path_closure_scale": "max(1_g_cm2, maximum_anchored_range)",
                "physical_tolerance_or_uncertainty": False,
                "source_spectrum_convolution_performed": False,
                "below_support_means_physically_stopped": False,
                "normative_export_allowed": False}


def build_candidate() -> dict[str, Any]:
    pins = _pins()
    verified = {catalog: replay._verify_report(catalog) for catalog in replay.REPORT_PINS}
    native_outputs = []
    selections = []
    for catalog, name in SELECTION:
        report, by_case = verified[catalog]
        case = next(c for c in report["cases"] if c["case"]["name"] == name)
        if case["status"] != "CONTROL_COMPLETE" or name not in by_case:
            raise ValueError(f"Required native candidate control is incomplete: {catalog}/{name}")
        native_outputs.append(by_case[name])
        selections.append({"catalog": catalog, "case": name, "raw_sha256": by_case[name].raw_sha256,
                           "semantic_sha256": by_case[name].semantic_sha256})
    tables = []
    for native in native_outputs:
        if native.target_z != 13:
            continue
        for interpolation in INTERPOLATIONS:
            table = AnchoredSrimRangeTable.from_native(native, interpolation=interpolation)
            tables.append({"candidate": asdict(table), "diagnostic_transport": _transport_checks(table, native)})
    coverage = audit_srim_native_coverage(tuple(native_outputs))
    # Recheck all parent identities, including failed runs; no re-execution.
    for catalog, (report, _) in verified.items():
        path = ROOT / f"outputs/srim_legacy_{catalog}_final_20261101/report.json"
        if replay._sha(path) != replay.REPORT_PINS[catalog]:
            raise ValueError("Native report changed during candidate construction.")
        for path_text, expected in report["protected_sha256_after"].items():
            if replay._sha(Path(path_text)) != expected:
                raise ValueError("Protected native-run dependency changed.")
        for case in report["cases"]:
            for repeat in case["repeats"]:
                for key in ("input", "output", "command_file"):
                    if repeat[key] is not None:
                        replay._check_identity(repeat[key])
    for relative, expected in pins.items():
        if replay._sha(ROOT / relative) != expected:
            raise ValueError("Source or implementation changed during candidate construction.")
    return {
        "schema_version": "radar.srim_anchored_range_candidate.v1",
        "protected_sha256_before": pins, "protected_sha256_after": pins.copy(),
        "native_catalog_sha256": replay.REPORT_PINS, "selected_native_controls": selections,
        "range_candidates": tables, "source_native_coverage": asdict(coverage),
        "unavailable_controls": [{"catalog": catalog, "case": c["case"], "status": c["status"]}
                                 for catalog, (report, _) in verified.items() for c in report["cases"]
                                 if c["status"] != "CONTROL_COMPLETE"],
        "diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
        "production_tables_replaced": False, "physical_accuracy_established": False,
        "source_spectra_changed": False, "isotope_policy_chosen": False,
        "limitations": [
            "Derived anchored range is an integral of declared interpolation, not native SRIM CSDA output.",
            "Zero anchored range refers to the first observed energy, not physical rest.",
            "Native projected ranges remain separate comparison data; no empirical rescaling.",
            "Only numerical transport on native nodes; no GCR/SEP/ERB spectral convolution or mission integration.",
            "Si outputs support a coverage inventory only; no production LET adapter is created.",
            "Mass equality and interval containment do not establish source isotope or transport qualification.",
            "No extrapolation, failed-case zero filling, nuclear reactions or fragmentation model added.",
            "Native H mass is retained, but proton transport uses total MeV, not a per-nucleon axis.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = build_candidate()
    payload = native_controls._json(report).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(),
                      "candidate_tables": len(report["range_candidates"]),
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
