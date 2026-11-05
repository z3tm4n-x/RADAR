"""Build an explicit, diagnostic-only OST projectile-mass experiment manifest.

Literal source A is passed on as a numeric experiment parameter, never as proof
of a physical isotope. The independent comparison roster comes only from a
SHA256-pinned v1 candidate; no rounding of source masses selects an isotope.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

from radar.sep.hze_spectrum import OST_SEP_HZE_B9_RESOURCE
from radar.validation.ion_mass_contract import (
    SOURCE_DEFINITION_REFERENCES,
    CandidateIsotope,
    IonMassAuditRow,
    audit_ost_ion_mass_contract,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = (
    "src/radar/gcr/parameters.py",
    "src/radar/sep/hze_spectrum.py",
    f"src/radar/data/normative/{OST_SEP_HZE_B9_RESOURCE}",
    "src/radar/data/integrity.py",
    "src/radar/validation/ion_mass_contract.py",
    "scripts/build_stopping_mass_manifest.py",
)
_SOURCE_ORDER = ("GCR", "SEP")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {relative: _sha(_ROOT / relative) for relative in _SOURCE_PATHS}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _inventory(payload: bytes) -> tuple[CandidateIsotope, ...]:
    raw = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    json.dumps(raw, allow_nan=False)  # Also rejects overflow literals such as 1e999.
    if not isinstance(raw, dict) or raw.get("schema_version") != "radar.ion_stopping_candidate.raw.v1":
        raise ValueError("Expected radar.ion_stopping_candidate.raw.v1 candidate JSON.")
    ions = raw.get("ions")
    if not isinstance(ions, list) or not ions:
        raise ValueError("Candidate ions must be a nonempty array.")
    if any(not isinstance(ion, dict) or set(ion) != {"z", "mass_number", "symbol"}
           for ion in ions):
        raise ValueError("Each candidate ion must declare exactly z, mass_number, symbol.")
    candidates = tuple(CandidateIsotope(**ion) for ion in ions)
    if len({ion.z for ion in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate Z; supply one explicit isotope per element.")
    return tuple(sorted(candidates, key=lambda ion: ion.z))


def _source_inventory(rows: Sequence[IonMassAuditRow]) -> dict[str, Any]:
    """Require complete literal OST rosters and consistent shared mass parameters."""
    by_source: dict[str, dict[int, IonMassAuditRow]] = {source: {} for source in _SOURCE_ORDER}
    for row in rows:
        if row.source not in by_source or type(row.z) is not int or not 2 <= row.z <= 92:
            raise ValueError("Unexpected source or HZE atomic number in source roster.")
        if row.z in by_source[row.source]:
            raise ValueError("Duplicate source/Z row in source roster.")
        mass = row.source_tabulated_mass_parameter
        if isinstance(mass, bool) or not isinstance(mass, (int, float)):
            raise ValueError("Source mass parameter must be numeric, not bool.")
        if not isfinite(mass) or mass < row.z:
            raise ValueError("Source mass parameter must be finite and >= Z.")
        if not isfinite(row.source_weight_parameter) or row.source_weight_parameter < 0:
            raise ValueError("Source weight parameter must be finite and nonnegative.")
        if not row.nonzero_source_components and row.source_weight_parameter != 0:
            raise ValueError("Declared-zero row has a nonzero source weight.")
        if row.source == "SEP" and (
            row.source_mass_to_charge is None or not isfinite(row.source_mass_to_charge)
            or row.source_mass_to_charge <= 0
        ):
            raise ValueError("SEP source A/Q must be preserved as a positive separate parameter.")
        by_source[row.source][row.z] = row
    if set(by_source["GCR"]) != set(range(2, 93)):
        raise ValueError("Complete OST GCR source roster Z=2..92 is required.")
    if set(by_source["SEP"]) != set(range(2, 29)):
        raise ValueError("Complete OST SEP source roster Z=2..28 is required.")

    for z, sep_row in by_source["SEP"].items():
        gcr_row = by_source["GCR"][z]
        if (gcr_row.symbol != sep_row.symbol
                or gcr_row.source_tabulated_mass_parameter != sep_row.source_tabulated_mass_parameter):
            raise ValueError(f"Conflicting literal GCR/SEP source mass or symbol at Z={z}.")
        if sep_row.nonzero_source_components and not gcr_row.nonzero_source_components:
            raise ValueError(f"Positive SEP Z={z} is absent from positive GCR projectiles.")

    projectiles = []
    source_rows = []
    declared_zero_sources = []
    source_counts: dict[str, dict[str, int]] = {}
    for source in _SOURCE_ORDER:
        selected = sorted(by_source[source].values(), key=lambda row: row.z)
        for row in selected:
            entry = {
                "source": source, "z": row.z, "symbol": row.symbol,
                "source_tabulated_mass_parameter": row.source_tabulated_mass_parameter,
                "source_mass_to_charge": row.source_mass_to_charge,
                "source_weight_parameter_name": row.source_weight_parameter_name,
                "source_weight_parameter": row.source_weight_parameter,
                "nonzero_source_components": list(row.nonzero_source_components),
            }
            source_rows.append(entry)
            if not row.nonzero_source_components:
                declared_zero_sources.append(entry.copy())
        positives = sum(bool(row.nonzero_source_components) for row in selected)
        source_counts[source] = {
            "hze_table_rows": len(selected), "positive_hze": positives,
            "declared_zero": len(selected) - positives,
        }
    if source_counts["GCR"]["positive_hze"] != 84 or source_counts["SEP"]["positive_hze"] != 24:
        raise ValueError("Unexpected positive OST roster size; re-audit source definitions explicitly.")
    for z, row in sorted(by_source["GCR"].items()):
        if row.nonzero_source_components:
            membership = ["GCR"]
            if z in by_source["SEP"] and by_source["SEP"][z].nonzero_source_components:
                membership.append("SEP")
            projectiles.append({
                "z": z, "symbol": row.symbol,
                "mass_parameter_u": row.source_tabulated_mass_parameter,
                "source_membership": membership,
            })
    return {
        "projectiles": projectiles, "source_rows": source_rows,
        "declared_zero_sources": declared_zero_sources, "source_counts": source_counts,
    }


def build_manifest(candidate: Path, *, expected_candidate_sha256: str) -> dict[str, Any]:
    if (not isinstance(expected_candidate_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_candidate_sha256) is None):
        raise ValueError("Candidate requires a lowercase SHA256 pin.")
    payload = candidate.read_bytes()
    before = hashlib.sha256(payload).hexdigest()
    if before != expected_candidate_sha256:
        raise ValueError("Candidate SHA256 does not match the supplied pin.")
    source_before = _source_hashes()
    pilots = _inventory(payload)
    audit = audit_ost_ion_mass_contract(pilots)
    inventory = _source_inventory(audit.rows)
    positive_z = {projectile["z"] for projectile in inventory["projectiles"]}
    if any(ion.z not in positive_z for ion in pilots):
        raise ValueError("Every comparison isotope must belong to the positive OST HZE roster.")
    source_after = _source_hashes()
    after = _sha(candidate)
    if before != after or source_before != source_after:
        raise ValueError("Candidate or source definitions changed during manifest generation.")
    return {
        "schema_version": "radar.stopping_mass_manifest.v1",
        **inventory,
        "pilot_isotopes": [
            {"z": ion.z, "symbol": ion.symbol, "mass_parameter_u": ion.mass_number}
            for ion in pilots
        ],
        "mass_parameter_policy": "literal_OST_source_parameter_used_as_diagnostic_projectile_mass",
        "pilot_isotope_policy": "explicit_pinned_candidate_inventory_not_inferred_by_rounding",
        "source_definition_references": SOURCE_DEFINITION_REFERENCES,
        "candidate_path": str(candidate.resolve()), "candidate_expected_sha256": expected_candidate_sha256,
        "candidate_sha256_before": before, "candidate_sha256_after": after,
        "candidate_payload_scope": "ions_inventory_only_stopping_rows_not_validated",
        "source_files_sha256_before": source_before, "source_files_sha256_after": source_after,
        "source_roster_inventory_complete": True,
        "normative_export_allowed": False,
        "physical_mass_policy_qualified": False,
        "transport_tables_generated": False,
        "production_tables_replaced": False,
        "limitations": [
            "Literal source A is a numeric experiment parameter, not a proven isotope or mass in u.",
            "The name mass_parameter_u describes the backend argument, not source mass qualification.",
            "SEP A/Q is preserved separately and must not be passed as the projectile nuclear charge.",
            "Full source roster inventory does not establish stopping-table coverage or backend validity.",
            "Declared zero sources are derived from source coefficients, not sampled numerical zeros.",
            "No source spectrum, production table or normative release gate is changed.",
        ],
    }


def write_manifest_exclusively(manifest: dict[str, Any], output: Path) -> None:
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        manifest = build_manifest(args.candidate, expected_candidate_sha256=args.candidate_sha256)
        write_manifest_exclusively(manifest, args.output)
        print(json.dumps({"output": str(args.output), "sha256": _sha(args.output),
                          "projectiles": len(manifest["projectiles"]),
                          "pilot_isotopes": len(manifest["pilot_isotopes"]),
                          "normative_export_allowed": False}))
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
