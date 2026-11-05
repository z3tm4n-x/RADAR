"""Validate an isolated candidate dataset without loading an external calculator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from radar.validation.ion_stopping_candidate import IonStoppingRow, validate_candidate_rows


_ROOT = Path(__file__).resolve().parents[1]
_RAW_SCHEMA = "radar.ion_stopping_candidate.raw.v1"
_REQUIRED_FIELDS = {
    "schema_version", "energies_mev_per_nucleon", "ions", "material_atomic_numbers",
    "rows", "source_provenance", "production_resources_sha256", "generation_checks",
}
_OPTIONAL_FIELDS = {
    "materials", "backend_configuration", "nonradiative_configuration",
    "backend_info_literal", "points_per_decade",
}
_SOURCE_ROOT = "research/ion_stopping_20261031"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _resource_hashes(raw_hashes: Any, repository_root: Path) -> dict[str, str]:
    if not isinstance(raw_hashes, dict) or not raw_hashes:
        raise ValueError("production_resources_sha256 must be a nonempty hash map.")
    actual: dict[str, str] = {}
    seen_paths: set[str] = set()
    root = repository_root.resolve(strict=True)
    for relative, expected in raw_hashes.items():
        if not isinstance(relative, str) or not relative or ":" in relative:
            raise ValueError("Resource paths must be nonempty repository-relative paths.")
        portable = PurePosixPath(relative.replace("\\", "/"))
        if (
            portable.is_absolute() or PureWindowsPath(relative).drive
            or ".." in portable.parts or portable.as_posix() == "."
        ):
            raise ValueError("Resource path escapes or is not relative to the repository.")
        target = root.joinpath(*portable.parts).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("Resource path escapes the repository or is not a file.")
        if str(target).casefold() in seen_paths:
            raise ValueError("Duplicate resolved resource path.")
        seen_paths.add(str(target).casefold())
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Missing or malformed SHA256 for {relative}.")
        actual[relative] = _sha(target)
        if actual[relative] != expected:
            raise ValueError(f"Production resource hash mismatch: {relative}")
    return actual


def _source_hashes(provenance: dict[str, Any], root: Path) -> dict[str, str]:
    if provenance.get("source_root") != _SOURCE_ROOT:
        raise ValueError(f"source_root must be exactly {_SOURCE_ROOT}.")
    source_root = (root / _SOURCE_ROOT).resolve(strict=True)
    if not source_root.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Source root escapes the repository.")
    source = _resource_hashes(provenance.get("hashes"), source_root)
    result = {f"{_SOURCE_ROOT}/{key}": value for key, value in source.items()}
    if "generator_sha256" in provenance:
        result.update(_resource_hashes(
            {"scripts/generate_catima_candidate.py": provenance["generator_sha256"]}, root
        ))
    return result


def run_validator(
    input_path: Path,
    *,
    relative_tolerance: float = 0.01,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Check immutable input and declared resources, then report range consistency.

    Source bytes are rehashed, but arbitrary provenance and generation-check
    claims are not established by this validator. Supplied hash maps scope the
    unchanged-file claims; they are not inferred to cover undeclared files.
    """
    root = _ROOT if repository_root is None else repository_root
    payload_bytes = input_path.read_bytes()
    before = hashlib.sha256(payload_bytes).hexdigest()
    raw = json.loads(payload_bytes, object_pairs_hook=_unique_object,
                     parse_constant=_reject_constant)
    # Also rejects overflow literals such as 1e999, including in metadata.
    json.dumps(raw, allow_nan=False)
    if (not isinstance(raw, dict) or not _REQUIRED_FIELDS <= set(raw)
            or set(raw) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS):
        raise ValueError("Candidate JSON must contain only the documented raw-schema fields.")
    if raw["schema_version"] != _RAW_SCHEMA:
        raise ValueError("Unsupported candidate raw schema_version.")
    for name in ("source_provenance", "generation_checks"):
        if not isinstance(raw[name], dict) or not raw[name]:
            raise ValueError(f"{name} must be a nonempty object.")
    for name in ("ions", "material_atomic_numbers", "energies_mev_per_nucleon", "rows"):
        if not isinstance(raw[name], list) or not raw[name]:
            raise ValueError(f"{name} must be a nonempty array.")
    ions = []
    for ion in raw["ions"]:
        if not isinstance(ion, dict) or set(ion) != {"z", "mass_number", "symbol"}:
            raise ValueError("Each ion must declare z, mass_number and symbol.")
        if not isinstance(ion["symbol"], str) or not ion["symbol"].strip():
            raise ValueError("Ion symbol must be nonempty text.")
        ions.append((ion["z"], ion["mass_number"]))
    if raw["material_atomic_numbers"] != [13, 14]:
        raise ValueError("This candidate matrix must declare materials [13, 14] (Al and Si).")
    resources_before = _resource_hashes(raw["production_resources_sha256"], root)
    source_before = _source_hashes(raw["source_provenance"], root)
    expected_row_fields = {item.name for item in fields(IonStoppingRow)}
    if any(not isinstance(row, dict) or set(row) != expected_row_fields for row in raw["rows"]):
        raise ValueError("Candidate rows must exactly match the IonStoppingRow schema.")
    try:
        rows = tuple(IonStoppingRow(**row) for row in raw["rows"])
    except TypeError as exc:
        raise ValueError("Candidate rows must match the IonStoppingRow schema.") from exc
    validation = validate_candidate_rows(
        rows, required_ions=ions, material_atomic_numbers=raw["material_atomic_numbers"],
        energy_nodes_mev_per_nucleon=raw["energies_mev_per_nucleon"],
        relative_tolerance=relative_tolerance,
    )
    resources_after = _resource_hashes(raw["production_resources_sha256"], root)
    source_after = _source_hashes(raw["source_provenance"], root)
    after = _sha(input_path)
    if before != after or resources_before != resources_after or source_before != source_after:
        raise ValueError("Candidate input, source or production resources changed during validation.")
    return {
        "schema_version": "radar.ion_stopping_candidate.validation.v1",
        "input_path": str(input_path.resolve()), "input_sha256_before": before,
        "input_sha256_after": after,
        "source_provenance": raw["source_provenance"],
        "generation_checks": raw["generation_checks"],
        "candidate_settings": {key: raw[key] for key in sorted(_OPTIONAL_FIELDS) if key in raw},
        "source_files_sha256_before": source_before,
        "source_files_sha256_after": source_after,
        "declared_source_file_hashes_verified": True,
        "production_resources_sha256_before": resources_before,
        "production_resources_sha256_after": resources_after,
        "source_provenance_claims_independently_verified": False,
        "generation_checks_independently_replayed": False,
        "production_hash_scope": "only_declared_production_resources_sha256_entries",
        "source_hash_scope": "declared_source_hash_entries_and_generator_if_present",
        "ions": raw["ions"], "material_atomic_numbers": raw["material_atomic_numbers"],
        "energies_mev_per_nucleon": raw["energies_mev_per_nucleon"],
        "validation": asdict(validation),
        "normative_release_eligible": False, "production_tables_replaced": False,
        "full_required_ion_coverage_established": False,
        "scope": "candidate_subset_only_not_a_full_GCR_or_SEP_transport_dataset",
        "limitations": [
            "The planned pilot covers 15 legacy isotope choices, not all GCR H-U ions.",
            "Numerical range-increment consistency does not validate source physics or licences.",
            "Absolute range from zero energy and normative LET definition remain unqualified.",
            "No production table, spectrum, audit status or normative release gate is modified.",
        ],
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = run_validator(args.input, relative_tolerance=args.relative_tolerance)
        write_report_exclusively(report, args.output)
        print(json.dumps({"output": str(args.output), "sha256": _sha(args.output),
                          "numerical_consistency_passed": report["validation"][
                              "numerical_consistency_passed"], "normative_release_eligible": False}))
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
