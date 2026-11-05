"""Audit a SHA256-pinned candidate ion inventory against the current OST sources."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from radar.sep.hze_spectrum import OST_SEP_HZE_B9_RESOURCE
from radar.validation.ion_mass_contract import (
    SOURCE_DEFINITION_REFERENCES, CandidateIsotope, audit_ost_ion_mass_contract,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = (
    "src/radar/gcr/parameters.py",
    "src/radar/gcr/source_spectra.py",
    "src/radar/sep/hze_spectrum.py",
    "src/radar/sep/model.py",
    f"src/radar/data/normative/{OST_SEP_HZE_B9_RESOURCE}",
    "src/radar/validation/ion_mass_contract.py",
    "scripts/audit_ion_mass_contract.py",
)


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


def _check_pin(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} requires a lowercase SHA256 pin.")


def _source_hashes() -> dict[str, str]:
    return {relative: _sha(_ROOT / relative) for relative in _SOURCE_PATHS}


def run_audit(
    input_path: Path, *, expected_candidate_sha256: str,
    source_document_path: Path | None = None, expected_source_document_sha256: str | None = None,
) -> dict[str, Any]:
    """Read only candidate inventory; explicitly do not validate stopping rows."""
    _check_pin(expected_candidate_sha256, "Candidate")
    payload = input_path.read_bytes()
    before = hashlib.sha256(payload).hexdigest()
    if before != expected_candidate_sha256:
        raise ValueError("Candidate SHA256 does not match the supplied pin.")
    source_before = _source_hashes()
    document = None
    if (source_document_path is None) != (expected_source_document_sha256 is None):
        raise ValueError("Source document path and SHA256 must be supplied together.")
    if source_document_path is not None:
        assert expected_source_document_sha256 is not None
        _check_pin(expected_source_document_sha256, "Source document")
        if _sha(source_document_path) != expected_source_document_sha256:
            raise ValueError("Source document SHA256 does not match the supplied pin.")
        document = {"path": str(source_document_path.resolve()),
                    "sha256": expected_source_document_sha256,
                    "contents_independently_read_by_runner": False}
    raw = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    json.dumps(raw, allow_nan=False)
    if not isinstance(raw, dict) or raw.get("schema_version") != "radar.ion_stopping_candidate.raw.v1":
        raise ValueError("Expected radar.ion_stopping_candidate.raw.v1 candidate JSON.")
    ions = raw.get("ions")
    if not isinstance(ions, list) or not ions:
        raise ValueError("Candidate ions must be a nonempty array.")
    if any(not isinstance(ion, dict) or set(ion) != {"z", "mass_number", "symbol"}
           for ion in ions):
        raise ValueError("Each candidate ion must declare exactly z, mass_number, symbol.")
    inventory = tuple(CandidateIsotope(**ion) for ion in ions)
    audit = audit_ost_ion_mass_contract(inventory)
    source_after = _source_hashes()
    after = _sha(input_path)
    if before != after or source_before != source_after:
        raise ValueError("Candidate input or source definitions changed during the audit.")
    if source_document_path is not None and _sha(source_document_path) != expected_source_document_sha256:
        raise ValueError("Source document changed during the audit.")
    return {
        "schema_version": "radar.ion_mass_contract.audit.v1",
        "candidate_path": str(input_path.resolve()),
        "candidate_sha256_before": before, "candidate_sha256_after": after,
        "candidate_expected_sha256": expected_candidate_sha256,
        "candidate_payload_scope": "ions_inventory_only_stopping_rows_not_validated",
        "source_files_sha256_before": source_before, "source_files_sha256_after": source_after,
        "source_definition_references": SOURCE_DEFINITION_REFERENCES,
        "source_document": document,
        "audit": asdict(audit),
        "normative_export_allowed": False,
        "limitations": [
            "Source tabulated fractional A is retained, not replaced by an isotope or atomic weight.",
            "Rounded metadata A is reported for diagnosis, never used to select a candidate isotope.",
            "Exact numeric A agreement does not establish a source isotope distribution.",
            "Candidate presence does not establish Al/Si, energy-domain or stopping-table coverage.",
            "Printed unusual source masses remain unchanged; no periodic-table correction is inferred.",
            "No source formula, production data or normative release gate is modified.",
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
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--source-document", type=Path)
    parser.add_argument("--source-document-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = run_audit(
            args.input, expected_candidate_sha256=args.candidate_sha256,
            source_document_path=args.source_document,
            expected_source_document_sha256=args.source_document_sha256,
        )
        write_report_exclusively(report, args.output)
        print(json.dumps({"output": str(args.output), "sha256": _sha(args.output),
                          "normative_export_allowed": False}))
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
