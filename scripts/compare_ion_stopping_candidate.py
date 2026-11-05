"""Compare three retained GSI responses; agreement is not normative qualification."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from radar.validation.ion_stopping_candidate import IonStoppingRow

_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = Path("research/ion_stopping_20261031")
_CASES = {
    "gsi_fe56_al_100.html": (26, 56, 100, 13),
    "gsi_fe56_si_1000.html": (26, 56, 1000, 14),
    "gsi_he4_si_100.html": (2, 4, 100, 14),
}
_N = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def _positive(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a positive finite number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a positive finite number.")
    return float(value)


def _one(pattern: str, text: str) -> tuple[str, ...]:
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError("GSI response field missing, duplicated or in unexpected units.")
    return matches[0].groups()


def parse_gsi_response(text: str, expected: tuple[int, int, int, int]) -> dict[str, Any]:
    """Parse inert HTML and verify the exact requested projectile, target and units."""
    z, a, energy, target = expected
    echo = _one(rf"Zp={_N} Mass_p={_N} E={_N} d={_N} Zt={_N} At={_N}\s*<br>", text)
    if tuple(map(float, echo)) != (z, a, energy, 1, target, 0):
        raise ValueError("GSI request readback does not match the fixed case.")
    projectile = _one(rf"projectile\s*:\s*Z={_N} A={_N} E={_N} MeV/u\s*<br>", text)
    if tuple(map(float, projectile)) != (z, a, energy):
        raise ValueError("GSI projectile readback mismatch.")
    material = _one(rf"target\s*:\s*Z={_N} A={_N} thickness={_N} mg/cm\^2\s*<br>", text)
    tz, mass, thickness = tuple(_positive(float(v)) for v in material)
    if tz != target or thickness != 1:
        raise ValueError("GSI target readback mismatch.")
    component = _one(rf"1: Z={_N}, A={_N}, w={_N}, pot={_N}\s*<br>", text)
    if tuple(map(float, component[:3])) != (tz, mass, 1):
        raise ValueError("GSI elemental composition mismatch.")
    potential = _positive(float(component[3]))
    version, = _one(r"<TITLE>\s*ATIMA ([0-9.]+), calculating ATomic Interaction with MAtter\s*</TITLE>", text)
    body_version, = _one(r"<br> Atima ([0-9.]+) <br>", text)
    if version != body_version:
        raise ValueError("GSI version labels disagree within the response.")
    stopping, = _one(r"dE/dx at entrance\s*:\s*(\d+\.\d{6}) MeV/\(mg/cm\^2\)<br>", text)
    range_text, = _one(r"\brange\s*:\s*(\d+\.\d{6}) mg/cm\^2\s*<br>", text)
    return {
        "z": z, "mass_number": a, "energy_mev_per_nucleon": energy, "material_z": target,
        "version_literal": version, "form_version_label": "1.41",
        "target_atomic_mass_u": mass, "target_ionization_potential_ev": potential,
        "target_thickness_mg_cm2": thickness,
        "stopping_total_mev_cm2_g": _positive(float(stopping)) * 1000,
        "range_g_cm2": _positive(float(range_text)) / 1000,
        "printed_stopping_mev_cm2_mg": stopping, "printed_range_mg_cm2": range_text,
        "printed_decimal_places": 6, "stopping_rounding_half_unit_mev_cm2_g": 0.0005,
        "range_rounding_half_unit_g_cm2": 5e-10,
    }


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_comparison(candidate: Path, *, repository_root: Path = _ROOT) -> dict[str, Any]:
    paths = [candidate, *(repository_root / _EVIDENCE / name for name in _CASES)]
    contents = {str(p.resolve()): p.read_bytes() for p in paths}
    before = {p: hashlib.sha256(data).hexdigest() for p, data in contents.items()}
    raw = json.loads(contents[str(candidate.resolve())], object_pairs_hook=_unique)
    json.dumps(raw, allow_nan=False)
    if not isinstance(raw, dict) or raw.get("schema_version") != "radar.ion_stopping_candidate.raw.v1":
        raise ValueError("Unsupported candidate schema.")
    rows = raw.get("rows")
    row_fields = {f.name for f in fields(IonStoppingRow)}
    if not isinstance(rows, list) or not rows:
        raise ValueError("Candidate rows must be nonempty.")
    if any(not isinstance(r, dict) or set(r) != row_fields for r in rows):
        raise ValueError("Candidate rows must explicitly declare all stopping components.")
    parsed = tuple(IonStoppingRow(**r) for r in rows)
    materials = raw.get("materials")
    if not isinstance(materials, list) or len(materials) != 2:
        raise ValueError("Both candidate material definitions are required.")
    if any(not isinstance(m, dict) or isinstance(m.get("z"), bool) for m in materials):
        raise ValueError("Invalid candidate material definition.")
    if {m.get("z") for m in materials} != {13, 14}:
        raise ValueError("Candidate materials must be Al and Si.")
    for material in materials:
        _positive(material.get("atomic_mass_u"))
    configuration = raw.get("backend_configuration")
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError("Explicit backend configuration is required.")
    comparisons = []
    for name, case in _CASES.items():
        key = str((repository_root / _EVIDENCE / name).resolve())
        reference = parse_gsi_response(contents[key].decode("utf-8"), case)
        matches = [r for r in parsed if (r.z, r.mass_number, r.energy_mev_per_nucleon, r.material_z) == case]
        if len(matches) != 1:
            raise ValueError(f"Candidate must contain exactly one matching node for {name}.")
        row = matches[0]
        comparisons.append({
            "reference_file": key, "reference": reference,
            "candidate_material": next(m for m in materials if m["z"] == row.material_z),
            "candidate_stopping_total_mev_cm2_g": row.stopping_total_mev_cm2_g,
            "candidate_range_g_cm2": row.range_g_cm2,
            "signed_relative_stopping_difference": row.stopping_total_mev_cm2_g / reference["stopping_total_mev_cm2_g"] - 1,
            "signed_relative_range_difference": row.range_g_cm2 / reference["range_g_cm2"] - 1,
        })
    after = {p: _sha(Path(p)) for p in before}
    if before != after:
        raise ValueError("Comparison inputs changed during the run.")
    return {
        "schema_version": "radar.ion_stopping_candidate.gsi_comparison.v1",
        "input_sha256_before": before, "input_sha256_after": after,
        "candidate_backend_configuration": configuration,
        "candidate_backend_info_literal": raw.get("backend_info_literal"),
        "comparisons": comparisons, "normative_release_eligible": False,
        "production_tables_replaced": False, "physics_validated": False,
        "scope": "three_cross_model_checkpoints_not_correctness_oracles",
        "limitations": ["Different effective-charge models and material constants may apply.",
                        "Printed precision is recorded, not used as a physical acceptance tolerance.",
                        "Candidate provenance claims and backend execution are not replayed here."],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_report(run_comparison(args.candidate), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
