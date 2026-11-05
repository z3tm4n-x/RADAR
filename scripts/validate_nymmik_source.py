"""Pinned NASA/OST formula comparison, not a HEASoft execution or qualification.

Downloaded Perl/C files are read as inert text. A deliberately narrow parser
extracts six sinusoid coefficient strings; no downloaded program or expression
is executed. The prefilter clock is a mathematical translation of inspected
source, explicitly separated from the documented apparent-solar-time clock.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import re
from typing import Any

from radar.geomagnetic.local_time import local_solar_time_hours
from radar.geomagnetic.ost_disturbance import (
    MIN_DISTURBANCE_CUTOFF_GV,
    ost_cutoff_disturbance_factor,
    ost_disturbance_coefficients,
    ost_disturbed_cutoff_rigidity_gv,
)

_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = Path("research/heasoft_nymmik_20261031")
_LIBRARY = _EVIDENCE / "archive_selected/heasoft-6.37.1/attitude/lib/atFunctions/src"
_SOURCES = {
    (_EVIDENCE / "cornymmik").as_posix():
        "fb684b53a8a0aaa494d5ce0ae104e5e863ed362631da947c01ff4827f4ea4a5c",
    (_EVIDENCE / "cornymmik.html").as_posix():
        "cb46c55d4be640875c609994995ee112a5781adfd4093e122996a13bc9462861",
    (_EVIDENCE / "prefilter_derive.c").as_posix():
        "58ce700d6748b6ae2fdfac2aa7a0835c88f9ad63a7f6456b1af6e5e9ef8426e5",
    (_EVIDENCE / "prefilter_prefilter.html").as_posix():
        "37658f9d1538d6d819fccbad0f60deb47d67890977f8815109cbca4177b1768c",
    (_LIBRARY / "atInvVect.c").as_posix():
        "eab9f5dbf070dc8ecda47d86d3c2ce53f1618d28ea4620e200072fa35a6cdb6f",
    (_LIBRARY / "atVectToPol.c").as_posix():
        "4c4efa5391aad6922863aa9c09180dce3212e6e22d5fc47a70346ab702894519",
}
_COEFFICIENT_NAMES = ("Aa", "Ba", "Ab", "Bb", "Ac", "Bc")
_HOURS = tuple(index / 4.0 for index in range(96))
_CUTOFFS_GV = (1e-4, 1e-3, 0.1, 1.0, 5.0, 15.0, 100.0)
_RELATIVE_TOLERANCE = 2e-13
_CLOCK_TOLERANCE_HOURS = 1e-12
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Sinusoid:
    amplitude: float
    phase_hours: float
    offset: float

    def value_at_degrees(self, local_time_degrees: float) -> float:
        return self.amplitude * math.sin(math.radians(
            local_time_degrees + self.phase_hours * 15.0,
        )) + self.offset


def _assignment(source: str, name: str) -> str:
    pattern = rf'^\s*\${re.escape(name)}\s*=\s*"([^"\r\n]*)"\s*;[^\r\n]*$'
    matches = re.findall(pattern, source, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one literal string assignment for ${name}.")
    return re.sub(r"\s+", "", matches[0])


def parse_coefficient_strings(source: str) -> dict[str, Sinusoid]:
    """Recognize only the inspected literal grammar, never execute Perl/FITS text."""
    required = {
        "hr": "15.0", "degr": "(#pi/180.0)",
        "a": "($Aa*$kp+$Ba)", "b": "($Ab*$kp+$Bb)", "c": "($Ac*$kp*$kp+$Bc)",
        "delta": "MIN(1.0+0.001*exp(($a)*($cor)^($b)-1.0),$c)",
        "cornym": "($cor/($delta))",
    }
    for name, expression in required.items():
        if _assignment(source, name) != expression:
            raise ValueError(f"Unrecognized NASA formula structure or units in ${name}.")
    number = r"\d+\.\d+"
    pattern = re.compile(
        rf"\((-?{number})\*sin\(\$degr\*\(\$lt([+-])({number})\*\$hr\)\)"
        rf"([+-])({number})\)",
    )
    coefficients = {}
    for name in _COEFFICIENT_NAMES:
        match = pattern.fullmatch(_assignment(source, name))
        if match is None:
            raise ValueError(f"Unrecognized sinusoid grammar in ${name}.")
        amplitude, phase_sign, phase, offset_sign, offset = match.groups()
        coefficients[name] = Sinusoid(
            float(amplitude), float(phase) * (1 if phase_sign == "+" else -1),
            float(offset) * (1 if offset_sign == "+" else -1),
        )
    return coefficients


def source_coefficients(
    coefficients: dict[str, Sinusoid], kp: int, local_time_degrees: float,
) -> tuple[float, float, float]:
    """Evaluate the documented NASA degree-domain coefficient organization C,A,B."""
    values = {name: coefficient.value_at_degrees(local_time_degrees)
              for name, coefficient in coefficients.items()}
    return (values["Ac"] * kp * kp + values["Bc"],
            values["Aa"] * kp + values["Ba"], values["Ab"] * kp + values["Bb"])


def source_delta(cutoff_gv: float, coefficients_c_a_b: tuple[float, float, float]) -> float:
    """Literal NASA arithmetic only on the bounded, positive comparison domain."""
    if not math.isfinite(cutoff_gv) or cutoff_gv < 1e-4:
        raise ValueError("Source comparison is restricted to finite cutoff >= 1e-4 GV.")
    c, a, b = coefficients_c_a_b
    return min(1.0 + 0.001 * math.exp(a * cutoff_gv**b - 1.0), c)


def _longitude(vector: Vector3) -> float:
    if len(vector) != 3 or any(isinstance(value, bool) or not math.isfinite(value)
                               for value in vector):
        raise ValueError("Clock comparison requires three finite vector components.")
    if vector[0] == 0.0 and vector[1] == 0.0:
        raise ValueError("Clock comparison excludes undefined polar longitude.")
    return math.atan2(vector[1], vector[0]) % (2 * math.pi)


def translated_prefilter_local_time_degrees(position: Vector3, sun: Vector3) -> float:
    """Translate derive_local_time lines 721--731 and two inspected helpers.

    This is not execution of NASA C, its ephemeris, its orbit provider or FITS.
    Position is geocentric outward; the inspected code negates it before
    subtracting the Sun longitude and adding 180 degrees.
    """
    _longitude(position)
    opposite_position = (-position[0], -position[1], -position[2])
    return (180.0 + math.degrees(_longitude(opposite_position) - _longitude(sun))) % 360.0


def _clock_row(name: str, position: Vector3, sun: Vector3) -> dict[str, Any]:
    documented_hours = (12.0 + math.degrees(_longitude(position) - _longitude(sun)) / 15) % 24
    code_degrees = translated_prefilter_local_time_degrees(position, sun)
    radar_hours = local_solar_time_hours(position_ecef=position, sun_direction_ecef=sun)
    shift = abs((code_degrees / 15.0 - documented_hours + 12.0) % 24.0 - 12.0)
    error = abs((radar_hours - documented_hours + 12.0) % 24.0 - 12.0)
    return {"case": name, "geocentric_outward_position": position, "sun_direction": sun,
            "documented_solar_hours": documented_hours, "radar_solar_hours": radar_hours,
            "translated_prefilter_degrees": code_degrees,
            "translated_prefilter_hours": code_degrees / 15.0,
            "absolute_circular_code_documentation_shift_hours": shift,
            "absolute_circular_radar_documentation_error_hours": error}


def clock_comparison() -> dict[str, Any]:
    sun = (1.0, 0.0, 0.0)
    cardinal = [_clock_row(name, position, sun) for name, position in (
        ("sunward", (1.0, 0.0, 0.0)), ("east", (0.0, 1.0, 0.0)),
        ("antisunward", (-1.0, 0.0, 0.0)), ("west", (0.0, -1.0, 0.0)),
    )]
    synthetic = []
    for index in range(360):
        p, s = math.radians(index), math.radians((index * 17 + 13) % 360)
        position = (math.cos(p), math.sin(p), 0.3)
        sun = (math.cos(s), math.sin(s), -0.2)
        synthetic.append(_clock_row(f"synthetic_{index}", position, sun))
    rows = [*cardinal, *synthetic]
    max_error = max(row["absolute_circular_radar_documentation_error_hours"] for row in rows)
    max_shift_error = max(abs(row["absolute_circular_code_documentation_shift_hours"] - 12.0)
                          for row in rows)
    if max_error > _CLOCK_TOLERANCE_HOURS or max_shift_error > _CLOCK_TOLERANCE_HOURS:
        raise ArithmeticError("Clock translation did not reproduce the inspected 12-hour conflict.")
    return {"cardinal_rows": cardinal, "synthetic_rows": synthetic, "case_count": len(rows),
            "clock_units": "NASA_degrees_15_per_hour_RADAR_hours",
            "source_code_documentation_conflict_detected": True,
            "expected_absolute_circular_shift_hours": 12.0,
            "maximum_radar_documentation_error_hours": max_error,
            "maximum_deviation_from_12_hour_shift": max_shift_error,
            "tolerance_hours": _CLOCK_TOLERANCE_HOURS,
            "full_prefilter_executed": False, "ephemeris_replayed": False,
            "source_translation_scope": "literal_C_longitude_arithmetic_not_full_HEASoft",
            "normative_clock_selection_performed": False}


def clock_shift_cutoff_examples(coefficients: dict[str, Sinusoid]) -> list[dict[str, Any]]:
    """Show consequences of the detected phase shift, not physical mission error."""
    rows = []
    for cutoff in (1.0, 5.0, 15.0):
        for hours in (0.0, 6.0, 12.0, 18.0):
            shifted_hours = (hours + 12.0) % 24.0
            reference = cutoff / source_delta(cutoff, source_coefficients(
                coefficients, 3, hours * 15.0,
            ))
            shifted = cutoff / source_delta(cutoff, source_coefficients(
                coefficients, 3, shifted_hours * 15.0,
            ))
            rows.append({
                "kp": 3, "quiet_cutoff_gv": cutoff, "documented_solar_hours": hours,
                "translated_producer_hours": shifted_hours,
                "documented_clock_disturbed_cutoff_gv": reference,
                "shifted_clock_disturbed_cutoff_gv": shifted,
                "signed_relative_change_vs_documented_clock": (shifted - reference) / reference,
                "denominator": "documented_clock_disturbed_cutoff_gv",
                "scope": "phase_counterfactual_not_physical_error_or_mission_spectrum",
            })
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path: _sha(_ROOT / path) for path in _SOURCES}


def _read_verified_sources() -> tuple[dict[str, str], dict[str, str]]:
    texts, hashes = {}, {}
    for relative, expected in _SOURCES.items():
        data = (_ROOT / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"Pinned NASA source hash mismatch: {relative}")
        texts[relative] = data.decode("ascii")
        hashes[relative] = digest
    return texts, hashes


def _model_hashes() -> dict[str, str]:
    files = [*_ROOT.joinpath("src/radar").rglob("*.py"), Path(__file__).resolve()]
    return {path.relative_to(_ROOT).as_posix(): _sha(path) for path in sorted(files)}


def _relative_error(actual: float, reference: float) -> float:
    if not math.isfinite(actual) or not math.isfinite(reference) or reference == 0.0:
        raise ValueError("This bounded formula comparison requires finite nonzero references.")
    return abs(actual - reference) / abs(reference)


def build_report() -> dict[str, Any]:
    texts, source_hashes_before = _read_verified_sources()
    models_before = _model_hashes()
    coefficients = parse_coefficient_strings(texts[(_EVIDENCE / "cornymmik").as_posix()])
    coefficient_rows = []
    cutoff_rows = []
    maximum_coefficient_error = 0.0
    maximum_delta_error = 0.0
    maximum_cutoff_error = 0.0
    for kp in range(10):
        for hours in _HOURS:
            reference = source_coefficients(coefficients, kp, hours * 15.0)
            actual = ost_disturbance_coefficients(kp=kp, magnetic_local_time_hours=hours)
            coefficient_error = max(_relative_error(a, b) for a, b in zip(actual, reference))
            maximum_coefficient_error = max(maximum_coefficient_error, coefficient_error)
            coefficient_rows.append({"kp": kp, "hours": hours, "NASA_degrees": hours * 15.0,
                                     "reference_C_A_B": reference, "radar_C_A_B": actual,
                                     "max_relative_error": coefficient_error})
            for cutoff in _CUTOFFS_GV:
                expected_delta = source_delta(cutoff, reference)
                actual_delta = ost_cutoff_disturbance_factor(
                    cutoff_rigidity_gv=cutoff, kp=kp, magnetic_local_time_hours=hours,
                )
                expected_cutoff = cutoff / expected_delta
                actual_cutoff = ost_disturbed_cutoff_rigidity_gv(
                    cutoff_rigidity_gv=cutoff, kp=kp, magnetic_local_time_hours=hours,
                )
                delta_error = _relative_error(actual_delta, expected_delta)
                cutoff_error = _relative_error(actual_cutoff, expected_cutoff)
                maximum_delta_error = max(maximum_delta_error, delta_error)
                maximum_cutoff_error = max(maximum_cutoff_error, cutoff_error)
                cutoff_rows.append({"kp": kp, "hours": hours, "quiet_cutoff_gv": cutoff,
                                    "reference_delta": expected_delta, "radar_delta": actual_delta,
                                    "reference_disturbed_cutoff_gv": expected_cutoff,
                                    "radar_disturbed_cutoff_gv": actual_cutoff,
                                    "delta_relative_error": delta_error,
                                    "cutoff_relative_error": cutoff_error})
    if max(maximum_coefficient_error, maximum_delta_error, maximum_cutoff_error) > _RELATIVE_TOLERANCE:
        raise ArithmeticError("RADAR and parsed NASA formula differ above numerical tolerance.")
    clocks = clock_comparison()
    source_hashes_after, models_after = _source_hashes(), _model_hashes()
    if source_hashes_after != source_hashes_before or models_after != models_before:
        raise ValueError("Sources or implementation changed during the diagnostic; discard run.")
    return {
        "schema": "radar_nymmik_source_comparison_v1",
        "normative_release_eligible": False, "mission_radiation_history_qualified": False,
        "stage4_spectral_convergence_established": False, "production_policy_changed": False,
        "NASA_external_code_executed": False, "complete_HEASoft_replay_performed": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "source_release": "NASA_HEASoft_6.37.1_cornymmik_task_1.0",
        "source_cutoff_column_unit_label": "GeV/c",
        "radar_rigidity_unit": "GV",
        "rigidity_numeric_scale_assumption": (
            "same_numeric_cutoff_scalar_as_in_matching_formula_"
            "not_general_particle_momentum_conversion"
        ),
        "ion_momentum_unit_conversion_verified": False,
        "source_base_url": "https://heasarc.gsfc.nasa.gov/FTP/software/lheasoft/lheasoft6.37.1/src/",
        "source_sha256_before": source_hashes_before, "source_sha256_after": source_hashes_after,
        "model_sha256_before": models_before, "model_sha256_after": models_after,
        "model_hash_scope": "all_src_radar_python_and_this_runner",
        "parsed_sinusoids": {name: asdict(value) for name, value in coefficients.items()},
        "formula_comparison": {
            "coefficient_order": "C_A_B", "kp_values": tuple(range(10)), "hours": _HOURS,
            "cutoff_grid_gv": _CUTOFFS_GV, "coefficient_case_count": len(coefficient_rows),
            "cutoff_case_count": len(cutoff_rows), "coefficient_rows": coefficient_rows,
            "cutoff_rows": cutoff_rows, "max_coefficient_relative_error": maximum_coefficient_error,
            "max_delta_relative_error": maximum_delta_error,
            "max_cutoff_relative_error": maximum_cutoff_error,
            "tolerance_relative": _RELATIVE_TOLERANCE, "numerical_comparison_passed": True,
            "clock_input_semantics_qualified_by_numeric_agreement": False,
        },
        "clock_comparison": clocks,
        "clock_shift_cutoff_examples": clock_shift_cutoff_examples(coefficients),
        "limitations": (
            "Matching coefficient arithmetic is not proof of physical or normative qualification.",
            "NASA help prescribes solar time, but inspected producer code conflicts by 12 hours.",
            "No author paper calibration, UTC epoch, Sun ephemeris or mission history is replayed.",
            "C and Perl files are inert evidence; this is a mathematical translation, not execution.",
            "The numerical grid excludes Rc=0 and Rc below the RADAR 1e-4 GV numerical floor.",
            "Fractional Kp and noncanonical local hours are outside this comparison's input grid.",
            "NASA labels its cutoff column GeV/c, RADAR uses GV; only identical numeric scalars "
            "are compared, with no FITS replay or general ion-momentum unit conversion verified.",
            "No extra mission duration factor is introduced and no spectrum is released by this test.",
        ),
        "radar_numerical_floor_gv": MIN_DISTURBANCE_CUTOFF_GV,
        "zero_cutoff_comparison_performed": False, "below_floor_comparison_performed": False,
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = build_report()
        write_report_exclusively(report, args.output)
        print(json.dumps({"output": str(args.output), "report_sha256": _sha(args.output),
                          "coefficient_cases": report["formula_comparison"]["coefficient_case_count"],
                          "cutoff_cases": report["formula_comparison"]["cutoff_case_count"],
                          "source_clock_conflict": True, "normative_release_eligible": False}))
    except (OSError, ValueError, ArithmeticError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
