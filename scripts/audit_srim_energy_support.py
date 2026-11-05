"""Replay retained SRIM evidence and audit source-energy support, without execution.

This is a table-domain ledger, not a source convolution or replacement transport
model. The native replay implementation is pinned; its private replay interface
is intentionally reused without rebuilding the unrelated Al candidate arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.validation.srim_energy_support import audit_srim_energy_support  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog  # noqa: E402

REPLAY_PATH = "scripts/summarize_srim_source_campaign.py"
REPLAY_SHA256 = "7ad98463fea65bcfb4f3919a28928c923ff1c14e204e472f85eac2ba2a43cf9f"
IMPLEMENTATION = (
    "scripts/audit_srim_energy_support.py",
    "src/radar/validation/srim_energy_support.py",
    "src/radar/validation/srim_native_coverage.py",
    "src/radar/validation/srim_source_catalog.py",
    "src/radar/validation/srim_output.py",
    "src/radar/core/constants.py",
    "src/radar/core/types.py",
    "src/radar/core/units.py",
)


def _load_replay() -> ModuleType:
    path = ROOT / REPLAY_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != REPLAY_SHA256:
        raise ValueError("Frozen native replay implementation changed.")
    spec = importlib.util.spec_from_file_location("srim_replay_for_energy_support", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the pinned read-only native replay.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_report(report_path: Path, *, expected_report_sha256: str) -> dict[str, Any]:
    """Authenticate both native repeats, then compare unchanged source grids."""
    replay = _load_replay()
    evidence = replay._Evidence()
    evidence.read(ROOT / REPLAY_PATH, REPLAY_SHA256)
    implementation = {
        name: hashlib.sha256(evidence.read(ROOT / name)).hexdigest()
        for name in IMPLEMENTATION
    }
    catalog = build_source_catalog(ROOT)
    campaign, observations = replay._replay(
        report_path, expected_report_sha256, catalog, evidence,
    )
    ledger = audit_srim_energy_support(observations)
    result = {
        "schema_version": "radar.srim_source_energy_support.v1",
        "campaign_report": {"path": str(report_path.resolve()),
                            "sha256": expected_report_sha256},
        "campaign_mode": campaign["mode"],
        "native_control_status_counts": campaign["native_control_status_counts"],
        "verified_native_control_count": len(observations),
        "unavailable_controls": [
            {"case": case["case"], "status": case["status"], "checkpoint": case["checkpoint"]}
            for case in campaign["cases"] if case["status"] != "CONTROL_COMPLETE"
        ],
        "energy_support": asdict(ledger),
        "implementation_sha256": implementation,
        "pinned_replay_sha256": REPLAY_SHA256,
        "diagnostic_status": "INCOMPLETE",
        "normative_export_allowed": False,
        "production_tables_replaced": False,
        "source_grids_changed": False,
        "source_spectrum_convolution_performed": False,
        "native_execution_performed": False,
        "physical_accuracy_verified": False,
        "limitations": [
            "Center counts and uncovered intervals do not measure missing particle fluence.",
            "Support refers to an explicit interpolation domain, not native samples at every energy.",
            "GCR finite integration closure is separate from its last evaluation center.",
            "SEP evaluated-grid endpoint is not claimed to be a physical upper model boundary.",
            "Low residual energies, absolute stopping origin and isotope interpretation remain separate.",
            "No extrapolation, source truncation, parameter correction or production connection is made.",
            "The runtime source grids are implementation choices, not newly asserted OST prescriptions.",
        ],
    }
    # Source inventories and pinned source files are checked again after the audit.
    replay._same([asdict(c) for c in build_source_catalog(ROOT)],
                 [asdict(c) for c in catalog], "source catalog after energy audit")
    before = evidence.pins.copy()
    evidence.verify_unchanged()
    result["evidence_sha256_before"] = before
    result["evidence_sha256_after"] = evidence.pins.copy()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_report(args.campaign_report, expected_report_sha256=args.report_sha256)
    # Use the pinned encoder: decimal text is retained and nonfinite values refused.
    encoded = _load_replay()._json(result).encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(f"Energy-support ledger saved: {args.output}; normative_export_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
