"""Package authenticated observed SRIM outputs for the opt-in spectral backend.

Text data only: no binaries, installer, native execution, extrapolation or
replacement of the default normative data. Historical evidence is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_srim_source_overlap import (  # noqa: E402
    EXTENSION_SHA256, SUMMARY_SHA256, reconstruct,
)
from radar.shielding.srim_dataset import POLICIES, SCHEMA, load_srim_hze_dataset  # noqa: E402


def build_bundle(output: Path) -> dict[str, str | int | bool]:
    if output.exists():
        raise FileExistsError(output)
    selected, evidence = reconstruct()
    by_sha: dict[str, list[str]] = {}
    for path, sha in evidence.pins.items():
        by_sha.setdefault(sha, []).append(path)
    members = {}
    records = []
    for name, native in sorted(selected.items(), key=lambda item: (item[1].ion_z, item[1].target_z)):
        path = Path(sorted(by_sha[native.raw_sha256])[0])
        raw = evidence.read(path, native.raw_sha256)
        member = f"native/{name}.txt"
        members[member] = raw
        records.append({"case_name": name, "z": native.ion_z,
                        "mass_parameter_u_decimal": format(native.ion_mass_u, "f"),
                        "material_z": native.target_z, "member": member, "sha256": native.raw_sha256})
    manifest = {"schema_version": SCHEMA, **POLICIES, "tables": records,
                "extension_report_sha256": EXTENSION_SHA256,
                "selection_summary_sha256": SUMMARY_SHA256,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "software_labels": sorted({(n.engine_label, n.version_label) for n in selected.values()}),
                "native_execution_performed": False,
                "binary_redistribution": False,
                "limitations": [
                    "Finite observed support only; source tails and below-anchor exits remain unresolved.",
                    "Native projected ranges are preserved as source text but not used as CSDA ranges.",
                    "An additive range anchor cancels only when both incident and exit energies are supported.",
                    "No heavy-ion nuclear attenuation or fragmentation model is added.",
                ]}
    members["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                           indent=2, allow_nan=False) + "\n").encode("utf-8")
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    evidence.verify_unchanged()
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    with output.open("xb") as target:
        target.write(payload)
    loaded = load_srim_hze_dataset(output, expected_sha256=sha)
    if len(loaded.tables_by_z) != 84:
        raise ValueError("Full observed HZE bundle must contain all 84 Al/Si pairs.")
    return {"output": str(output), "sha256": sha, "ions": len(loaded.tables_by_z),
            "native_tables": len(records), "normative_export_allowed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
