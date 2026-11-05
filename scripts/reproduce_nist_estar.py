"""Build unchanged NIST STAR 1.2 ESTAR and verify the bundled electron/Al table.

Runs on the host of the supplied GNU Fortran compiler (Linux/WSL is supported).
No package installation, network access, production writes or automatic status
promotion is performed. Each invocation requires a new output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.validation.nist_estar import compare_electron_al_csv, parse_estar_output  # noqa: E402


ARCHIVE_SHA256 = "b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6"
ARCHIVE_BYTES = 444183
SOURCE_HASHES = {
    "ESTAR.f": "ed2a9f66fb4ba904490da4c6fcf297053bc28fbb07d910f8f8961b6781ef907e",
    "EDCONV.f": "06354bed87e1d2564db73b66f47884c48e728e726e1fc1a96d78c3159ecc946b",
    "CONVERT.f": "35509e220d1d08ac5ca83ef09ddaafcc01c68c9372e929546b68bd0103ef001a",
    "FCOMP": "7303575f8923254323990bda5d5d1f46b8a111918f9600509e023dd1cbce85fc",
    "FEDAT": "6386e421808f6438e4a5de9be31a2ffeb7bef5c27bbac34395a8985b3287383c",
    "ENG.ELE": "06f8c6d355666724cfe9afd473ac041d97ea3d231b79c0217ed11d0e8148ac4f",
}
STDIN = "1\n1\n13\n1\nal.out\n"
IMPLEMENTATION_PATHS = (
    "scripts/reproduce_nist_estar.py", "src/radar/validation/nist_estar.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_members(archive: bytes) -> dict[str, bytes]:
    if len(archive) != ARCHIVE_BYTES or _sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Official NIST STAR 1.2 archive identity mismatch.")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = bundle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names are not allowed.")
        result = {}
        for name, expected in SOURCE_HASHES.items():
            info = bundle.getmember("ESTAR/" + name)
            if not info.isfile():
                raise ValueError("Expected a regular ESTAR source member.")
            stream = bundle.extractfile(info)
            if stream is None:
                raise ValueError("Missing ESTAR source member.")
            data = stream.read()
            if _sha(data) != expected:
                raise ValueError(f"Official archive source member changed: {name}.")
            result[name] = data
    return result


def _json(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, default=encode) + "\n"


def _write_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": _sha(data), "bytes": len(data)}


def run_build(
    *, archive: Path, production_csv: Path, compiler: Path, output: Path,
    compiler_prefixes: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_bytes = archive.read_bytes()
    members = archive_members(source_bytes)
    csv_bytes = production_csv.read_bytes()
    compiler = compiler.resolve(strict=True)
    if not compiler.is_file():
        raise ValueError("A regular Fortran compiler file is required.")
    prefixes = tuple(p.resolve(strict=True) for p in compiler_prefixes)
    if any(not p.is_dir() for p in prefixes):
        raise ValueError("Compiler prefixes must be existing directories.")
    pins = {p: _sha((ROOT / p).read_bytes()) for p in IMPLEMENTATION_PATHS}
    compiler_before = _identity(compiler)
    output.mkdir(parents=False)
    output = output.resolve()
    commands: list[dict[str, Any]] = []

    def run(argv: list[str], cwd: Path, stdin: str = "") -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(argv, cwd=cwd, input=stdin, text=True,
                                   capture_output=True, timeout=120, check=False)
        record = {"argv": argv, "cwd": str(cwd), "stdin": stdin,
                  "returncode": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr}
        commands.append(record)
        _write_new(output / f"command_{len(commands):02d}.json", _json(record))
        completed.check_returncode()
        return completed

    version = run([str(compiler), "--version"], output).stdout
    if "GNU Fortran" not in version:
        raise ValueError("This replay requires explicit GNU Fortran semantics.")
    prefix_flags = ["-B" + str(p) + "/" for p in prefixes]
    # Capture the actual compiler pass, assembler, linker and Fortran archive.
    dependencies: dict[str, dict[str, Any]] = {}
    for query in ("-print-prog-name=f951", "-print-prog-name=as", "-print-prog-name=ld",
                  "-print-file-name=libgfortran.a", "-print-libgcc-file-name"):
        name = run([str(compiler), *prefix_flags, query], output).stdout.strip()
        resolved = Path(name) if Path(name).is_absolute() else Path(shutil.which(name) or name)
        if not resolved.is_file():
            raise ValueError(f"Unable to identify compiler dependency: {query} = {name}")
        dependencies[query] = _identity(resolved)
    builds: list[dict[str, Any]] = []
    for level in ("-O0", "-O2"):
        work = output / level[1:]
        work.mkdir()
        for name, data in members.items():
            with (work / name).open("xb") as stream:
                stream.write(data)
        flags = ["-std=legacy", level, "-ffixed-line-length-72", "-static-libgfortran", *prefix_flags]
        executable_suffix = ".exe" if sys.platform == "win32" else ""
        for name in ("EDCONV", "CONVERT", "ESTAR"):
            run([str(compiler), *flags, name + ".f", "-o", name.lower() + executable_suffix], work)
        for name in ("edconv", "convert"):
            run([str(work / (name + executable_suffix))], work)
        for name, expected_size in (("UEDAT", 122400), ("UCOMP", 74772)):
            if (work / name).stat().st_size != expected_size:
                raise ValueError(f"Unexpected native direct-access record layout: {name}.")
        run([str(work / ("estar" + executable_suffix))], work, STDIN)
        native_bytes = (work / "al.out").read_bytes()
        parsed = parse_estar_output(native_bytes)
        comparison = compare_electron_al_csv(parsed, csv_bytes)
        # Native output is preserved regardless of whether it matches production.
        builds.append({
            "optimization": level, "flags": flags,
            "native_output": _identity(work / "al.out"),
            "parsed_output": asdict(parsed), "comparison": asdict(comparison),
            "generated_files": {p.name: _identity(p) for p in work.iterdir()
                                if p.is_file() and p.name not in members},
        })
        for name, expected in SOURCE_HASHES.items():
            if _sha((work / name).read_bytes()) != expected:
                raise ValueError("A native source file changed during execution.")
        print(f"{level}: {len(parsed.rows)} ESTAR rows; exact primary match={comparison.exact_primary_match}",
              flush=True)
    after = {p: _sha((ROOT / p).read_bytes()) for p in IMPLEMENTATION_PATHS}
    if pins != after or compiler_before != _identity(compiler):
        raise ValueError("Compiler or replay implementation changed during execution.")
    for evidence in dependencies.values():
        if evidence != _identity(Path(evidence["path"])):
            raise ValueError("Compiler dependency changed during execution.")
    if archive.read_bytes() != source_bytes or production_csv.read_bytes() != csv_bytes:
        raise ValueError("Source archive or production CSV changed during execution.")
    report = {
        "schema_version": "radar.nist_estar_native_reproduction.v1",
        "upstream": "https://physics.nist.gov/PhysRefData/Star/STAR.tar.gz",
        "redistribution_notice": "https://physics.nist.gov/PhysRefData/Star/Text/download.html",
        "source_archive": _identity(archive), "source_members_sha256": SOURCE_HASHES,
        "production_csv": _identity(production_csv), "compiler": compiler_before,
        "compiler_version": version, "compiler_dependencies": dependencies,
        "implementation_sha256_before": pins, "implementation_sha256_after": after,
        "stdin": STDIN, "builds": builds,
        "native_O0_O2_output_bytes_identical":
            builds[0]["native_output"]["sha256"] == builds[1]["native_output"]["sha256"],
        "all_primary_values_exact": all(b["comparison"]["exact_primary_match"] for b in builds),
        "all_csv_values_exact": all(
            b["comparison"]["exact_primary_match"]
            and b["comparison"]["exact_range_mg_match"]
            and b["comparison"]["range_mg_internal_consistent"] for b in builds
        ),
        "production_data_modified": False, "release_status_changed_automatically": False,
        "production_data_check_scope": "The selected electron Al CSV only, not the entire worktree.",
        "stage4_complete": False,
        "limitations": [
            "Reproduces NIST STAR 1.2, not a claim of identity to every online ESTAR revision.",
            "Source and compiler warnings are unchanged and retained; no bounds-clean claim.",
            "No fast-math, type promotion, material adjustment or physics correction is used.",
            "Internal ESTAR range quadrature is preserved, not reintegrated from printed rows.",
            "Table provenance is separate from full spectral transport/physical qualification.",
            "Selected compiler dependencies are pinned; this is not a hermetic toolchain build.",
            "Environment, startup objects and shared/system libraries are not comprehensively pinned.",
        ],
    }
    _write_new(output / "report.json", _json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "research/nist_star_20261031/STAR.tar.gz")
    parser.add_argument("--production-csv", type=Path,
                        default=ROOT / "src/radar/data/normative/stopping/electron_al_range.csv")
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-prefix", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_build(archive=args.archive, production_csv=args.production_csv,
                       compiler=args.compiler, compiler_prefixes=tuple(args.compiler_prefix),
                       output=args.output)
    print(_json({"output": str(args.output / "report.json"),
                 "all_primary_values_exact": report["all_primary_values_exact"],
                 "all_csv_values_exact": report["all_csv_values_exact"],
                 "native_O0_O2_output_bytes_identical": report["native_O0_O2_output_bytes_identical"]}))
    return 0 if report["all_csv_values_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
