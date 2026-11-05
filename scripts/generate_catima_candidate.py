"""Generate an isolated, non-production CATIMA stopping/range candidate.

Run with CPython 3.12 and the pinned, unpacked wheel. No RADAR or downloaded
build scripts are imported. Nuclear stopping is a mathematical replay of the
pinned CATIMA v1.91 src/calculations.cpp (dedx_n); it is independently checked
against the exposed low-energy electronic term, not presented as a new model.
Source/derived-data redistribution and normative qualification remain open.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/ion_stopping_20261031"
WHEEL = "pycatima-1.982-cp312-cp312-win_amd64.whl"
PINS = {
    WHEEL: "24dab5c194e383bdd68df1d6ffbe8e768703169e0178c946232cee239bf1af9c",
    "pycatima-1.982.tar.gz":
        "79f239f8c7425b20fcf19dd756b9cf39b24d970423c2c03da7b43b1ea3395971",
    "catima-v1.91.zip":
        "c2eabcc1b83d121e82d0cb9a71ed2731198c4e2477492be804fd32bc8e85db4e",
    "source/catima-1.91/src/calculations.cpp":
        "ff08e35799c06cc7b5d0a6c95a73171482c0aac3be6120200ae059f35a3ecbe5",
    "source/catima-1.91/src/catima.cpp":
        "2ef012e5d38a1f7da1fc5e4c04effda1eaa16d0bafdeedc45b33254efec7bcf7",
    "source/catima-1.91/include/catima/config.h":
        "1eeb808d3902368086021366132a9188a51f15c8271dfe9138bf072ef75c6789",
}
# Explicit diagnostic isotope choices matching the legacy transport set.
# These are NOT inferred by rounding the spectral model's effective masses.
IONS = (
    (2, 4, "He"), (6, 12, "C"), (7, 14, "N"), (8, 16, "O"),
    (10, 20, "Ne"), (12, 24, "Mg"), (14, 28, "Si"), (16, 32, "S"),
    (18, 40, "Ar"), (20, 40, "Ca"), (24, 52, "Cr"), (25, 55, "Mn"),
    (26, 56, "Fe"), (27, 59, "Co"), (28, 58, "Ni"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_hashes() -> dict[str, str]:
    """Pin all bundled normative resources and production Python, not just HZE."""
    files = set((ROOT / "src/radar").rglob("*.py"))
    files.update(p for p in (ROOT / "src/radar/data/normative").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts)
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def energy_grid(points_per_decade: int) -> tuple[float, ...]:
    if type(points_per_decade) is not int or points_per_decade not in (50, 100, 200, 400):
        raise ValueError("Use 50, 100, 200 or 400 points per decade.")
    # Explicit transition nodes. No extrapolation below 0.01 or above 1e5 MeV/u.
    return tuple(sorted({10.0 ** (-2 + i / points_per_decade)
                         for i in range(7 * points_per_decade + 1)}
                        | {10.0, 30.0, 50000.0, 100000.0}))


def nuclear_stopping(z: int, a: float, target_z: int, target_a: float, e: float) -> float:
    """Replay the source's elemental nuclear term, in whole-ion MeV cm2/g.

    e is MeV/u; A*e*1000 is total projectile kinetic energy in keV.
    6.022140857 is the pinned library's Avogadro constant scaled by 1e23.
    This is not an independently calibrated replacement for CATIMA.
    """
    for value in (z, a, target_z, target_a, e):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Nuclear stopping inputs must be positive finite numbers.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Nuclear stopping inputs must be positive finite numbers.")
    screening = z**0.23 + target_z**0.23
    mass_sum = a + target_a
    reduced_energy = 32530.0 * target_a * e * a / (z * target_z * mass_sum * screening)
    if reduced_energy <= 30.0:
        reduced_stopping = math.log1p(1.1383 * reduced_energy) / (
            2.0 * (reduced_energy + 0.01321 * reduced_energy**0.21226
                   + 0.19593 * math.sqrt(reduced_energy)))
    else:
        reduced_stopping = math.log(reduced_energy) / (2.0 * reduced_energy)
    return float(846.21 * z * target_z * a * 6.022140857 * reduced_stopping / (
        mass_sum * screening * target_a))


def verify_sources() -> dict[str, str]:
    hashes = {name: sha256(SOURCE / name) for name in PINS}
    if hashes != PINS:
        raise ValueError("Pinned source/package hash mismatch.")
    # Verify every unpacked member before loading any Python/native code.
    runtime = (SOURCE / "runtime").resolve()
    if any(p.name == "__pycache__" or p.suffix == ".pyc" for p in runtime.rglob("*")):
        raise ValueError("Bytecode caches are not allowed in the pinned backend runtime.")
    with zipfile.ZipFile(SOURCE / WHEEL) as archive:
        expected_members = set()
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (runtime / member.filename).resolve()
            if not target.is_relative_to(runtime):
                raise ValueError("Unsafe wheel member path.")
            expected_members.add(target)
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            if sha256(target) != expected:
                raise ValueError(f"Unpacked wheel mismatch: {member.filename}")
    actual_members = {p.resolve() for p in runtime.rglob("*") if p.is_file()}
    if actual_members != expected_members:
        raise ValueError("Unexpected files in the isolated backend runtime.")
    return hashes


def load_backend() -> Any:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The retained native wheel requires CPython 3.12.")
    if any(name == "pycatima" or name.startswith("pycatima.") for name in sys.modules):
        raise ValueError("Refuse a previously imported, unverified backend.")
    verify_sources()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE / "runtime"))
    backend = importlib.import_module("pycatima")
    if Path(backend.__file__).resolve() != (SOURCE / "runtime/pycatima/__init__.py").resolve():
        raise ValueError("Unexpected backend import path.")
    extension = importlib.import_module("pycatima._ext")
    expected_extension = SOURCE / "runtime/pycatima/_ext.cp312-win_amd64.pyd"
    if Path(extension.__file__).resolve() != expected_extension.resolve():
        raise ValueError("Unexpected native backend import path.")
    return backend


def configuration(backend: Any, *, no_radiative: bool = False) -> Any:
    config = backend.Config()
    # Explicitly use the available Pierce-Blann charge model. GLOBAL is OFF in
    # the retained binary; do not label this as ATIMA1.41's charge prescription.
    config.z_effective = 1
    config.corrections = 8 if no_radiative else 0
    config.calculation = 1
    config.low_energy = 0
    config.scattering = 255
    config.scattering_factor = 1.0
    return config


def generate(backend: Any, points_per_decade: int) -> dict[str, Any]:
    energies = energy_grid(points_per_decade)
    config = configuration(backend)
    no_radiative = configuration(backend, no_radiative=True)
    rows = []
    materials = []
    closure_errors = []
    derivative_errors = []
    for material_z in (13, 14):
        material = backend.get_material(material_z)
        target = material.get_element(0)
        if material.ncomponents() != 1 or target.Z != material_z or target.stn != 1:
            raise ValueError("Expected an elemental target.")
        materials.append({"z": material_z, "atomic_mass_u": target.A,
                          "density_g_cm3": material.density(),
                          "ionization_potential_override_ev": material.I(),
                          "ionization_potential_policy": "pinned_internal_element_table"})
        for z, a, _symbol in IONS:
            projectile = backend.Projectile(a, z, z)
            for e in energies:
                projectile.T(e)
                total = backend.dedx(projectile, material, config)
                nonradiative = backend.dedx(projectile, material, no_radiative)
                nuclear = nuclear_stopping(z, a, material_z, target.A, e)
                electronic = nonradiative - nuclear
                radiative = total - nonradiative
                mass_range = backend.range(projectile, material, config)
                values = (total, electronic, nuclear, radiative, mass_range)
                if any(not math.isfinite(v) for v in values) or min(values) < 0:
                    raise ValueError("Invalid backend stopping/range; no clipping permitted.")
                if min(total, electronic, mass_range) <= 0:
                    raise ValueError("Positive stopping and range required.")
                if e <= 10:
                    independent_electronic = backend.sezi_dedx_e(projectile, material, config)
                    if not math.isfinite(independent_electronic) or independent_electronic <= 0:
                        raise ValueError("Invalid independent electronic stopping check.")
                    closure_errors.append(abs(electronic / independent_electronic - 1))
                derivative_stopping = backend.dedx_from_range(projectile, material, config)
                if not math.isfinite(derivative_stopping) or derivative_stopping <= 0:
                    raise ValueError("Invalid range derivative stopping check.")
                derivative_errors.append({
                    "z": z, "mass_number": a, "material_z": material_z, "energy": e,
                    "relative_error": abs(derivative_stopping / total - 1),
                })
                rows.append({
                    "z": z, "mass_number": a, "material_z": material_z,
                    "energy_mev_per_nucleon": e,
                    "stopping_electronic_mev_cm2_g": electronic,
                    "stopping_nuclear_mev_cm2_g": nuclear,
                    "stopping_radiative_mev_cm2_g": radiative,
                    "range_g_cm2": mass_range,
                })
    closure_max = max(closure_errors)
    if closure_max > 1e-10:
        raise ValueError("Nuclear replay failed exposed low-energy electronic closure.")
    return {
        "schema_version": "radar.ion_stopping_candidate.raw.v1",
        "ions": [{"z": z, "mass_number": a, "symbol": symbol} for z, a, symbol in IONS],
        "material_atomic_numbers": [13, 14], "materials": materials,
        "energies_mev_per_nucleon": energies, "rows": rows,
        "generation_checks": {
            "low_energy_electronic_closure_count": len(closure_errors),
            "low_energy_electronic_closure_max_relative_error": closure_max,
            "range_derivative_worst": max(derivative_errors, key=lambda r: r["relative_error"]),
            "range_derivative_checks": len(derivative_errors),
            "range_origin_energy_mev_per_nucleon": 0.001,
            "absolute_range_from_zero_verified": False,
            "normative_export_allowed": False,
            "all_required_gcr_ions_covered": False,
            "all_required_sep_ions_covered": False,
            "source_model_mass_policy": "not_used_to_infer_transport_isotopes",
            "electronic_let_definition_normatively_qualified": False,
            "higher_energy_radiative_component_separated": True,
        },
        "backend_configuration": config.get(),
        "nonradiative_configuration": no_radiative.get(),
        "backend_info_literal": backend.catima_info(),
        "points_per_decade": points_per_decade,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-decade", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    energy_grid(args.points_per_decade)
    pinned_before = verify_sources()
    resources_before = production_hashes()
    runner_before = sha256(Path(__file__))
    report = generate(load_backend(), args.points_per_decade)
    if verify_sources() != pinned_before or production_hashes() != resources_before:
        raise ValueError("Source or RADAR resources changed during generation.")
    if sha256(Path(__file__)) != runner_before:
        raise ValueError("Generator changed during execution.")
    report["production_resources_sha256"] = resources_before
    report["source_provenance"] = {
        "package": "pycatima 1.982", "catima_source_tag": "v1.91",
        "package_publishing_commit": "17c915cd38da58300e52ab5facb2727629fc55db",
        "package_url": "https://pypi.org/project/pycatima/1.982/",
        "source_url": "https://github.com/hrosiak/catima/tree/v1.91",
        "source_root": "research/ion_stopping_20261031",
        "hashes": pinned_before, "generator_sha256": runner_before,
        "python": sys.version, "wheel_locally_rebuilt": False,
        "source_license": "AGPL-3.0; redistribution assessment not completed",
        "same_model_as_gsi_atima_1_41": False,
        "normative_export_allowed": False,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(report["rows"]), "output": str(args.output),
                      "checks": report["generation_checks"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
