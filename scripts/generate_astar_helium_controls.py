"""Sample pinned CATIMA He4 exactly at ASTAR archive nodes; no table replacement.

Run with CPython 3.12 -B. The ASTAR source table supplies only the energy grid;
no Fortran executable or ASTAR interpolation is replayed. These native samples
separate candidate output-table interpolation from inter-model differences.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import generate_catima_candidate as native

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from radar.validation.nist_astar import load_astar_archive  # noqa: E402


_HEADER_PATH = (
    "research/ion_stopping_20261031/runtime/include/catima/data_ionisation_potential.h"
)
_IMPLEMENTATION_PATHS = (
    "scripts/generate_astar_helium_controls.py",
    "scripts/generate_catima_candidate.py",
    "src/radar/validation/nist_astar.py",
    _HEADER_PATH,
)
_CONFIGURATION = {
    "z_effective": 1, "corrections": 0, "calculation": 1, "low_energy": 0,
    "scattering": 255, "scattering_factor": 1.0,
}


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number, not bool.")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    return float(value)


def _implementation_hashes() -> dict[str, str]:
    return {p: native.sha256(ROOT / p) for p in _IMPLEMENTATION_PATHS}


def _elemental_i_values() -> dict[int, float]:
    """Read the wheel-verified header; never override backend material I."""
    text = (ROOT / _HEADER_PATH).read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    match = re.search(r"ionisation_potentials_z\[[^]]+\]\s*=\s*\{([^}]+)\}", text)
    if match is None:
        raise ValueError("Pinned elemental ionization-potential array is missing.")
    values = [float(token.strip()) for token in match.group(1).split(",") if token.strip()]
    if len(values) != 121:
        raise ValueError("Unexpected elemental ionization-potential array length.")
    return {z: _number(values[z], "elemental I", positive=True) for z in (13, 14)}


def _configuration_values(config: Any, *, nonradiative: bool) -> dict[str, Any]:
    values = dict(config.get())
    expected = {**_CONFIGURATION, "corrections": 8 if nonradiative else 0}
    if set(values) != set(expected):
        raise ValueError("Unexpected backend configuration keys.")
    for key, value in expected.items():
        if key == "scattering_factor":
            if _number(values[key], key, positive=True) != value:
                raise ValueError("Unexpected backend configuration.")
        elif type(values[key]) is not int or values[key] != value:
            raise ValueError("Unexpected backend configuration.")
    return values


def generate_samples(
    backend: Any, energies_mev_total: tuple[float, ...], *, elemental_i_ev: dict[int, float],
) -> dict[str, Any]:
    """Whole-ion stopping and relative CSDA range at explicit E_total/4 nodes."""
    if not isinstance(energies_mev_total, tuple) or not energies_mev_total:
        raise ValueError("An increasing immutable total-energy grid is required.")
    for value in energies_mev_total:
        value = _number(value, "total energy", positive=True)
        if not 0.04 <= value <= 1000.0:
            raise ValueError("Only the retained candidate/ASTAR overlap .04..1000 MeV is allowed.")
    if any(a >= b for a, b in zip(energies_mev_total, energies_mev_total[1:])):
        raise ValueError("An increasing immutable total-energy grid is required.")
    if set(elemental_i_ev) != {13, 14}:
        raise ValueError("Both elemental Al/Si ionization potentials are required.")
    for value in elemental_i_ev.values():
        _number(value, "elemental I", positive=True)
    config = native.configuration(backend)
    nonradiative = native.configuration(backend, no_radiative=True)
    configuration_before = _configuration_values(config, nonradiative=False)
    nonradiative_before = _configuration_values(nonradiative, nonradiative=True)
    rows: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    closure_count = 0
    closure_max = 0.0
    for material_z in (13, 14):
        material = backend.get_material(material_z)
        target = material.get_element(0)
        if (material.ncomponents() != 1 or target.Z != material_z or target.stn != 1
                or isinstance(target.Z, bool) or isinstance(target.stn, bool)):
            raise ValueError("Expected the unchanged elemental Al/Si material.")
        target_mass = _number(target.A, "target mass", positive=True)
        density = _number(material.density(), "material density", positive=True)
        override = _number(material.I(), "material I override")
        if override != 0:
            raise ValueError("Material I override is not permitted in this matched-control run.")
        materials.append({
            "z": material_z, "atomic_mass_u": target_mass, "density_g_cm3": density,
            "ionization_potential_override_ev": override,
            "elemental_ionization_potential_ev": float(elemental_i_ev[material_z]),
            "ionization_potential_policy": "unchanged_backend_internal_elemental_table",
        })
        projectile = backend.Projectile(4.0, 2, 2)
        previous_range = -1.0
        for total_energy in energies_mev_total:
            energy = total_energy / 4.0
            projectile.T(energy)
            total = _number(backend.dedx(projectile, material, config), "total stopping", positive=True)
            nonrad = _number(backend.dedx(projectile, material, nonradiative),
                             "nonradiative stopping", positive=True)
            nuclear = _number(native.nuclear_stopping(2, 4.0, material_z, target_mass, energy),
                              "nuclear stopping")
            electronic = _number(nonrad - nuclear, "electronic stopping", positive=True)
            radiative = _number(total - nonrad, "radiative stopping")
            if radiative != 0:
                raise ValueError("Radiative term must be zero below the pinned 50000-MeV/u threshold.")
            mass_range = _number(backend.range(projectile, material, config),
                                 "relative range", positive=True)
            if mass_range <= previous_range:
                raise ValueError("Native relative range must increase on the declared grid.")
            previous_range = mass_range
            if energy <= 10.0:
                check = _number(backend.sezi_dedx_e(projectile, material, config),
                                "low-energy electronic closure", positive=True)
                error = abs(electronic / check - 1.0)
                closure_max = max(closure_max, error)
                closure_count += 1
                if error > 1e-10:
                    raise ValueError("Pinned nuclear subtraction failed exposed electronic closure.")
            rows.append({
                "material_z": material_z, "energy_mev_total": float(total_energy),
                "energy_mev_per_nucleon": energy,
                "stopping_electronic_mev_cm2_g": electronic,
                "stopping_nuclear_mev_cm2_g": nuclear,
                "stopping_radiative_mev_cm2_g": radiative, "range_g_cm2": mass_range,
            })
        if (target_mass != _number(material.get_element(0).A, "final target mass", positive=True)
                or material.ncomponents() != 1 or material.get_element(0).Z != material_z
                or material.get_element(0).stn != 1
                or density != _number(material.density(), "final density", positive=True)
                or override != _number(material.I(), "final material I")):
            raise ValueError("Backend material changed during matched sampling.")
    if (_configuration_values(config, nonradiative=False) != configuration_before
            or _configuration_values(nonradiative, nonradiative=True) != nonradiative_before):
        raise ValueError("Backend configuration changed during matched sampling.")
    return {
        "rows": rows, "materials": materials,
        "backend_configuration": configuration_before,
        "nonradiative_configuration": nonradiative_before,
        "low_energy_electronic_closure_count": closure_count,
        "low_energy_electronic_closure_max_relative_error": closure_max,
    }


def generate_controls(archive_path: Path) -> dict[str, Any]:
    implementation_before = _implementation_hashes()
    archive = load_astar_archive(archive_path)
    if native.sha256(archive_path) != archive.archive_sha256:
        raise ValueError("ASTAR archive changed after parsing.")
    energies = tuple(row.energy_mev_total for row in archive.material(13).rows
                     if row.energy_mev_total >= 0.04)
    silicon_energies = tuple(row.energy_mev_total for row in archive.material(14).rows
                            if row.energy_mev_total >= 0.04)
    if len(energies) != 101 or energies != silicon_energies:
        raise ValueError("Expected the same 101 native ASTAR overlap nodes in Al and Si.")
    sources_before = native.verify_sources()
    production_before = native.production_hashes()
    elemental_i = _elemental_i_values()
    backend = native.load_backend()
    samples = generate_samples(backend, energies, elemental_i_ev=elemental_i)
    sources_after = native.verify_sources()
    production_after = native.production_hashes()
    implementation_after = _implementation_hashes()
    if (sources_before != sources_after or production_before != production_after
            or implementation_before != implementation_after
            or native.sha256(archive_path) != archive.archive_sha256):
        raise ValueError("Archive, source, implementation or production inputs changed during sampling.")
    return {
        "schema_version": "radar.astar_helium_native_controls.v1",
        "archive_path": str(archive_path.resolve()), "archive_sha256": archive.archive_sha256,
        "archive_member_sha256": dict(archive.member_sha256), "archive_version": archive.version,
        "implementation_sha256_before": implementation_before,
        "implementation_sha256_after": implementation_after,
        "native_source_pins_before": sources_before, "native_source_pins_after": sources_after,
        "production_resources_sha256_before": production_before,
        "production_resources_sha256_after": production_after,
        "python": sys.version, "backend_info_literal": backend.catima_info(),
        "projectile": {"z": 2, "mass_parameter_u": 4.0, "charge_input": 2},
        "energies_mev_total": energies, "material_atomic_numbers": (13, 14),
        "range_origin_energy_mev_per_nucleon": 0.001,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "energy_conversion": "backend_energy_mev_per_u=ASTAR_whole_alpha_energy_mev/4.0",
        "native_sampling_policy": "exact_ASTAR_source_energy_nodes_not_candidate_table_interpolation",
        "normative_export_allowed": False, "physical_accuracy_verified": False,
        "physical_mass_policy_qualified": False, "production_tables_replaced": False,
        "fortran_executed": False, "absolute_range_origin_verified": False,
        "limitations": [
            "The ASTAR archive supplies energy nodes only; no Fortran or APOL interpolation is run.",
            "Mass 4.0 is the unchanged candidate parameter, not a precision alpha-particle mass.",
            "CATIMA exact-node sampling separates interpolation error, not physical model accuracy.",
            "The native range starts at 0.001 MeV/u, not a verified physical zero-energy origin.",
            "The retained wheel is source-pinned but has not been independently rebuilt locally.",
        ],
        **samples,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = generate_controls(args.archive)
    write_report(report, args.output)
    print(json.dumps({"rows": len(report["rows"]), "output_sha256": native.sha256(args.output),
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
