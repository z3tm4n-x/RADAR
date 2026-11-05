"""Fixed SRIM controls for literal positive OST HZE source parameters.

The manifest and its six source dependencies are checked before and after roster
reconstruction. A source A is a numerical experiment input, not an isotope claim.
These controls cover an observed total-energy interval, not a source spectrum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
from importlib import resources
import json
from math import isfinite
from pathlib import Path
from typing import Any

from radar.data.integrity import validate_normative_resource_integrity
from radar.gcr.parameters import (
    get_ost_akl_parameters, get_ost_gcr_ion_parameters, get_ost_nek_parameter,
)
from radar.sep.hze_spectrum import OST_SEP_HZE_B9_RESOURCE, load_ost_sep_hze_ion_records


MANIFEST_PATH = "outputs/stopping_mass_manifest_20261031.json"
MANIFEST_SHA256 = "f367a1b5f047451e40ea37d4cb03139ff6cbdb7c7eb9333888b44c11d46250fc"
SOURCE_HASHES = {
    "src/radar/gcr/parameters.py": "ff623c9b8276d881b8f1deb319c9583152050719f7c90af1b451359fc28884c7",
    "src/radar/sep/hze_spectrum.py": "1dcb09ee7e2d37377e12241ec6607701e3e38b2de804d1eb8a2751a80c6b84cc",
    "src/radar/data/normative/sep_hze/ost_134_1044_2007_b9.csv": "d15632e66fb0daff36915c023a9454025ac15a2ffe1846685b8be514c0022e5f",
    "src/radar/data/integrity.py": "948de704e35149a4e55705a13b827f02ec2947482fcdc76660d966d61a8a994e",
    "src/radar/validation/ion_mass_contract.py": "698ef44f4ba08c2e866a442a349acb5586555799fc8452b48a4440b218d4fe60",
    "scripts/build_stopping_mass_manifest.py": "2e48fa2632f0cf825382838e311969a2e979f695732f6ac6a4ce109d8ef4352f",
}
_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_TARGETS = {13: ("al", Decimal("26.981539"), Decimal("2.70")),
            14: ("si", Decimal("28.0855"), Decimal("2.3290"))}
_PILOT_Z = (9, 10, 26, 92)


def _number(value: float, label: str, *, positive: bool = False) -> Decimal:
    if (type(value) not in (int, float) or not isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"Invalid literal source parameter {label}.")
    return Decimal(str(value))


def _gcr_parameters() -> dict[int, tuple[str, Decimal, Decimal, tuple[str, ...]]]:
    result = {}
    akl = get_ost_akl_parameters()
    for ion in get_ost_gcr_ion_parameters().values():
        if ion.z == 1:
            continue
        if type(ion.z) is not int or not 2 <= ion.z <= 92 or ion.z in result:
            raise ValueError("Duplicate or invalid GCR source charge.")
        mass = _number(ion.mass_number, "A", positive=True)
        if mass < ion.z:
            raise ValueError("Source mass parameter must be at least Z.")
        weight = _number(ion.c, "C")
        nek = get_ost_nek_parameter(ion.z)
        components = tuple(name for name, coefficient in (
            ("main", weight),
            ("NEK", Decimal(0) if nek is None else _number(nek.ac, "NEK ac")),
            ("AKL", Decimal(0) if ion.symbol not in akl else _number(akl[ion.symbol].c, "AKL c")),
        ) if coefficient > 0)
        result[ion.z] = (ion.symbol, mass, weight, components)
    if set(result) != set(range(2, 93)) or sum(bool(r[3]) for r in result.values()) != 84:
        raise ValueError("Complete fixed OST GCR roster must contain 84 positive HZE species.")
    return result


def _source_inventory() -> dict[str, Any]:
    """Independently reconstruct all source fields without selecting isotopes."""
    gcr = _gcr_parameters()
    rows = [{"source": "GCR", "z": z, "symbol": symbol,
             "source_tabulated_mass_parameter": mass, "source_mass_to_charge": None,
             "source_weight_parameter_name": "main_component_C", "source_weight_parameter": weight,
             "nonzero_source_components": list(components)}
            for z, (symbol, mass, weight, components) in sorted(gcr.items())]
    validate_normative_resource_integrity(OST_SEP_HZE_B9_RESOURCE)
    resource = resources.files("radar.data.normative").joinpath(OST_SEP_HZE_B9_RESOURCE)
    with resources.as_file(resource) as path:
        sep = load_ost_sep_hze_ion_records(path)
    seen: set[int] = set()
    positive_sep: set[int] = set()
    for ion in sorted(sep, key=lambda ion: ion.z):
        if type(ion.z) is not int or ion.z in seen or ion.z not in range(2, 29):
            raise ValueError("Duplicate or invalid SEP source charge.")
        seen.add(ion.z)
        mass = _number(ion.mass_number, "SEP A", positive=True)
        weight = _number(ion.relative_abundance, "SEP K")
        if (ion.symbol, mass) != gcr[ion.z][:2]:
            raise ValueError("GCR/SEP literal source masses or symbols disagree.")
        if weight > 0:
            if not gcr[ion.z][3]:
                raise ValueError("Positive SEP species absent from positive GCR roster.")
            positive_sep.add(ion.z)
        rows.append({"source": "SEP", "z": ion.z, "symbol": ion.symbol,
                     "source_tabulated_mass_parameter": mass,
                     "source_mass_to_charge": _number(ion.mass_to_charge, "SEP A/Q", positive=True),
                     "source_weight_parameter_name": "relative_abundance_K",
                     "source_weight_parameter": weight,
                     "nonzero_source_components": ["SEP"] if weight > 0 else []})
    if seen != set(range(2, 29)) or len(positive_sep) != 24:
        raise ValueError("Complete fixed OST SEP roster must contain 24 positive HZE species.")
    return {
        "projectiles": [{"z": z, "symbol": symbol, "mass_parameter_u": mass,
                         "source_membership": ["GCR", "SEP"] if z in positive_sep else ["GCR"]}
                        for z, (symbol, mass, _, components) in sorted(gcr.items()) if components],
        "source_rows": rows,
        "declared_zero_sources": [row.copy() for row in rows if not row["nonzero_source_components"]],
        "source_counts": {
            "GCR": {"hze_table_rows": 91, "positive_hze": 84, "declared_zero": 7},
            "SEP": {"hze_table_rows": 27, "positive_hze": 24, "declared_zero": 3},
        },
    }


def _name(symbol: str, mass: Decimal, target_z: int) -> str:
    text = format(mass, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{symbol.lower()}{text.replace('.', 'p')}_{_TARGETS[target_z][0]}"


@dataclass(frozen=True)
class SourceControl:
    name: str
    ion_z: int
    symbol: str
    ion_mass_u: Decimal
    target_z: int
    target_mass_u: Decimal
    density_g_cm3: Decimal
    energy_min_mev: Decimal = Decimal(1)
    energy_max_mev: Decimal = Decimal(12000)
    mass_policy: str = field(default="literal_OST_source_parameter_diagnostic_not_isotope", init=False)
    energy_basis: str = field(default="total_ion_kinetic_energy_MeV", init=False)
    source_energy_domain_covered: bool = field(default=False, init=False)
    physical_mass_policy_qualified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _check_control(self)


def _check_control(control: SourceControl) -> None:
    """Fixed inputs only; pin validation remains build_source_catalog's duty."""
    if (type(control) is not SourceControl or type(control.ion_z) is not int
            or type(control.target_z) is not int or control.target_z not in _TARGETS
            or any(type(v) is not Decimal or not v.is_finite() for v in (
                control.ion_mass_u, control.target_mass_u, control.density_g_cm3,
                control.energy_min_mev, control.energy_max_mev,
            ))):
        raise ValueError("Only typed fixed source controls are allowed.")
    sources = _gcr_parameters()
    source = sources.get(control.ion_z)
    if source is None or not source[3] or (control.symbol, control.ion_mass_u) != source[:2]:
        raise ValueError("Control must use an exact positive OST source mass parameter.")
    _, target_mass, density = _TARGETS[control.target_z]
    if (control.target_mass_u != target_mass or control.density_g_cm3 != density
            or control.energy_min_mev != 1 or control.energy_max_mev != 12000
            or control.name != _name(control.symbol, control.ion_mass_u, control.target_z)):
        raise ValueError("Control differs from fixed material, name or total-energy limits.")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_catalog(root: Path) -> tuple[SourceControl, ...]:
    """Verify the pinned source inventory and produce exactly 168 native cases."""
    path = root / MANIFEST_PATH
    if _sha(path) != MANIFEST_SHA256:
        raise ValueError("Pinned source-mass manifest changed.")
    manifest = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    if (manifest["schema_version"] != "radar.stopping_mass_manifest.v1"
            or manifest["source_files_sha256_before"] != SOURCE_HASHES
            or manifest["source_files_sha256_after"] != SOURCE_HASHES
            or manifest["source_roster_inventory_complete"] is not True
            or manifest["mass_parameter_policy"] != (
                "literal_OST_source_parameter_used_as_diagnostic_projectile_mass")
            or any(manifest[key] is not False for key in (
                "normative_export_allowed", "physical_mass_policy_qualified",
                "transport_tables_generated", "production_tables_replaced",
            ))):
        raise ValueError("Source manifest schema or six-file pin inventory differs.")
    for relative, expected in SOURCE_HASHES.items():
        if _sha(root / relative) != expected:
            raise ValueError(f"Pinned source file changed: {relative}")
    inventory = _source_inventory()
    if any(manifest[key] != value for key, value in inventory.items()):
        raise ValueError("Current source roster differs from the complete pinned manifest.")
    catalog = tuple(SourceControl(
        _name(row["symbol"], row["mass_parameter_u"], target), row["z"], row["symbol"],
        row["mass_parameter_u"], target, mass, density,
    ) for row in inventory["projectiles"] for target, (_, mass, density) in _TARGETS.items())
    if len(catalog) != 168 or len({c.name for c in catalog}) != 168:
        raise ValueError("Source control catalog must contain 168 unique cases.")
    for relative, expected in {MANIFEST_PATH: MANIFEST_SHA256, **SOURCE_HASHES}.items():
        if _sha(root / relative) != expected:
            raise ValueError("Source or manifest changed during catalog construction.")
    return catalog


def pilot_controls(catalog: tuple[SourceControl, ...]) -> tuple[SourceControl, ...]:
    """Fixed fractional/new-species pilot: F, Ne, Fe and U, both materials."""
    if type(catalog) is not tuple or len(catalog) != 168:
        raise ValueError("Pilot requires the complete fixed catalog.")
    for control in catalog:
        _check_control(control)
    if len({(c.ion_z, c.target_z) for c in catalog}) != 168:
        raise ValueError("Duplicate source control in catalog.")
    selected = tuple(c for c in catalog if c.ion_z in _PILOT_Z)
    return tuple(sorted(selected, key=lambda c: (c.ion_z, c.target_z)))


def expected_control_grid(control: SourceControl) -> tuple[Decimal, ...]:
    _check_control(control)
    with localcontext(_CONTEXT):
        mantissas = tuple(Decimal(v) for v in (
            "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2",
            "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4", "4.5", "5",
            "5.5", "6", "6.5", "7", "8", "9",
        ))
        return tuple(e for exponent in range(5) for v in mantissas
                     if control.energy_min_mev <= (e := v * Decimal(10)**exponent)
                     <= control.energy_max_mev)


def input_bytes(control: SourceControl) -> bytes:
    _check_control(control)
    with localcontext(_CONTEXT):
        target = "Aluminum" if control.target_z == 13 else "Silicon"
        lines = (
            "---Stopping/Range Input Data (Number-format: Period = Decimal Point)",
            "---Output File Name", '"srim_output.txt"', "---Ion(Z), Ion Mass(u)",
            f"{control.ion_z}   {control.ion_mass_u}",
            "---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.",
            f"0    {control.density_g_cm3}    1", "---Number of Target Elements", "1",
            "---Target Elements: (Z), Target name, Stoich, Target Mass(u)",
            f'{control.target_z}   "{target}"   1   {control.target_mass_u}',
            "---Output Stopping Units (1-8)", "5",
            "---Ion Energy : E-Min(keV), E-Max(keV)",
            f"{control.energy_min_mev * 1000}    {control.energy_max_mev * 1000}",
        )
    return ("\r\n".join(lines) + "\r\n").encode("ascii")
