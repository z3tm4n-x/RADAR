"""Fixed normal-mode SRIM extensions below a declared diagnostic energy ceiling.

The 1000 MeV per literal source-A unit ceiling bounds this numerical experiment;
it is not an asserted physical validity limit. The original source parameters,
lower endpoint and normal grid are preserved. Existing tables whose upper
endpoint is already higher are not shortened or scheduled for another run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
from pathlib import Path

from radar.validation.srim_source_catalog import (
    SourceControl, build_source_catalog, expected_control_grid,
)


SOURCE_CATALOG_PATH = "src/radar/validation/srim_source_catalog.py"
SOURCE_CATALOG_SHA256 = "ea57ea410ce10fc9f2fc317d4ec967cfdb7531eecfcd6ada9c069c1308f1982a"
DIAGNOSTIC_CEILING_MEV_PER_SOURCE_A = Decimal(1000)
ENGINEERING_INPUT_MAX_KEV = Decimal(2147483647)
_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_PILOT_Z = (9, 10, 26, 92)


def _normal_grid(base: SourceControl) -> tuple[Decimal, ...]:
    """Extend the pinned normal-mode mantissas, not the source spectrum grid."""
    mantissas = tuple(value for value in expected_control_grid(base) if value < 10)
    with localcontext(_CONTEXT):
        return tuple(value * Decimal(10) ** exponent
                     for exponent in range(7) for value in mantissas)


def _upper_endpoint(base: SourceControl) -> Decimal:
    with localcontext(_CONTEXT):
        ceiling = base.ion_mass_u * DIAGNOSTIC_CEILING_MEV_PER_SOURCE_A
    # Every selected ceiling node is an integer MeV value; canonicalize only
    # its textual scale, never the mass parameter or a noninteger energy.
    selected = max(value for value in _normal_grid(base) if value <= ceiling)
    return Decimal(int(selected))


@dataclass(frozen=True)
class ExtensionControl:
    base_source_name: str
    name: str
    ion_z: int
    symbol: str
    ion_mass_u: Decimal
    target_z: int
    target_mass_u: Decimal
    density_g_cm3: Decimal
    energy_max_mev: Decimal
    energy_min_mev: Decimal = Decimal(1)
    baseline_energy_max_mev: Decimal = field(default=Decimal(12000), init=False)
    diagnostic_ceiling_mev_per_source_a: Decimal = field(default=Decimal(1000), init=False)
    ceiling_policy: str = field(default="bounded_numerical_experiment_not_physical_validity_limit",
                               init=False)
    mass_policy: str = field(default="literal_OST_source_parameter_diagnostic_not_isotope", init=False)
    energy_basis: str = field(default="total_ion_kinetic_energy_MeV", init=False)
    native_execution_verified: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    source_energy_domain_covered: bool = field(default=False, init=False)
    physical_mass_policy_qualified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _check_control(self)


def _check_control(control: ExtensionControl) -> SourceControl:
    """Validate only exact fixed extensions; disk pins are checked by the builder."""
    if (type(control) is not ExtensionControl or type(control.ion_z) is not int
            or type(control.target_z) is not int
            or any(type(value) is not Decimal or not value.is_finite() for value in (
                control.ion_mass_u, control.target_mass_u, control.density_g_cm3,
                control.energy_min_mev, control.energy_max_mev,
            ))):
        raise ValueError("Only typed fixed source extension controls are allowed.")
    with localcontext(_CONTEXT):
        maximum_input_mev = ENGINEERING_INPUT_MAX_KEV / Decimal(1000)
        if (control.energy_min_mev <= 0 or control.energy_max_mev <= 0
                or control.energy_min_mev > maximum_input_mev
                or control.energy_max_mev > maximum_input_mev):
            raise ValueError("Energy exceeds the declared engineering input bound; not an overflow diagnosis.")
    base = SourceControl(
        name=control.base_source_name, ion_z=control.ion_z, symbol=control.symbol,
        ion_mass_u=control.ion_mass_u, target_z=control.target_z,
        target_mass_u=control.target_mass_u, density_g_cm3=control.density_g_cm3,
    )
    maximum = _upper_endpoint(base)
    if (maximum <= base.energy_max_mev or control.energy_min_mev != base.energy_min_mev
            or control.energy_max_mev != maximum
            or control.name != f"{base.name}_upper{maximum:f}"):
        raise ValueError("Control differs from the fixed nonshrinking normal-grid extension.")
    return base


def _from_base(base: SourceControl) -> ExtensionControl:
    maximum = _upper_endpoint(base)
    return ExtensionControl(
        base_source_name=base.name, name=f"{base.name}_upper{maximum:f}",
        ion_z=base.ion_z, symbol=base.symbol, ion_mass_u=base.ion_mass_u,
        target_z=base.target_z, target_mass_u=base.target_mass_u,
        density_g_cm3=base.density_g_cm3, energy_min_mev=base.energy_min_mev,
        energy_max_mev=maximum,
    )


def _verify_catalog_implementation(root: Path) -> None:
    path = root / SOURCE_CATALOG_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_CATALOG_SHA256:
        raise ValueError("Pinned base source-catalog implementation changed.")


def build_extension_catalog(root: Path) -> tuple[ExtensionControl, ...]:
    """Pin sources before/after and select exactly 79 extending species in Al/Si."""
    _verify_catalog_implementation(root)
    baseline = build_source_catalog(root)
    selected = tuple(_from_base(base) for base in baseline
                     if _upper_endpoint(base) > base.energy_max_mev)
    if (len(selected) != 158 or len({case.ion_z for case in selected}) != 79
            or len({case.name for case in selected}) != 158):
        raise ValueError("Fixed extension roster must contain 79 species / 158 unique controls.")
    if build_source_catalog(root) != baseline:
        raise ValueError("Pinned source catalog changed during extension construction.")
    _verify_catalog_implementation(root)
    return selected


def pilot_extension_controls(
    catalog: tuple[ExtensionControl, ...],
) -> tuple[ExtensionControl, ...]:
    """Fixed F, Ne, Fe and U pilot in both materials, not an adjustable grid."""
    if type(catalog) is not tuple or len(catalog) != 158:
        raise ValueError("Extension pilot requires the complete fixed extension catalog.")
    for control in catalog:
        _check_control(control)
    if len({(case.ion_z, case.target_z) for case in catalog}) != 158:
        raise ValueError("Duplicate source extension control.")
    return tuple(sorted((case for case in catalog if case.ion_z in _PILOT_Z),
                        key=lambda case: (case.ion_z, case.target_z)))


def expected_extension_grid(control: ExtensionControl) -> tuple[Decimal, ...]:
    base = _check_control(control)
    return tuple(value for value in _normal_grid(base)
                 if control.energy_min_mev <= value <= control.energy_max_mev)


def extension_input_bytes(control: ExtensionControl) -> bytes:
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
