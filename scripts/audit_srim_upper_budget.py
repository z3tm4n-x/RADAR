"""Count upper source tails without extrapolating stopping or discarding particles.

Prospective coverage is a planning quantity, not a successful native run, LET,
dose, SEE or physical accuracy bound. Only the declared sampled interval enters.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, localcontext
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radar.shielding.srim_dataset import load_srim_hze_dataset  # noqa: E402

CUTOFFS = (1000., 3000., 5000., 10000.)
MANTISSAS = tuple(Decimal(v) for v in (
    "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2", "2.25",
    "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4", "4.5", "5", "5.5", "6", "6.5", "7", "8", "9"))


def prospective_upper(mass: Decimal) -> Decimal:
    if not isinstance(mass, Decimal) or not mass.is_finite() or not 2 <= mass <= 238:
        raise ValueError("Expected a bounded positive source mass.")
    with localcontext() as context:
        context.prec = 80
        upper = max(m * Decimal(10)**exponent for exponent in range(7) for m in MANTISSAS
                    if m * Decimal(10)**exponent <= 5000 * mass)
        if upper * 1000 > 2147483647:
            raise ValueError("Prospective endpoint exceeds the explicit engineering keV guard.")
    return upper


def tail(x: tuple[float, ...], y: tuple[float, ...], lower: float) -> float:
    """Integrate piecewise-linear differential density on the finite source grid."""
    if (len(x) < 2 or len(x) != len(y) or any(not isfinite(v) or v <= 0 for v in x)
            or any(not isfinite(v) or v < 0 for v in y)
            or any(b <= a for a, b in zip(x, x[1:])) or not isfinite(lower) or lower < 0):
        raise ValueError("Invalid finite source grid/density/cutoff.")
    pieces = []
    for a, b, ya, yb in zip(x, x[1:], y, y[1:]):
        start = max(a, lower)
        if start < b:
            ys = ya + (yb-ya) * ((start-a)/(b-a))
            pieces.append((b-start) * (ys+yb)/2)
    return fsum(pieces)


def audit(bundle: Path, bundle_sha: str, products: Path, products_sha: str) -> dict[str, Any]:
    raw = products.read_bytes()
    if hashlib.sha256(raw).hexdigest() != products_sha:
        raise ValueError("Source products fingerprint mismatch.")
    report = json.loads(raw)
    dataset = load_srim_hze_dataset(bundle, expected_sha256=bundle_sha)
    if (report.get("schema_version") != "radar.srim_hze_external_products.v1"
            or report.get("bundle_sha256") != bundle_sha or report.get("source_product_count") != 216
            or report.get("requested_symbols") is not None or len(dataset.tables_by_z) != 84
            or report.get("normative_export_allowed") is not False
            or report["implementation_and_input_sha256_before"] != report["implementation_and_input_sha256_after"]):
        raise ValueError("A complete, pinned non-normative external-source report is required.")
    cases, seen = [], set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report["cases"]:
        if row["thickness_g_cm2"] != 0:
            continue
        source, mode, z = row["source"], row["mode"], row["z"]
        key = (source, mode, z)
        if key in seen or source not in ("gcr", "sep") or mode not in ("peak", "cumulative"):
            raise ValueError("Duplicate or invalid zero-shield source identity.")
        seen.add(key)
        s = row["energy_product"]["spectrum"]
        species = s.get("species")
        if (s.get("particle") != "hze" or s.get("source") != source
                or not isinstance(species, dict) or species.get("is_aggregate") is not False
                or species.get("atomic_number") != z or species.get("symbol") != row["symbol"]
                or s.get("bin_edges") is not None):
            raise ValueError("Source row/species identity or point-sample convention mismatch.")
        unit = "cm^-2 s^-1 (MeV/nucleon)^-1" if mode == "peak" else "cm^-2 (MeV/nucleon)^-1"
        quantities = ("maximum_differential_flux", "peak_differential_flux") if mode == "peak" else ("differential_fluence",)
        if (s["x_unit"] != "MeV/nucleon" or s["y_unit"] != unit or s["quantity"] not in quantities
                or s["ordinate_basis"] != "point_sample" or s["assessment_mode"] != mode
                or s["angular_basis"] != "omnidirectional_4pi"
                or s["axis_kind"] != "kinetic_energy_per_nucleon"):
            raise ValueError("Unexpected differential source or temporal/angular convention.")
        x, y = tuple(s["x"]), tuple(s["y"])
        if any(type(v) not in (int, float) for v in (*x, *y)):
            raise ValueError("Numeric source samples required.")
        total = tail(x, y, 0.)
        if total <= 0:
            raise ValueError("Positive source required; zero records must remain separate.")
        table = dataset.tables_by_z[z]
        mass = table.mass_parameter_u_decimal
        declared = [float(value) for step in s["transformation_chain"]
                    for key, value in step["parameters"] if key == "source_tabulated_mass_parameter"]
        if not declared or any(not isfinite(v) or v != float(mass) for v in declared):
            raise ValueError("Source mass trace differs from the native literal mass.")
        upper = prospective_upper(mass)
        actual = table.al_energy[-1]
        if x[0] < table.al_energy[0]:
            raise ValueError("Upper-only budget requires no source below native Al support.")
        thresholds = {f"above_{int(e)}_MeV_per_source_A": e for e in CUTOFFS}
        with localcontext() as context:
            context.prec = 80
            prospective_per_a = float(upper/mass)
        thresholds.update(current_native_upper=actual, prospective_5000_normal_grid=prospective_per_a)
        integrals = {name: tail(x, y, value) for name, value in thresholds.items()}
        case = {"source": source, "mode": mode, "z": z, "symbol": row["symbol"],
                "mass_parameter_u": str(mass), "declared_source_bounds": [x[0], x[-1]],
                "source_integral": total, "thresholds_mev_per_source_A": thresholds,
                "prospective_total_mev": str(upper), "tail_integrals": integrals,
                "tail_fractions": {k: v/total for k, v in integrals.items()}}
        cases.append(case)
        groups[source, mode].append(case)
    ions = set(dataset.tables_by_z)
    if len(cases) != 216 or any({r["z"] for r in groups["gcr", mode]} != ions for mode in ("peak", "cumulative")):
        raise ValueError("Full GCR roster / both modes missing.")
    if (len(groups["sep", "peak"]) != 24
            or {r["z"] for r in groups["sep", "peak"]} != {r["z"] for r in groups["sep", "cumulative"]}):
        raise ValueError("Full positive SEP roster / both modes missing.")
    summaries = []
    for (source, mode), rows in sorted(groups.items()):
        total = fsum(r["source_integral"] for r in rows)
        names = rows[0]["tail_integrals"]
        tails = {name: fsum(r["tail_integrals"][name] for r in rows) for name in names}
        he = next(r for r in rows if r["z"] == 2)
        summaries.append({"source": source, "mode": mode, "source_integral": total,
                          "tail_integrals": tails, "tail_fractions": {k: v/total for k, v in tails.items()},
                          "helium_fraction_of_each_tail": {k: he["tail_integrals"][k]/v if v else None
                                                           for k, v in tails.items()}})
    if (hashlib.sha256(products.read_bytes()).hexdigest() != products_sha
            or hashlib.sha256(bundle.read_bytes()).hexdigest() != bundle_sha):
        raise ValueError("Budget inputs changed.")
    return {"schema_version": "radar.srim_upper_source_budget.v1", "status": "INCOMPLETE",
            "bundle_sha256": bundle_sha, "source_products_sha256": products_sha,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "summaries": summaries, "cases": cases, "normative_export_allowed": False,
            "native_extension_calculated": False, "dose_or_see_error_bound": False,
            "limitations": ["Only the finite declared piecewise-linear source measure; unknown outer tails excluded.",
                            "Prospective normal-grid coverage is a planning estimate, not native execution or physical accuracy.",
                            "No stopping continuation, LET mapping, angular or mission-time factor is applied."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "products"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.bundle, args.bundle_sha256, args.products, args.products_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(result["cases"]), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
