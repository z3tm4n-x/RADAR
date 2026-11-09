"""Re-evaluate OST GCR only for common threshold rows that changed >1% at the first doubling."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(folder))
import build_paired_stopping_controls as control  # noqa: E402
import compare_srim_source_overlap as io  # noqa: E402
import paired_stopping_impact as op  # noqa: E402
import run_paired_stopping_impact as pilot  # noqa: E402
import run_srim_hze_products as sources  # noqa: E402
from report_paired_stopping_impact import index_report  # noqa: E402
from radar.core.constants import DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON  # noqa: E402
from radar.core.project import MethodologyConfig, MissionConfig  # noqa: E402
from radar.core.spectra import Spectrum1D  # noqa: E402
from radar.core.types import Particle, SpectrumQuantity  # noqa: E402
from radar.gcr.model import GcrModelInput, OstGcrModel  # noqa: E402
from radar.solar_activity.model import build_mission_solar_activity  # noqa: E402
from radar.solar_activity.ost import ost_wolf_number_cycle_table  # noqa: E402
from radar.validation.candidate_transport import _source_integral, _source_support  # noqa: E402

BASE = "390b4b3d4810970f101d0b9a31d9a68caf5a7db08e0d6b5998c3426ee1c0a000"
DENSE = "2cd06248c868f883deba8af280e65fb689f4dd9d39679f0779fa14a56fe9ca45"
CONTROLS = "82cfa0d857020bc30f10971b5b18de97e47f71b75419085add85241c54f4de22"
FILES = (("paired_impact_base_20261109.json", BASE), ("paired_impact_dense_20261109.json", DENSE),
         ("paired_stopping_controls_20261109.json", CONTROLS))
SYMBOLS = ("He", "Ne", "Fe")
# Fixed pilot ceiling: 8->16 still changed four originally selected Fe rows
# above 1%. The final 16->32 probe retains all 118, not just the survivors.
LEVELS = (1, 2, 4, 8, 16, 32)
Key = tuple[str, str, int, float, str, str, str]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def grids(nodes: tuple[float, ...]) -> dict[int, tuple[float, ...]]:
    if len(nodes) < 2 or any(not isfinite(x) or x <= 0 for x in nodes) or any(a >= b for a, b in zip(nodes, nodes[1:])):
        raise ValueError("A finite positive increasing grid is required.")
    result = {1: nodes}
    for level in LEVELS[1:]:
        old = result[level//2]
        new = tuple(v for i in range(len(old)-1) for v in (old[i], sqrt(old[i]*old[i+1]))) + (old[-1],)
        if new[::2] != old or any(a >= b for a, b in zip(new, new[1:])):
            raise ValueError("Midpoint refinement failed to nest strictly.")
        result[level] = new
    return result


def changes(a: float, b: float, full: float) -> dict[str, float | None]:
    if (min(a, b) < 0 or full <= 0 or any(not isfinite(x) for x in (a, b, full))):
        raise ValueError("Positive finite source denominator and valid counts required.")
    return {"difference": b-a, "percent_of_previous": 100*(b-a)/a if a else None,
            "percent_of_previous_full_source": 100*(b-a)/full}


def exceptional(a: dict[tuple[Any, ...], Any], b: dict[tuple[Any, ...], Any]) -> tuple[Key, ...]:
    if a.keys() != b.keys():
        raise ValueError("The two pilot inventories must match.")
    keys: list[Key] = []
    for case in sorted(a):
        for v in ("SS", "CS", "SC", "CC"):
            for axis in ("energy_ge", "let_ge"):
                x, y = a[case]["variants"][v][axis], b[case]["variants"][v][axis]
                if x.keys() != y.keys():
                    raise ValueError("Threshold inventory changed.")
                for t, val in x.items():
                    q = changes(val, y[t], a[case]["source_integral"])["percent_of_previous"]
                    if q is None and y[t] != 0:
                        raise ValueError("Newly nonzero row requires separate zero-base review.")
                    if q is not None and abs(q) > 1.:
                        k: Key = case[0], case[1], case[2], case[3], v, axis, t
                        keys.append(k)
    return tuple(keys)


def generated(nodes: tuple[float, ...]) -> dict[tuple[str, int], Spectrum1D]:
    mission = MissionConfig(launch_year=2026, lifetime_years=5, sep_exceedance_probability=.1)
    policy = MethodologyConfig()
    solar = build_mission_solar_activity(mission, ost_wolf_number_cycle_table(mission.solar_activity_level),
        policy.solar_cycle_reference_start_year, reference_cycle_number=policy.solar_cycle_reference_number,
        wolf_number_version=policy.solar_activity_wolf_version, monthly_policy=policy.solar_activity_monthly_policy)
    gcr = OstGcrModel(energy_grid_mev_per_nucleon=nodes, symbols=SYMBOLS).calculate(GcrModelInput(mission, solar))
    out = {}
    for product in gcr.products:
        s = product.spectrum
        if s.particle is not Particle.HZE or s.assessment_mode is None or s.quantity not in (
                SpectrumQuantity.DIFFERENTIAL_FLUENCE, SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX):
            continue
        if s.species is None or s.species.atomic_number not in (2, 10, 26) or s.x != nodes:
            raise ValueError("Unexpected GCR projectile/energy nodes.")
        assert s.species.atomic_number is not None
        key = (s.assessment_mode.value, s.species.atomic_number)
        if key in out:
            raise ValueError("Duplicate GCR input.")
        out[key] = s
    if set(out) != {(m, z) for m in ("peak", "cumulative") for z in (2, 10, 26)}:
        raise ValueError("Expected six actual OST model spectra.")
    return out


def pins() -> dict[str, str]:
    return {name: digest(ROOT / "scripts" / name) for name in (Path(__file__).name,
        "build_paired_stopping_controls.py", "run_paired_stopping_impact.py", "paired_stopping_impact.py",
        "run_srim_hze_products.py", "report_paired_stopping_impact.py", "compare_srim_source_overlap.py")}


def run() -> dict[str, Any]:
    before, sources_before = pins(), sources._code_pins()
    archive = [io.read_pinned(ROOT / "outputs" / name, sha) for name, sha in FILES[:2]]
    controls = control.read_verified(ROOT / "outputs" / FILES[2][0], CONTROLS)
    indexed = [index_report(r, level, 80) for r, level in zip(archive, (1, 2), strict=True)]
    for r in archive:
        io.same(r["controls_sha256"], CONTROLS, "same pinned stopping controls")
        io.same(r["production_and_source_sha256"], sources_before, "source and resource hashes")
        io.same(r["implementation_sha256"], {k: before[k] for k in (
            "run_paired_stopping_impact.py", "build_paired_stopping_controls.py", "paired_stopping_impact.py")}, "paired operator pins")
    keys = exceptional(*indexed)
    if (len(keys) != 118 or len(set(k[:4] for k in keys)) != 12 or sum(k[4] == "CC" for k in keys) != 28
            or any(k[0] != "gcr" or k[2] not in (2, 10, 26) or k[5] != "let_ge" for k in keys)):
        raise ValueError("Pinned 118-row exceptional inventory changed.")
    pairs = {z: pilot.prepare(controls, z, 80) for z in (2, 10, 26)}
    settings = grids(DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON)
    by_key: dict[Key, list[float]] = {k: [] for k in keys}
    full: dict[tuple[str, int], list[float]] = {}
    previous: dict[tuple[str, int], Spectrum1D] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for level in LEVELS:
        print(f"Actual OST GCR source refinement {level}: {len(settings[level])} energy nodes", flush=True)
        spectra = generated(settings[level])
        input_hashes[str(level)] = {}
        for key, s in spectra.items():
            if level in (1, 2):
                evidence = [e for e in archive[level-1]["input_spectra"] if e["source"] == "gcr"
                            and e["assessment_mode"] == key[0] and e["species"]["atomic_number"] == key[1]]
                if len(evidence) != 1:
                    raise ValueError("Missing archived GCR input.")
                row = asdict(s)
                for f in ("species", "x", "y", "x_unit", "y_unit", "assessment_mode", "quantity", "temporal_basis",
                          "axis_kind", "angular_basis", "bin_edges", "mission_duration_s", "ordinate_basis", "location"):
                    io.same(row[f], evidence[0][f], f"original OST input field {f}")
            if previous and (s.x[::2] != previous[key].x or s.y[::2] != previous[key].y):
                raise ValueError("Actual model changed a pre-existing input ordinate.")
            op._source(s, pairs[key[1]])
            full.setdefault(key, []).append(_source_integral(s, *_source_support(s)))
            input_hashes[str(level)][str(key)] = io.digest(io.encode(asdict(s)))
        previous = spectra
        cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        for k in keys:
            _, mode, z, t, v, axis, thr = k
            s, pair = spectra[mode, z], pairs[z]
            if k[:5] not in cache:
                al = pair.srim_al if v[0] == "S" else pair.catima_al
                si = pair.srim_si if v[1] == "S" else pair.catima_si
                lo, hi = al.curve.energy[0], al.curve.energy[-1]
                cache[k[:5]] = op._variant(s, al, si, t, lo, hi, hi)
            val = cache[k[:5]][axis][thr]
            if level in (1, 2):
                expected = indexed[level-1][k[:4]]
                io.same(val, expected["variants"][v][axis][thr], "replayed archived threshold")
                io.same(full[mode,z][-1], expected["source_integral"], "replayed archived full source")
            by_key[k].append(val)
    rows: list[dict[str, Any]] = []
    for k in keys:
        vals, totals = by_key[k], full[k[1], k[2]]
        steps = [changes(vals[i], vals[i+1], totals[i]) for i in range(len(LEVELS)-1)]
        rows.append({"key": list(k), "values": vals, "full_source_integrals": totals, "steps": steps})
    summary = []
    for i in range(len(LEVELS)-1):
        vals = [r["steps"][i]["percent_of_previous"] for r in rows]
        summary.append({"from": LEVELS[i], "to": LEVELS[i+1], "above_one_percent": sum(q is not None and abs(q) > 1 for q in vals),
            "undefined_previous_zero": sum(q is None for q in vals), "max_abs_percent": max((abs(q) for q in vals if q is not None),default=None),
            "max_abs_percent_of_full_source": max(abs(r["steps"][i]["percent_of_previous_full_source"]) for r in rows)})
    io.same(before, pins(), "probe helper hashes after calculation")
    io.same(sources_before, sources._code_pins(), "production hashes after calculation")
    for name, sha in FILES:
        if digest(ROOT / "outputs" / name) != sha:
            raise ValueError("Archive changed during evaluation.")
    return {"schema_version": "radar.paired_source_refinement_probe.v1", **pilot.FLAGS,
        "scope": "only 118 common threshold rows that changed >1% at first doubling; not a global accuracy test",
        "method": "actual monthly OST GCR on nested sqrt midpoints, restricted symbol roster; archived 1/2 inputs and counts exactly replayed",
        "levels": list(LEVELS), "node_counts": [len(settings[level]) for level in LEVELS], "row_count": len(keys),
        "refinement_ceiling": 32, "global_convergence_established": False,
        "ion_counts": dict(Counter(str(k[2]) for k in keys)), "variant_counts": dict(Counter(k[4] for k in keys)),
        "ion_and_let_threshold_counts": dict(Counter(f"{k[2]}:{k[6]}" for k in keys)),
        "archive_sha256": dict(FILES), "implementation_sha256": before, "source_and_resource_sha256": sources_before,
        "input_spectrum_sha256_by_level": input_hashes, "summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError(path)
    report = run()
    io.write_new(path, report)
    print(report["summary"], digest(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
