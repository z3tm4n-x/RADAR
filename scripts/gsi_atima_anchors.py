"""Retain and compare 32 original-GSI web anchors; diagnostic, not LET qualification.

Only explicit public physics parameters are sent to the original HTTPS CGI.
The archived response is generic stopping, not an identified electronic LET.
No native code is downloaded; no model/table/profile is changed.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_srim_source_overlap as shared  # noqa: E402
import generate_high_energy_anchors as anchors  # noqa: E402

BASE = "https://web-docs.gsi.de/~weick/atima/"
CGI = BASE + "atima14_script.cgi"
SOURCE_URLS = tuple(BASE + p for p in ("", "atima14.html", "atima_readme.txt", "atima.txt"))
MAX_BYTES = 131072
NATURAL_TARGET_MASS = {13: Decimal("26.981541"), 14: Decimal("28.085501")}
NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
FLAGS = {"status": "INCOMPLETE", "normative_export_allowed": False,
         "release_ready": False, "production_replacement_allowed": False,
         "electronic_let_equivalence_verified": False, "experimental_validation": False}


def acquisition_pins() -> dict[str, str]:
    return {p: digest((ROOT / p).read_bytes()) for p in
            ("scripts/gsi_atima_anchors.py", "scripts/fetch_gsi_atima_evidence.ps1")}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _one(pattern: str, text: str) -> tuple[str, ...]:
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one ATIMA field: {pattern}")
    return matches[0].groups()


def _decimal(token: str) -> Decimal:
    if re.fullmatch(NUMBER, token) is None:
        raise ValueError("Invalid numeric token.")
    value = Decimal(token)
    if not value.is_finite():
        raise ValueError("Nonfinite ATIMA field.")
    return value


def _rounds_to(value: Decimal, printed: str) -> bool:
    number = _decimal(printed)
    # The observed CGI fields use six decimal places. Retain the actual token
    # resolution instead of treating a printed endpoint as exact energy.
    with localcontext() as ctx:
        ctx.prec = 50
        return abs(value - number) <= _half_resolution(number)


def _half_resolution(value: Decimal) -> Decimal:
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError("Nonfinite printed decimal.")
    return Decimal(5).scaleb(exponent - 1)


def request_url(request: dict[str, Any]) -> str:
    return CGI + "?" + urlencode({"zp": request["z"],
        "massp": request["mass_parameter_u_decimal"],
        "energy": repr(request["energy_mev_per_nucleon"]),
        "Zt": request["material_z"], "at": "0", "thickness": "1"})


def parse_response(text: str, request: dict[str, Any]) -> dict[str, Any]:
    """Parse one response and check inputs/whole-ion unit convention independently.

    The finite-foil energy loss is a convention check with a declared 0.1%
    tolerance, not an equality between a derivative and a finite difference or
    a physical-accuracy acceptance threshold. Printed rounding is propagated.
    """
    if len(text.encode("utf-8")) > MAX_BYTES or not text.rstrip().endswith("</BODY></HTML>"):
        raise ValueError("Truncated, oversized or unexpected GSI response.")
    if not re.search(r"\bdone\s+ende\b", text):
        raise ValueError("GSI completion markers missing.")
    n = f"({NUMBER})"
    echo = _one(rf"Zp={n} Mass_p={n} E={n} d={n} Zt={n} At={n}", text)
    energy = Decimal(repr(request["energy_mev_per_nucleon"]))
    mass = Decimal(request["mass_parameter_u_decimal"])
    expected = tuple(map(Decimal, (str(request["z"]), str(mass), str(energy),
                                 "1", str(request["material_z"]), "0")))
    if tuple(map(_decimal, echo)) != expected:
        raise ValueError("GSI input echo changed an input, including literal source A.")
    proj = _one(rf"projectile\s*:\s*Z={n}\s+A={n}\s+E={n}\s+MeV/u", text)
    target = _one(rf"target\s*:\s*Z={n}\s+A={n}\s+thickness={n}\s+mg/cm\^2", text)
    material = _one(rf"1:\s*Z={n},\s*A={n},\s*w={n},\s*pot={n}", text)
    symbol = _one(r"read ATIMA splines for ([A-Za-z]+)\.", text)[0]
    expected_symbol = {13: "Al", 14: "Si"}[request["material_z"]]
    if (Decimal(proj[0]) != request["z"] or Decimal(proj[1]) != mass
            or not _rounds_to(energy, proj[2]) or Decimal(target[0]) != request["material_z"]
            or Decimal(target[2]) != 1 or Decimal(material[0]) != request["material_z"]
            or Decimal(material[1]) != Decimal(target[1]) or Decimal(material[2]) != 1
            or Decimal(material[3]) != {13: 166, 14: 173}[request["material_z"]]
            or symbol != expected_symbol):
        raise ValueError("GSI calculated projectile/target differs from the request.")
    if Decimal(target[1]) != NATURAL_TARGET_MASS[request["material_z"]]:
        raise ValueError("Reviewed original-GSI natural target mass changed.")
    if len(re.findall(r"(?:^|\n)\s*[0-9]+:\s*Z=", text)) != 1:
        raise ValueError("Exactly one elemental GSI target component is required.")
    stopping = _one(rf"dE/dx at entrance\s*:\s*{n}\s+MeV/\(mg/cm\^2\)", text)[0]
    exit_energy = _one(rf"exit energy\s*:\s*{n}\s+MeV/u", text)[0]
    range_value = _one(rf"(?:^|\n)\s*range\s*:\s*{n}\s+mg/cm\^2", text)[0]
    charge = _one(rf"q_p in - out\s*:\s*{n}\s+-\s+{n}", text)
    label = _one(r"<br>\s*(Atima [0-9.]+)\s*<br>", text)[0]
    with localcontext() as ctx:
        ctx.prec = 50
        s, out = Decimal(stopping), Decimal(exit_energy)
        if s <= 0 or out <= 0 or out >= energy or Decimal(range_value) <= 0:
            raise ValueError("Nonphysical GSI stopping/energy/range.")
        if any(Decimal(q) <= 0 for q in charge):
            raise ValueError("Invalid GSI charge.")
        # The original spline output can print q slightly above Z. Preserve
        # and flag it; do not clamp it or turn this diagnostic into acceptance.
        charge_excess = [max(Decimal(0), Decimal(q) - request["z"]) for q in charge]
        mean = (energy - out) * mass  # thickness is exactly 1 mg/cm2
        half_exit = _half_resolution(out)
        half_s = _half_resolution(s)
        ratio_lo = (mean - half_exit * mass) / (s + half_s)
        ratio_hi = (mean + half_exit * mass) / (s - half_s)
        if ratio_hi < Decimal("0.999") or ratio_lo > Decimal("1.001"):
            raise ValueError("Whole-ion printed stopping fails finite-foil unit check.")
        return {"response_model_label": label, "target_symbol": symbol,
            "target_mass_u_decimal": target[1], "target_ionization_potential_ev": material[3],
            "calculated_projectile_mass_decimal": proj[1],
            "calculated_energy_mev_per_u_decimal": proj[2],
            "stopping_mev_cm2_mg_decimal": stopping,
            "stopping_mev_cm2_g_decimal": str(s * 1000),
            "exit_energy_mev_per_u_decimal": exit_energy, "range_mg_cm2_decimal": range_value,
            "charge_in_out_decimal": list(charge),
            "charge_excess_above_z_decimal": [str(v) for v in charge_excess],
            "charge_physical_bound_warning": any(v > 0 for v in charge_excess),
            "finite_foil_mean_stopping_mev_cm2_mg_decimal": str(mean),
            "mean_to_entrance_ratio_rounding_interval": [str(ratio_lo), str(ratio_hi)],
            "unit_check_relative_tolerance": "0.001",
            "generic_stopping_operator_not_identified_as_electronic_let": True}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:
        raise ValueError("Refuse redirected original-GSI evidence URL.")


def _fetch_once(url: str) -> dict[str, Any]:
    parsed = urlsplit(url)
    if (url not in SOURCE_URLS and not url.startswith(CGI + "?")) or (
            parsed.scheme != "https" or parsed.netloc != "web-docs.gsi.de"):
        raise ValueError("Only the declared public original-GSI paths are permitted.")
    req = Request(url, headers={"User-Agent": "RADAR-scientific-validation/0.1"})
    with build_opener(_NoRedirect()).open(req, timeout=25) as response:
        raw = response.read(MAX_BYTES + 1)
        if response.status != 200 or response.geturl() != url or len(raw) > MAX_BYTES:
            raise ValueError("Unexpected GSI HTTP response.")
        content_type = response.headers.get("Content-Type", "")
        if not content_type.lower().startswith(("text/html", "text/plain")):
            raise ValueError("Unexpected GSI content type.")
        # The official landing page uses legacy Latin-1 bytes. Never hash a
        # transcoded representation in place of the received evidence.
        encoding = "utf-8"
        try:
            text = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            encoding = "iso-8859-1"
            text = raw.decode(encoding, errors="strict")
        return {"url": url, "http_status": 200, "content_type": content_type,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": digest(raw), "raw_base64": base64.b64encode(raw).decode("ascii"),
            "text_encoding": encoding, "text": text}


def fetch(url: str) -> dict[str, Any]:
    for attempt in range(2):
        try:
            return _fetch_once(url)
        except (URLError, TimeoutError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            if attempt or not isinstance(reason, TimeoutError):
                raise
            print("One GSI timeout; retrying this public request once.", flush=True)
    raise AssertionError("Unreachable retry state")


def cached_fetch(url: str, cache: Path | None) -> dict[str, Any]:
    if cache is None:
        return fetch(url)
    cache.mkdir(exist_ok=True)
    path = cache / (digest(url.encode("ascii")) + ".json")
    if path.exists():
        value = shared.read_pinned(path, digest(path.read_bytes()))
        _validate_evidence(value, url)
        return dict(value)
    value = fetch(url)
    _validate_evidence(value, url)
    shared.write_new(path, value)
    return value


def native_report(path: Path, sha: str) -> dict[str, Any]:
    report = shared.read_pinned(path, sha)
    anchors.validate_report(report)
    return dict(report)


def _core(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in report["rows"] if row["core_anchor"]]
    if len(rows) != 32:
        raise ValueError("Exactly 32 original-GSI core anchors are required.")
    return rows


def collect(path: Path, sha: str, cache: Path | None = None) -> dict[str, Any]:
    report = native_report(path, sha)
    implementation_sha = digest(Path(__file__).read_bytes())
    helpers = acquisition_pins()
    source_evidence = [cached_fetch(url, cache) for url in SOURCE_URLS]
    form = source_evidence[1]["text"]
    if 'action="atima14_script.cgi"' not in form or "V1.41" not in form:
        raise ValueError("Original-GSI V1.41 interface no longer matches reviewed route.")
    evidence = []
    for row in _core(report):
        response = cached_fetch(request_url(row), cache)
        parse_response(response["text"], row)
        evidence.append(response)
        print(f"GSI anchor {len(evidence)}/32: Z={row['z']} target={row['material_z']} "
              f"{row['anchor_kind']}", flush=True)
    shared.same(native_report(path, sha), report, "native anchors after GSI collection")
    if digest(Path(__file__).read_bytes()) != implementation_sha:
        raise ValueError("GSI collector implementation changed during collection.")
    shared.same(acquisition_pins(), helpers, "acquisition helper hashes after collection")
    return {"schema_version": "radar.original_gsi_atima_anchors.raw.v1", **FLAGS,
        "native_path": str(path.resolve()), "native_sha256": sha,
        "implementation_sha256": implementation_sha,
        "acquisition_helpers_sha256": helpers,
        "request_count": 32, "interface_version_label": "V1.41",
        "source_evidence": source_evidence, "responses": evidence}


def _validate_evidence(evidence: Any, url: str) -> str:
    if (not isinstance(evidence, dict) or evidence.get("url") != url
            or type(evidence.get("http_status")) is not int or evidence["http_status"] != 200
            or not isinstance(evidence.get("text"), str)
            or not isinstance(evidence.get("raw_base64"), str)
            or len(evidence["raw_base64"]) > 4 * (MAX_BYTES // 3 + 1)
            or evidence.get("text_encoding") not in ("utf-8", "iso-8859-1")
            or not isinstance(evidence.get("content_type"), str)
            or not evidence["content_type"].lower().startswith(("text/html", "text/plain"))):
        raise ValueError("GSI response URL, status, size or content pin mismatch.")
    raw = base64.b64decode(evidence["raw_base64"], validate=True)
    if (len(raw) > MAX_BYTES or digest(raw) != evidence.get("sha256")
            or raw.decode(evidence["text_encoding"], errors="strict") != evidence["text"]):
        raise ValueError("GSI raw bytes, decoded text or hash changed.")
    stamp = datetime.fromisoformat(evidence["retrieved_utc"])
    if stamp.tzinfo != timezone.utc:
        raise ValueError("UTC evidence timestamp required.")
    return str(evidence["text"])


def compare(raw_path: Path, raw_sha: str, native_path: Path, native_sha: str) -> dict[str, Any]:
    native = native_report(native_path, native_sha)
    raw = shared.read_pinned(raw_path, raw_sha)
    if (raw.get("schema_version") != "radar.original_gsi_atima_anchors.raw.v1"
            or any(raw.get(k) != v or type(raw.get(k)) is not type(v) for k, v in FLAGS.items())
            or raw.get("native_path") != str(native_path.resolve()) or raw.get("native_sha256") != native_sha
            or raw.get("implementation_sha256") != digest(Path(__file__).read_bytes())
            or raw.get("acquisition_helpers_sha256") != acquisition_pins()
            or type(raw.get("request_count")) is not int or raw["request_count"] != 32
            or raw.get("interface_version_label") != "V1.41"
            or len(raw.get("source_evidence", [])) != len(SOURCE_URLS)
            or len(raw.get("responses", [])) != 32):
        raise ValueError("GSI collection identity, flags, code pin or completeness changed.")
    for evidence, url in zip(raw["source_evidence"], SOURCE_URLS, strict=True):
        _validate_evidence(evidence, url)
    if 'action="atima14_script.cgi"' not in raw["source_evidence"][1]["text"]:
        raise ValueError("Original CGI route missing from retained form.")
    rows = []
    for evidence, row in zip(raw["responses"], _core(native), strict=True):
        parsed = parse_response(_validate_evidence(evidence, request_url(row)), row)
        s = float(parsed["stopping_mev_cm2_g_decimal"])
        differences = {component: 100 * (row[f"stopping_{component}_mev_cm2_g"] / s - 1)
                       for component in ("electronic", "nonradiative", "total")}
        half = float(_half_resolution(Decimal(parsed["stopping_mev_cm2_mg_decimal"])) * 1000)
        intervals = {component: [100 * (row[f"stopping_{component}_mev_cm2_g"] / (s + half) - 1),
                                100 * (row[f"stopping_{component}_mev_cm2_g"] / (s - half) - 1)]
                     for component in differences}
        rows.append({"native_anchor": row, "gsi_response_sha256": evidence["sha256"],
                     "gsi": parsed, "catima_vs_generic_gsi_percent": differences,
                     "gsi_print_half_resolution_mev_cm2_g": half,
                     "catima_vs_gsi_percent_print_rounding_interval": intervals})
    shared.same(native_report(native_path, native_sha), native, "native anchors after comparison")
    shared.same(shared.read_pinned(raw_path, raw_sha), raw, "GSI evidence after comparison")
    return {"schema_version": "radar.original_gsi_atima_anchors.comparison.v1", **FLAGS,
        "raw_sha256": raw_sha, "native_sha256": native_sha,
        "implementation_sha256": digest(Path(__file__).read_bytes()), "anchor_count": 32,
        "acquisition_helpers_sha256": acquisition_pins(),
        "difference_definition": "100*(CATIMA_component/GSI_generic_printed_stopping-1)",
        "limitations": ["GSI interface says V1.41; response label retained separately, not relabelled.",
            "Six-decimal printed generic stopping is not resolved into electronic/nuclear/radiative parts.",
            "Model-to-model comparison, not experiment; shared LS ancestry remains.",
            "Natural target constants differ slightly; not fitted to force equality.",
            "No SRIM splice, source transport, dose, SEE or normative qualification."], "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "compare"), required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--raw-sha256")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "collect":
        if args.raw is not None or args.raw_sha256 is not None:
            parser.error("Collection does not consume an existing GSI raw report.")
        cache = args.cache or args.output.with_name(args.output.stem + "_responses")
        result = collect(args.native, args.native_sha256, cache)
    else:
        if args.raw is None or args.raw_sha256 is None:
            parser.error("Comparison requires a pinned GSI raw report.")
        if args.cache is not None:
            parser.error("Offline comparison does not acquire or cache remote data.")
        result = compare(args.raw, args.raw_sha256, args.native, args.native_sha256)
    shared.write_new(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output.read_bytes())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
