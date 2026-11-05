"""Offline original-GSI evidence and unit contracts; no network or native execution."""
from __future__ import annotations

import base64
from contextlib import nullcontext
from copy import deepcopy
from decimal import Decimal, localcontext
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.error import URLError

import pytest


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("gsi_atima_anchors")


def request(z: int = 2, target: int = 13, energy: float = 5000.0) -> dict[str, Any]:
    return {"z": z, "mass_parameter_u_decimal": {2: "4", 10: "20.2", 26: "55.8", 92: "238"}[z],
            "material_z": target, "energy_mev_per_nucleon": energy, "core_anchor": True,
            "anchor_kind": "synthetic_anchor",
            "stopping_electronic_mev_cm2_g": 6.0,
            "stopping_nonradiative_mev_cm2_g": 6.5, "stopping_total_mev_cm2_g": 7.0}


def response(row: dict[str, Any], stopping: str = "0.006681") -> str:
    """Synthetic CGI output, not a retained physical observation."""
    mass = Decimal(row["mass_parameter_u_decimal"])
    energy = Decimal(repr(row["energy_mev_per_nucleon"]))
    z, target = row["z"], row["material_z"]
    symbol, atomic, potential = {13: ("Al", "26.981541", 166),
                                 14: ("Si", "28.085501", 173)}[target]
    with localcontext() as ctx:
        ctx.prec = 50
        out = energy - Decimal(stopping) / mass
        return (f"<BODY>\nZp={z} Mass_p={mass} E={energy} d=1 Zt={target} At=0\n"
                f"read ATIMA splines for {symbol}.\n"
                f"1: Z={target}, A={atomic}, w=1.000000, pot={potential}.000000\n"
                "<br> Atima 1.4 <br>\n"
                f"projectile : Z={z} A={mass:.6f} E={energy:.6f} MeV/u\n"
                f"target : Z={target} A={atomic} thickness=1.000000 mg/cm^2\n"
                f"dE/dx at entrance : {stopping} MeV/(mg/cm^2)\n"
                f"exit energy : {out:.6f} MeV/u\n"
                "range : 10000.000000 mg/cm^2\n"
                f"q_p in - out : {z}.000000 - {z}.000000\n"
                "done ende\n</BODY></HTML>")


def evidence(worker: ModuleType, url: str, text: str,
             encoding: str = "utf-8") -> dict[str, Any]:
    raw = text.encode(encoding)
    return {"url": url, "http_status": 200, "content_type": "text/html; charset=UTF-8",
            "retrieved_utc": "2026-11-05T12:00:00+00:00", "sha256": worker.digest(raw),
            "raw_base64": base64.b64encode(raw).decode("ascii"),
            "text_encoding": encoding, "text": text}


@pytest.mark.parametrize("z,target", [(2, 13), (10, 14), (26, 13), (92, 14)])
def test_literal_mass_unit_conversion_and_rounding(worker: ModuleType, z: int, target: int) -> None:
    row = request(z, target, 79432.82347242821)
    parsed = worker.parse_response(response(row, "0.173790"), row)
    assert Decimal(parsed["calculated_projectile_mass_decimal"]) == Decimal(
        row["mass_parameter_u_decimal"])
    assert Decimal(parsed["stopping_mev_cm2_g_decimal"]) == Decimal("173.790")
    low, high = map(Decimal, parsed["mean_to_entrance_ratio_rounding_interval"])
    assert low <= 1 <= high
    assert parsed["response_model_label"] == "Atima 1.4"
    assert parsed["generic_stopping_operator_not_identified_as_electronic_let"] is True


@pytest.mark.parametrize("z", [10, 26])
def test_fractional_mass_request_url_is_not_integerized(worker: ModuleType, z: int) -> None:
    row = request(z, 14, 4950.495049504951)
    url = worker.request_url(row)
    assert urlsplit(url).scheme == "https"
    assert urlsplit(url).netloc == "web-docs.gsi.de"
    assert parse_qs(urlsplit(url).query) == {
        "zp": [str(z)], "massp": [row["mass_parameter_u_decimal"]],
        "energy": ["4950.495049504951"], "Zt": ["14"], "at": ["0"], "thickness": ["1"]}


@pytest.mark.parametrize("old,new", [
    ("Mass_p=4", "Mass_p=3"), ("Zp=2", "Zp=3"), ("E=5000.0 d=", "E=5001.0 d="),
    ("d=1 Zt=", "d=2 Zt="), ("Zt=13 At=", "Zt=14 At="), ("At=0", "At=27"),
    ("A=4.000000", "A=4.000001"), ("E=5000.000000 MeV/u", "E=5000.000001 MeV/u"),
    ("pot=166.000000", "pot=173.000000"), ("w=1.000000", "w=0.500000"),
    ("26.981541", "26.981500"), ("for Al.", "for Si."),
    ("MeV/(mg/cm^2)", "MeV/(g/cm^2)"), ("MeV/(mg/cm^2)", "MeV/u/(mg/cm^2)"),
    ("0.006681 MeV", "0.000000 MeV"), ("0.006681 MeV", "NaN MeV"),
    ("0.006681 MeV", "Infinity MeV"), ("range : 10000.000000", "range : -1.000000"),
    ("2.000000 - 2.000000", "0.000000 - 2.000000"),
    ("done ende", "done"), ("</BODY></HTML>", "</BODY>"),
])
def test_changed_inputs_units_and_incomplete_response_rejected(
    worker: ModuleType, old: str, new: str
) -> None:
    row = request()
    text = response(row)
    assert old in text
    with pytest.raises(ValueError):
        worker.parse_response(text.replace(old, new), row)


@pytest.mark.parametrize("field", ["Zp=", "projectile :", "target :", "1: Z=",
                                 "dE/dx at entrance", "exit energy", "range :",
                                 "q_p in - out", "<br> Atima"])
def test_duplicate_fields_fail_closed(worker: ModuleType, field: str) -> None:
    row = request()
    text = response(row)
    line = next(line for line in text.splitlines() if field in line)
    with pytest.raises(ValueError):
        worker.parse_response(text.replace(line, line + "\n" + line), row)


def test_extra_material_component_rejected(worker: ModuleType) -> None:
    row = request()
    text = response(row).replace("done ende", "2: Z=14, A=28.085501, w=0.5\ndone ende")
    with pytest.raises(ValueError, match="one elemental"):
        worker.parse_response(text, row)


@pytest.mark.parametrize("factor", ["4", "0.25", "1000", "0.001"])
def test_wrong_whole_ion_or_mass_unit_factor_fails_foil_check(worker: ModuleType, factor: str) -> None:
    row = request()
    text = response(row).replace("0.006681 MeV", f"{Decimal('0.006681') * Decimal(factor)} MeV")
    with pytest.raises(ValueError, match="finite-foil"):
        worker.parse_response(text, row)


def test_observed_helium_rounding_interval(worker: ModuleType) -> None:
    row = request()
    parsed = worker.parse_response(response(row), row)
    assert parsed["exit_energy_mev_per_u_decimal"] == "4999.998330"
    assert Decimal(parsed["finite_foil_mean_stopping_mev_cm2_mg_decimal"]) == Decimal("0.006680")
    low, high = map(Decimal, parsed["mean_to_entrance_ratio_rounding_interval"])
    assert low < 1 < high
    assert parsed["unit_check_relative_tolerance"] == "0.001"


@pytest.mark.parametrize("encoding,text", [("utf-8", "original λ"), ("iso-8859-1", "Sörensen")])
def test_raw_evidence_keeps_actual_bytes(worker: ModuleType, encoding: str, text: str) -> None:
    record = evidence(worker, worker.SOURCE_URLS[0], text, encoding)
    assert worker._validate_evidence(record, worker.SOURCE_URLS[0]) == text
    assert record["sha256"] == worker.digest(text.encode(encoding))


@pytest.mark.parametrize("key,value", [
    ("url", "https://example.org"), ("http_status", True), ("http_status", 201),
    ("sha256", "0" * 64), ("text", "changed"), ("raw_base64", "%%%"),
    ("text_encoding", "utf-16"), ("content_type", "application/octet-stream"),
    ("retrieved_utc", "2026-11-05T12:00:00"),
])
def test_evidence_tampering_rejected(worker: ModuleType, key: str, value: Any) -> None:
    record = evidence(worker, worker.SOURCE_URLS[0], "original")
    record[key] = value
    with pytest.raises(ValueError):
        worker._validate_evidence(record, worker.SOURCE_URLS[0])


@pytest.mark.parametrize("url", ["http://web-docs.gsi.de/~weick/atima/",
    "https://example.org/atima14_script.cgi?zp=2", "https://web-docs.gsi.de/~weick/atima/other"])
def test_fetch_rejects_unapproved_route_without_network(worker: ModuleType, url: str,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("No network should be reached")
    monkeypatch.setattr(worker, "build_opener", no_network)
    with pytest.raises(ValueError):
        worker.fetch(url)


@pytest.fixture
def fake_campaign(worker: ModuleType, monkeypatch: pytest.MonkeyPatch,
                  tmp_path: Path) -> tuple[dict[str, Any], Path, str]:
    rows = [request(z, target, energy) for z in (2, 10, 26, 92) for target in (13, 14)
            for energy in (5000.0, 10000.0, 50000.0, 79432.82347242821)]
    rows += [{**request(), "core_anchor": False} for _ in range(16)]
    report = {"rows": rows}
    path = tmp_path / "native.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    sha = worker.digest(path.read_bytes())
    monkeypatch.setattr(worker, "native_report", lambda p, s: deepcopy(report))
    by_url = {worker.request_url(row): row for row in rows if row["core_anchor"]}
    calls: list[str] = []
    def fake_fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        if url in worker.SOURCE_URLS:
            text = 'V1.41 <form action="atima14_script.cgi">'
        else:
            text = response(by_url[url], "0.173790")
        return evidence(worker, url, text)
    monkeypatch.setattr(worker, "fetch", fake_fetch)
    return {"report": report, "calls": calls}, path, sha


def test_collect_is_bounded_and_keeps_diagnostic_flags(worker: ModuleType,
    fake_campaign: tuple[dict[str, Any], Path, str]) -> None:
    fixture, path, sha = fake_campaign
    raw = worker.collect(path, sha)
    assert len(fixture["calls"]) == 36
    assert fixture["calls"][:4] == list(worker.SOURCE_URLS)
    assert len(raw["responses"]) == 32
    assert raw["interface_version_label"] == "V1.41"
    assert all(raw[key] == value for key, value in worker.FLAGS.items())


def test_compare_components_are_not_implicitly_equated(worker: ModuleType, tmp_path: Path,
    fake_campaign: tuple[dict[str, Any], Path, str]) -> None:
    _, path, sha = fake_campaign
    raw = worker.collect(path, sha)
    raw_path = tmp_path / "gsi.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    result = worker.compare(raw_path, worker.digest(raw_path.read_bytes()), path, sha)
    assert result["anchor_count"] == 32
    first = result["rows"][0]
    assert first["gsi"]["response_model_label"] == "Atima 1.4"
    assert first["catima_vs_generic_gsi_percent"] == pytest.approx({
        "electronic": 100 * (6 / 173.79 - 1),
        "nonradiative": 100 * (6.5 / 173.79 - 1), "total": 100 * (7 / 173.79 - 1)})
    assert all(result[key] == value for key, value in worker.FLAGS.items())


@pytest.mark.parametrize("field,value", [("release_ready", True), ("normative_export_allowed", 0),
    ("implementation_sha256", "0" * 64), ("native_sha256", "0" * 64),
    ("request_count", True), ("interface_version_label", "V1.4")])
def test_compare_rejects_changed_policies_and_pins(worker: ModuleType, tmp_path: Path,
    fake_campaign: tuple[dict[str, Any], Path, str], field: str, value: Any) -> None:
    _, path, sha = fake_campaign
    raw = worker.collect(path, sha)
    raw[field] = value
    raw_path = tmp_path / "gsi.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        worker.compare(raw_path, worker.digest(raw_path.read_bytes()), path, sha)


def test_collect_rejects_missing_core_anchor(worker: ModuleType,
    fake_campaign: tuple[dict[str, Any], Path, str]) -> None:
    fixture, path, sha = fake_campaign
    fixture["report"]["rows"][0]["core_anchor"] = False
    with pytest.raises(ValueError, match="32"):
        worker.collect(path, sha)
    assert len(fixture["calls"]) == 4


def test_cli_refuses_existing_output_before_collection(worker: ModuleType,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "existing.json"
    target.write_text("keep", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["test", "--mode", "collect", "--native", "missing.json",
        "--native-sha256", "a" * 64, "--output", str(target)])
    def no_collect(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Existing-output guard must run before network/native work")
    monkeypatch.setattr(worker, "collect", no_collect)
    with pytest.raises(FileExistsError):
        worker.main()
    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("wrapped", [False, True])
def test_one_timeout_retry_then_success(worker: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                       wrapped: bool) -> None:
    calls: list[str] = []
    expected = {"evidence": "retained"}
    def once(url: str) -> dict[str, str]:
        calls.append(url)
        if len(calls) == 1:
            error = TimeoutError("bounded synthetic timeout")
            raise URLError(error) if wrapped else error
        return expected
    monkeypatch.setattr(worker, "_fetch_once", once)
    assert worker.fetch(worker.SOURCE_URLS[0]) == expected
    assert len(calls) == 2


@pytest.mark.parametrize("failure,expected_calls", [(TimeoutError("timeout"), 2),
    (URLError(TimeoutError("timeout")), 2), (URLError("DNS failure"), 1),
    (ValueError("invalid data"), 1)])
def test_retry_is_bounded_and_only_for_timeouts(worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch, failure: Exception, expected_calls: int) -> None:
    calls: list[str] = []
    def fail(url: str) -> None:
        calls.append(url)
        raise failure
    monkeypatch.setattr(worker, "_fetch_once", fail)
    with pytest.raises(type(failure)):
        worker.fetch(worker.SOURCE_URLS[0])
    assert len(calls) == expected_calls


def test_cache_replays_identical_raw_bytes_without_network(worker: ModuleType, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    url = worker.SOURCE_URLS[0]
    original = evidence(worker, url, "Sörensen", "iso-8859-1")
    calls: list[str] = []
    def fetch_once(value: str) -> dict[str, Any]:
        calls.append(value)
        return deepcopy(original)
    monkeypatch.setattr(worker, "fetch", fetch_once)
    cache = tmp_path / "cache"
    assert worker.cached_fetch(url, cache) == original
    assert worker.cached_fetch(url, cache) == original
    assert calls == [url]
    files = list(cache.iterdir())
    assert len(files) == 1
    assert files[0].name == worker.digest(url.encode("ascii")) + ".json"


@pytest.mark.parametrize("key,value", [("url", "https://example.org"),
    ("text", "changed"), ("sha256", "a" * 64)])
def test_corrupt_cache_rejected_without_network(worker: ModuleType, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    url = worker.SOURCE_URLS[0]
    record = evidence(worker, url, "saved")
    record[key] = value
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / (worker.digest(url.encode("ascii")) + ".json")).write_text(
        json.dumps(record), encoding="utf-8")
    def no_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Do not overwrite a failed evidence cache")
    monkeypatch.setattr(worker, "fetch", no_network)
    with pytest.raises(ValueError):
        worker.cached_fetch(url, cache)


@pytest.mark.parametrize("mutation", ["good_latin1", "oversize", "status", "redirect", "content_type"])
def test_fetch_enforces_http_bounds_and_preserves_encoding(worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    url = worker.SOURCE_URLS[0]
    payload = b"S\xf6rensen" if mutation != "oversize" else b"a" * (worker.MAX_BYTES + 1)
    read_sizes: list[int] = []
    def read(size: int) -> bytes:
        read_sizes.append(size)
        return payload
    http = SimpleNamespace(status=201 if mutation == "status" else 200,
        geturl=lambda: "https://example.org" if mutation == "redirect" else url,
        headers={"Content-Type": "application/octet-stream" if mutation == "content_type"
                 else "text/html; charset=iso-8859-1"}, read=read)
    calls: list[tuple[str, int]] = []
    def open_request(req: Any, timeout: int) -> Any:
        calls.append((req.full_url, timeout))
        return nullcontext(http)
    monkeypatch.setattr(worker, "build_opener", lambda *a: SimpleNamespace(open=open_request))
    if mutation == "good_latin1":
        got = worker._fetch_once(url)
        assert got["text_encoding"] == "iso-8859-1"
        assert got["text"] == "Sörensen"
        assert base64.b64decode(got["raw_base64"]) == payload
        assert worker._validate_evidence(got, url) == "Sörensen"
    else:
        with pytest.raises(ValueError):
            worker._fetch_once(url)
    assert calls == [(url, 25)]
    assert read_sizes == [worker.MAX_BYTES + 1]


def test_redirect_handler_fails_closed(worker: ModuleType) -> None:
    with pytest.raises(ValueError, match="redirected"):
        worker._NoRedirect().redirect_request(None, None, 302, "moved", {}, "https://example.org")


def test_native_change_during_collection_rejected(worker: ModuleType,
    fake_campaign: tuple[dict[str, Any], Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, path, sha = fake_campaign
    count = 0
    def read_native(*args: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        report: dict[str, Any] = deepcopy(fixture["report"])
        if count > 1:
            report["rows"][0]["stopping_total_mev_cm2_g"] += 1
        return report
    monkeypatch.setattr(worker, "native_report", read_native)
    with pytest.raises(ValueError):
        worker.collect(path, sha)
    assert count == 2


@pytest.mark.parametrize("mutation", ["missing", "reordered", "raw_hash"])
def test_compare_rejects_incomplete_or_reordered_evidence(worker: ModuleType, tmp_path: Path,
    fake_campaign: tuple[dict[str, Any], Path, str], mutation: str) -> None:
    _, path, sha = fake_campaign
    raw = worker.collect(path, sha)
    if mutation == "missing":
        raw["responses"].pop()
    elif mutation == "reordered":
        raw["responses"][0], raw["responses"][1] = raw["responses"][1], raw["responses"][0]
    raw_path = tmp_path / "gsi.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    pin = "0" * 64 if mutation == "raw_hash" else worker.digest(raw_path.read_bytes())
    with pytest.raises(ValueError):
        worker.compare(raw_path, pin, path, sha)


@pytest.mark.parametrize("component,numerator", [("electronic", "6"),
    ("nonradiative", "6.5"), ("total", "7")])
def test_print_rounding_interval_formula_without_extra_mass_factor(worker: ModuleType,
    tmp_path: Path, fake_campaign: tuple[dict[str, Any], Path, str],
    component: str, numerator: str) -> None:
    _, path, sha = fake_campaign
    raw = worker.collect(path, sha)
    raw_path = tmp_path / "gsi.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    result = worker.compare(raw_path, worker.digest(raw_path.read_bytes()), path, sha)
    with localcontext() as ctx:
        ctx.prec = 50
        expected = [float(100 * (Decimal(numerator) / denominator - 1)) for denominator in
                    (Decimal("173.7905"), Decimal("173.7895"))]
    for row in result["rows"]:
        interval = row["catima_vs_gsi_percent_print_rounding_interval"][component]
        assert row["gsi_print_half_resolution_mev_cm2_g"] == 0.0005
        assert interval == pytest.approx(expected, rel=0, abs=2e-12)
        assert interval[0] < row["catima_vs_generic_gsi_percent"][component] < interval[1]
    assert {row["native_anchor"]["mass_parameter_u_decimal"] for row in result["rows"]} == {
        "4", "20.2", "55.8", "238"}


def test_changed_acquisition_helper_pin_rejected(worker: ModuleType, tmp_path: Path,
    fake_campaign: tuple[dict[str, Any], Path, str]) -> None:
    _, path, sha = fake_campaign
    raw = worker.collect(path, sha)
    raw["acquisition_helpers_sha256"]["scripts/fetch_gsi_atima_evidence.ps1"] = "0" * 64
    raw_path = tmp_path / "gsi.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        worker.compare(raw_path, worker.digest(raw_path.read_bytes()), path, sha)


def test_helper_change_during_collection_rejected(worker: ModuleType,
    fake_campaign: tuple[dict[str, Any], Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _, path, sha = fake_campaign
    count = 0
    def pins() -> dict[str, str]:
        nonlocal count
        count += 1
        return {"helper": ("a" if count == 1 else "b") * 64}
    monkeypatch.setattr(worker, "acquisition_pins", pins)
    with pytest.raises(ValueError, match="acquisition helper"):
        worker.collect(path, sha)
    assert count == 2


def test_implementation_change_during_collection_rejected(worker: ModuleType,
    fake_campaign: tuple[dict[str, Any], Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _, path, sha = fake_campaign
    original_digest = worker.digest
    assert worker.__file__ is not None
    source_bytes = Path(worker.__file__).read_bytes()
    count = 0
    def changing_hash(raw: bytes) -> str:
        nonlocal count
        if raw == source_bytes:
            count += 1
            if count >= 3:  # initial implementation, initial helpers, final implementation
                return "0" * 64
        return str(original_digest(raw))
    monkeypatch.setattr(worker, "digest", changing_hash)
    with pytest.raises(ValueError, match="implementation changed"):
        worker.collect(path, sha)


@pytest.mark.parametrize("incoming,outgoing,expected", [
    ("26.000002", "26.000006", ["0.000002", "0.000006"]),
    ("26.000000000000000001", "26.000000", ["0.000000000000000001", "0"]),
    ("27.000000", "25.000000", ["1", "0"]),
    ("25.999999", "26.000004", ["0", "0.000004"]),
])
def test_charge_bound_excess_is_retained_and_warned_without_tolerance_or_clamp(
    worker: ModuleType, incoming: str, outgoing: str, expected: list[str]
) -> None:
    row = request(26)
    baseline = worker.parse_response(response(row, "0.173790"), row)
    text = response(row, "0.173790").replace("26.000000 - 26.000000", f"{incoming} - {outgoing}")
    parsed = worker.parse_response(text, row)
    assert parsed["charge_in_out_decimal"] == [incoming, outgoing]
    assert list(map(Decimal, parsed["charge_excess_above_z_decimal"])) == list(map(Decimal, expected))
    assert parsed["charge_physical_bound_warning"] is True
    for key in ("stopping_mev_cm2_mg_decimal", "stopping_mev_cm2_g_decimal",
                "mean_to_entrance_ratio_rounding_interval"):
        assert parsed[key] == baseline[key]
    assert baseline["charge_physical_bound_warning"] is False
    assert list(map(Decimal, baseline["charge_excess_above_z_decimal"])) == [0, 0]
    assert parsed["generic_stopping_operator_not_identified_as_electronic_let"] is True


@pytest.mark.parametrize("incoming,outgoing", [("0", "26"), ("26", "0"),
                                             ("-0.000001", "26"), ("26", "-1")])
def test_nonpositive_charge_is_rejected(worker: ModuleType, incoming: str, outgoing: str) -> None:
    row = request(26)
    text = response(row, "0.173790").replace("26.000000 - 26.000000", f"{incoming} - {outgoing}")
    with pytest.raises(ValueError, match="charge"):
        worker.parse_response(text, row)
