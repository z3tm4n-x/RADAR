"""ESTAR build orchestration guards with a fake compiler, never native CI execution."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts/reproduce_nist_estar.py"
    spec = importlib.util.spec_from_file_location("nist_estar_reproduction_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _energies() -> tuple[Decimal, ...]:
    # Independently specified source grid, not borrowed from the parser.
    mantissas = ("1", "1.25", "1.5", "1.75", "2", "2.5", "3", "3.5",
                 "4", "4.5", "5", "5.5", "6", "7", "8", "9")
    return tuple(Decimal(m) * Decimal(10) ** exponent
                 for exponent in range(-2, 3) for m in mantissas) + (Decimal(1000),)


def _native_output(*, energy_change: bool = False, short: bool = False) -> bytes:
    header = (
        "Electrons in ALUMINUM\nDensity, g/cm3 I, eV\n2.6989E+00 166.0\n"
        "KINETIC STOPPING POWER CSDA RADIATION DENSITY\n"
        "ENERGY COLLISION RADIATIVE TOTAL RANGE YIELD EFFECT\nDELTA\n"
        "MeV MeV cm2/g MeV cm2/g MeV cm2/g g/cm2\n"
    )
    rows = []
    for i, energy in enumerate(_energies()):
        actual = Decimal(".01001") if energy_change and i == 0 else energy
        rows.append(f"{float(actual):.3E} 1.000E+00 1.000E+00 2.000E+00 "
                    f"{float(energy):.3E} 2.000E-01 1.000E-01\n")
    if short:
        rows.pop()
    return (header + "".join(rows[:40]) + "\f" + header + "".join(rows[40:])).encode("ascii")


def _csv(*, primary_difference: bool = False) -> bytes:
    rows = ["energy_mev,stopping_mev_cm2_g,range_g_cm2,range_mg_cm2,source_sheet\n"]
    for i, energy in enumerate(_energies()):
        stop = "2.0000000000000001" if primary_difference and i == 0 else "2"
        rows.append(f"{energy},{stop},{energy},{energy * 1000},e_Al\n")
    return "".join(rows).encode("ascii")


def _tar(entries: list[tuple[str, bytes, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.linkname = "../../unrelated"
                archive.addfile(info)
    return stream.getvalue()


class BuildHarness:
    """Filesystem effects occur only in pytest's fresh temporary directory."""

    def __init__(self, runner: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.runner, self.root = runner, root
        self.members = {name: f"synthetic unchanged {name}\n".encode("ascii")
                        for name in runner.SOURCE_HASHES}
        self.members["ENG.ELE"] = (
            "81\n" + " ".join(str(e) for e in _energies()) + "\n"
        ).encode("ascii")
        self.archive = root / "STAR.tar.gz"
        self.csv = root / "electron_al.csv"
        self.compiler = root / "fake-gfortran"
        self.output = root / "new-replay"
        self.csv.write_bytes(_csv())
        self.compiler.write_bytes(b"fake compiler identity")
        self.prefix = root / "compiler-prefix"
        self.prefix.mkdir()
        self.dependencies = {query: root / f"dependency-{i}"
                             for i, query in enumerate((
                                 "-print-prog-name=f951", "-print-prog-name=as",
                                 "-print-prog-name=ld", "-print-file-name=libgfortran.a",
                                 "-print-libgcc-file-name",
                             ))}
        for i, path in enumerate(self.dependencies.values()):
            path.write_bytes(f"fake selected dependency {i}".encode("ascii"))
        for name in runner.IMPLEMENTATION_PATHS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"pinned implementation {name}".encode("ascii"))
        self.calls: list[dict[str, Any]] = []
        self.fail_at: str | None = None
        self.timeout_at: str | None = None
        self.change: str | None = None
        self.wrong_version = False
        self.missing_dependency = False
        self.bare_dependency = False
        self.wrong_layout: str | None = None
        self.changed_grid = False
        self.short_grid = False
        self.o2_whitespace_difference = False
        self.precreate_report = False
        monkeypatch.setattr(runner, "ROOT", root)
        monkeypatch.setattr(runner, "SOURCE_HASHES", {k: _sha(v) for k, v in self.members.items()})
        self.rebuild_archive(monkeypatch)
        monkeypatch.setattr(runner.subprocess, "run", self.fake_run)

    def rebuild_archive(
        self, monkeypatch: pytest.MonkeyPatch,
        entries: list[tuple[str, bytes, bytes]] | None = None,
    ) -> None:
        if entries is None:
            entries = [("ESTAR/" + name, data, tarfile.REGTYPE)
                       for name, data in self.members.items()]
        data = _tar(entries)
        self.archive.write_bytes(data)
        monkeypatch.setattr(self.runner, "ARCHIVE_BYTES", len(data))
        monkeypatch.setattr(self.runner, "ARCHIVE_SHA256", _sha(data))

    def build(self) -> dict[str, Any]:
        result: dict[str, Any] = self.runner.run_build(
            archive=self.archive, production_csv=self.csv, compiler=self.compiler,
            output=self.output, compiler_prefixes=(self.prefix,),
        )
        return result

    def fake_run(
        self, argv: list[str], *, cwd: Path, input: str, text: bool,
        capture_output: bool, timeout: int, check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert text and capture_output and timeout == 120 and not check
        assert isinstance(argv, list) and all(isinstance(arg, str) for arg in argv)
        self.calls.append({"argv": argv.copy(), "cwd": cwd, "stdin": input})
        action = "version" if "--version" in argv else (
            "query" if argv[-1] in self.dependencies else (
                "compile" if "-o" in argv else Path(argv[0]).stem
            )
        )
        if self.timeout_at == action:
            raise subprocess.TimeoutExpired(argv, timeout, output="partial")
        if self.fail_at == action:
            return subprocess.CompletedProcess(argv, 7, "partial stdout", "intentional failure")
        stdout = ""
        if action == "version":
            stdout = "unrelated compiler\n" if self.wrong_version else "GNU Fortran fake test\n"
        elif action == "query":
            path = self.dependencies[argv[-1]]
            stdout = "missing-dependency" if self.missing_dependency else (
                path.name if self.bare_dependency else str(path)
            )
        elif action == "compile":
            (cwd / argv[argv.index("-o") + 1]).write_bytes(b"synthetic executable")
        elif action == "edconv":
            size = 122399 if self.wrong_layout == "UEDAT" else 122400
            (cwd / "UEDAT").write_bytes(b"0" * size)
        elif action == "convert":
            size = 74771 if self.wrong_layout == "UCOMP" else 74772
            (cwd / "UCOMP").write_bytes(b"0" * size)
        elif action == "estar":
            assert input == self.runner.STDIN == "1\n1\n13\n1\nal.out\n"
            data = _native_output(energy_change=self.changed_grid, short=self.short_grid)
            if cwd.name == "O2" and self.o2_whitespace_difference:
                data = b"\n" + data
            (cwd / "al.out").write_bytes(data)
            targets = {
                "source": cwd / "ESTAR.f", "archive": self.archive, "csv": self.csv,
                "compiler": self.compiler, "dependency": next(iter(self.dependencies.values())),
                "implementation": self.root / self.runner.IMPLEMENTATION_PATHS[0],
            }
            if self.change is not None:
                targets[self.change].write_bytes(b"changed during native execution")
            if self.precreate_report:
                (self.output / "report.json").write_bytes(b"must not overwrite")
        else:
            raise AssertionError(f"Unexpected fake subprocess: {argv}")
        return subprocess.CompletedProcess(argv, 0, stdout, "retained compiler warning")


@pytest.fixture
def harness(runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BuildHarness:
    return BuildHarness(runner, tmp_path, monkeypatch)


def test_known_upstream_pins_are_explicit(runner: ModuleType) -> None:
    assert runner.ARCHIVE_BYTES == 444183
    assert runner.ARCHIVE_SHA256 == "b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6"
    assert set(runner.SOURCE_HASHES) == {
        "ESTAR.f", "EDCONV.f", "CONVERT.f", "FCOMP", "FEDAT", "ENG.ELE",
    }
    assert runner.SOURCE_HASHES["ENG.ELE"] == (
        "06f8c6d355666724cfe9afd473ac041d97ea3d231b79c0217ed11d0e8148ac4f"
    )


def test_archive_returns_exact_unchanged_source_bytes(harness: BuildHarness) -> None:
    assert harness.runner.archive_members(harness.archive.read_bytes()) == harness.members


@pytest.mark.parametrize("change", ["size", "sha"])
def test_archive_identity_mismatch_rejected(harness: BuildHarness, change: str) -> None:
    data = harness.archive.read_bytes()
    data = data + b"x" if change == "size" else bytes([data[0] ^ 1]) + data[1:]
    with pytest.raises(ValueError, match="archive identity mismatch"):
        harness.runner.archive_members(data)


def test_member_hash_checked_separately(harness: BuildHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    harness.members["ESTAR.f"] = b"modified even though outer archive is repinned"
    harness.rebuild_archive(monkeypatch)
    with pytest.raises(ValueError, match="source member changed"):
        harness.build()
    assert not harness.output.exists()


@pytest.mark.parametrize("duplicate_name", ["ESTAR/ESTAR.f", "unused/path"])
def test_duplicate_archive_members_rejected(
    harness: BuildHarness, monkeypatch: pytest.MonkeyPatch, duplicate_name: str,
) -> None:
    entries = [("ESTAR/" + name, data, tarfile.REGTYPE)
               for name, data in harness.members.items()]
    entries.extend([(duplicate_name, b"first", tarfile.REGTYPE),
                    (duplicate_name, b"second", tarfile.REGTYPE)])
    harness.rebuild_archive(monkeypatch, entries)
    with pytest.raises(ValueError, match="Duplicate"):
        harness.build()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_source_symlink_hardlink_or_directory_rejected(
    harness: BuildHarness, monkeypatch: pytest.MonkeyPatch, kind: bytes,
) -> None:
    entries = [("ESTAR/" + name, data, kind if name == "ESTAR.f" else tarfile.REGTYPE)
               for name, data in harness.members.items()]
    harness.rebuild_archive(monkeypatch, entries)
    with pytest.raises(ValueError, match="regular ESTAR source"):
        harness.build()


def test_both_optimization_levels_use_unmodified_sources_and_exact_grid(
    harness: BuildHarness,
) -> None:
    report = harness.build()
    assert report["schema_version"] == "radar.nist_estar_native_reproduction.v1"
    assert report["all_primary_values_exact"]
    assert report["all_csv_values_exact"]
    assert report["native_O0_O2_output_bytes_identical"]
    assert not report["production_data_modified"]
    assert not report["release_status_changed_automatically"]
    assert not report["stage4_complete"]
    assert report["implementation_sha256_before"] == report["implementation_sha256_after"]
    assert set(report["compiler_dependencies"]) == set(harness.dependencies)
    for build, level in zip(report["builds"], ("-O0", "-O2"), strict=True):
        assert build["optimization"] == level
        assert build["flags"] == [
            "-std=legacy", level, "-ffixed-line-length-72", "-static-libgfortran",
            "-B" + str(harness.prefix.resolve()) + "/",
        ]
        parsed = build["parsed_output"]
        assert len(parsed["rows"]) == 81
        assert tuple(row["energy_mev"] for row in parsed["rows"]) == _energies()
        assert not parsed["physical_accuracy_verified"]
        assert not parsed["normative_export_allowed"]
        assert not parsed["fortran_execution_verified"]  # reader alone does not certify execution
        assert build["comparison"]["exact_primary_match"]
        assert build["comparison"]["primary_differences"] == ()
        for name, data in harness.members.items():
            assert (harness.output / level[1:] / name).read_bytes() == data
    assert len(harness.calls) == 18  # version+five dependency queries+two*(three builds+three runs)
    compile_calls = [call for call in harness.calls if "-o" in call["argv"]]
    assert len(compile_calls) == 6
    assert all("-ffast-math" not in call["argv"] for call in compile_calls)
    logs = sorted(harness.output.glob("command_*.json"))
    assert len(logs) == len(harness.calls)
    assert all(json.loads(p.read_text("utf-8"))["stderr"] == "retained compiler warning"
               for p in logs)
    serialized = json.loads((harness.output / "report.json").read_text("utf-8"))
    assert serialized["builds"][0]["parsed_output"]["rows"][0]["energy_mev"] == "0.01000"


def test_selected_dependency_identities_are_recorded_not_a_hermetic_claim(
    harness: BuildHarness,
) -> None:
    report = harness.build()
    for query, path in harness.dependencies.items():
        assert report["compiler_dependencies"][query] == {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": _sha(path.read_bytes()),
        }
    # The runner captures these five artifacts, not every system library,
    # startup object, compiler environment variable or build-host property.
    assert "hermetic_build_verified" not in report or not report["hermetic_build_verified"]
    assert any("not a hermetic" in text for text in report["limitations"])
    assert "selected electron Al CSV only" in report["production_data_check_scope"]


def test_tiny_primary_difference_is_not_hidden_by_float_tolerance(harness: BuildHarness) -> None:
    harness.csv.write_bytes(_csv(primary_difference=True))
    report = harness.build()
    assert not report["all_primary_values_exact"]
    assert not report["all_csv_values_exact"]
    for build in report["builds"]:
        assert not build["comparison"]["exact_primary_match"]
        assert len(build["comparison"]["primary_differences"]) == 1
        assert (harness.output / build["optimization"][1:] / "al.out").exists()
    assert (harness.output / "report.json").exists()


@pytest.mark.parametrize("change, primary, mg, internal", [
    ("mg_only", True, False, False),
    ("range_only", False, True, False),
    ("range_and_mg", False, False, True),
])
def test_mg_exactness_and_internal_consistency_are_not_silently_ignored(
    harness: BuildHarness, change: str, primary: bool, mg: bool, internal: bool,
) -> None:
    lines = _csv().decode("ascii").splitlines()
    fields = lines[1].split(",")
    if change in ("range_only", "range_and_mg"):
        fields[2] = "0.01001"
    if change in ("mg_only", "range_and_mg"):
        fields[3] = "10.01"
    lines[1] = ",".join(fields)
    harness.csv.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    report = harness.build()
    assert report["all_primary_values_exact"] is primary
    assert not report["all_csv_values_exact"]
    for build in report["builds"]:
        comparison = build["comparison"]
        assert comparison["exact_primary_match"] is primary
        assert comparison["exact_range_mg_match"] is mg
        assert comparison["range_mg_internal_consistent"] is internal


@pytest.mark.parametrize("bad_grid", ["changed", "short"])
def test_native_source_grid_is_strict_not_interpolated(harness: BuildHarness, bad_grid: str) -> None:
    harness.changed_grid = bad_grid == "changed"
    harness.short_grid = bad_grid == "short"
    with pytest.raises(ValueError, match="81-node|row/header count"):
        harness.build()
    assert (harness.output / "O0/al.out").exists()
    assert not (harness.output / "report.json").exists()


def test_output_byte_difference_is_distinct_from_primary_exactness(harness: BuildHarness) -> None:
    harness.o2_whitespace_difference = True
    report = harness.build()
    assert report["all_primary_values_exact"]
    assert not report["native_O0_O2_output_bytes_identical"]


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_output_must_be_new_before_any_subprocess(harness: BuildHarness, kind: str) -> None:
    if kind == "directory":
        harness.output.mkdir()
    else:
        harness.output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        harness.build()
    assert harness.calls == []
    if kind == "file":
        assert harness.output.read_bytes() == b"original"


def test_report_write_is_exclusive_even_if_created_during_execution(harness: BuildHarness) -> None:
    harness.precreate_report = True
    with pytest.raises(FileExistsError):
        harness.build()
    assert (harness.output / "report.json").read_bytes() == b"must not overwrite"


@pytest.mark.parametrize("change", ["source", "archive", "csv", "compiler", "dependency",
                                    "implementation"])
def test_before_after_pins_detect_mutation(harness: BuildHarness, change: str) -> None:
    harness.change = change
    with pytest.raises(ValueError, match="changed during execution"):
        harness.build()
    assert not (harness.output / "report.json").exists()


@pytest.mark.parametrize("action", ["version", "query", "compile", "edconv", "convert", "estar"])
def test_nonzero_subprocess_exit_is_logged_and_raised(harness: BuildHarness, action: str) -> None:
    harness.fail_at = action
    with pytest.raises(subprocess.CalledProcessError) as caught:
        harness.build()
    assert caught.value.returncode == 7
    last = sorted(harness.output.glob("command_*.json"))[-1]
    record = json.loads(last.read_text("utf-8"))
    assert record["returncode"] == 7
    assert record["stdout"] == "partial stdout"
    assert record["stderr"] == "intentional failure"
    assert not (harness.output / "report.json").exists()


def test_subprocess_timeout_propagates_without_success_report(harness: BuildHarness) -> None:
    harness.timeout_at = "compile"
    with pytest.raises(subprocess.TimeoutExpired):
        harness.build()
    assert not (harness.output / "report.json").exists()


@pytest.mark.parametrize("name", ["UEDAT", "UCOMP"])
def test_native_direct_access_record_sizes_are_checked(harness: BuildHarness, name: str) -> None:
    harness.wrong_layout = name
    with pytest.raises(ValueError, match="direct-access record layout"):
        harness.build()


def test_unrecognized_compiler_rejected(harness: BuildHarness) -> None:
    harness.wrong_version = True
    with pytest.raises(ValueError, match="GNU Fortran"):
        harness.build()
    assert len(harness.calls) == 1


def test_unidentified_selected_dependency_rejected(harness: BuildHarness) -> None:
    harness.missing_dependency = True
    with pytest.raises(ValueError, match="Unable to identify compiler dependency"):
        harness.build()


def test_bare_dependency_name_uses_resolved_search_path(
    harness: BuildHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.bare_dependency = True
    by_name = {p.name: str(p) for p in harness.dependencies.values()}
    monkeypatch.setattr(harness.runner.shutil, "which", lambda name: by_name.get(name))
    assert harness.build()["all_primary_values_exact"]


def test_compiler_directory_is_not_a_compiler(harness: BuildHarness) -> None:
    harness.compiler = harness.prefix
    with pytest.raises(ValueError, match="regular Fortran compiler"):
        harness.build()
    assert harness.calls == []


def test_compiler_prefix_file_is_not_a_directory(harness: BuildHarness) -> None:
    harness.prefix = harness.compiler
    with pytest.raises(ValueError, match="prefixes must be existing directories"):
        harness.build()
    assert harness.calls == []


def test_write_new_and_json_are_strict(runner: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "exclusive.json"
    runner._write_new(target, runner._json({"exact": Decimal(".00000000000001")}))
    assert json.loads(target.read_text("utf-8")) == {"exact": "1E-14"}
    with pytest.raises(FileExistsError):
        runner._write_new(target, "replace")
    with pytest.raises(ValueError):
        runner._json({"bad": float("nan")})
    with pytest.raises(TypeError):
        runner._json({"bad": object()})


@pytest.mark.parametrize("primary_exact, csv_exact, exit_code", [
    (True, True, 0), (True, False, 1), (False, False, 1),
])
def test_cli_exit_code_requires_all_csv_values_not_just_primary_fields(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    primary_exact: bool, csv_exact: bool, exit_code: int,
) -> None:
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"all_primary_values_exact": primary_exact, "all_csv_values_exact": csv_exact,
                "native_O0_O2_output_bytes_identical": True}

    monkeypatch.setattr(runner, "run_build", fake_build)
    monkeypatch.setattr(runner.sys, "argv", [
        "reproduce_nist_estar.py", "--compiler", str(tmp_path / "compiler"),
        "--output", str(tmp_path / "out"), "--compiler-prefix", str(tmp_path / "prefix"),
    ])
    assert runner.main() == exit_code
    assert captured["compiler"] == tmp_path / "compiler"
    assert captured["output"] == tmp_path / "out"
    assert captured["compiler_prefixes"] == (tmp_path / "prefix",)
