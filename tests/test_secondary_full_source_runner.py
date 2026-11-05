"""Mocked bounded full-source runner checks; no actual TENDL integration."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import fsum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.core.spectra import Spectrum1D, TransformationStep
from radar.core.types import (
    AssessmentMode,
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import log_bin_edges_from_centers


@pytest.fixture
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_secondary_full_source.py"
    spec = spec_from_file_location("independent_full_source_runner", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_sources(nodes: int) -> dict[str, Spectrum1D]:
    grid = tuple(10.0 ** (-1.0 + 6.0 * index / (nodes - 1)) for index in range(nodes))
    return {
        mode: Spectrum1D(
            x=grid,
            y=(1.0 if mode == "peak" else 7.0,) * nodes,
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUX if mode == "peak" else Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX if mode == "peak"
            else SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.GCR,
            model="mock_full_source",
            assessment_mode=AssessmentMode(mode),
            temporal_basis=TemporalBasis.SOURCE_MAXIMUM if mode == "peak"
            else TemporalBasis.MISSION_TOTAL,
            mission_duration_s=None if mode == "peak" else 157788000.0,
            location=SpectrumLocation.ON_ORBIT,
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=("mock H-only incomplete ion roster",),
        )
        for mode in ("peak", "cumulative")
    }


@pytest.fixture
def mocked_api(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"source_requests": [], "operator_calls": []}

    def sources(nodes: int) -> tuple[dict[str, Spectrum1D], dict[str, Any]]:
        state["source_requests"].append(nodes)
        state["sources"] = _synthetic_sources(nodes)
        return state["sources"], {"solar_months": 60, "mock_source_provenance": "unmodified"}

    kernel = SimpleNamespace(
        products=tuple(SimpleNamespace(mt=5, product_index=index) for index in range(10)),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=1.0, maximum_mev=200.0, dataset_ref="mock_support",
        ),
    )
    stopping = object()
    state.update(kernel=kernel, stopping=stopping)

    def response(**kwargs: Any) -> Any:
        return runner.secondary._SingleIncidentResponse(
            counts=(1.0,), reaction_energies=(10.0,), attenuation_energies=(10.0,),
            stopped_below_range_support=False, range_domain_mappings=0,
            production_samples=1, attenuation_missing=False,
        )

    def calculate(**kwargs: Any) -> Spectrum1D:
        state["operator_calls"].append(kwargs)
        source = kwargs["incident_spectrum"]
        for _ in range(4):
            runner.secondary._single_incident_secondary_response(output_centers_mev=(source.x[0],))
        return replace(
            source,
            y=(2.0 * source.y[0],) + (0.0,) * (len(source.x) - 1),
            ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
            bin_edges=log_bin_edges_from_centers(source.x),
            incomplete_reasons=(*source.incomplete_reasons,
                               "mock budget-unresolved intervals", "mock unsupported contribution"),
            transformation_chain=(TransformationStep(
                "mock_secondary", "9_mock", parameters=(
                    ("incident_quadrature_response_evaluations", "4"),
                    ("incident_quadrature_status", "incomplete"),
                    ("incident_quadrature_supported_status", "incomplete"),
                    ("incident_quadrature_estimated_absolute_error", "unavailable"),
                    ("incident_quadrature_unresolved_intervals", "9999"),
                    ("prequadrature_omitted_unsupported_base_intervals", "7"),
                ),
            ),),
        )

    monkeypatch.setattr(runner, "_sources", sources)
    monkeypatch.setattr(runner, "load_normative_proton_al_range_table", lambda: stopping)
    monkeypatch.setattr(runner, "load_normative_secondary_proton_kernel", lambda: kernel)
    monkeypatch.setattr(runner.secondary, "_single_incident_secondary_response", response)
    monkeypatch.setattr(runner.secondary, "calculate_secondary_proton_spectrum_through_al", calculate)
    monkeypatch.setattr(runner, "_input_hashes", lambda: {"mock-input": "unchanged"})
    state["original_response"] = response
    return state


def test_defaults_are_explicit_bounded_peak_with_required_new_output(runner: ModuleType) -> None:
    args = runner._parser().parse_args(["--thickness", "0.01", "--output", "new.json"])
    assert (args.grid, args.mode, args.depth_steps, args.max_response_evaluations) == (
        81, "peak", 1, 4096,
    )
    options = runner._incident_options(args)
    assert options.partition_strategy == "per_output_bin_active_cells"
    assert options.prequadrature_screening and options.reuse_birth_survival_partition
    assert options.relative_tolerance == 0.001


@pytest.mark.parametrize("mode", ["peak", "cumulative"])
@pytest.mark.parametrize("nodes", [81, 161])
def test_full_source_and_time_contract_are_forwarded_without_trimming_or_rescaling(
    runner: ModuleType, mocked_api: dict[str, Any], tmp_path: Path,
    capsys: pytest.CaptureFixture[str], mode: str, nodes: int,
) -> None:
    output = tmp_path / "nested" / "partial.json"
    runner.main([
        "--grid", str(nodes), "--mode", mode, "--thickness", "0.01", "--depth-steps", "2",
        "--max-response-evaluations", "4", "--output", str(output),
    ])
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text(encoding="utf-8"))
    case = report["calculation"]
    source = mocked_api["sources"][mode]
    call = mocked_api["operator_calls"][0]
    assert mocked_api["source_requests"] == [nodes]
    assert call["incident_spectrum"] is source
    assert source.x[-1] > mocked_api["kernel"].incident_energy_support.maximum_mev
    assert source.y[-1] > 0  # Unknown high-energy source is still present.
    assert call["kernel"] is mocked_api["kernel"]
    assert call["stopping_table"] is mocked_api["stopping"]
    assert call["depth_steps"] == 2 and call["survival_integration_steps"] == 64
    assert call["incident_quadrature"].max_response_evaluations == 4
    assert call["incident_quadrature"].relative_tolerance == 0.001
    assert call["incident_quadrature"].prequadrature_screening is True
    assert call["incident_quadrature"].reuse_birth_survival_partition is True
    assert call["depth_quadrature"].max_depth_knots == 8192
    assert call["depth_quadrature"].max_coupled_incident_knots == 65536
    assert call["depth_quadrature"].max_active_cell_records == 32768
    assert case["input_spectrum"]["x"] == list(source.x)
    assert case["input_spectrum"]["y"] == list(source.y)
    assert case["output_spectrum"]["y_unit"] == source.y_unit.value
    assert case["output_spectrum"]["mission_duration_s"] == source.mission_duration_s
    assert case["output_ordinates"][0] == 2.0 * source.y[0]
    assert case["implicit_time_factor_applied"] is False
    assert case["source_generation"]["solar_months"] == 60
    assert case["retained_kernel_product_count"] == 10
    assert len(case["retained_kernel_products"]) == 10
    assert case["retained_diagnostic_count_sum"] == fsum(case["output_counts_by_bin"])
    assert "total_count" not in case
    assert "unknown, not zero" in case["retained_diagnostic_count_scope"]
    assert case["response_budget_reached"] is True
    assert case["observed_response_calls"] == case["observed_response_bin_calls"] == 4
    assert case["observed_response_calls_by_output_bin"] == [4] + [0] * (nodes - 1)
    assert len(case["single_response_seconds_by_output_bin"]) == nodes
    assert case["parameters"]["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert case["parameters"]["incident_quadrature_unresolved_intervals"] == "9999"
    assert case["completeness"] == "incomplete"
    assert source.incomplete_reasons[0] in case["incomplete_reasons"]
    assert report["file_sha256_at_start"] == report["file_sha256_at_end"]
    assert report["input_hashes_consistent"] is True
    assert report["normative_certification"] is case["normative_certification"] is False
    assert report["full_spectrum_convergence_claim"] is False
    assert runner.secondary._single_incident_secondary_response is mocked_api["original_response"]


@pytest.mark.parametrize("option,value", [
    ("--thickness", "0"), ("--thickness", "-1"), ("--thickness", "nan"),
    ("--thickness", "inf"), ("--depth-steps", "0"), ("--depth-steps", "-2"),
    ("--max-response-evaluations", "3"), ("--grid", "80"), ("--mode", "mean"),
])
def test_invalid_options_fail_before_hashing_or_calculation(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, option: str, value: str,
    tmp_path: Path,
) -> None:
    def forbidden() -> None:
        pytest.fail("Invalid configuration must not start data access or calculation.")

    monkeypatch.setattr(runner, "_input_hashes", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(["--thickness", "0.01", "--output", str(tmp_path / "unused.json"), option, value])
    assert error.value.code == 2


def test_existing_output_is_not_overwritten_before_input_access(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(runner, "_input_hashes", lambda: pytest.fail("Must not read inputs"))
    with pytest.raises(SystemExit):
        runner.main(["--thickness", "0.01", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "preserve"


def test_pinned_input_cannot_be_an_output(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_input_hashes", lambda: pytest.fail("Must not read inputs"))
    with pytest.raises(SystemExit):
        runner.main([
            "--thickness", "0.01", "--output",
            str(runner._PROJECT_ROOT / "scripts/validate_secondary_full_source.py"),
        ])


def test_hash_drift_refuses_publication(
    runner: ModuleType, mocked_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    hashes = iter(({"input": "before"}, {"input": "after"}))
    monkeypatch.setattr(runner, "_input_hashes", lambda: next(hashes))
    output = tmp_path / "changed.json"
    with pytest.raises(RuntimeError, match="changed during"):
        runner.main(["--thickness", "0.01", "--output", str(output)])
    assert not output.exists()


def test_response_hook_is_restored_if_operator_raises(
    runner: ModuleType, mocked_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def failed_operator(**kwargs: Any) -> None:
        raise RuntimeError("mock operator failed")

    monkeypatch.setattr(runner.secondary, "calculate_secondary_proton_spectrum_through_al",
                        failed_operator)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="mock operator failed"):
        runner.main(["--thickness", "0.01", "--output", str(output)])
    assert not output.exists()
    assert runner.secondary._single_incident_secondary_response is mocked_api["original_response"]


@pytest.mark.parametrize("mutation", ["warning_removed", "time_changed", "counter_changed"])
def test_inconsistent_operator_output_is_not_published(
    runner: ModuleType, mocked_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, mutation: str,
) -> None:
    original = runner.secondary.calculate_secondary_proton_spectrum_through_al

    def inconsistent(**kwargs: Any) -> Spectrum1D:
        result = original(**kwargs)
        if mutation == "warning_removed":
            return replace(result, incomplete_reasons=("only another warning",))
        if mutation == "time_changed":
            return replace(result, mission_duration_s=1.0)
        step = result.transformation_chain[-1]
        parameters = tuple((key, "5" if key == "incident_quadrature_response_evaluations" else value)
                           for key, value in step.parameters)
        return replace(result, transformation_chain=(replace(step, parameters=parameters),))

    monkeypatch.setattr(runner.secondary, "calculate_secondary_proton_spectrum_through_al",
                        inconsistent)
    output = tmp_path / "inconsistent.json"
    with pytest.raises(RuntimeError):
        runner.main(["--thickness", "0.01", "--output", str(output)])
    assert not output.exists()


def test_existing_preflight_source_helper_is_reused_without_options(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (_synthetic_sources(161), {"original_provenance": True})
    requests = []

    def sources(nodes: int) -> Any:
        requests.append(nodes)
        return expected

    monkeypatch.setattr(runner._PREFLIGHT, "_sources", sources)
    assert runner._sources(161) is expected
    assert requests == [161]


def test_full_source_hash_registry_extends_preflight_and_pins_this_runner(runner: ModuleType) -> None:
    assert set(runner._PREFLIGHT._FILES).issubset(runner._INPUT_FILES)
    assert "scripts/validate_secondary_full_source.py" in runner._INPUT_FILES
    assert "src/radar/validation/stage4_spectra.py" in runner._INPUT_FILES
    assert len(runner._INPUT_FILES) == len(set(runner._INPUT_FILES))
    assert all((runner._PROJECT_ROOT / name).is_file() for name in runner._INPUT_FILES)


def test_nan_metadata_cannot_be_published(
    runner: ModuleType, mocked_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = runner._calculate

    def nonfinite(args: Any) -> dict[str, Any]:
        return {**original(args), "mock_nonfinite": float("nan")}

    monkeypatch.setattr(runner, "_calculate", nonfinite)
    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        runner.main(["--thickness", "0.01", "--output", str(output)])
    assert not output.exists()


def test_output_created_during_calculation_is_not_overwritten(
    runner: ModuleType, mocked_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = runner._calculate
    output = tmp_path / "race.json"

    def racing_calculation(args: Any) -> dict[str, Any]:
        result = original(args)
        output.write_text("created by another actor", encoding="utf-8")
        return result

    monkeypatch.setattr(runner, "_calculate", racing_calculation)
    with pytest.raises(FileExistsError):
        runner.main(["--thickness", "0.01", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "created by another actor"
