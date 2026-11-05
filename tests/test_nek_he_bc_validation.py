import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, fields, replace
from math import exp
from pathlib import Path

import pytest

import radar.gcr.nek_he_bc_validation as validation
from radar.gcr.components import ost_nek_differential_flux_directional_cm2_s_sr_mev
from radar.gcr import source_spectra
from radar.gcr.nek_he_bc_validation import (
    NEK_HE_BC_COMPARISON_ENERGIES_MEV_PER_NUCLEON,
    NEK_HE_COMPONENT_ID,
    NEK_HE_DIRECTIONAL_FLUX_UNITS,
    CosradNekHeReferenceMetadata,
    CosradNekHeReferencePoint,
    CosradNekHeReferenceSet,
    calculate_nek_he_bc_candidate_grid,
    evaluate_nek_he_bc_candidates,
    load_cosrad_nek_he_reference,
)
from radar.gcr.parameters import (
    NekParameters,
    OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE,
    OST_NEK_HE_BC_PRINTED,
    OST_NEK_HE_BC_RELEASE_ELIGIBLE,
    OST_NEK_HE_BC_USED,
    OST_NEK_HE_BC_VALIDATION_STATUS,
)


def _metadata(*, source_artifact: Path) -> CosradNekHeReferenceMetadata:
    return CosradNekHeReferenceMetadata(
        cosrad_version="4.3.1.1",
        source_artifact=str(source_artifact),
        source_sha256=hashlib.sha256(source_artifact.read_bytes()).hexdigest(),
        calculation_configuration="NEK He; W=0,30,60,100; E=10,30,100,300 MeV/n",
        exported_at_utc="2026-08-11T09:00:00+00:00",
        component=NEK_HE_COMPONENT_ID,
        units=NEK_HE_DIRECTIONAL_FLUX_UNITS,
    )


def test_candidate_grid_uses_the_isolated_ost_nek_he_formula() -> None:
    wolf_numbers = (0.0, 30.0, 60.0, 100.0)

    points = calculate_nek_he_bc_candidate_grid(wolf_numbers=wolf_numbers)

    assert len(points) == len(wolf_numbers) * 4
    assert {point.energy_mev_per_nucleon for point in points} == set(
        NEK_HE_BC_COMPARISON_ENERGIES_MEV_PER_NUCLEON,
    )
    for point in points:
        common_factor = 3.6e-2 * point.energy_mev_per_nucleon ** (-2.45)
        expected_printed = common_factor * exp(
            OST_NEK_HE_BC_PRINTED * point.wolf_number,
        )
        expected_alternative = common_factor * exp(
            OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE * point.wolf_number,
        )
        expected_ratio = exp(
            (OST_NEK_HE_BC_PRINTED - OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE)
            * point.wolf_number,
        )

        assert point.ost_printed_flux == pytest.approx(expected_printed)
        assert point.engineering_alternative_flux == pytest.approx(expected_alternative)
        assert point.printed_to_alternative_ratio == pytest.approx(expected_ratio)
        assert point.alternative_to_printed_ratio == pytest.approx(1.0 / expected_ratio)


def test_bench_never_calls_total_gcr_spectrum_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar_calls: list[tuple[int, float]] = []
    original_scalar = ost_nek_differential_flux_directional_cm2_s_sr_mev

    def forbidden_total(*args: object, **kwargs: object) -> tuple[float, ...]:
        del args, kwargs
        raise AssertionError("total GCR spectrum must not be used by the NEK-He bench")

    def observed_scalar(
        *,
        kinetic_energy_mev_per_nucleon: float,
        parameters: NekParameters,
        wolf_number: float,
    ) -> float:
        scalar_calls.append((parameters.z, parameters.bc))
        return original_scalar(
            kinetic_energy_mev_per_nucleon=kinetic_energy_mev_per_nucleon,
            parameters=parameters,
            wolf_number=wolf_number,
        )

    monkeypatch.setattr(source_spectra, "_total_values", forbidden_total)
    monkeypatch.setattr(
        validation,
        "ost_nek_differential_flux_directional_cm2_s_sr_mev",
        observed_scalar,
    )

    report = evaluate_nek_he_bc_candidates(wolf_numbers=(10.0, 80.0))

    assert report.component == NEK_HE_COMPONENT_ID
    assert len(scalar_calls) == 2 * 4 * 2
    assert {z for z, _ in scalar_calls} == {2}
    assert {coefficient for _, coefficient in scalar_calls} == {
        OST_NEK_HE_BC_PRINTED,
        OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE,
    }


def test_report_makes_no_selection_without_cosrad_reference() -> None:
    report = evaluate_nek_he_bc_candidates(wolf_numbers=(0.0, 50.0, 100.0))

    assert report.validation_status == "pending_cosrad_comparison"
    assert report.cosrad_comparison is None
    assert OST_NEK_HE_BC_USED == pytest.approx(OST_NEK_HE_BC_PRINTED)
    _assert_comparison_never_authorizes_acceptance(report)


def _assert_comparison_never_authorizes_acceptance(
    report: validation.NekHeBcValidationReport,
) -> None:
    assert report.validation_status == "pending_cosrad_comparison"
    assert OST_NEK_HE_BC_VALIDATION_STATUS == "pending_cosrad_comparison"
    assert OST_NEK_HE_BC_USED == OST_NEK_HE_BC_PRINTED == 6.6e-2
    assert OST_NEK_HE_BC_RELEASE_ELIGIBLE is False
    objects: list[validation.NekHeBcValidationReport | validation.CosradNekHeComparison] = [
        report,
    ]
    if report.cosrad_comparison is not None:
        objects.append(report.cosrad_comparison)
        assert report.cosrad_comparison.best_candidate_meaning == "closest_match_only"
    for result in objects:
        serialized = json.loads(json.dumps(asdict(result)))
        assert serialized["evidence_role"] == "independent_model_comparison_only"
        assert serialized["normative_qualification"] == "not_established"
        assert serialized["coefficient_change_authorized"] is False


def _synthetic_reference(
    tmp_path: Path,
    *,
    points: tuple[validation.NekHeBcComparisonPoint, ...],
    candidate: validation.NekHeBcCandidate,
    scale: float = 1.0,
) -> CosradNekHeReferenceSet:
    artifact_path = tmp_path / "synthetic_comparison_fixture.txt"
    artifact_path.write_text("Synthetic unit-test data, not a COSRAD run.\n", encoding="utf-8")
    return CosradNekHeReferenceSet(
        metadata=_metadata(source_artifact=artifact_path),
        points=tuple(
            CosradNekHeReferencePoint(
                wolf_number=point.wolf_number,
                energy_mev_per_nucleon=point.energy_mev_per_nucleon,
                flux=scale * (
                    point.ost_printed_flux
                    if candidate == "ost_printed"
                    else point.engineering_alternative_flux
                ),
            )
            for point in points
        ),
        source_artifact_path=artifact_path,
    )


@pytest.mark.parametrize("candidate", ["ost_printed", "engineering_alternative"])
def test_exact_match_is_only_a_ranking_not_coefficient_approval(
    tmp_path: Path,
    candidate: validation.NekHeBcCandidate,
) -> None:
    wolf_numbers = (30.0, 100.0)
    points = calculate_nek_he_bc_candidate_grid(wolf_numbers=wolf_numbers)
    report = evaluate_nek_he_bc_candidates(
        wolf_numbers=wolf_numbers,
        cosrad_reference=_synthetic_reference(tmp_path, points=points, candidate=candidate),
    )
    comparison = report.cosrad_comparison
    assert comparison is not None
    assert comparison.selection_status == "selected"
    assert comparison.best_candidate == candidate
    summary = (
        comparison.ost_printed
        if candidate == "ost_printed"
        else comparison.engineering_alternative
    )
    assert summary.root_mean_square_relative_error == 0.0
    _assert_comparison_never_authorizes_acceptance(report)


def test_zero_wolf_tie_does_not_authorize_either_coefficient(tmp_path: Path) -> None:
    points = calculate_nek_he_bc_candidate_grid(wolf_numbers=(0.0,))
    report = evaluate_nek_he_bc_candidates(
        wolf_numbers=(0.0,),
        cosrad_reference=_synthetic_reference(
            tmp_path, points=points, candidate="ost_printed",
        ),
    )
    comparison = report.cosrad_comparison
    assert comparison is not None
    assert comparison.selection_status == "tie"
    assert comparison.best_candidate is None
    assert comparison.ost_printed.root_mean_square_relative_error == 0.0
    assert comparison.engineering_alternative.root_mean_square_relative_error == 0.0
    _assert_comparison_never_authorizes_acceptance(report)


def test_closest_candidate_with_poor_absolute_fit_is_not_accepted(tmp_path: Path) -> None:
    points = calculate_nek_he_bc_candidate_grid(wolf_numbers=(30.0, 100.0))
    report = evaluate_nek_he_bc_candidates(
        wolf_numbers=(30.0, 100.0),
        cosrad_reference=_synthetic_reference(
            tmp_path, points=points, candidate="ost_printed", scale=100.0,
        ),
    )
    comparison = report.cosrad_comparison
    assert comparison is not None
    assert comparison.selection_status == "selected"
    assert comparison.best_candidate == "ost_printed"
    assert comparison.ost_printed.root_mean_square_relative_error > 0.98
    assert comparison.engineering_alternative.root_mean_square_relative_error > 0.98
    _assert_comparison_never_authorizes_acceptance(report)


def test_comparison_acceptance_metadata_cannot_be_overridden(tmp_path: Path) -> None:
    points = calculate_nek_he_bc_candidate_grid(wolf_numbers=(30.0,))
    report = evaluate_nek_he_bc_candidates(
        wolf_numbers=(30.0,),
        cosrad_reference=_synthetic_reference(
            tmp_path, points=points, candidate="ost_printed",
        ),
    )
    comparison = report.cosrad_comparison
    assert comparison is not None
    for result in (report, comparison):
        guarded_fields = {
            "evidence_role", "normative_qualification", "coefficient_change_authorized",
        }
        if isinstance(result, validation.CosradNekHeComparison):
            guarded_fields.add("best_candidate_meaning")
        for definition in fields(result):
            if definition.name in guarded_fields:
                assert definition.init is False
                # Python 3.13 raises TypeError; earlier supported versions use ValueError.
                with pytest.raises((TypeError, ValueError), match="init=False"):
                    replace(result, **{definition.name: "override"})
                with pytest.raises(FrozenInstanceError):
                    setattr(result, definition.name, "override")
    _assert_comparison_never_authorizes_acceptance(report)


def test_provenance_checked_cosrad_reference_can_select_lower_rms_error(
    tmp_path: Path,
) -> None:
    candidate_points = calculate_nek_he_bc_candidate_grid(wolf_numbers=(30.0, 100.0))
    artifact_path = tmp_path / "isolated_nek_he_export.csv"
    artifact_path.write_text("isolated COSRAD NEK-He export\n", encoding="utf-8")
    reference = CosradNekHeReferenceSet(
        metadata=_metadata(source_artifact=artifact_path),
        points=tuple(
            CosradNekHeReferencePoint(
                wolf_number=point.wolf_number,
                energy_mev_per_nucleon=point.energy_mev_per_nucleon,
                flux=point.engineering_alternative_flux,
            )
            for point in candidate_points
        ),
        source_artifact_path=artifact_path,
    )

    report = evaluate_nek_he_bc_candidates(
        wolf_numbers=(30.0, 100.0),
        cosrad_reference=reference,
    )

    comparison = report.cosrad_comparison
    assert comparison is not None
    assert comparison.selection_metric == "root_mean_square_relative_error"
    assert comparison.selection_status == "selected"
    assert comparison.best_candidate == "engineering_alternative"
    assert (
        comparison.engineering_alternative.root_mean_square_relative_error
        == pytest.approx(0.0)
    )
    assert comparison.ost_printed.root_mean_square_relative_error > 0.0


def test_cosrad_reference_must_cover_the_complete_requested_grid(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "isolated_nek_he_export.csv"
    artifact_path.write_text("isolated COSRAD NEK-He export\n", encoding="utf-8")
    reference = CosradNekHeReferenceSet(
        metadata=_metadata(source_artifact=artifact_path),
        points=(
            CosradNekHeReferencePoint(
                wolf_number=30.0,
                energy_mev_per_nucleon=10.0,
                flux=1.0,
            ),
        ),
        source_artifact_path=artifact_path,
    )

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_nek_he_bc_candidates(
            wolf_numbers=(30.0,),
            cosrad_reference=reference,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"source_sha256": "unknown"}, "SHA-256"),
        ({"exported_at_utc": "2026-08-11T09:00:00"}, "UTC offset"),
        ({"exported_at_utc": "2026-08-11T12:00:00+03:00"}, "expressed in UTC"),
        ({"component": "total_gcr"}, "total GCR spectra"),
        ({"units": "cm^-2 s^-1"}, "reference units"),
    ],
)
def test_cosrad_metadata_rejects_missing_or_non_component_provenance(
    override: dict[str, str],
    message: str,
) -> None:
    values = {
        "cosrad_version": "4.3.1.1",
        "source_artifact": "isolated_nek_he_export.csv",
        "source_sha256": "a" * 64,
        "calculation_configuration": "isolated NEK He",
        "exported_at_utc": "2026-08-11T09:00:00+00:00",
        "component": NEK_HE_COMPONENT_ID,
        "units": NEK_HE_DIRECTIONAL_FLUX_UNITS,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        CosradNekHeReferenceMetadata(**values)


@pytest.mark.parametrize("wolf_numbers", [(), (-1.0,), (float("nan"),), (10.0, 10.0)])
def test_candidate_grid_rejects_invalid_wolf_numbers(
    wolf_numbers: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="Wolf|At least"):
        calculate_nek_he_bc_candidate_grid(wolf_numbers=wolf_numbers)


def test_json_loader_requires_and_preserves_cosrad_provenance(tmp_path: Path) -> None:
    candidate_points = calculate_nek_he_bc_candidate_grid(wolf_numbers=(30.0,))
    artifact_path = tmp_path / "isolated_nek_he_export.csv"
    artifact_path.write_text("isolated COSRAD NEK-He export\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    payload = {
        "metadata": {
            "cosrad_version": "4.3.1.1",
            "source_artifact": "isolated_nek_he_export.csv",
            "source_sha256": artifact_sha256,
            "calculation_configuration": "NEK He; W=30",
            "exported_at_utc": "2026-08-11T09:00:00Z",
            "component": NEK_HE_COMPONENT_ID,
            "units": NEK_HE_DIRECTIONAL_FLUX_UNITS,
        },
        "points": [
            {
                "wolf_number": point.wolf_number,
                "energy_mev_per_nucleon": point.energy_mev_per_nucleon,
                "flux": point.ost_printed_flux,
            }
            for point in candidate_points
        ],
    }
    path = tmp_path / "cosrad_nek_he.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reference = load_cosrad_nek_he_reference(path)

    assert reference.metadata.cosrad_version == "4.3.1.1"
    assert reference.metadata.source_sha256 == artifact_sha256
    assert reference.source_artifact_path == artifact_path
    assert len(reference.points) == 4


def test_json_loader_rejects_source_artifact_digest_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "isolated_nek_he_export.csv"
    artifact_path.write_text("actual COSRAD export\n", encoding="utf-8")
    payload = {
        "metadata": {
            "cosrad_version": "4.3.1.1",
            "source_artifact": artifact_path.name,
            "source_sha256": "b" * 64,
            "calculation_configuration": "NEK He; W=30",
            "exported_at_utc": "2026-08-11T09:00:00Z",
            "component": NEK_HE_COMPONENT_ID,
            "units": NEK_HE_DIRECTIONAL_FLUX_UNITS,
        },
        "points": [
            {
                "wolf_number": 30.0,
                "energy_mev_per_nucleon": 10.0,
                "flux": 1.0,
            },
        ],
    }
    path = tmp_path / "cosrad_nek_he.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match metadata"):
        load_cosrad_nek_he_reference(path)
