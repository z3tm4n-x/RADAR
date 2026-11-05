"""Isolated comparison bench for the unresolved OST NEK helium ``b_c`` value.

The bench deliberately evaluates only the low-energy cosmic-ray (NEK) helium
component.  It does not call the main GCR, AKL or total-spectrum builders and
does not change the coefficient selected by the production model. COSRAD is
an independent model comparison, not a truth oracle. A closer match does not
establish physical accuracy, normative qualification or permission to change
the production coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isclose, isfinite, sqrt
from pathlib import Path
from typing import Final, Literal, cast

from radar.gcr.components import ost_nek_differential_flux_directional_cm2_s_sr_mev
from radar.gcr.parameters import (
    NekParameters,
    OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE,
    OST_NEK_HE_BC_PRINTED,
    OST_NEK_HE_BC_VALIDATION_STATUS,
    get_ost_nek_parameter,
)

NekHeBcCandidate = Literal["ost_printed", "engineering_alternative"]
NekHeBcSelectionStatus = Literal["selected", "tie"]

NEK_HE_COMPONENT_ID: Final[str] = "ost_nek_he_only"
NEK_HE_DIRECTIONAL_FLUX_UNITS: Final[str] = (
    "cm^-2 s^-1 sr^-1 (MeV/nucleon)^-1"
)
NEK_HE_BC_COMPARISON_ENERGIES_MEV_PER_NUCLEON: Final[tuple[float, ...]] = (
    10.0,
    30.0,
    100.0,
    300.0,
)
NEK_HE_BC_SELECTION_METRIC: Final[str] = "root_mean_square_relative_error"


def _require_non_empty(value: str, *, name: str) -> None:
    if not value.strip():
        msg = f"{name} must not be empty."
        raise ValueError(msg)


def _require_finite_non_negative(value: float, *, name: str) -> None:
    if not isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative."
        raise ValueError(msg)


def _require_finite_positive(value: float, *, name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive."
        raise ValueError(msg)


@dataclass(frozen=True)
class CosradNekHeReferenceMetadata:
    """Mandatory provenance for an external COSRAD NEK-He export."""

    cosrad_version: str
    source_artifact: str
    source_sha256: str
    calculation_configuration: str
    exported_at_utc: str
    component: str
    units: str

    def __post_init__(self) -> None:
        for name, value in (
            ("COSRAD version", self.cosrad_version),
            ("COSRAD source artifact", self.source_artifact),
            ("COSRAD calculation configuration", self.calculation_configuration),
            ("COSRAD export time", self.exported_at_utc),
        ):
            _require_non_empty(value, name=name)

        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            msg = "COSRAD source SHA-256 must contain exactly 64 hexadecimal characters."
            raise ValueError(msg)

        try:
            exported_at = datetime.fromisoformat(self.exported_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            msg = "COSRAD export time must be an ISO-8601 date-time."
            raise ValueError(msg) from exc

        utc_offset = exported_at.utcoffset()
        if exported_at.tzinfo is None or utc_offset is None:
            msg = "COSRAD export time must include an explicit UTC offset."
            raise ValueError(msg)

        if utc_offset.total_seconds() != 0.0:
            msg = "COSRAD export time must be expressed in UTC (+00:00 or Z)."
            raise ValueError(msg)

        if self.component != NEK_HE_COMPONENT_ID:
            msg = (
                "COSRAD reference must identify the isolated OST NEK helium component as "
                f"{NEK_HE_COMPONENT_ID!r}; total GCR spectra are not accepted."
            )
            raise ValueError(msg)

        if self.units != NEK_HE_DIRECTIONAL_FLUX_UNITS:
            msg = (
                "COSRAD reference units must be "
                f"{NEK_HE_DIRECTIONAL_FLUX_UNITS!r}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class CosradNekHeReferencePoint:
    """One isolated COSRAD NEK-He directional differential-flux value."""

    wolf_number: float
    energy_mev_per_nucleon: float
    flux: float

    def __post_init__(self) -> None:
        _require_finite_non_negative(self.wolf_number, name="COSRAD Wolf number")
        _require_finite_positive(
            self.energy_mev_per_nucleon,
            name="COSRAD energy per nucleon",
        )
        _require_finite_positive(self.flux, name="COSRAD NEK-He flux")


@dataclass(frozen=True)
class CosradNekHeReferenceSet:
    """COSRAD points plus the provenance required to use them for selection."""

    metadata: CosradNekHeReferenceMetadata
    points: tuple[CosradNekHeReferencePoint, ...]
    source_artifact_path: Path

    def __post_init__(self) -> None:
        if not self.source_artifact_path.is_file():
            msg = (
                "COSRAD source artifact does not exist or is not a regular file: "
                f"{self.source_artifact_path}."
            )
            raise ValueError(msg)

        with self.source_artifact_path.open("rb") as source_file:
            actual_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
        if actual_sha256 != self.metadata.source_sha256.lower():
            msg = (
                "COSRAD source artifact SHA-256 does not match metadata: "
                f"expected {self.metadata.source_sha256.lower()}, got {actual_sha256}."
            )
            raise ValueError(msg)

        if not self.points:
            msg = "COSRAD NEK-He reference set must contain at least one point."
            raise ValueError(msg)

        keys = tuple(
            (point.wolf_number, point.energy_mev_per_nucleon) for point in self.points
        )
        if len(set(keys)) != len(keys):
            msg = "COSRAD NEK-He reference points must have unique Wolf/energy pairs."
            raise ValueError(msg)


@dataclass(frozen=True)
class NekHeBcComparisonPoint:
    """Both candidate predictions at one Wolf/energy pair."""

    wolf_number: float
    energy_mev_per_nucleon: float
    ost_printed_flux: float
    engineering_alternative_flux: float
    printed_to_alternative_ratio: float
    alternative_to_printed_ratio: float


@dataclass(frozen=True)
class CosradNekHePointResidual:
    """Relative candidate errors against one COSRAD reference point."""

    wolf_number: float
    energy_mev_per_nucleon: float
    reference_flux: float
    ost_printed_relative_error: float
    engineering_alternative_relative_error: float


@dataclass(frozen=True)
class NekHeBcCandidateErrorSummary:
    """Aggregate relative errors for one candidate coefficient."""

    candidate: NekHeBcCandidate
    coefficient: float
    mean_absolute_relative_error: float
    root_mean_square_relative_error: float
    maximum_absolute_relative_error: float


@dataclass(frozen=True)
class CosradNekHeComparison:
    """Candidate ranking against a provenance-checked COSRAD data set.

    The compatibility fields ``selection_status`` and ``best_candidate``
    describe relative closeness only, with no absolute acceptance threshold.
    Even an exact match does not establish that either model is correct.
    """

    metadata: CosradNekHeReferenceMetadata
    residuals: tuple[CosradNekHePointResidual, ...]
    ost_printed: NekHeBcCandidateErrorSummary
    engineering_alternative: NekHeBcCandidateErrorSummary
    selection_metric: str
    selection_status: NekHeBcSelectionStatus
    best_candidate: NekHeBcCandidate | None
    evidence_role: str = field(default="independent_model_comparison_only", init=False)
    normative_qualification: str = field(default="not_established", init=False)
    coefficient_change_authorized: bool = field(default=False, init=False)
    best_candidate_meaning: str = field(default="closest_match_only", init=False)


@dataclass(frozen=True)
class NekHeBcValidationReport:
    """Complete isolated comparison report, optionally fitted to COSRAD."""

    component: str
    units: str
    validation_status: str
    energies_mev_per_nucleon: tuple[float, ...]
    wolf_numbers: tuple[float, ...]
    points: tuple[NekHeBcComparisonPoint, ...]
    cosrad_comparison: CosradNekHeComparison | None
    evidence_role: str = field(default="independent_model_comparison_only", init=False)
    normative_qualification: str = field(default="not_established", init=False)
    coefficient_change_authorized: bool = field(default=False, init=False)


def _nek_he_parameters_with_bc(coefficient: float) -> NekParameters:
    base = get_ost_nek_parameter(2)
    if base is None:  # pragma: no cover - guarded by the normative parameter table
        msg = "OST NEK helium parameters are missing."
        raise RuntimeError(msg)

    return NekParameters(
        z=base.z,
        ac=base.ac,
        bc=coefficient,
        ag=base.ag,
        bg=base.bg,
    )


def _normalise_wolf_numbers(wolf_numbers: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in wolf_numbers)
    if not values:
        msg = "At least one Wolf number is required for the NEK-He comparison."
        raise ValueError(msg)

    for value in values:
        _require_finite_non_negative(value, name="NEK-He comparison Wolf number")

    if len(set(values)) != len(values):
        msg = "NEK-He comparison Wolf numbers must be unique."
        raise ValueError(msg)

    return values


def calculate_nek_he_bc_candidate_grid(
    *,
    wolf_numbers: Sequence[float],
) -> tuple[NekHeBcComparisonPoint, ...]:
    """Evaluate the two unresolved coefficients for isolated OST NEK helium.

    The energy grid is intentionally fixed at 10, 30, 100 and
    300 MeV/nucleon so a COSRAD comparison cannot silently use different
    checkpoints.  Wolf numbers remain explicit inputs because solar activity
    is part of the disputed exponential factor.
    """

    normalised_wolf_numbers = _normalise_wolf_numbers(wolf_numbers)
    printed_parameters = _nek_he_parameters_with_bc(OST_NEK_HE_BC_PRINTED)
    alternative_parameters = _nek_he_parameters_with_bc(
        OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE,
    )
    points: list[NekHeBcComparisonPoint] = []

    for wolf_number in normalised_wolf_numbers:
        for energy in NEK_HE_BC_COMPARISON_ENERGIES_MEV_PER_NUCLEON:
            printed_flux = ost_nek_differential_flux_directional_cm2_s_sr_mev(
                kinetic_energy_mev_per_nucleon=energy,
                parameters=printed_parameters,
                wolf_number=wolf_number,
            )
            alternative_flux = ost_nek_differential_flux_directional_cm2_s_sr_mev(
                kinetic_energy_mev_per_nucleon=energy,
                parameters=alternative_parameters,
                wolf_number=wolf_number,
            )
            points.append(
                NekHeBcComparisonPoint(
                    wolf_number=wolf_number,
                    energy_mev_per_nucleon=energy,
                    ost_printed_flux=printed_flux,
                    engineering_alternative_flux=alternative_flux,
                    printed_to_alternative_ratio=printed_flux / alternative_flux,
                    alternative_to_printed_ratio=alternative_flux / printed_flux,
                ),
            )

    return tuple(points)


def _error_summary(
    *,
    candidate: NekHeBcCandidate,
    coefficient: float,
    errors: Sequence[float],
) -> NekHeBcCandidateErrorSummary:
    count = len(errors)
    return NekHeBcCandidateErrorSummary(
        candidate=candidate,
        coefficient=coefficient,
        mean_absolute_relative_error=sum(errors) / count,
        root_mean_square_relative_error=sqrt(
            sum(error * error for error in errors) / count,
        ),
        maximum_absolute_relative_error=max(errors),
    )


def _compare_to_cosrad(
    *,
    candidate_points: tuple[NekHeBcComparisonPoint, ...],
    reference: CosradNekHeReferenceSet,
) -> CosradNekHeComparison:
    candidate_by_key = {
        (point.wolf_number, point.energy_mev_per_nucleon): point
        for point in candidate_points
    }
    reference_by_key = {
        (point.wolf_number, point.energy_mev_per_nucleon): point
        for point in reference.points
    }

    if reference_by_key.keys() != candidate_by_key.keys():
        missing = sorted(candidate_by_key.keys() - reference_by_key.keys())
        extra = sorted(reference_by_key.keys() - candidate_by_key.keys())
        msg = (
            "COSRAD NEK-He reference grid must exactly match the requested comparison grid; "
            f"missing={missing!r}, extra={extra!r}."
        )
        raise ValueError(msg)

    residuals: list[CosradNekHePointResidual] = []
    printed_errors: list[float] = []
    alternative_errors: list[float] = []

    for candidate_point in candidate_points:
        key = (candidate_point.wolf_number, candidate_point.energy_mev_per_nucleon)
        reference_point = reference_by_key[key]
        printed_error = abs(candidate_point.ost_printed_flux - reference_point.flux) / (
            reference_point.flux
        )
        alternative_error = abs(
            candidate_point.engineering_alternative_flux - reference_point.flux
        ) / reference_point.flux
        printed_errors.append(printed_error)
        alternative_errors.append(alternative_error)
        residuals.append(
            CosradNekHePointResidual(
                wolf_number=candidate_point.wolf_number,
                energy_mev_per_nucleon=candidate_point.energy_mev_per_nucleon,
                reference_flux=reference_point.flux,
                ost_printed_relative_error=printed_error,
                engineering_alternative_relative_error=alternative_error,
            ),
        )

    printed_summary = _error_summary(
        candidate="ost_printed",
        coefficient=OST_NEK_HE_BC_PRINTED,
        errors=printed_errors,
    )
    alternative_summary = _error_summary(
        candidate="engineering_alternative",
        coefficient=OST_NEK_HE_BC_ENGINEERING_ALTERNATIVE,
        errors=alternative_errors,
    )
    printed_metric = printed_summary.root_mean_square_relative_error
    alternative_metric = alternative_summary.root_mean_square_relative_error

    if isclose(printed_metric, alternative_metric, rel_tol=1.0e-12, abs_tol=0.0):
        selection_status: NekHeBcSelectionStatus = "tie"
        best_candidate: NekHeBcCandidate | None = None
    elif printed_metric < alternative_metric:
        selection_status = "selected"
        best_candidate = "ost_printed"
    else:
        selection_status = "selected"
        best_candidate = "engineering_alternative"

    return CosradNekHeComparison(
        metadata=reference.metadata,
        residuals=tuple(residuals),
        ost_printed=printed_summary,
        engineering_alternative=alternative_summary,
        selection_metric=NEK_HE_BC_SELECTION_METRIC,
        selection_status=selection_status,
        best_candidate=best_candidate,
    )


def evaluate_nek_he_bc_candidates(
    *,
    wolf_numbers: Sequence[float],
    cosrad_reference: CosradNekHeReferenceSet | None = None,
) -> NekHeBcValidationReport:
    """Build the candidate grid and, only when supplied, fit it to COSRAD."""

    normalised_wolf_numbers = _normalise_wolf_numbers(wolf_numbers)
    points = calculate_nek_he_bc_candidate_grid(
        wolf_numbers=normalised_wolf_numbers,
    )
    comparison = None
    if cosrad_reference is not None:
        comparison = _compare_to_cosrad(
            candidate_points=points,
            reference=cosrad_reference,
        )

    return NekHeBcValidationReport(
        component=NEK_HE_COMPONENT_ID,
        units=NEK_HE_DIRECTIONAL_FLUX_UNITS,
        validation_status=OST_NEK_HE_BC_VALIDATION_STATUS,
        energies_mev_per_nucleon=NEK_HE_BC_COMPARISON_ENERGIES_MEV_PER_NUCLEON,
        wolf_numbers=normalised_wolf_numbers,
        points=points,
        cosrad_comparison=comparison,
    )


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        msg = f"COSRAD reference field {key!r} must be a string."
        raise ValueError(msg)
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"COSRAD reference field {key!r} must be numeric."
        raise ValueError(msg)
    return float(value)


def load_cosrad_nek_he_reference(path: Path) -> CosradNekHeReferenceSet:
    """Load the strict JSON exchange format documented for this bench."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        msg = "COSRAD NEK-He reference root must be a JSON object."
        raise ValueError(msg)

    metadata_raw = raw.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        msg = "COSRAD NEK-He reference must contain a metadata object."
        raise ValueError(msg)

    metadata_mapping = cast(Mapping[str, object], metadata_raw)
    metadata = CosradNekHeReferenceMetadata(
        cosrad_version=_required_string(metadata_mapping, "cosrad_version"),
        source_artifact=_required_string(metadata_mapping, "source_artifact"),
        source_sha256=_required_string(metadata_mapping, "source_sha256"),
        calculation_configuration=_required_string(
            metadata_mapping,
            "calculation_configuration",
        ),
        exported_at_utc=_required_string(metadata_mapping, "exported_at_utc"),
        component=_required_string(metadata_mapping, "component"),
        units=_required_string(metadata_mapping, "units"),
    )

    points_raw = raw.get("points")
    if not isinstance(points_raw, list):
        msg = "COSRAD NEK-He reference must contain a points array."
        raise ValueError(msg)

    points: list[CosradNekHeReferencePoint] = []
    for index, point_raw in enumerate(points_raw):
        if not isinstance(point_raw, Mapping):
            msg = f"COSRAD NEK-He reference point {index} must be an object."
            raise ValueError(msg)
        point_mapping = cast(Mapping[str, object], point_raw)
        points.append(
            CosradNekHeReferencePoint(
                wolf_number=_required_float(point_mapping, "wolf_number"),
                energy_mev_per_nucleon=_required_float(
                    point_mapping,
                    "energy_mev_per_nucleon",
                ),
                flux=_required_float(point_mapping, "flux"),
            ),
        )

    declared_artifact_path = Path(metadata.source_artifact)
    source_artifact_path = (
        declared_artifact_path
        if declared_artifact_path.is_absolute()
        else path.parent / declared_artifact_path
    )
    return CosradNekHeReferenceSet(
        metadata=metadata,
        points=tuple(points),
        source_artifact_path=source_artifact_path,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wolf",
        type=float,
        nargs="+",
        required=True,
        help="Explicit Wolf numbers, for example: --wolf 0 30 60 100",
    )
    parser.add_argument(
        "--cosrad-reference",
        type=Path,
        help="Optional provenance-checked isolated NEK-He COSRAD JSON export.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated comparison bench and print a JSON report."""

    args = _build_argument_parser().parse_args(argv)
    reference = None
    if args.cosrad_reference is not None:
        reference = load_cosrad_nek_he_reference(args.cosrad_reference)
    report = evaluate_nek_he_bc_candidates(
        wolf_numbers=args.wolf,
        cosrad_reference=reference,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())


__all__ = [
    "NEK_HE_BC_COMPARISON_ENERGIES_MEV_PER_NUCLEON",
    "NEK_HE_BC_SELECTION_METRIC",
    "NEK_HE_COMPONENT_ID",
    "NEK_HE_DIRECTIONAL_FLUX_UNITS",
    "CosradNekHeComparison",
    "CosradNekHePointResidual",
    "CosradNekHeReferenceMetadata",
    "CosradNekHeReferencePoint",
    "CosradNekHeReferenceSet",
    "NekHeBcCandidate",
    "NekHeBcCandidateErrorSummary",
    "NekHeBcComparisonPoint",
    "NekHeBcValidationReport",
    "calculate_nek_he_bc_candidate_grid",
    "evaluate_nek_he_bc_candidates",
    "load_cosrad_nek_he_reference",
    "main",
]
