"""Machine audit for RADAR's bundled physical resource registry."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib import resources
from importlib.resources.abc import Traversable
from io import StringIO
import json
from typing import Iterable

from radar.core.profiles import (
    MethodologyProfile,
    SourceModelFamily,
    source_model_contract_for_profile,
)
from radar.data.integrity import (
    NormativeResourceStatus,
    load_normative_integrity_manifest,
    validate_normative_resource_integrity,
)

_NORMATIVE_PACKAGE = "radar.data.normative"
_MANIFEST_PATH = "integrity_manifest.json"
_HZE_SI_LET_PATH = "stopping/hze_si_let.csv"

_COMMON_OST_RUNTIME_RESOURCES = frozenset(
    {
        "index.json",
        "ost_134_1044_2007/manifest.json",
        "ost_134_1044_2007/table_g_1_wolf_numbers.csv",
    }
)
_SHIELDING_RUNTIME_RESOURCES = frozenset(
    {
        "stopping/electron_al_range.csv",
        "stopping/proton_al_range.csv",
        "stopping/proton_si_let.csv",
        "stopping/hze_al_range.csv",
        _HZE_SI_LET_PATH,
        "tendl/p_al27_mf3_xs.csv",
        "tendl/p_al27_mf6_proton_yield.csv",
        "tendl/p_al27_mf6_proton_pdf.csv",
    }
)
_OST_SEP_RUNTIME_RESOURCES = frozenset(
    {
        "sep_protons/ost_134_1044_2007_coefficients.csv",
        "sep_hze/ost_134_1044_2007_b9.csv",
    }
)
_GOST_SEP_RUNTIME_RESOURCES = frozenset(
    {
        "sep_protons/gost_r_25645_165_2025_coefficients.csv",
        # The current GOST proton profile still uses OST Appendix B.9 for HZE.
        "sep_hze/ost_134_1044_2007_b9.csv",
    }
)
_OST_ERB_RUNTIME_RESOURCES = frozenset(
    {
        "erb/ost_appendix_a_tables.json",
        "erb/igrf14coeffs.txt",
    }
)
_AE8_AP8_ERB_RUNTIME_RESOURCES = frozenset(
    {
        "erb/ae8max.asc",
        "erb/ae8min.asc",
        "erb/ap8max.asc",
        "erb/ap8min.asc",
        "erb/igrf14coeffs.txt",
    }
)


class DataAuditScopeKind(StrEnum):
    """Kind of resource set covered by one audit."""

    GLOBAL = "global"
    METHODOLOGY_PROFILE = "methodology_profile"


@dataclass(frozen=True)
class DataAuditScope:
    """Typed global or methodology-profile audit scope."""

    kind: DataAuditScopeKind
    profile: MethodologyProfile | None = None

    def __post_init__(self) -> None:
        if self.kind is DataAuditScopeKind.GLOBAL and self.profile is not None:
            msg = "Global data-audit scope must not select a methodology profile."
            raise ValueError(msg)
        if self.kind is DataAuditScopeKind.METHODOLOGY_PROFILE:
            if self.profile is None:
                msg = "Methodology-profile data-audit scope requires a profile."
                raise ValueError(msg)
            if self.profile is MethodologyProfile.CUSTOM:
                msg = "Custom methodology has no deterministic normative-resource scope."
                raise ValueError(msg)

    @classmethod
    def global_scope(cls) -> DataAuditScope:
        """Return the complete bundled-resource audit scope."""

        return cls(kind=DataAuditScopeKind.GLOBAL)

    @classmethod
    def for_profile(cls, profile: MethodologyProfile) -> DataAuditScope:
        """Return the runtime dependency scope of one methodology profile."""

        return cls(
            kind=DataAuditScopeKind.METHODOLOGY_PROFILE,
            profile=profile,
        )


GLOBAL_DATA_AUDIT_SCOPE = DataAuditScope.global_scope()


class AuditSeverity(StrEnum):
    """Severity of a normative-data audit finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class DataAuditFinding:
    """One reproducible machine-audit finding."""

    severity: AuditSeverity
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class DataAuditReport:
    """Complete audit outcome for one manifest release."""

    release_id: str
    scope: DataAuditScope
    audited_paths: tuple[str, ...]
    findings: tuple[DataAuditFinding, ...]

    @property
    def error_count(self) -> int:
        return sum(item.severity is AuditSeverity.ERROR for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity is AuditSeverity.WARNING for item in self.findings)

    @property
    def release_ready(self) -> bool:
        return not self.findings

    def to_json(self) -> str:
        """Return deterministic JSON suitable for CI artifacts."""

        return json.dumps(
            {
                "release_id": self.release_id,
                "scope": self.scope.kind.value,
                "profile": (
                    None if self.scope.profile is None else self.scope.profile.value
                ),
                "audited_resource_count": len(self.audited_paths),
                "audited_paths": self.audited_paths,
                "release_ready": self.release_ready,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "findings": [asdict(item) for item in self.findings],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def normative_resource_paths_for_profile(
    profile: MethodologyProfile,
) -> frozenset[str]:
    """Return bundled resources loaded by the profile's default calculation path."""

    if profile is MethodologyProfile.CUSTOM:
        msg = "Custom methodology has no deterministic normative-resource scope."
        raise ValueError(msg)

    contract = source_model_contract_for_profile(profile)
    selected = set(_COMMON_OST_RUNTIME_RESOURCES | _SHIELDING_RUNTIME_RESOURCES)

    if contract.sep_model_family is SourceModelFamily.GOST_SEP:
        selected.update(_GOST_SEP_RUNTIME_RESOURCES)
    else:
        selected.update(_OST_SEP_RUNTIME_RESOURCES)

    if contract.erb_model_family is SourceModelFamily.AE8_AP8:
        selected.update(_AE8_AP8_ERB_RUNTIME_RESOURCES)
    else:
        selected.update(_OST_ERB_RUNTIME_RESOURCES)

    # OST and the current GOST GCR implementations both use the common Wolf
    # number table; they have no additional bundled runtime table.
    return frozenset(selected)


def normative_resource_paths_for_scope(scope: DataAuditScope) -> frozenset[str] | None:
    """Return selected paths, or None when every bundled resource is in scope."""

    if scope.kind is DataAuditScopeKind.GLOBAL:
        return None

    assert scope.profile is not None
    return normative_resource_paths_for_profile(scope.profile)


def _walk_resource_files(node: Traversable, prefix: str = "") -> Iterable[str]:
    """Yield package-relative physical/registry files from a Traversable tree."""

    for child in node.iterdir():
        name = child.name
        relative_path = f"{prefix}/{name}" if prefix else name
        if name == "__pycache__":
            continue
        if child.is_dir():
            yield from _walk_resource_files(child, relative_path)
        elif name != "__init__.py" and relative_path != _MANIFEST_PATH:
            yield relative_path


def _hze_let_discontinuity_findings(
    audited_paths: frozenset[str],
) -> tuple[DataAuditFinding, ...]:
    """Detect the known multi-decade discontinuity independently of row sums."""

    if _HZE_SI_LET_PATH not in audited_paths:
        return ()

    resource = resources.files(_NORMATIVE_PACKAGE).joinpath(_HZE_SI_LET_PATH)
    reader = csv.DictReader(StringIO(resource.read_text(encoding="utf-8")))
    previous_by_z: dict[int, tuple[float, float]] = {}
    affected_z: set[int] = set()

    for row in reader:
        z = int(row["z"])
        energy = float(row["energy_mev_per_nucleon"])
        electronic_let = float(row["let_electronic_mev_cm2_mg"])
        previous = previous_by_z.get(z)
        if previous is not None:
            previous_energy, previous_let = previous
            if (
                energy > previous_energy
                and electronic_let > 0.0
                and previous_let / electronic_let >= 1.0e3
            ):
                affected_z.add(z)
        previous_by_z[z] = (energy, electronic_let)

    if not affected_z:
        return ()

    charges = ",".join(str(value) for value in sorted(affected_z))
    return (
        DataAuditFinding(
            severity=AuditSeverity.ERROR,
            code="hze_let_multidecade_discontinuity",
            path=_HZE_SI_LET_PATH,
            message=f"Electronic LET collapses by >=3 decades for Z={charges}.",
        ),
    )


def audit_normative_resources(
    *,
    strict: bool = False,
    scope: DataAuditScope = GLOBAL_DATA_AUDIT_SCOPE,
) -> DataAuditReport:
    """Audit coverage, hashes, statuses and physical invariants in one scope."""

    manifest = load_normative_integrity_manifest()
    findings: list[DataAuditFinding] = []
    discovered_paths = set(_walk_resource_files(resources.files(_NORMATIVE_PACKAGE)))
    registered_paths = {item.path for item in manifest.resources}
    selected_paths = normative_resource_paths_for_scope(scope)

    if selected_paths is None:
        audited_paths = frozenset(discovered_paths | registered_paths)
        unregistered_paths = discovered_paths - registered_paths
        missing_paths = registered_paths - discovered_paths
        entries = manifest.resources
    else:
        audited_paths = selected_paths
        unregistered_paths = set(selected_paths & (discovered_paths - registered_paths))
        missing_paths = set(selected_paths - discovered_paths)
        entries = tuple(item for item in manifest.resources if item.path in selected_paths)

    for path in sorted(unregistered_paths):
        findings.append(
            DataAuditFinding(
                severity=AuditSeverity.ERROR,
                code="unregistered_resource",
                path=path,
                message="Bundled resource is absent from the integrity manifest.",
            )
        )
    for path in sorted(missing_paths):
        findings.append(
            DataAuditFinding(
                severity=AuditSeverity.ERROR,
                code="missing_resource",
                path=path,
                message="Manifest entry has no bundled resource.",
            )
        )

    for entry in entries:
        try:
            validate_normative_resource_integrity(entry.path)
        except (FileNotFoundError, ValueError) as exc:
            findings.append(
                DataAuditFinding(
                    severity=AuditSeverity.ERROR,
                    code="integrity_mismatch",
                    path=entry.path,
                    message=str(exc),
                )
            )
            continue

        if entry.status is NormativeResourceStatus.QUARANTINED:
            findings.append(
                DataAuditFinding(
                    severity=AuditSeverity.ERROR,
                    code="quarantined_resource",
                    path=entry.path,
                    message=entry.verification_note,
                )
            )
        elif entry.status in (
            NormativeResourceStatus.PROVISIONAL,
            NormativeResourceStatus.VERIFIED_AGAINST_DRAFT,
        ):
            findings.append(
                DataAuditFinding(
                    severity=(AuditSeverity.ERROR if strict else AuditSeverity.WARNING),
                    code="non_release_resource",
                    path=entry.path,
                    message=entry.verification_note,
                )
            )

    findings.extend(_hze_let_discontinuity_findings(audited_paths))

    return DataAuditReport(
        release_id=manifest.release_id,
        scope=scope,
        audited_paths=tuple(sorted(audited_paths)),
        findings=tuple(findings),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the audit and print a JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat provisional and draft resources as errors.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(
            profile.value
            for profile in MethodologyProfile
            if profile is not MethodologyProfile.CUSTOM
        ),
        default=None,
        help=(
            "Audit only normative resources used by this methodology profile; "
            "omit for the complete bundled-resource audit."
        ),
    )
    arguments = parser.parse_args(argv)
    scope = (
        GLOBAL_DATA_AUDIT_SCOPE
        if arguments.profile is None
        else DataAuditScope.for_profile(MethodologyProfile(arguments.profile))
    )
    report = audit_normative_resources(strict=arguments.strict, scope=scope)
    print(report.to_json())
    return 0 if report.release_ready else 1


__all__ = [
    "AuditSeverity",
    "DataAuditFinding",
    "DataAuditReport",
    "DataAuditScope",
    "DataAuditScopeKind",
    "GLOBAL_DATA_AUDIT_SCOPE",
    "audit_normative_resources",
    "main",
    "normative_resource_paths_for_profile",
    "normative_resource_paths_for_scope",
]


if __name__ == "__main__":
    raise SystemExit(main())
