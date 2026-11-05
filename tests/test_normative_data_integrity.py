from dataclasses import replace
import json
from pathlib import Path

import pytest

import radar.data.integrity as integrity_module
from radar.core.profiles import MethodologyProfile
from radar.data.audit import (
    DataAuditScope,
    DataAuditScopeKind,
    audit_normative_resources,
    main as audit_main,
    normative_resource_paths_for_profile,
)
from radar.data.integrity import (
    NormativeResourceStatus,
    load_normative_integrity_manifest,
    require_release_eligible_normative_resource,
    validate_normative_resource_integrity,
)


def _tracked_normative_resource_paths() -> set[str]:
    root = Path("src/radar/data/normative")
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"__init__.py", "integrity_manifest.json"}
        and "__pycache__" not in path.parts
    }


def test_integrity_manifest_covers_every_bundled_resource_exactly_once() -> None:
    manifest = load_normative_integrity_manifest()

    assert {entry.path for entry in manifest.resources} == (
        _tracked_normative_resource_paths()
    )


def test_every_manifest_digest_and_byte_count_matches_current_resource() -> None:
    manifest = load_normative_integrity_manifest()

    for entry in manifest.resources:
        assert validate_normative_resource_integrity(entry.path) == entry


def test_verified_ost_and_official_igrf_resources_pass_release_gate() -> None:
    for path in (
        "ost_134_1044_2007/table_g_1_wolf_numbers.csv",
        "sep_protons/ost_134_1044_2007_coefficients.csv",
        "sep_hze/ost_134_1044_2007_b9.csv",
        "erb/ost_appendix_a_tables.json",
        "erb/igrf14coeffs.txt",
        "stopping/electron_al_range.csv",
        "tendl/p_al27_mf3_xs.csv",
        "tendl/p_al27_mf6_proton_yield.csv",
        "tendl/p_al27_mf6_proton_pdf.csv",
    ):
        assert require_release_eligible_normative_resource(path).release_eligible


@pytest.mark.parametrize(
    "path",
    (
        "stopping/hze_si_let.csv",
        "stopping/proton_al_range.csv",
        "stopping/proton_si_let.csv",
        "stopping/hze_al_range.csv",
        "erb/ae8max.asc",
        "sep_protons/gost_r_25645_165_2025_coefficients.csv",
    ),
)
def test_quarantined_provisional_and_draft_resources_fail_release_gate(
    path: str,
) -> None:
    with pytest.raises(ValueError, match="not eligible"):
        require_release_eligible_normative_resource(path)


def test_hze_let_is_quarantined_and_histogram_law_is_explicit() -> None:
    manifest = load_normative_integrity_manifest()
    hze_let = manifest.by_path("stopping/hze_si_let.csv")
    mf6_pdf = manifest.by_path("tendl/p_al27_mf6_proton_pdf.csv")

    assert hze_let.status is NormativeResourceStatus.QUARANTINED
    assert hze_let.release_eligible is False
    assert mf6_pdf.interpolation_law == "histogram_left"


def test_tendl_derivatives_retain_their_official_source_fingerprint() -> None:
    manifest = load_normative_integrity_manifest()
    expected_digest = (
        "f2218940baef62cf9fb72ed3ab4a0326e69f92faabb0b7ee69fed7b0bb4c41a7"
    )

    for path in (
        "tendl/p_al27_mf3_xs.csv",
        "tendl/p_al27_mf6_proton_yield.csv",
        "tendl/p_al27_mf6_proton_pdf.csv",
    ):
        entry = manifest.by_path(path)
        assert entry.source_artifact_sha256 == expected_digest
        assert entry.source_artifact_byte_count == 2_712_771
        assert entry.derivation_command is not None


def test_remaining_stopping_tables_retain_workbook_fingerprint_without_release_claim() -> None:
    manifest = load_normative_integrity_manifest()
    expected_digest = (
        "d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684"
    )

    for path in (
        "stopping/proton_al_range.csv",
        "stopping/proton_si_let.csv",
        "stopping/hze_al_range.csv",
        "stopping/hze_si_let.csv",
    ):
        entry = manifest.by_path(path)
        assert entry.source_artifact_sha256 == expected_digest
        assert entry.source_artifact_byte_count == 522_688
        assert entry.derivation_command is not None
        assert not entry.release_eligible


def test_electron_table_has_official_derived_passport_without_numeric_changes() -> None:
    entry = require_release_eligible_normative_resource("stopping/electron_al_range.csv")

    assert entry.status is NormativeResourceStatus.VERIFIED_DERIVED
    assert entry.calculation_input is True
    assert entry.release_eligible is True
    assert entry.sha256 == "b53413110cd515bb11bc15fc074acfbb5eb0f5dcc473279af8b25ecc7f2f7d71"
    assert entry.byte_count == 2304
    assert entry.record_count == 81
    assert entry.interpolation_law == "log_log_monotone"
    assert entry.energy_basis == "per_particle"
    assert entry.source_artifact_sha256 == (
        "b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6"
    )
    assert entry.source_artifact_byte_count == 444_183
    assert "https://physics.nist.gov/PhysRefData/Star/STAR.tar.gz" in entry.source_ref
    assert entry.derivation_command is not None
    for required_argument in ("reproduce_nist_estar.py", "--compiler", "--output"):
        assert required_argument in entry.derivation_command
    assert "nist_estar_reproduction_final_20261101/report.json" in entry.verification_note
    assert "d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684" in (
        entry.verification_note
    )


def test_proton_let_interpolation_metadata_does_not_promote_or_replace_data() -> None:
    entry = validate_normative_resource_integrity("stopping/proton_si_let.csv")

    assert entry.interpolation_law == "linear_linear_piecewise"
    assert entry.status is NormativeResourceStatus.PROVISIONAL
    assert entry.calculation_input is True
    assert entry.release_eligible is False
    assert entry.sha256 == "e8b536e3a1eac9eefad062fd12d763f24d1a7f7cc84f311d510a323e2d0832d8"
    assert entry.byte_count == 4098
    assert entry.record_count == 105
    assert entry.energy_basis == "per_particle"
    assert entry.source_artifact_sha256 == (
        "d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684"
    )
    assert entry.source_artifact_byte_count == 522688
    with pytest.raises(ValueError, match="not eligible"):
        require_release_eligible_normative_resource(entry.path)


@pytest.mark.parametrize("missing", ("source_fingerprint", "derivation_command"))
def test_electron_derived_passport_cannot_drop_reproduction_evidence(missing: str) -> None:
    entry = load_normative_integrity_manifest().by_path("stopping/electron_al_range.csv")

    with pytest.raises(ValueError, match="source-artifact digest and a deterministic"):
        if missing == "source_fingerprint":
            replace(entry, source_artifact_sha256=None, source_artifact_byte_count=None)
        else:
            replace(entry, derivation_command=None)


@pytest.mark.parametrize(
    "status",
    (NormativeResourceStatus.PROVISIONAL, NormativeResourceStatus.QUARANTINED),
)
def test_electron_passport_cannot_keep_release_permission_without_verified_status(
    status: NormativeResourceStatus,
) -> None:
    entry = load_normative_integrity_manifest().by_path("stopping/electron_al_range.csv")

    with pytest.raises(ValueError, match="explicit verified status"):
        replace(entry, status=status)


def test_electron_provenance_upgrade_preserves_remaining_strict_ost_blockers() -> None:
    scope = DataAuditScope.for_profile(MethodologyProfile.OST_134_1044_2007)
    report = audit_normative_resources(strict=True, scope=scope)

    assert len(report.audited_paths) == 15
    assert report.error_count == 5
    assert report.warning_count == 0
    assert report.release_ready is False
    assert {(finding.code, finding.path) for finding in report.findings} == {
        ("non_release_resource", "stopping/proton_al_range.csv"),
        ("non_release_resource", "stopping/proton_si_let.csv"),
        ("non_release_resource", "stopping/hze_al_range.csv"),
        ("quarantined_resource", "stopping/hze_si_let.csv"),
        ("hze_let_multidecade_discontinuity", "stopping/hze_si_let.csv"),
    }


def test_machine_audit_reports_known_physical_blocker_without_hash_errors() -> None:
    report = audit_normative_resources()
    codes = {finding.code for finding in report.findings}

    assert "integrity_mismatch" not in codes
    assert "unregistered_resource" not in codes
    assert "missing_resource" not in codes
    assert "quarantined_resource" in codes
    assert "hze_let_multidecade_discontinuity" in codes
    assert report.scope.kind is DataAuditScopeKind.GLOBAL
    # The original 14 findings lose three verified TENDL derivatives and one
    # electron table independently reproduced from the official STAR archive.
    assert len(report.findings) == 10
    assert report.release_ready is False


def test_plain_ost_profile_audits_only_runtime_dependencies() -> None:
    scope = DataAuditScope.for_profile(MethodologyProfile.OST_134_1044_2007)
    report = audit_normative_resources(scope=scope)
    finding_paths = {finding.path for finding in report.findings}

    assert report.scope == scope
    assert set(report.audited_paths) == normative_resource_paths_for_profile(
        MethodologyProfile.OST_134_1044_2007,
    )
    assert "sep_protons/ost_134_1044_2007_coefficients.csv" in report.audited_paths
    assert "erb/ost_appendix_a_tables.json" in report.audited_paths
    assert "stopping/hze_si_let.csv" in report.audited_paths
    assert "sep_protons/gost_r_25645_165_2025_coefficients.csv" not in report.audited_paths
    assert not any(path.startswith("erb/ae8") for path in report.audited_paths)
    assert not any(path.startswith("erb/ap8") for path in report.audited_paths)
    assert "sep_protons/gost_r_25645_165_2025_coefficients.csv" not in finding_paths
    assert not any(path.startswith("erb/ae8") for path in finding_paths)
    assert not any(path.startswith("erb/ap8") for path in finding_paths)


def test_experimental_profile_scope_selects_only_its_runtime_model_resources() -> None:
    paths = normative_resource_paths_for_profile(
        MethodologyProfile.OST_WITH_AE8_AP8_ERB,
    )

    assert "erb/ae8max.asc" in paths
    assert "erb/ap8max.asc" in paths
    assert "erb/ost_appendix_a_tables.json" not in paths
    assert "sep_protons/gost_r_25645_165_2025_coefficients.csv" not in paths


def test_custom_profile_has_no_deterministic_audit_scope() -> None:
    with pytest.raises(ValueError, match="no deterministic"):
        DataAuditScope.for_profile(MethodologyProfile.CUSTOM)


def test_profile_audit_cli_emits_typed_scope_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = audit_main(
        ["--profile", MethodologyProfile.OST_134_1044_2007.value],
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["scope"] == DataAuditScopeKind.METHODOLOGY_PROFILE.value
    assert payload["profile"] == MethodologyProfile.OST_134_1044_2007.value
    assert payload["audited_resource_count"] == len(payload["audited_paths"])
    assert "erb/ae8max.asc" not in payload["audited_paths"]


def test_strict_audit_promotes_provisional_resources_to_errors() -> None:
    report = audit_normative_resources(strict=True)

    non_release_findings = tuple(
        finding for finding in report.findings if finding.code == "non_release_resource"
    )
    assert non_release_findings
    assert all(finding.severity.value == "error" for finding in non_release_findings)


@pytest.mark.parametrize("field", ("byte_count", "record_count"))
def test_integrity_manifest_rejects_boolean_integer_fields(
    field: str,
) -> None:
    manifest_path = Path("src/radar/data/normative/integrity_manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["resources"][0][field] = True

    with pytest.raises(ValueError, match=field):
        integrity_module._resource_from_dict(payload["resources"][0])
