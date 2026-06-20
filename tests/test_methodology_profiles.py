import pytest

from radar.core.types import RadiationSource
from radar.core.profiles import (
    DEFAULT_METHODOLOGY_PROFILE,
    GOST_GCR_DOCUMENT,
    GOST_SEP_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    MethodologyProfile,
    MethodologyProfileSpec,
    MethodologySourceModelContract,
    SourceModelFamily,
    SourceModelMetadata,
    documents_for_methodology_profile,
    expected_source_model_family_for_profile,
    methodology_profile_spec,
    profile_uses_gost_gcr,
    profile_uses_gost_sep,
    profile_uses_ost_134_1044_2007,
    source_model_contract_for_profile,
    validate_source_model_family_for_profile,
    validate_source_model_metadata,
    validate_source_model_metadata_for_profile,
)


def test_default_methodology_profile_is_plain_ost() -> None:
    assert DEFAULT_METHODOLOGY_PROFILE is MethodologyProfile.OST_134_1044_2007


def test_plain_ost_profile_spec() -> None:
    spec = methodology_profile_spec(MethodologyProfile.OST_134_1044_2007)

    assert spec.profile is MethodologyProfile.OST_134_1044_2007
    assert spec.documents == (OST_134_1044_2007_DOCUMENT,)
    assert spec.uses_ost_134_1044_2007 is True
    assert spec.uses_gost_sep is False
    assert spec.uses_gost_gcr is False
    assert spec.is_custom is False


def test_ost_with_gost_sep_profile_spec() -> None:
    spec = methodology_profile_spec(MethodologyProfile.OST_WITH_GOST_SEP)

    assert spec.documents == (
        OST_134_1044_2007_DOCUMENT,
        GOST_SEP_DOCUMENT,
    )
    assert spec.uses_ost_134_1044_2007 is True
    assert spec.uses_gost_sep is True
    assert spec.uses_gost_gcr is False


def test_ost_with_gost_gcr_profile_spec() -> None:
    spec = methodology_profile_spec(MethodologyProfile.OST_WITH_GOST_GCR)

    assert spec.documents == (
        OST_134_1044_2007_DOCUMENT,
        GOST_GCR_DOCUMENT,
    )
    assert spec.uses_ost_134_1044_2007 is True
    assert spec.uses_gost_sep is False
    assert spec.uses_gost_gcr is True


def test_ost_with_gost_sep_gcr_profile_spec() -> None:
    spec = methodology_profile_spec(MethodologyProfile.OST_WITH_GOST_SEP_GCR)

    assert spec.documents == (
        OST_134_1044_2007_DOCUMENT,
        GOST_SEP_DOCUMENT,
        GOST_GCR_DOCUMENT,
    )
    assert spec.uses_ost_134_1044_2007 is True
    assert spec.uses_gost_sep is True
    assert spec.uses_gost_gcr is True


def test_custom_profile_spec_is_explicitly_custom() -> None:
    spec = methodology_profile_spec(MethodologyProfile.CUSTOM)

    assert spec.profile is MethodologyProfile.CUSTOM
    assert spec.documents == ("custom",)
    assert spec.uses_ost_134_1044_2007 is False
    assert spec.uses_gost_sep is False
    assert spec.uses_gost_gcr is False
    assert spec.is_custom is True


def test_methodology_profile_document_helper() -> None:
    assert documents_for_methodology_profile(MethodologyProfile.OST_WITH_GOST_SEP) == (
        OST_134_1044_2007_DOCUMENT,
        GOST_SEP_DOCUMENT,
    )


def test_methodology_profile_predicates() -> None:
    assert profile_uses_ost_134_1044_2007(MethodologyProfile.OST_WITH_GOST_SEP)
    assert profile_uses_gost_sep(MethodologyProfile.OST_WITH_GOST_SEP)
    assert not profile_uses_gost_gcr(MethodologyProfile.OST_WITH_GOST_SEP)

    assert profile_uses_ost_134_1044_2007(MethodologyProfile.OST_WITH_GOST_GCR)
    assert not profile_uses_gost_sep(MethodologyProfile.OST_WITH_GOST_GCR)
    assert profile_uses_gost_gcr(MethodologyProfile.OST_WITH_GOST_GCR)


def test_methodology_profile_spec_requires_documents() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        MethodologyProfileSpec(
            profile=MethodologyProfile.OST_134_1044_2007,
            documents=(),
            uses_ost_134_1044_2007=True,
            uses_gost_sep=False,
            uses_gost_gcr=False,
        )


def test_gost_sep_profile_must_reference_gost_sep_document() -> None:
    with pytest.raises(ValueError, match="GOST SEP"):
        MethodologyProfileSpec(
            profile=MethodologyProfile.OST_WITH_GOST_SEP,
            documents=(OST_134_1044_2007_DOCUMENT,),
            uses_ost_134_1044_2007=True,
            uses_gost_sep=True,
            uses_gost_gcr=False,
        )


def test_gost_gcr_profile_must_reference_gost_gcr_document() -> None:
    with pytest.raises(ValueError, match="GOST GCR"):
        MethodologyProfileSpec(
            profile=MethodologyProfile.OST_WITH_GOST_GCR,
            documents=(OST_134_1044_2007_DOCUMENT,),
            uses_ost_134_1044_2007=True,
            uses_gost_sep=False,
            uses_gost_gcr=True,
        )


def test_custom_profile_must_be_marked_custom() -> None:
    with pytest.raises(ValueError, match="Custom methodology profile"):
        MethodologyProfileSpec(
            profile=MethodologyProfile.CUSTOM,
            documents=("custom",),
            uses_ost_134_1044_2007=False,
            uses_gost_sep=False,
            uses_gost_gcr=False,
            is_custom=False,
        )

def test_plain_ost_source_model_contract() -> None:
    contract = source_model_contract_for_profile(MethodologyProfile.OST_134_1044_2007)

    assert contract.sep_model_family is SourceModelFamily.OST_134_1044_2007
    assert contract.gcr_model_family is SourceModelFamily.OST_134_1044_2007
    assert contract.erb_model_family is SourceModelFamily.OST_134_1044_2007


def test_ost_with_gost_sep_source_model_contract() -> None:
    contract = source_model_contract_for_profile(MethodologyProfile.OST_WITH_GOST_SEP)

    assert contract.sep_model_family is SourceModelFamily.GOST_SEP
    assert contract.gcr_model_family is SourceModelFamily.OST_134_1044_2007
    assert contract.erb_model_family is SourceModelFamily.OST_134_1044_2007


def test_ost_with_gost_gcr_source_model_contract() -> None:
    contract = source_model_contract_for_profile(MethodologyProfile.OST_WITH_GOST_GCR)

    assert contract.sep_model_family is SourceModelFamily.OST_134_1044_2007
    assert contract.gcr_model_family is SourceModelFamily.GOST_GCR
    assert contract.erb_model_family is SourceModelFamily.OST_134_1044_2007


def test_ost_with_gost_sep_gcr_source_model_contract() -> None:
    contract = source_model_contract_for_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    assert contract.sep_model_family is SourceModelFamily.GOST_SEP
    assert contract.gcr_model_family is SourceModelFamily.GOST_GCR
    assert contract.erb_model_family is SourceModelFamily.OST_134_1044_2007


def test_custom_source_model_contract() -> None:
    contract = source_model_contract_for_profile(MethodologyProfile.CUSTOM)

    assert contract.sep_model_family is SourceModelFamily.CUSTOM
    assert contract.gcr_model_family is SourceModelFamily.CUSTOM
    assert contract.erb_model_family is SourceModelFamily.CUSTOM


def test_expected_source_model_family_for_profile() -> None:
    assert (
        expected_source_model_family_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR,
            source=RadiationSource.SEP,
        )
        is SourceModelFamily.GOST_SEP
    )
    assert (
        expected_source_model_family_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR,
            source=RadiationSource.GCR,
        )
        is SourceModelFamily.GOST_GCR
    )
    assert (
        expected_source_model_family_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR,
            source=RadiationSource.ERB,
        )
        is SourceModelFamily.OST_134_1044_2007
    )


def test_validate_source_model_family_for_profile_accepts_expected_family() -> None:
    validate_source_model_family_for_profile(
        profile=MethodologyProfile.OST_WITH_GOST_SEP,
        source=RadiationSource.SEP,
        model_family=SourceModelFamily.GOST_SEP,
    )


def test_validate_source_model_family_for_profile_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match methodology profile"):
        validate_source_model_family_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_SEP,
            source=RadiationSource.GCR,
            model_family=SourceModelFamily.GOST_GCR,
        )


def test_custom_contract_requires_custom_families() -> None:
    with pytest.raises(ValueError, match="custom source model families"):
        MethodologySourceModelContract(
            profile=MethodologyProfile.CUSTOM,
            sep_model_family=SourceModelFamily.CUSTOM,
            gcr_model_family=SourceModelFamily.GOST_GCR,
            erb_model_family=SourceModelFamily.CUSTOM,
        )


def test_normative_contract_requires_ost_erb_family() -> None:
    with pytest.raises(ValueError, match="ERB model family"):
        MethodologySourceModelContract(
            profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR,
            sep_model_family=SourceModelFamily.GOST_SEP,
            gcr_model_family=SourceModelFamily.GOST_GCR,
            erb_model_family=SourceModelFamily.CUSTOM,
        )

def test_source_model_metadata_accepts_valid_source_family_pair() -> None:
    metadata = SourceModelMetadata(
        source=RadiationSource.SEP,
        model_family=SourceModelFamily.GOST_SEP,
        name="gost_sep_model",
        document=GOST_SEP_DOCUMENT,
    )

    validate_source_model_metadata(metadata)


def test_source_model_metadata_rejects_invalid_source_family_pair() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.GOST_GCR,
            name="wrong",
            document=GOST_GCR_DOCUMENT,
        )


def test_validate_source_model_metadata_for_profile_accepts_custom_metadata() -> None:
    metadata = SourceModelMetadata(
        source=RadiationSource.SEP,
        model_family=SourceModelFamily.CUSTOM,
        name="static_sep_model",
        document="test",
    )

    validate_source_model_metadata_for_profile(
        metadata=metadata,
        profile=MethodologyProfile.CUSTOM,
    )


def test_validate_source_model_metadata_for_profile_rejects_mismatch() -> None:
    metadata = SourceModelMetadata(
        source=RadiationSource.GCR,
        model_family=SourceModelFamily.CUSTOM,
        name="static_gcr_model",
        document="test",
    )

    with pytest.raises(ValueError, match="does not match methodology profile"):
        validate_source_model_metadata_for_profile(
            metadata=metadata,
            profile=MethodologyProfile.OST_134_1044_2007,
        )
