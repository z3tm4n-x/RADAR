import pytest

from radar.core.profiles import (
    DEFAULT_METHODOLOGY_PROFILE,
    GOST_GCR_DOCUMENT,
    GOST_SEP_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    MethodologyProfile,
    MethodologyProfileSpec,
    documents_for_methodology_profile,
    methodology_profile_spec,
    profile_uses_gost_gcr,
    profile_uses_gost_sep,
    profile_uses_ost_134_1044_2007,
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
