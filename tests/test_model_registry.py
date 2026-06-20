import pytest

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.types import RadiationSource
from radar.erb.model import OstErbModel
from radar.gcr.model import GostGcrModel, OstGcrModel
from radar.model_registry import (
    REGISTERED_SOURCE_MODELS,
    SourceModelRegistration,
    SourceModelBundle,
    registered_source_model_families,
    source_model_class,
    source_model_bundle_for_profile,
    source_model_class_for_profile,
    source_model_classes_for_profile,
    source_model_registration,
    source_model_registration_for_profile,
)
from radar.sep.model import GostSepModel, OstSepModel


def test_registered_source_models_are_unique() -> None:
    keys = [registration.key for registration in REGISTERED_SOURCE_MODELS]

    assert len(keys) == len(set(keys))


def test_registered_source_model_families_for_sep() -> None:
    assert registered_source_model_families(RadiationSource.SEP) == (
        SourceModelFamily.OST_134_1044_2007,
        SourceModelFamily.GOST_SEP,
    )


def test_registered_source_model_families_for_gcr() -> None:
    assert registered_source_model_families(RadiationSource.GCR) == (
        SourceModelFamily.OST_134_1044_2007,
        SourceModelFamily.GOST_GCR,
    )


def test_registered_source_model_families_for_erb() -> None:
    assert registered_source_model_families(RadiationSource.ERB) == (
        SourceModelFamily.OST_134_1044_2007,
    )


def test_source_model_registration_for_source_and_family() -> None:
    registration = source_model_registration(
        source=RadiationSource.SEP,
        model_family=SourceModelFamily.GOST_SEP,
    )

    assert isinstance(registration, SourceModelRegistration)
    assert registration.source is RadiationSource.SEP
    assert registration.model_family is SourceModelFamily.GOST_SEP
    assert registration.model_class is GostSepModel


def test_source_model_class_for_source_and_family() -> None:
    assert (
        source_model_class(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.OST_134_1044_2007,
        )
        is OstSepModel
    )
    assert (
        source_model_class(
            source=RadiationSource.GCR,
            model_family=SourceModelFamily.OST_134_1044_2007,
        )
        is OstGcrModel
    )
    assert (
        source_model_class(
            source=RadiationSource.GCR,
            model_family=SourceModelFamily.GOST_GCR,
        )
        is GostGcrModel
    )
    assert (
        source_model_class(
            source=RadiationSource.ERB,
            model_family=SourceModelFamily.OST_134_1044_2007,
        )
        is OstErbModel
    )


def test_source_model_registration_rejects_unregistered_custom_family() -> None:
    with pytest.raises(ValueError, match="No registered source model"):
        source_model_registration(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.CUSTOM,
        )


def test_source_model_class_for_plain_ost_profile() -> None:
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_134_1044_2007,
            source=RadiationSource.SEP,
        )
        is OstSepModel
    )
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_134_1044_2007,
            source=RadiationSource.GCR,
        )
        is OstGcrModel
    )
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_134_1044_2007,
            source=RadiationSource.ERB,
        )
        is OstErbModel
    )


def test_source_model_class_for_gost_sep_profile() -> None:
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_SEP,
            source=RadiationSource.SEP,
        )
        is GostSepModel
    )
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_SEP,
            source=RadiationSource.GCR,
        )
        is OstGcrModel
    )


def test_source_model_class_for_gost_gcr_profile() -> None:
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_GCR,
            source=RadiationSource.SEP,
        )
        is OstSepModel
    )
    assert (
        source_model_class_for_profile(
            profile=MethodologyProfile.OST_WITH_GOST_GCR,
            source=RadiationSource.GCR,
        )
        is GostGcrModel
    )


def test_source_model_registration_for_custom_profile_has_no_default() -> None:
    with pytest.raises(ValueError, match="No registered source model"):
        source_model_registration_for_profile(
            profile=MethodologyProfile.CUSTOM,
            source=RadiationSource.SEP,
        )

def test_source_model_bundle_for_plain_ost_profile() -> None:
    bundle = source_model_bundle_for_profile(MethodologyProfile.OST_134_1044_2007)

    assert isinstance(bundle, SourceModelBundle)
    assert bundle.profile is MethodologyProfile.OST_134_1044_2007
    assert bundle.sep.model_class is OstSepModel
    assert bundle.gcr.model_class is OstGcrModel
    assert bundle.erb.model_class is OstErbModel


def test_source_model_bundle_for_gost_sep_profile() -> None:
    bundle = source_model_bundle_for_profile(MethodologyProfile.OST_WITH_GOST_SEP)

    assert bundle.sep.model_class is GostSepModel
    assert bundle.gcr.model_class is OstGcrModel
    assert bundle.erb.model_class is OstErbModel


def test_source_model_bundle_for_gost_gcr_profile() -> None:
    bundle = source_model_bundle_for_profile(MethodologyProfile.OST_WITH_GOST_GCR)

    assert bundle.sep.model_class is OstSepModel
    assert bundle.gcr.model_class is GostGcrModel
    assert bundle.erb.model_class is OstErbModel


def test_source_model_bundle_for_gost_sep_gcr_profile() -> None:
    bundle = source_model_bundle_for_profile(MethodologyProfile.OST_WITH_GOST_SEP_GCR)

    assert bundle.sep.model_class is GostSepModel
    assert bundle.gcr.model_class is GostGcrModel
    assert bundle.erb.model_class is OstErbModel


def test_source_model_bundle_registration_order() -> None:
    bundle = source_model_bundle_for_profile(MethodologyProfile.OST_134_1044_2007)

    assert tuple(registration.source for registration in bundle.registrations) == (
        RadiationSource.SEP,
        RadiationSource.GCR,
        RadiationSource.ERB,
    )


def test_source_model_bundle_model_classes_order() -> None:
    bundle = source_model_bundle_for_profile(MethodologyProfile.OST_134_1044_2007)

    assert bundle.model_classes == (
        OstSepModel,
        OstGcrModel,
        OstErbModel,
    )


def test_source_model_classes_for_profile() -> None:
    assert source_model_classes_for_profile(MethodologyProfile.OST_WITH_GOST_SEP_GCR) == (
        GostSepModel,
        GostGcrModel,
        OstErbModel,
    )


def test_source_model_bundle_for_custom_profile_has_no_default_models() -> None:
    with pytest.raises(ValueError, match="No registered source model"):
        source_model_bundle_for_profile(MethodologyProfile.CUSTOM)
