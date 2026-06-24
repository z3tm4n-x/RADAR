import pytest

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.types import RadiationSource
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    SourceModelSelectionConfig,
)
from radar.erb.model import OstErbModel
from radar.gcr.model import GostGcrModel, OstGcrModel
from radar.model_registry import (
    REGISTERED_SOURCE_MODELS,
    SourceModelRegistration,
    SourceModelBundle,
    registered_source_model_families,
    source_model_class,
    source_model_bundle_for_profile,
    source_model_bundle_for_selection,
    source_model_class_for_profile,
    source_model_classes_for_profile,
    source_model_classes_for_selection,
    source_model_instances_for_profile,
    source_model_instances_for_config,
    source_model_instances_for_selection,
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

def test_source_model_bundle_for_explicit_selection() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    bundle = source_model_bundle_for_selection(selection)

    assert bundle.profile is MethodologyProfile.OST_WITH_GOST_SEP_GCR
    assert bundle.sep.model_class is GostSepModel
    assert bundle.gcr.model_class is GostGcrModel
    assert bundle.erb.model_class is OstErbModel


def test_source_model_classes_for_explicit_selection() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    assert source_model_classes_for_selection(selection) == (
        GostSepModel,
        GostGcrModel,
        OstErbModel,
    )


def test_source_model_bundle_for_custom_selection_has_no_default_models() -> None:
    selection = SourceModelSelectionConfig.from_profile(MethodologyProfile.CUSTOM)

    with pytest.raises(ValueError, match="No registered source model"):
        source_model_bundle_for_selection(selection)



def test_source_model_instances_for_profile_returns_default_instances() -> None:
    sep_model, gcr_model, erb_model = source_model_instances_for_profile(
        MethodologyProfile.OST_134_1044_2007
    )

    assert isinstance(sep_model, OstSepModel)
    assert isinstance(gcr_model, OstGcrModel)
    assert isinstance(erb_model, OstErbModel)
    assert sep_model.version == "not_implemented"


def test_source_model_instances_for_selection_configures_ost_sep_model() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_134_1044_2007,
    )

    sep_model, gcr_model, erb_model = source_model_instances_for_selection(
        selection,
        sep_kwargs={
            "energy_grid_mev": (10.0, 20.0),
            "monthly_smoothed_wolf_numbers": (100.0, 120.0),
            "version": "protons_only_v1",
        },
    )

    assert isinstance(sep_model, OstSepModel)
    assert sep_model.energy_grid_mev == (10.0, 20.0)
    assert sep_model.monthly_smoothed_wolf_numbers == (100.0, 120.0)
    assert sep_model.version == "protons_only_v1"

    assert isinstance(gcr_model, OstGcrModel)
    assert isinstance(erb_model, OstErbModel)


def test_source_model_instances_for_selection_respects_gost_sep_profile() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP,
    )

    sep_model, gcr_model, erb_model = source_model_instances_for_selection(selection)

    assert isinstance(sep_model, GostSepModel)
    assert isinstance(gcr_model, OstGcrModel)
    assert isinstance(erb_model, OstErbModel)



def _calculation_config_for_model_builder(
    profile: MethodologyProfile = MethodologyProfile.OST_134_1044_2007,
    *,
    lifetime_years: int = 2,
) -> CalculationConfig:
    return CalculationConfig(
        mission=MissionConfig(
            launch_year=2027,
            lifetime_years=lifetime_years,
        ),
        orbit=OrbitConfig.circular(
            altitude_km=35786.0,
            inclination_deg=0.0,
        ),
        methodology=MethodologyConfig(profile=profile),
    )


def test_source_model_instances_for_config_returns_default_instances_without_sep_grid() -> None:
    config = _calculation_config_for_model_builder()

    sep_model, gcr_model, erb_model = source_model_instances_for_config(config)

    assert isinstance(sep_model, OstSepModel)
    assert sep_model.energy_grid_mev == ()
    assert sep_model.monthly_smoothed_wolf_numbers == ()
    assert sep_model.version == "not_implemented"

    assert isinstance(gcr_model, OstGcrModel)
    assert isinstance(erb_model, OstErbModel)


def test_source_model_instances_for_config_configures_ost_sep_model() -> None:
    config = _calculation_config_for_model_builder(lifetime_years=2)

    sep_model, gcr_model, erb_model = source_model_instances_for_config(
        config,
        sep_energy_grid_mev=(10.0, 20.0, 100.0),
    )

    assert isinstance(sep_model, OstSepModel)
    assert sep_model.energy_grid_mev == (10.0, 20.0, 100.0)
    assert len(sep_model.monthly_smoothed_wolf_numbers) == 24
    assert sep_model.monthly_smoothed_wolf_numbers[:12] == (
        sep_model.monthly_smoothed_wolf_numbers[0],
    ) * 12
    assert sep_model.version == "protons_only_v1"

    assert isinstance(gcr_model, OstGcrModel)
    assert isinstance(erb_model, OstErbModel)


def test_source_model_instances_for_config_rejects_empty_sep_grid() -> None:
    config = _calculation_config_for_model_builder()

    with pytest.raises(ValueError, match="SEP energy grid must not be empty"):
        source_model_instances_for_config(
            config,
            sep_energy_grid_mev=(),
        )


def test_source_model_instances_for_config_does_not_configure_gost_sep_model() -> None:
    config = _calculation_config_for_model_builder(
        profile=MethodologyProfile.OST_WITH_GOST_SEP,
    )

    sep_model, gcr_model, erb_model = source_model_instances_for_config(
        config,
        sep_energy_grid_mev=(10.0, 20.0),
    )

    assert isinstance(sep_model, GostSepModel)
    assert isinstance(gcr_model, OstGcrModel)
    assert isinstance(erb_model, OstErbModel)
