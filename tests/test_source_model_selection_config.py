import pytest

from radar.core.profiles import MethodologyProfile, SourceModelFamily
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    SourceModelSelectionConfig,
)
from radar.core.types import RadiationSource


def test_source_model_selection_from_plain_ost_profile() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_134_1044_2007,
    )

    assert selection.profile is MethodologyProfile.OST_134_1044_2007
    assert selection.sep_model_family is SourceModelFamily.OST_134_1044_2007
    assert selection.gcr_model_family is SourceModelFamily.OST_134_1044_2007
    assert selection.erb_model_family is SourceModelFamily.OST_134_1044_2007


def test_source_model_selection_from_gost_sep_gcr_profile() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    assert selection.sep_model_family is SourceModelFamily.GOST_SEP
    assert selection.gcr_model_family is SourceModelFamily.GOST_GCR
    assert selection.erb_model_family is SourceModelFamily.OST_134_1044_2007


def test_source_model_selection_from_custom_profile() -> None:
    selection = SourceModelSelectionConfig.from_profile(MethodologyProfile.CUSTOM)

    assert selection.sep_model_family is SourceModelFamily.CUSTOM
    assert selection.gcr_model_family is SourceModelFamily.CUSTOM
    assert selection.erb_model_family is SourceModelFamily.CUSTOM


def test_source_model_selection_families_by_source() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP_GCR,
    )

    assert selection.families_by_source == {
        RadiationSource.SEP: SourceModelFamily.GOST_SEP,
        RadiationSource.GCR: SourceModelFamily.GOST_GCR,
        RadiationSource.ERB: SourceModelFamily.OST_134_1044_2007,
    }


def test_source_model_selection_model_family_for_source() -> None:
    selection = SourceModelSelectionConfig.from_profile(
        MethodologyProfile.OST_WITH_GOST_SEP,
    )

    assert selection.model_family_for_source(RadiationSource.SEP) is SourceModelFamily.GOST_SEP
    assert (
        selection.model_family_for_source(RadiationSource.GCR)
        is SourceModelFamily.OST_134_1044_2007
    )


def test_source_model_selection_rejects_profile_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match methodology profile"):
        SourceModelSelectionConfig(
            profile=MethodologyProfile.OST_WITH_GOST_SEP,
            sep_model_family=SourceModelFamily.OST_134_1044_2007,
            gcr_model_family=SourceModelFamily.OST_134_1044_2007,
            erb_model_family=SourceModelFamily.OST_134_1044_2007,
        )


def test_methodology_config_exposes_source_model_selection() -> None:
    methodology = MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_GCR)

    assert methodology.source_model_selection.gcr_model_family is SourceModelFamily.GOST_GCR
    assert (
        methodology.source_model_selection.sep_model_family
        is SourceModelFamily.OST_134_1044_2007
    )


def test_calculation_config_exposes_source_model_selection() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=5)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)
    config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_WITH_GOST_SEP),
    )

    assert config.source_model_selection.sep_model_family is SourceModelFamily.GOST_SEP
    assert (
        config.source_model_selection.gcr_model_family
        is SourceModelFamily.OST_134_1044_2007
    )
