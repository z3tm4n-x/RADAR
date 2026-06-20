"""Methodology profiles supported by RADAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from radar.core.types import RadiationSource


class MethodologyProfile(StrEnum):
    """Calculation methodology profile."""

    OST_134_1044_2007 = "ost_134_1044_2007"
    OST_WITH_GOST_SEP = "ost_with_gost_sep"
    OST_WITH_GOST_GCR = "ost_with_gost_gcr"
    OST_WITH_GOST_SEP_GCR = "ost_with_gost_sep_gcr"
    CUSTOM = "custom"


class SourceModelFamily(StrEnum):
    """Source model family selected by a methodology profile."""

    OST_134_1044_2007 = "ost_134_1044_2007"
    GOST_SEP = "gost_sep"
    GOST_GCR = "gost_gcr"
    CUSTOM = "custom"


DEFAULT_METHODOLOGY_PROFILE = MethodologyProfile.OST_134_1044_2007

OST_134_1044_2007_DOCUMENT = "OST 134-1044-2007"
GOST_SEP_DOCUMENT = "GOST SEP"
GOST_GCR_DOCUMENT = "GOST GCR"


@dataclass(frozen=True)
class MethodologyProfileSpec:
    """Explicit calculation-methodology specification."""

    profile: MethodologyProfile
    documents: tuple[str, ...]
    uses_ost_134_1044_2007: bool
    uses_gost_sep: bool
    uses_gost_gcr: bool
    is_custom: bool = False

    def __post_init__(self) -> None:
        if not self.documents:
            msg = "Methodology profile must reference at least one document."
            raise ValueError(msg)

        if self.is_custom and self.profile is not MethodologyProfile.CUSTOM:
            msg = "Only custom methodology profile may be marked custom."
            raise ValueError(msg)

        if not self.is_custom and self.profile is MethodologyProfile.CUSTOM:
            msg = "Custom methodology profile must be marked custom."
            raise ValueError(msg)

        if self.uses_gost_sep and GOST_SEP_DOCUMENT not in self.documents:
            msg = "GOST SEP profile must reference GOST SEP document."
            raise ValueError(msg)

        if self.uses_gost_gcr and GOST_GCR_DOCUMENT not in self.documents:
            msg = "GOST GCR profile must reference GOST GCR document."
            raise ValueError(msg)

        if self.uses_ost_134_1044_2007 and OST_134_1044_2007_DOCUMENT not in self.documents:
            msg = "OST methodology profile must reference OST 134-1044-2007."
            raise ValueError(msg)


METHODOLOGY_PROFILE_SPECS: dict[MethodologyProfile, MethodologyProfileSpec] = {
    MethodologyProfile.OST_134_1044_2007: MethodologyProfileSpec(
        profile=MethodologyProfile.OST_134_1044_2007,
        documents=(OST_134_1044_2007_DOCUMENT,),
        uses_ost_134_1044_2007=True,
        uses_gost_sep=False,
        uses_gost_gcr=False,
    ),
    MethodologyProfile.OST_WITH_GOST_SEP: MethodologyProfileSpec(
        profile=MethodologyProfile.OST_WITH_GOST_SEP,
        documents=(
            OST_134_1044_2007_DOCUMENT,
            GOST_SEP_DOCUMENT,
        ),
        uses_ost_134_1044_2007=True,
        uses_gost_sep=True,
        uses_gost_gcr=False,
    ),
    MethodologyProfile.OST_WITH_GOST_GCR: MethodologyProfileSpec(
        profile=MethodologyProfile.OST_WITH_GOST_GCR,
        documents=(
            OST_134_1044_2007_DOCUMENT,
            GOST_GCR_DOCUMENT,
        ),
        uses_ost_134_1044_2007=True,
        uses_gost_sep=False,
        uses_gost_gcr=True,
    ),
    MethodologyProfile.OST_WITH_GOST_SEP_GCR: MethodologyProfileSpec(
        profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR,
        documents=(
            OST_134_1044_2007_DOCUMENT,
            GOST_SEP_DOCUMENT,
            GOST_GCR_DOCUMENT,
        ),
        uses_ost_134_1044_2007=True,
        uses_gost_sep=True,
        uses_gost_gcr=True,
    ),
    MethodologyProfile.CUSTOM: MethodologyProfileSpec(
        profile=MethodologyProfile.CUSTOM,
        documents=("custom",),
        uses_ost_134_1044_2007=False,
        uses_gost_sep=False,
        uses_gost_gcr=False,
        is_custom=True,
    ),
}


def methodology_profile_spec(profile: MethodologyProfile) -> MethodologyProfileSpec:
    """Return explicit specification for a methodology profile."""

    try:
        return METHODOLOGY_PROFILE_SPECS[profile]
    except KeyError as exc:
        msg = f"Unsupported methodology profile: {profile}"
        raise ValueError(msg) from exc


def documents_for_methodology_profile(profile: MethodologyProfile) -> tuple[str, ...]:
    """Return source documents referenced by a methodology profile."""

    return methodology_profile_spec(profile).documents


def profile_uses_gost_sep(profile: MethodologyProfile) -> bool:
    """Return True if the profile uses GOST SEP model data."""

    return methodology_profile_spec(profile).uses_gost_sep


def profile_uses_gost_gcr(profile: MethodologyProfile) -> bool:
    """Return True if the profile uses GOST GCR model data."""

    return methodology_profile_spec(profile).uses_gost_gcr


def profile_uses_ost_134_1044_2007(profile: MethodologyProfile) -> bool:
    """Return True if the profile uses OST 134-1044-2007 methodology."""

    return methodology_profile_spec(profile).uses_ost_134_1044_2007

@dataclass(frozen=True)
class MethodologySourceModelContract:
    """Source-model family contract for a methodology profile."""

    profile: MethodologyProfile
    sep_model_family: SourceModelFamily
    gcr_model_family: SourceModelFamily
    erb_model_family: SourceModelFamily

    def __post_init__(self) -> None:
        if self.profile is MethodologyProfile.CUSTOM:
            expected = SourceModelFamily.CUSTOM
            if (
                self.sep_model_family is not expected
                or self.gcr_model_family is not expected
                or self.erb_model_family is not expected
            ):
                msg = "Custom methodology profile must use custom source model families."
                raise ValueError(msg)
            return

        if self.erb_model_family is not SourceModelFamily.OST_134_1044_2007:
            msg = "ERB model family must remain OST 134-1044-2007 for normative profiles."
            raise ValueError(msg)


METHODOLOGY_SOURCE_MODEL_CONTRACTS: dict[
    MethodologyProfile,
    MethodologySourceModelContract,
] = {
    MethodologyProfile.OST_134_1044_2007: MethodologySourceModelContract(
        profile=MethodologyProfile.OST_134_1044_2007,
        sep_model_family=SourceModelFamily.OST_134_1044_2007,
        gcr_model_family=SourceModelFamily.OST_134_1044_2007,
        erb_model_family=SourceModelFamily.OST_134_1044_2007,
    ),
    MethodologyProfile.OST_WITH_GOST_SEP: MethodologySourceModelContract(
        profile=MethodologyProfile.OST_WITH_GOST_SEP,
        sep_model_family=SourceModelFamily.GOST_SEP,
        gcr_model_family=SourceModelFamily.OST_134_1044_2007,
        erb_model_family=SourceModelFamily.OST_134_1044_2007,
    ),
    MethodologyProfile.OST_WITH_GOST_GCR: MethodologySourceModelContract(
        profile=MethodologyProfile.OST_WITH_GOST_GCR,
        sep_model_family=SourceModelFamily.OST_134_1044_2007,
        gcr_model_family=SourceModelFamily.GOST_GCR,
        erb_model_family=SourceModelFamily.OST_134_1044_2007,
    ),
    MethodologyProfile.OST_WITH_GOST_SEP_GCR: MethodologySourceModelContract(
        profile=MethodologyProfile.OST_WITH_GOST_SEP_GCR,
        sep_model_family=SourceModelFamily.GOST_SEP,
        gcr_model_family=SourceModelFamily.GOST_GCR,
        erb_model_family=SourceModelFamily.OST_134_1044_2007,
    ),
    MethodologyProfile.CUSTOM: MethodologySourceModelContract(
        profile=MethodologyProfile.CUSTOM,
        sep_model_family=SourceModelFamily.CUSTOM,
        gcr_model_family=SourceModelFamily.CUSTOM,
        erb_model_family=SourceModelFamily.CUSTOM,
    ),
}


def source_model_contract_for_profile(
    profile: MethodologyProfile,
) -> MethodologySourceModelContract:
    """Return source-model family contract for a methodology profile."""

    try:
        return METHODOLOGY_SOURCE_MODEL_CONTRACTS[profile]
    except KeyError as exc:
        msg = f"Unsupported methodology profile: {profile}"
        raise ValueError(msg) from exc


def expected_source_model_family_for_profile(
    profile: MethodologyProfile,
    source: RadiationSource,
) -> SourceModelFamily:
    """Return expected source model family for a profile and radiation source."""

    contract = source_model_contract_for_profile(profile)

    if source is RadiationSource.SEP:
        return contract.sep_model_family

    if source is RadiationSource.GCR:
        return contract.gcr_model_family

    if source is RadiationSource.ERB:
        return contract.erb_model_family

    msg = f"Unsupported radiation source: {source}"
    raise ValueError(msg)


def validate_source_model_family_for_profile(
    profile: MethodologyProfile,
    source: RadiationSource,
    model_family: SourceModelFamily,
) -> None:
    """Validate source model family against a methodology profile."""

    expected_family = expected_source_model_family_for_profile(
        profile=profile,
        source=source,
    )

    if model_family is expected_family:
        return

    msg = (
        f"{source.value} model family {model_family.value} does not match "
        f"methodology profile {profile.value}. "
        f"Expected model family: {expected_family.value}."
    )
    raise ValueError(msg)
