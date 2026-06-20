"""Methodology profiles supported by RADAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MethodologyProfile(StrEnum):
    """Calculation methodology profile."""

    OST_134_1044_2007 = "ost_134_1044_2007"
    OST_WITH_GOST_SEP = "ost_with_gost_sep"
    OST_WITH_GOST_GCR = "ost_with_gost_gcr"
    OST_WITH_GOST_SEP_GCR = "ost_with_gost_sep_gcr"
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
