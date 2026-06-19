"""Methodology profiles supported by RADAR."""

from enum import StrEnum


class MethodologyProfile(StrEnum):
    """Calculation methodology profile."""

    OST_134_1044_2007 = "ost_134_1044_2007"
    OST_WITH_GOST_SEP = "ost_with_gost_sep"
    OST_WITH_GOST_GCR = "ost_with_gost_gcr"
    OST_WITH_GOST_SEP_GCR = "ost_with_gost_sep_gcr"
    CUSTOM = "custom"


DEFAULT_METHODOLOGY_PROFILE = MethodologyProfile.OST_134_1044_2007
