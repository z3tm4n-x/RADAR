"""Registered source model classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from radar.core.profiles import (
    MethodologyProfile,
    SourceModelFamily,
    expected_source_model_family_for_profile,
)
from radar.core.types import RadiationSource
from radar.erb.model import OstErbModel
from radar.gcr.model import GostGcrModel, OstGcrModel
from radar.sep.model import GostSepModel, OstSepModel


@dataclass(frozen=True)
class SourceModelRegistration:
    """Registered source model class."""

    source: RadiationSource
    model_family: SourceModelFamily
    model_class: type[Any]

    @property
    def key(self) -> tuple[RadiationSource, SourceModelFamily]:
        """Return registry key."""

        return (self.source, self.model_family)


REGISTERED_SOURCE_MODELS: tuple[SourceModelRegistration, ...] = (
    SourceModelRegistration(
        source=RadiationSource.SEP,
        model_family=SourceModelFamily.OST_134_1044_2007,
        model_class=OstSepModel,
    ),
    SourceModelRegistration(
        source=RadiationSource.SEP,
        model_family=SourceModelFamily.GOST_SEP,
        model_class=GostSepModel,
    ),
    SourceModelRegistration(
        source=RadiationSource.GCR,
        model_family=SourceModelFamily.OST_134_1044_2007,
        model_class=OstGcrModel,
    ),
    SourceModelRegistration(
        source=RadiationSource.GCR,
        model_family=SourceModelFamily.GOST_GCR,
        model_class=GostGcrModel,
    ),
    SourceModelRegistration(
        source=RadiationSource.ERB,
        model_family=SourceModelFamily.OST_134_1044_2007,
        model_class=OstErbModel,
    ),
)

_SOURCE_MODEL_REGISTRY: dict[
    tuple[RadiationSource, SourceModelFamily],
    SourceModelRegistration,
] = {
    registration.key: registration
    for registration in REGISTERED_SOURCE_MODELS
}


def registered_source_model_families(
    source: RadiationSource,
) -> tuple[SourceModelFamily, ...]:
    """Return registered model families for a radiation source."""

    return tuple(
        registration.model_family
        for registration in REGISTERED_SOURCE_MODELS
        if registration.source is source
    )


def source_model_registration(
    source: RadiationSource,
    model_family: SourceModelFamily,
) -> SourceModelRegistration:
    """Return registered source model class metadata."""

    try:
        return _SOURCE_MODEL_REGISTRY[(source, model_family)]
    except KeyError as exc:
        msg = (
            f"No registered source model for source {source.value} "
            f"and model family {model_family.value}."
        )
        raise ValueError(msg) from exc


def source_model_class(
    source: RadiationSource,
    model_family: SourceModelFamily,
) -> type[Any]:
    """Return registered source model class."""

    return source_model_registration(
        source=source,
        model_family=model_family,
    ).model_class


def source_model_registration_for_profile(
    profile: MethodologyProfile,
    source: RadiationSource,
) -> SourceModelRegistration:
    """Return registered source model class for a profile and source."""

    model_family = expected_source_model_family_for_profile(
        profile=profile,
        source=source,
    )
    return source_model_registration(
        source=source,
        model_family=model_family,
    )


def source_model_class_for_profile(
    profile: MethodologyProfile,
    source: RadiationSource,
) -> type[Any]:
    """Return registered source model class for a profile and source."""

    return source_model_registration_for_profile(
        profile=profile,
        source=source,
    ).model_class
