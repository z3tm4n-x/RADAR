"""Registered source model classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from radar.core.project import CalculationConfig, SourceModelSelectionConfig
from radar.core.profiles import (
    MethodologyProfile,
    SourceModelFamily,
    expected_source_model_family_for_profile,
)
from radar.core.types import RadiationSource
from radar.erb.model import OstErbModel
from radar.gcr.model import GostGcrModel, OstGcrModel
from radar.sep.model import GostSepModel, OstSepModel
from radar.solar_activity.model import (
    build_mission_solar_activity,
    mission_solar_activity_monthly_wolf_numbers,
)
from radar.solar_activity.ost import ost_wolf_number_cycle_table


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


@dataclass(frozen=True)
class SourceModelBundle:
    """Registered source model classes selected for a methodology profile."""

    profile: MethodologyProfile
    sep: SourceModelRegistration
    gcr: SourceModelRegistration
    erb: SourceModelRegistration

    @property
    def registrations(self) -> tuple[SourceModelRegistration, ...]:
        """Return registrations in calculation-source order."""

        return (self.sep, self.gcr, self.erb)

    @property
    def model_classes(self) -> tuple[type[Any], ...]:
        """Return model classes in calculation-source order."""

        return tuple(registration.model_class for registration in self.registrations)


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


def source_model_bundle_for_profile(profile: MethodologyProfile) -> SourceModelBundle:
    """Return registered source model bundle for a methodology profile."""

    return SourceModelBundle(
        profile=profile,
        sep=source_model_registration_for_profile(
            profile=profile,
            source=RadiationSource.SEP,
        ),
        gcr=source_model_registration_for_profile(
            profile=profile,
            source=RadiationSource.GCR,
        ),
        erb=source_model_registration_for_profile(
            profile=profile,
            source=RadiationSource.ERB,
        ),
    )


def _model_instance(
    registration: SourceModelRegistration,
    kwargs: dict[str, Any] | None,
) -> Any:
    """Instantiate a registered source model with optional constructor kwargs."""

    return registration.model_class(**(kwargs or {}))


def source_model_instances_for_profile(
    profile: MethodologyProfile,
    *,
    sep_kwargs: dict[str, Any] | None = None,
    gcr_kwargs: dict[str, Any] | None = None,
    erb_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Return SEP, GCR and ERB model instances for a methodology profile."""

    bundle = source_model_bundle_for_profile(profile)
    return (
        _model_instance(bundle.sep, sep_kwargs),
        _model_instance(bundle.gcr, gcr_kwargs),
        _model_instance(bundle.erb, erb_kwargs),
    )


def _normalized_sep_energy_grid_mev(
    sep_energy_grid_mev: tuple[float, ...],
) -> tuple[float, ...]:
    """Return a validated SEP proton energy grid."""

    if not sep_energy_grid_mev:
        msg = "SEP energy grid must not be empty."
        raise ValueError(msg)

    return tuple(float(value) for value in sep_energy_grid_mev)


def _ost_sep_kwargs_from_config(
    config: CalculationConfig,
    *,
    sep_energy_grid_mev: tuple[float, ...],
) -> dict[str, Any]:
    """Return constructor kwargs for configured OST SEP proton source model."""

    solar_activity = build_mission_solar_activity(
        mission=config.mission,
        cycle_table=ost_wolf_number_cycle_table(config.mission.solar_activity_level),
        reference_start_year=config.mission.launch_year,
    )

    return {
        "energy_grid_mev": _normalized_sep_energy_grid_mev(sep_energy_grid_mev),
        "monthly_smoothed_wolf_numbers": mission_solar_activity_monthly_wolf_numbers(
            solar_activity,
        ),
        "version": "proton_only",
    }


def source_model_instances_for_config(
    config: CalculationConfig,
    *,
    sep_energy_grid_mev: tuple[float, ...] | None = None,
) -> tuple[Any, Any, Any]:
    """Return SEP, GCR and ERB model instances configured from calculation input.

    Only the OST SEP proton source model is configurable at this layer for now.
    Other selected models are instantiated with their default constructor.
    """

    sep_kwargs: dict[str, Any] | None = None

    if (
        sep_energy_grid_mev is not None
        and config.source_model_selection.sep_model_family
        is SourceModelFamily.OST_134_1044_2007
    ):
        sep_kwargs = _ost_sep_kwargs_from_config(
            config,
            sep_energy_grid_mev=sep_energy_grid_mev,
        )

    return source_model_instances_for_selection(
        config.source_model_selection,
        sep_kwargs=sep_kwargs,
    )


def source_model_classes_for_profile(
    profile: MethodologyProfile,
) -> tuple[type[Any], type[Any], type[Any]]:
    """Return SEP, GCR and ERB model classes for a methodology profile."""

    bundle = source_model_bundle_for_profile(profile)
    return (
        bundle.sep.model_class,
        bundle.gcr.model_class,
        bundle.erb.model_class,
    )

def source_model_bundle_for_selection(
    selection: SourceModelSelectionConfig,
) -> SourceModelBundle:
    """Return registered source model bundle for explicit model family selection."""

    return SourceModelBundle(
        profile=selection.profile,
        sep=source_model_registration(
            source=RadiationSource.SEP,
            model_family=selection.sep_model_family,
        ),
        gcr=source_model_registration(
            source=RadiationSource.GCR,
            model_family=selection.gcr_model_family,
        ),
        erb=source_model_registration(
            source=RadiationSource.ERB,
            model_family=selection.erb_model_family,
        ),
    )


def source_model_instances_for_selection(
    selection: SourceModelSelectionConfig,
    *,
    sep_kwargs: dict[str, Any] | None = None,
    gcr_kwargs: dict[str, Any] | None = None,
    erb_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Return SEP, GCR and ERB model instances for explicit model family selection."""

    bundle = source_model_bundle_for_selection(selection)
    return (
        _model_instance(bundle.sep, sep_kwargs),
        _model_instance(bundle.gcr, gcr_kwargs),
        _model_instance(bundle.erb, erb_kwargs),
    )


def source_model_classes_for_selection(
    selection: SourceModelSelectionConfig,
) -> tuple[type[Any], type[Any], type[Any]]:
    """Return SEP, GCR and ERB model classes for explicit model family selection."""

    bundle = source_model_bundle_for_selection(selection)
    return (
        bundle.sep.model_class,
        bundle.gcr.model_class,
        bundle.erb.model_class,
    )
