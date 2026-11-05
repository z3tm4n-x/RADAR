"""Safe operations on typed spectra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite, log
from typing import TypeVar, cast

from radar.core.spectra import (
    ModelReference,
    ParticleSpecies,
    ShieldingDescriptor,
    Spectrum1D,
    TransformationStep,
)
from radar.core.types import (
    AssessmentMode,
    LetSpectrumRole,
    Particle,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit


_T = TypeVar("_T")
_UNCHANGED = object()


def _updated_model_reference(
    spectrum: Spectrum1D,
    *,
    model: str,
    additional_document_refs: tuple[str, ...],
    additional_dataset_refs: tuple[str, ...],
) -> ModelReference | None:
    reference = spectrum.model_reference
    if reference is None:
        return None

    return replace(
        reference,
        name=model,
        document_refs=tuple(
            dict.fromkeys((*reference.document_refs, *additional_document_refs))
        ),
        dataset_refs=tuple(
            dict.fromkeys((*reference.dataset_refs, *additional_dataset_refs))
        ),
    )


def _value_or_existing(value: _T | object, existing: _T) -> _T:
    if value is _UNCHANGED:
        return existing
    return cast(_T, value)


def derive_spectrum(
    spectrum: Spectrum1D,
    *,
    x: tuple[float, ...] | object = _UNCHANGED,
    y: tuple[float, ...] | object = _UNCHANGED,
    x_unit: Unit | object = _UNCHANGED,
    y_unit: Unit | object = _UNCHANGED,
    quantity: SpectrumQuantity | object = _UNCHANGED,
    particle: Particle | object = _UNCHANGED,
    model: str | None = None,
    axis_kind: SpectrumAxisKind | None | object = _UNCHANGED,
    temporal_basis: TemporalBasis | None | object = _UNCHANGED,
    location: SpectrumLocation | None | object = _UNCHANGED,
    species: ParticleSpecies | None | object = _UNCHANGED,
    shielding: ShieldingDescriptor | None | object = _UNCHANGED,
    assessment_mode: AssessmentMode | None | object = _UNCHANGED,
    mission_duration_s: float | None | object = _UNCHANGED,
    transformation: TransformationStep | None = None,
    additional_document_refs: tuple[str, ...] = (),
    additional_dataset_refs: tuple[str, ...] = (),
    completeness: SpectrumCompleteness | None | object = _UNCHANGED,
    incomplete_reasons: tuple[str, ...] | object = _UNCHANGED,
    let_role: LetSpectrumRole | None | object = _UNCHANGED,
    ordinate_basis: SpectrumOrdinateBasis | object = _UNCHANGED,
    bin_edges: tuple[float, ...] | None | object = _UNCHANGED,
) -> Spectrum1D:
    """Create a derived spectrum without dropping its physical metadata.

    This is the canonical operation boundary for attenuation, shielding,
    rebinning and coordinate transforms.  The legacy ``model`` string remains
    synchronized with ``ModelReference.name`` while the structured
    transformation chain records how the values were produced.
    """

    output_model = model or spectrum.model
    output_chain = spectrum.transformation_chain
    if transformation is not None:
        output_chain = (*output_chain, transformation)

    return replace(
        spectrum,
        x=_value_or_existing(x, spectrum.x),
        y=_value_or_existing(y, spectrum.y),
        x_unit=_value_or_existing(x_unit, spectrum.x_unit),
        y_unit=_value_or_existing(y_unit, spectrum.y_unit),
        quantity=_value_or_existing(quantity, spectrum.quantity),
        particle=_value_or_existing(particle, spectrum.particle),
        model=output_model,
        axis_kind=_value_or_existing(axis_kind, spectrum.axis_kind),
        temporal_basis=_value_or_existing(temporal_basis, spectrum.temporal_basis),
        location=_value_or_existing(location, spectrum.location),
        species=_value_or_existing(species, spectrum.species),
        shielding=_value_or_existing(shielding, spectrum.shielding),
        assessment_mode=_value_or_existing(
            assessment_mode,
            spectrum.assessment_mode,
        ),
        mission_duration_s=_value_or_existing(
            mission_duration_s,
            spectrum.mission_duration_s,
        ),
        model_reference=_updated_model_reference(
            spectrum,
            model=output_model,
            additional_document_refs=additional_document_refs,
            additional_dataset_refs=additional_dataset_refs,
        ),
        transformation_chain=output_chain,
        completeness=_value_or_existing(completeness, spectrum.completeness),
        incomplete_reasons=_value_or_existing(
            incomplete_reasons,
            spectrum.incomplete_reasons,
        ),
        let_role=_value_or_existing(let_role, spectrum.let_role),
        ordinate_basis=_value_or_existing(ordinate_basis, spectrum.ordinate_basis),
        bin_edges=_value_or_existing(bin_edges, spectrum.bin_edges),
    )


def mark_spectrum_incomplete(
    spectrum: Spectrum1D,
    *reasons: str,
) -> Spectrum1D:
    """Return ``spectrum`` explicitly quarantined from production release."""

    merged_reasons = tuple(dict.fromkeys((*spectrum.incomplete_reasons, *reasons)))
    if not merged_reasons:
        msg = "At least one incomplete-spectrum reason is required."
        raise ValueError(msg)

    return derive_spectrum(
        spectrum,
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=merged_reasons,
    )


def _validate_differential_integration_grid(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
    context: str,
) -> None:
    if not energies:
        msg = f"{context} energy grid must not be empty."
        raise ValueError(msg)

    if len(energies) != len(differential_values):
        msg = f"{context} energy and differential-value grids must have the same length."
        raise ValueError(msg)

    if any(not isfinite(energy) or energy <= 0.0 for energy in energies):
        msg = f"{context} energy grid values must be finite and positive."
        raise ValueError(msg)

    if tuple(sorted(energies)) != tuple(energies):
        msg = f"{context} energy grid must be sorted."
        raise ValueError(msg)

    if len(set(energies)) != len(energies):
        msg = f"{context} energy grid values must be unique."
        raise ValueError(msg)

    if any(not isfinite(value) or value < 0.0 for value in differential_values):
        msg = f"{context} differential values must be finite and non-negative."
        raise ValueError(msg)


def _power_law_interval_integral(
    *,
    lower_energy: float,
    upper_energy: float,
    lower_value: float,
    upper_value: float,
) -> float:
    if lower_value <= 0.0 and upper_value <= 0.0:
        return 0.0

    if lower_value > 0.0 and upper_value > 0.0:
        exponent = -log(upper_value / lower_value) / log(upper_energy / lower_energy)

        if abs(exponent - 1.0) < 1.0e-8:
            return lower_value * lower_energy * log(upper_energy / lower_energy)

        return float(
            lower_value
            * lower_energy**exponent
            * (
                upper_energy ** (1.0 - exponent)
                - lower_energy ** (1.0 - exponent)
            )
            / (1.0 - exponent)
        )

    return 0.5 * (lower_value + upper_value) * (upper_energy - lower_energy)


def _power_law_tail_integral(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
) -> float:
    positive_indices = [
        index
        for index, value in enumerate(differential_values)
        if value > 0.0
    ]

    if len(positive_indices) < 2:
        return 0.0

    upper_index = positive_indices[-1]
    lower_index = positive_indices[-2]
    lower_value = differential_values[lower_index]
    upper_value = differential_values[upper_index]
    lower_energy = energies[lower_index]
    upper_energy = energies[upper_index]

    exponent = -log(upper_value / lower_value) / log(upper_energy / lower_energy)

    if exponent <= 1.05:
        return 0.0

    return upper_value * upper_energy / (exponent - 1.0)


def integrate_differential_spectrum_tail_power_law(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
    context: str = "Spectrum",
) -> tuple[float, ...]:
    """Return integral values from tabulated differential values.

    Positive intervals use power-law interpolation in energy. Mixed zero/non-zero
    intervals use trapezoidal integration. The high-energy tail is extrapolated
    from the last two positive points only when the inferred power-law exponent
    gives a finite integral.
    """

    _validate_differential_integration_grid(
        energies=energies,
        differential_values=differential_values,
        context=context,
    )

    integral_values = [0.0 for _ in differential_values]

    accumulated = _power_law_tail_integral(
        energies=energies,
        differential_values=differential_values,
    )
    integral_values[-1] = accumulated

    for index in range(len(energies) - 2, -1, -1):
        accumulated += _power_law_interval_integral(
            lower_energy=energies[index],
            upper_energy=energies[index + 1],
            lower_value=differential_values[index],
            upper_value=differential_values[index + 1],
        )
        integral_values[index] = accumulated

    return tuple(integral_values)


def integrate_differential_spectrum_finite_upper_bound(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
    upper_energy_bound: float,
    context: str = "Spectrum",
) -> tuple[float, ...]:
    """Return integral values over a finite, explicitly declared domain.

    Tabulated intervals use the same local power-law interpolation as
    :func:`integrate_differential_spectrum_tail_power_law`.  When the declared
    upper boundary is above the final sampled energy, only that finite terminal
    interval is continued using the slope of the final two samples.  No
    integration to infinity is performed.

    A positive terminal value requires two positive terminal samples so that
    the bounded continuation is defined.  A terminal zero contributes no
    continuation beyond the final sample; callers must separately preserve any
    incompleteness flag that caused a physical spectrum to be truncated to
    zero.
    """

    _validate_differential_integration_grid(
        energies=energies,
        differential_values=differential_values,
        context=context,
    )

    if not isfinite(upper_energy_bound) or upper_energy_bound <= 0.0:
        msg = f"{context} finite upper energy bound must be finite and positive."
        raise ValueError(msg)

    final_energy = energies[-1]
    if upper_energy_bound < final_energy:
        msg = (
            f"{context} finite upper energy bound must not be below the final "
            "sampled energy."
        )
        raise ValueError(msg)

    accumulated = 0.0
    final_value = differential_values[-1]
    if upper_energy_bound > final_energy and final_value > 0.0:
        if len(energies) < 2 or differential_values[-2] <= 0.0:
            msg = (
                f"{context} finite terminal interval requires two positive "
                "terminal differential samples."
            )
            raise ValueError(msg)

        previous_energy = energies[-2]
        previous_value = differential_values[-2]
        exponent = -log(final_value / previous_value) / log(
            final_energy / previous_energy
        )
        boundary_value = final_value * (upper_energy_bound / final_energy) ** (
            -exponent
        )
        accumulated = _power_law_interval_integral(
            lower_energy=final_energy,
            upper_energy=upper_energy_bound,
            lower_value=final_value,
            upper_value=boundary_value,
        )

    integral_values = [0.0 for _ in differential_values]
    integral_values[-1] = accumulated

    for index in range(len(energies) - 2, -1, -1):
        accumulated += _power_law_interval_integral(
            lower_energy=energies[index],
            upper_energy=energies[index + 1],
            lower_value=differential_values[index],
            upper_value=differential_values[index + 1],
        )
        integral_values[index] = accumulated

    return tuple(integral_values)


def _check_same_grid(left: Spectrum1D, right: Spectrum1D) -> None:
    if left.x != right.x:
        msg = "Spectra must have the same grid."
        raise ValueError(msg)


def _check_compatible_metadata(left: Spectrum1D, right: Spectrum1D) -> None:
    if left.x_unit is not right.x_unit:
        msg = "Spectra must have the same x unit."
        raise ValueError(msg)

    if left.y_unit is not right.y_unit:
        msg = "Spectra must have the same y unit."
        raise ValueError(msg)

    if left.quantity is not right.quantity:
        msg = "Spectra must have the same quantity."
        raise ValueError(msg)

    if left.particle is not right.particle:
        msg = "Spectra must have the same particle."
        raise ValueError(msg)

    if left.source is not right.source:
        msg = "Spectra must have the same radiation source."
        raise ValueError(msg)

    structured_semantics = (
        "axis_kind",
        "angular_basis",
        "temporal_basis",
        "location",
        "species",
        "shielding",
        "assessment_mode",
        "mission_duration_s",
        "let_role",
        "ordinate_basis",
        "bin_edges",
    )
    for field_name in structured_semantics:
        if getattr(left, field_name) != getattr(right, field_name):
            msg = f"Spectra must have the same {field_name}."
            raise ValueError(msg)


def check_spectra_compatible(left: Spectrum1D, right: Spectrum1D) -> None:
    """Validate that two spectra may be combined pointwise."""

    _check_same_grid(left=left, right=right)
    _check_compatible_metadata(left=left, right=right)


def scale_spectrum(
    spectrum: Spectrum1D,
    factor: float,
    model: str | None = None,
) -> Spectrum1D:
    """Return a spectrum multiplied by a non-negative finite factor."""

    if not isfinite(factor):
        msg = "Spectrum scale factor must be finite."
        raise ValueError(msg)

    if factor < 0.0:
        msg = "Spectrum scale factor must be non-negative."
        raise ValueError(msg)

    output_model = model

    if output_model is None:
        output_model = f"{spectrum.model}*{factor:g}"

    return derive_spectrum(
        spectrum,
        y=tuple(value * factor for value in spectrum.y),
        model=output_model,
        transformation=TransformationStep(
            name="scale_spectrum",
            version="1",
            parameters=(("factor", f"{factor:.17g}"),),
        ),
    )


def add_spectra(
    left: Spectrum1D,
    right: Spectrum1D,
    model: str | None = None,
) -> Spectrum1D:
    """Return pointwise sum of two compatible spectra."""

    check_spectra_compatible(left=left, right=right)

    output_model = model

    if output_model is None:
        output_model = f"{left.model}+{right.model}"

    completeness = left.completeness
    incomplete_reasons = left.incomplete_reasons
    if right.completeness is SpectrumCompleteness.INCOMPLETE:
        completeness = SpectrumCompleteness.INCOMPLETE
        incomplete_reasons = tuple(
            dict.fromkeys((*incomplete_reasons, *right.incomplete_reasons))
        )

    additional_document_refs: tuple[str, ...] = ()
    additional_dataset_refs: tuple[str, ...] = ()
    if right.model_reference is not None:
        additional_document_refs = right.model_reference.document_refs
        additional_dataset_refs = right.model_reference.dataset_refs

    return derive_spectrum(
        left,
        y=tuple(left_value + right_value for left_value, right_value in zip(left.y, right.y)),
        model=output_model,
        transformation=TransformationStep(
            name="pointwise_spectrum_sum",
            version="1",
            parameters=(("right_model", right.model),),
        ),
        additional_document_refs=additional_document_refs,
        additional_dataset_refs=additional_dataset_refs,
        completeness=completeness,
        incomplete_reasons=incomplete_reasons,
    )
