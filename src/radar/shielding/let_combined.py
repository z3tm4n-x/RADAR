"""Combination utilities for LET spectra."""

from __future__ import annotations

from math import exp, log
from typing import Iterable

from radar.core.spectra import ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.core.types import (
    LetSpectrumRole,
    Particle,
    SpectrumCompleteness,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit

HZE_AGGREGATE_LET_MODEL = "hze_direct_ion_aggregate_let"
MIXED_LET_DIAGNOSTIC_MODEL = "mixed_proton_hze_let_diagnostic"
# Compatibility name for callers that intentionally request the mixed
# diagnostic.  It no longer identifies a canonical assessment spectrum.
COMBINED_LET_MODEL = MIXED_LET_DIAGNOSTIC_MODEL

_LET_QUANTITIES = (
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)

_LET_UNITS = (
    Unit.DIFFERENTIAL_LET_FLUENCE,
    Unit.DIFFERENTIAL_LET_FLUX,
)


def _log_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    if len(centers) < 2:
        msg = "At least two LET centers are required to build logarithmic edges."
        raise ValueError(msg)

    if any(center <= 0.0 for center in centers):
        msg = "LET centers must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(centers, centers[1:])):
        msg = "LET centers must be strictly increasing."
        raise ValueError(msg)

    log_centers = tuple(log(center) for center in centers)
    edges: list[float] = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (log_centers[index - 1] + log_centers[index]))

    edges[0] = exp(log_centers[0] - 0.5 * (log_centers[1] - log_centers[0]))
    edges[-1] = exp(log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2]))

    return tuple(edges)


def _validate_let_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.x_unit is not Unit.LET:
        msg = "LET spectrum x unit must be Unit.LET."
        raise ValueError(msg)

    if spectrum.y_unit not in _LET_UNITS:
        msg = "LET spectrum y unit must be a differential LET unit."
        raise ValueError(msg)

    if spectrum.quantity not in _LET_QUANTITIES:
        msg = "LET spectrum quantity must be a LET differential quantity."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "LET spectrum must contain at least two LET points."
        raise ValueError(msg)


def _integral_piecewise_constant(spectrum: Spectrum1D) -> float:
    edges = _log_edges_from_centers(spectrum.x)

    return sum(
        value * (edges[index + 1] - edges[index])
        for index, value in enumerate(spectrum.y)
    )


def rebin_let_spectrum_to_grid(
    *,
    spectrum: Spectrum1D,
    let_centers: tuple[float, ...],
) -> Spectrum1D:
    """Conservatively rebin a LET spectrum to a target LET grid."""

    _validate_let_spectrum(spectrum)

    source_edges = (
        spectrum.bin_edges
        if spectrum.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
        else _log_edges_from_centers(spectrum.x)
    )
    assert source_edges is not None
    target_edges = _log_edges_from_centers(let_centers)
    target_counts = [0.0 for _ in let_centers]

    for source_index, source_value in enumerate(spectrum.y):
        source_left = source_edges[source_index]
        source_right = source_edges[source_index + 1]
        source_width = source_right - source_left

        if source_width <= 0.0 or source_value == 0.0:
            continue

        for target_index in range(len(let_centers)):
            target_left = target_edges[target_index]
            target_right = target_edges[target_index + 1]
            overlap_left = max(source_left, target_left)
            overlap_right = min(source_right, target_right)
            overlap = overlap_right - overlap_left

            if overlap <= 0.0:
                continue

            target_counts[target_index] += source_value * overlap

    target_values = tuple(
        count / (target_edges[index + 1] - target_edges[index])
        for index, count in enumerate(target_counts)
    )

    return derive_spectrum(
        spectrum,
        x=let_centers,
        y=target_values,
        model=f"{spectrum.model}+rebin_let",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=target_edges,
        transformation=TransformationStep(
            name="conservative_let_rebin",
            version="1",
            parameters=(("target_points", str(len(let_centers))),),
        ),
    )


def _as_tuple(spectra: Iterable[Spectrum1D]) -> tuple[Spectrum1D, ...]:
    result = tuple(spectra)

    if not result:
        msg = "At least one LET spectrum is required."
        raise ValueError(msg)

    return result


def combine_let_spectra(
    *,
    spectra: Iterable[Spectrum1D],
    let_centers: tuple[float, ...] | None = None,
    particle: Particle | None = None,
    model: str | None = None,
    let_role: LetSpectrumRole | None = None,
) -> Spectrum1D:
    """Return a combined LET spectrum on a common LET grid."""

    spectra_tuple = _as_tuple(spectra)
    first = spectra_tuple[0]
    _validate_let_spectrum(first)

    for spectrum in spectra_tuple[1:]:
        _validate_let_spectrum(spectrum)

        if spectrum.source is not first.source:
            msg = "Combined LET spectra must have the same radiation source."
            raise ValueError(msg)

        if spectrum.quantity is not first.quantity:
            msg = "Combined LET spectra must have the same quantity."
            raise ValueError(msg)

        if spectrum.y_unit is not first.y_unit:
            msg = "Combined LET spectra must have the same y unit."
            raise ValueError(msg)

        for field_name in (
            "angular_basis",
            "temporal_basis",
            "location",
            "shielding",
            "assessment_mode",
            "mission_duration_s",
        ):
            if getattr(spectrum, field_name) != getattr(first, field_name):
                msg = f"Combined LET spectra must have the same {field_name}."
                raise ValueError(msg)

    output_centers = let_centers or first.x
    _ = _log_edges_from_centers(output_centers)

    rebinned = tuple(
        rebin_let_spectrum_to_grid(spectrum=spectrum, let_centers=output_centers)
        for spectrum in spectra_tuple
    )
    total_values = tuple(
        sum(spectrum.y[index] for spectrum in rebinned)
        for index in range(len(output_centers))
    )

    additional_document_refs: list[str] = []
    additional_dataset_refs: list[str] = []
    incomplete_reasons: list[str] = []
    completeness = first.completeness
    for spectrum in spectra_tuple:
        if spectrum.model_reference is not None:
            additional_document_refs.extend(spectrum.model_reference.document_refs)
            additional_dataset_refs.extend(spectrum.model_reference.dataset_refs)
        if spectrum.completeness is SpectrumCompleteness.INCOMPLETE:
            completeness = SpectrumCompleteness.INCOMPLETE
            incomplete_reasons.extend(spectrum.incomplete_reasons)

    input_particles = {spectrum.particle for spectrum in spectra_tuple}
    if particle is None:
        particle = (
            next(iter(input_particles))
            if len(input_particles) == 1
            else Particle.MIXED_ION
        )

    if let_role is None:
        if particle is Particle.HZE and all(
            spectrum.particle is Particle.HZE for spectrum in spectra_tuple
        ):
            let_role = (
                LetSpectrumRole.HZE_DIRECT_ION
                if len(spectra_tuple) > 1
                else LetSpectrumRole.HZE_SPECIES
            )
        elif particle is Particle.MIXED_ION:
            let_role = LetSpectrumRole.MIXED_DIAGNOSTIC
        elif particle is Particle.PROTON:
            let_role = LetSpectrumRole.PROTON_DIAGNOSTIC
        else:
            msg = "Combined LET spectrum requires an explicit compatible LET role."
            raise ValueError(msg)

    species = first.species
    if let_role is LetSpectrumRole.HZE_DIRECT_ION:
        species = ParticleSpecies(symbol="HZE aggregate", is_aggregate=True)
    elif let_role is LetSpectrumRole.MIXED_DIAGNOSTIC:
        species = ParticleSpecies(symbol="proton+HZE aggregate", is_aggregate=True)

    output = derive_spectrum(
        rebinned[0],
        x=output_centers,
        y=total_values,
        particle=particle,
        model=model or COMBINED_LET_MODEL,
        species=species,
        let_role=let_role,
        transformation=TransformationStep(
            name="combine_let_spectra",
            version="1",
            parameters=(
                ("input_count", str(len(spectra_tuple))),
                ("input_models", "|".join(spectrum.model for spectrum in spectra_tuple)),
            ),
        ),
        additional_document_refs=tuple(dict.fromkeys(additional_document_refs)),
        additional_dataset_refs=tuple(dict.fromkeys(additional_dataset_refs)),
        completeness=completeness,
        incomplete_reasons=tuple(dict.fromkeys(incomplete_reasons)),
    )
    return output


def combine_hze_let_spectra(
    *,
    hze_let_spectra: Iterable[Spectrum1D],
    let_centers: tuple[float, ...] | None = None,
    model: str | None = None,
) -> Spectrum1D:
    """Return the HZE-only aggregate used by the direct-ion SEE model."""

    hze_tuple = tuple(hze_let_spectra)
    if not hze_tuple:
        msg = "HZE aggregate LET requires at least one HZE LET spectrum."
        raise ValueError(msg)
    if any(spectrum.particle is not Particle.HZE for spectrum in hze_tuple):
        msg = "HZE aggregate LET accepts only HZE LET spectra."
        raise ValueError(msg)

    return combine_let_spectra(
        spectra=hze_tuple,
        let_centers=let_centers,
        particle=Particle.HZE,
        model=model or HZE_AGGREGATE_LET_MODEL,
        let_role=LetSpectrumRole.HZE_DIRECT_ION,
    )


def combine_proton_and_hze_let_spectra(
    *,
    proton_let: Spectrum1D,
    hze_let_spectra: Iterable[Spectrum1D],
    let_centers: tuple[float, ...] | None = None,
    model: str | None = None,
) -> Spectrum1D:
    """Return combined proton + HZE LET spectrum."""

    if proton_let.particle is not Particle.PROTON:
        msg = "Combined proton/HZE LET requires a proton LET spectrum."
        raise ValueError(msg)

    hze_tuple = tuple(hze_let_spectra)

    if not hze_tuple:
        msg = "Combined proton/HZE LET requires at least one HZE LET spectrum."
        raise ValueError(msg)

    for spectrum in hze_tuple:
        if spectrum.particle is not Particle.HZE:
            msg = "Combined proton/HZE LET requires HZE LET spectra."
            raise ValueError(msg)

    combined = combine_let_spectra(
        spectra=(proton_let, *hze_tuple),
        let_centers=let_centers,
        particle=Particle.MIXED_ION,
        model=model or MIXED_LET_DIAGNOSTIC_MODEL,
        let_role=LetSpectrumRole.MIXED_DIAGNOSTIC,
    )
    return mark_spectrum_incomplete(
        combined,
        "mixed proton+HZE LET is not a canonical HZE production product",
    )


__all__ = [
    "COMBINED_LET_MODEL",
    "HZE_AGGREGATE_LET_MODEL",
    "MIXED_LET_DIAGNOSTIC_MODEL",
    "combine_hze_let_spectra",
    "combine_let_spectra",
    "combine_proton_and_hze_let_spectra",
    "rebin_let_spectrum_to_grid",
]
