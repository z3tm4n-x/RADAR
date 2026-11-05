"""GCR mission product aggregation policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D, TransformationStep
from radar.core.spectrum_ops import (
    derive_spectrum,
    integrate_differential_spectrum_finite_upper_bound,
)
from radar.core.types import (
    AssessmentMode,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit

GCR_MISSION_PRODUCTS_MODEL_SUFFIX: Final[str] = "mission_products"
GCR_FINITE_DOMAIN_INTEGRATION_STEP: Final[str] = "gcr_finite_domain_integration"


@dataclass(frozen=True)
class GcrFluxTimeBin:
    """One time bin of a GCR differential flux spectrum."""

    spectrum: Spectrum1D
    duration_seconds: float

    def __post_init__(self) -> None:
        _validate_gcr_flux_spectrum(self.spectrum)

        if not isfinite(self.duration_seconds):
            msg = "GCR time-bin duration must be finite."
            raise ValueError(msg)

        if self.duration_seconds <= 0.0:
            msg = "GCR time-bin duration must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class GcrMissionProducts:
    """Mission products derived from time-binned GCR flux spectra."""

    mean_flux: SpectrumProduct
    mean_integral_flux: SpectrumProduct
    maximum_flux: SpectrumProduct
    maximum_integral_flux: SpectrumProduct
    mission_fluence: SpectrumProduct
    mission_integral_fluence: SpectrumProduct

    @property
    def products(
        self,
    ) -> tuple[
        SpectrumProduct,
        SpectrumProduct,
        SpectrumProduct,
        SpectrumProduct,
        SpectrumProduct,
        SpectrumProduct,
    ]:
        """Return products in deterministic order."""

        return (
            self.mean_flux,
            self.mean_integral_flux,
            self.maximum_flux,
            self.maximum_integral_flux,
            self.mission_fluence,
            self.mission_integral_fluence,
        )

    def __post_init__(self) -> None:
        if self.mean_flux.kind is not RadiationProductKind.MEAN_FLUX:
            msg = "GCR mean-flux product kind must be MEAN_FLUX."
            raise ValueError(msg)

        if self.mean_integral_flux.kind is not RadiationProductKind.MEAN_FLUX:
            msg = "GCR mean-integral-flux product kind must be MEAN_FLUX."
            raise ValueError(msg)

        if self.maximum_flux.kind is not RadiationProductKind.MAXIMUM_FLUX:
            msg = "GCR maximum-flux product kind must be MAXIMUM_FLUX."
            raise ValueError(msg)

        if self.maximum_integral_flux.kind is not RadiationProductKind.MAXIMUM_FLUX:
            msg = "GCR maximum-integral-flux product kind must be MAXIMUM_FLUX."
            raise ValueError(msg)

        if self.mission_fluence.kind is not RadiationProductKind.MISSION_FLUENCE:
            msg = "GCR mission-fluence product kind must be MISSION_FLUENCE."
            raise ValueError(msg)

        if self.mission_integral_fluence.kind is not RadiationProductKind.MISSION_FLUENCE:
            msg = "GCR mission-integral-fluence product kind must be MISSION_FLUENCE."
            raise ValueError(msg)


def _validate_gcr_flux_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source is not RadiationSource.GCR:
        msg = "GCR mission aggregation input spectrum source must be GCR."
        raise ValueError(msg)

    if spectrum.particle not in (Particle.PROTON, Particle.HZE):
        msg = "GCR mission aggregation supports only proton and HZE spectra."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUX:
        msg = "GCR mission aggregation input spectrum quantity must be differential flux."
        raise ValueError(msg)

    if spectrum.y_unit not in (
        Unit.DIFFERENTIAL_FLUX,
        Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,
    ):
        msg = "GCR mission aggregation input spectrum unit must be differential flux."
        raise ValueError(msg)

    if spectrum.x_unit not in (Unit.MEV, Unit.MEV_PER_NUCLEON, Unit.GEV_PER_NUCLEON):
        msg = "GCR mission aggregation input energy grid must be in MeV or GeV/nucleon."
        raise ValueError(msg)


def _validate_matching_bins(time_bins: tuple[GcrFluxTimeBin, ...]) -> None:
    if not time_bins:
        msg = "GCR mission aggregation requires at least one time bin."
        raise ValueError(msg)

    reference = time_bins[0].spectrum

    for time_bin in time_bins[1:]:
        spectrum = time_bin.spectrum

        if spectrum.x != reference.x:
            msg = "GCR time-binned spectra must share the same energy grid."
            raise ValueError(msg)

        if spectrum.x_unit is not reference.x_unit:
            msg = "GCR time-binned spectra must share the same energy unit."
            raise ValueError(msg)

        if spectrum.particle is not reference.particle:
            msg = "GCR time-binned spectra must share the same particle group."
            raise ValueError(msg)

        if spectrum.source is not reference.source:
            msg = "GCR time-binned spectra must share the same source."
            raise ValueError(msg)


def _total_duration_seconds(time_bins: tuple[GcrFluxTimeBin, ...]) -> float:
    return sum(time_bin.duration_seconds for time_bin in time_bins)


def _time_integral_values(time_bins: tuple[GcrFluxTimeBin, ...]) -> tuple[float, ...]:
    point_count = len(time_bins[0].spectrum.y)
    values: list[float] = []

    for index in range(point_count):
        values.append(
            sum(time_bin.spectrum.y[index] * time_bin.duration_seconds for time_bin in time_bins)
        )

    return tuple(values)


def _time_weighted_mean_values(time_bins: tuple[GcrFluxTimeBin, ...]) -> tuple[float, ...]:
    total_duration = _total_duration_seconds(time_bins)
    fluence_values = _time_integral_values(time_bins)

    return tuple(value / total_duration for value in fluence_values)


def _pointwise_maximum_values(time_bins: tuple[GcrFluxTimeBin, ...]) -> tuple[float, ...]:
    point_count = len(time_bins[0].spectrum.y)
    values: list[float] = []

    for index in range(point_count):
        values.append(max(time_bin.spectrum.y[index] for time_bin in time_bins))

    return tuple(values)


def _combined_time_bin_reference(time_bins: tuple[GcrFluxTimeBin, ...]) -> Spectrum1D:
    """Merge input status and references even from zero or non-maximum bins."""

    spectra = tuple(time_bin.spectrum for time_bin in time_bins)
    reasons = tuple(
        dict.fromkeys(reason for spectrum in spectra for reason in spectrum.incomplete_reasons)
    )
    completeness: SpectrumCompleteness | None
    if reasons or any(
        spectrum.completeness is SpectrumCompleteness.INCOMPLETE for spectrum in spectra
    ):
        completeness = SpectrumCompleteness.INCOMPLETE
    elif all(spectrum.completeness is SpectrumCompleteness.COMPLETE for spectrum in spectra):
        completeness = SpectrumCompleteness.COMPLETE
    else:
        # Legacy unknown status must not become complete because of bin order.
        completeness = None
    references = tuple(
        spectrum.model_reference for spectrum in spectra if spectrum.model_reference is not None
    )
    return derive_spectrum(
        spectra[0],
        completeness=completeness,
        incomplete_reasons=reasons,
        additional_document_refs=tuple(
            dict.fromkeys(ref for reference in references for ref in reference.document_refs)
        ),
        additional_dataset_refs=tuple(
            dict.fromkeys(ref for reference in references for ref in reference.dataset_refs)
        ),
    )


def _product_spectrum(
    *,
    reference: Spectrum1D,
    values: tuple[float, ...],
    y_unit: Unit,
    quantity: SpectrumQuantity,
    model_component: str,
    temporal_basis: TemporalBasis,
    assessment_mode: AssessmentMode | None,
    mission_duration_s: float | None,
    time_bin_count: int,
) -> Spectrum1D:
    return derive_spectrum(
        reference,
        y=values,
        y_unit=y_unit,
        quantity=quantity,
        model=f"{reference.model}:{GCR_MISSION_PRODUCTS_MODEL_SUFFIX}:{model_component}",
        temporal_basis=temporal_basis,
        assessment_mode=assessment_mode,
        mission_duration_s=mission_duration_s,
        transformation=TransformationStep(
            name=GCR_MISSION_PRODUCTS_MODEL_SUFFIX,
            version="3",
            parameters=(
                ("aggregation", model_component),
                ("time_bin_count", str(time_bin_count)),
                ("input_provenance_policy", "all_bins_including_zero_and_nonmaximum_bins"),
            ),
        ),
    )


def _integral_product_spectrum(
    *,
    reference: Spectrum1D,
    differential_values: tuple[float, ...],
    y_unit: Unit,
    quantity: SpectrumQuantity,
    model_component: str,
    temporal_basis: TemporalBasis,
    assessment_mode: AssessmentMode | None,
    mission_duration_s: float | None,
    upper_energy_bound: float,
    time_bin_count: int,
) -> Spectrum1D:
    differential_product = _product_spectrum(
        reference=reference,
        values=integrate_differential_spectrum_finite_upper_bound(
            energies=reference.x,
            differential_values=differential_values,
            upper_energy_bound=upper_energy_bound,
            context="GCR",
        ),
        y_unit=y_unit,
        quantity=quantity,
        model_component=model_component,
        temporal_basis=temporal_basis,
        assessment_mode=assessment_mode,
        mission_duration_s=mission_duration_s,
        time_bin_count=time_bin_count,
    )
    return derive_spectrum(
        differential_product,
        transformation=TransformationStep(
            name=GCR_FINITE_DOMAIN_INTEGRATION_STEP,
            version="1",
            parameters=(
                ("integration", "local_power_law_finite_domain"),
                ("upper_energy_bound", f"{upper_energy_bound:.17g}"),
                ("upper_energy_bound_unit", reference.x_unit.value),
            ),
        ),
    )


def calculate_gcr_mission_products(
    time_bins: tuple[GcrFluxTimeBin, ...],
    *,
    upper_energy_bound: float | None = None,
) -> GcrMissionProducts:
    """Aggregate time-binned GCR flux spectra into mission products.

    Policy:
        MEAN_FLUX = time-weighted mean flux.
        MAXIMUM_FLUX = pointwise maximum flux.
        MISSION_FLUENCE = time integral of flux in seconds.
    """

    _validate_matching_bins(time_bins)

    reference = _combined_time_bin_reference(time_bins)
    time_bin_count = len(time_bins)
    finite_upper_energy_bound = (
        reference.x[-1] if upper_energy_bound is None else upper_energy_bound
    )
    total_duration_seconds = _total_duration_seconds(time_bins)
    differential_flux_unit = (
        Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY
        if reference.x_unit is Unit.MEV_PER_NUCLEON
        else Unit.DIFFERENTIAL_FLUX
    )
    differential_fluence_unit = (
        Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY
        if reference.x_unit is Unit.MEV_PER_NUCLEON
        else Unit.DIFFERENTIAL_FLUENCE
    )

    mean_flux_values = _time_weighted_mean_values(time_bins)
    maximum_flux_values = _pointwise_maximum_values(time_bins)
    mission_fluence_values = _time_integral_values(time_bins)

    mean_flux_spectrum = _product_spectrum(
        reference=reference,
        values=mean_flux_values,
        y_unit=differential_flux_unit,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        model_component="mean_flux",
        temporal_basis=TemporalBasis.MEAN_INTERNAL,
        assessment_mode=None,
        mission_duration_s=None,
        time_bin_count=time_bin_count,
    )
    mean_integral_flux_spectrum = _integral_product_spectrum(
        reference=reference,
        differential_values=mean_flux_values,
        y_unit=Unit.INTEGRAL_FLUX,
        quantity=SpectrumQuantity.MEAN_INTEGRAL_FLUX,
        model_component="mean_integral_flux",
        temporal_basis=TemporalBasis.MEAN_INTERNAL,
        assessment_mode=None,
        mission_duration_s=None,
        upper_energy_bound=finite_upper_energy_bound,
        time_bin_count=time_bin_count,
    )
    maximum_flux_spectrum = _product_spectrum(
        reference=reference,
        values=maximum_flux_values,
        y_unit=differential_flux_unit,
        quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        model_component="maximum_flux",
        temporal_basis=TemporalBasis.SOURCE_MAXIMUM,
        assessment_mode=AssessmentMode.PEAK,
        mission_duration_s=None,
        time_bin_count=time_bin_count,
    )
    maximum_integral_flux_spectrum = _integral_product_spectrum(
        reference=reference,
        differential_values=maximum_flux_values,
        y_unit=Unit.INTEGRAL_FLUX,
        quantity=SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
        model_component="maximum_integral_flux",
        temporal_basis=TemporalBasis.SOURCE_MAXIMUM,
        assessment_mode=AssessmentMode.PEAK,
        mission_duration_s=None,
        upper_energy_bound=finite_upper_energy_bound,
        time_bin_count=time_bin_count,
    )
    mission_fluence_spectrum = _product_spectrum(
        reference=reference,
        values=mission_fluence_values,
        y_unit=differential_fluence_unit,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        model_component="mission_fluence",
        temporal_basis=TemporalBasis.MISSION_TOTAL,
        assessment_mode=AssessmentMode.CUMULATIVE,
        mission_duration_s=total_duration_seconds,
        time_bin_count=time_bin_count,
    )
    mission_integral_fluence_spectrum = _integral_product_spectrum(
        reference=reference,
        differential_values=mission_fluence_values,
        y_unit=Unit.INTEGRAL_FLUENCE,
        quantity=SpectrumQuantity.INTEGRAL_FLUENCE,
        model_component="mission_integral_fluence",
        temporal_basis=TemporalBasis.MISSION_TOTAL,
        assessment_mode=AssessmentMode.CUMULATIVE,
        mission_duration_s=total_duration_seconds,
        upper_energy_bound=finite_upper_energy_bound,
        time_bin_count=time_bin_count,
    )

    return GcrMissionProducts(
        mean_flux=SpectrumProduct(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=mean_flux_spectrum,
            label=f"GCR {reference.particle.value} mean flux",
        ),
        mean_integral_flux=SpectrumProduct(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=mean_integral_flux_spectrum,
            label=f"GCR {reference.particle.value} mean integral flux",
        ),
        maximum_flux=SpectrumProduct(
            kind=RadiationProductKind.MAXIMUM_FLUX,
            spectrum=maximum_flux_spectrum,
            label=f"GCR {reference.particle.value} maximum flux",
        ),
        maximum_integral_flux=SpectrumProduct(
            kind=RadiationProductKind.MAXIMUM_FLUX,
            spectrum=maximum_integral_flux_spectrum,
            label=f"GCR {reference.particle.value} maximum integral flux",
        ),
        mission_fluence=SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=mission_fluence_spectrum,
            label=f"GCR {reference.particle.value} mission fluence",
        ),
        mission_integral_fluence=SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=mission_integral_fluence_spectrum,
            label=f"GCR {reference.particle.value} mission integral fluence",
        ),
    )


__all__ = [
    "GCR_MISSION_PRODUCTS_MODEL_SUFFIX",
    "GCR_FINITE_DOMAIN_INTEGRATION_STEP",
    "GcrFluxTimeBin",
    "GcrMissionProducts",
    "calculate_gcr_mission_products",
]
