"""Secondary proton production in aluminium shielding from compact TENDL CSV data."""

from __future__ import annotations

import csv
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import StrEnum
from math import exp, inf, isfinite, log1p, nextafter
from pathlib import Path
from typing import Literal, Mapping, TypeVar

from radar.core.spectra import ShieldingDescriptor, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import add_spectra, derive_spectrum, mark_spectrum_incomplete
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumLocation,
    SpectrumOrdinateBasis,
)
from radar.nuclear.tendl import (
    DEFAULT_TENDL_MF6_INTERPOLATION,
    EndfInterpolationLaw,
    TendlIncidentEnergySupport,
    TendlMf6InterpolationMetadata,
    assess_tendl_incident_energy_coverage,
)
from radar.shielding.proton_al import (
    ProtonAlRangeEnergyTable,
    _interpolate_zero_outside,
    log_bin_edges_from_centers,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
    Al27CrossSectionDiscontinuity,
    Al27NonelasticCrossSectionTable,
    IncidentEnergyLimit,
    canonicalize_al27_cross_section_rows,
    interpolate_al27_cross_section,
    load_al27_nonelastic_cross_section_table,
    proton_nonelastic_survival_factor,
    validate_al27_cross_section_discontinuities,
)

PROTON_AL_SECONDARY_MODEL = "proton_secondary_al_tendl_kernel"
PROTON_AL_TOTAL_MODEL = "proton_total_al_primary_plus_secondary"
PROTON_AL_SECONDARY_SUPPORT_OPERATOR = "TENDL p+Al-27 secondary-proton production"
PROTON_AL_SECONDARY_ATTENUATION_OPERATOR = (
    "TENDL p+Al-27 secondary-proton single-generation nonelastic attenuation"
)
PROTON_AL_SECONDARY_ATTENUATION_MISSING_REASON = (
    "TENDL p+Al-27 secondary-proton single-generation nonelastic attenuation "
    "is incomplete: the MF=3/MT=3 nonelastic cross-section table is unavailable"
)
PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX = (
    "Proton Al secondary transport domain incomplete"
)
PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON = (
    "secondary angular distribution/path-length transport not included"
)
PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON = (
    "secondary MF=6 centre-of-mass to laboratory energy-angle transformation not included"
)
PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON = (
    "higher-generation secondary-proton cascade not included"
)
PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON = (
    "secondary-proton incident-energy bin quadrature not included: point-sampled "
    "ordinate is represented by its derived logarithmic-bin centre"
)
PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON = (
    "secondary-proton incident-energy quadrature did not meet its requested "
    "accuracy within the evaluation/refinement budget"
)
PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON = (
    "secondary-proton depth/output-crossing quadrature accuracy not established: "
    "the full moving birth-PDF/exit-bin support crossing partition is not included"
)
PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON = (
    "secondary-proton depth quadrature accuracy has not been established by refinement"
)
PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON = (
    "secondary-proton crossing partition exceeds its configured knot budget; "
    "the retained diagnostic quadrature does not resolve all crossing events"
)
PROTON_AL_SECONDARY_ACTIVE_PDF_SUPPORT_REASON = (
    "secondary-proton active-cell PDF support is incomplete; all retained PDF "
    "edges were conservatively retained where incident PDF coverage is unavailable"
)
PROTON_AL_SECONDARY_SCREENED_DOMAIN_REASON = (
    "secondary-proton pre-quadrature screening omitted unsupported intervals; "
    "their physical contribution and full-result uncertainty are unknown, not zero"
)
_InterpolationValue = TypeVar("_InterpolationValue")

# Four-point Gauss-Legendre quadrature is applied only after the integration
# interval has been split at every retained reaction, PDF, range-energy and
# uniform-depth knot.  Keeping the rule and the splitting separate is
# important: a Gaussian rule must never straddle an ENDF one-sided jump.
_GAUSS_LEGENDRE_4_NODES = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)
_GAUSS_LEGENDRE_4_WEIGHTS = (
    0.34785484513745385,
    0.6521451548625461,
    0.6521451548625461,
    0.34785484513745385,
)


@dataclass(frozen=True)
class DepthCrossingQuadrature:
    """Bound geometry partitions without silently discarding required events.

    On exceeding a limit, the affected partition reverts to the previous
    diagnostic rule and records INCOMPLETE. No sampled prefix of the knots is
    treated as complete. A zero limit explicitly requests that bounded fallback.
    Limits apply to each partition: the whole exit grid for global_union, or
    one original exit bin for per_output_bin. Response work has a separate,
    strictly global budget in IncidentEnergyQuadrature.
    Active-cell partitions additionally bound attached (cell, depth) records;
    max_depth_knots still bounds distinct geometric depths per output bin.
    """

    max_depth_knots: int = 128
    max_coupled_incident_knots: int = 4096
    max_active_cell_records: int = 32768

    def __post_init__(self) -> None:
        for value in (
            self.max_depth_knots, self.max_coupled_incident_knots, self.max_active_cell_records
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Crossing partition knot budgets must be non-negative integers.")


@dataclass(frozen=True)
class IncidentEnergyQuadrature:
    """Explicit diagnostic point-source quadrature, not the production default.

    The tolerance controls the L1 difference of all exit-bin counts between
    open GL4 and two half-interval GL4 rules. It is an estimator for this outer
    integral only, not a bound on depth, MF6 or physical-model errors. Two zero
    estimates require a reaction-zero proof (or, for per_output_bin, a fully
    supported PDF-envelope non-overlap proof); otherwise a separate bounded
    refinement search retains an unknown-error blocker rather than certifying
    a potentially missed narrow response.

    global_union retains the vector reference operator. per_output_bin uses
    each original exit bin's own complete crossing partition. The response
    budget is global in either case, counting vector or scalar responses,
    respectively; these different work units are recorded in provenance.
    per_output_bin_active_cells further retains PDF crossings only on complete
    primary-energy cells where their interpolation slices can participate.
    Its physical operator and all three strategies' production default are
    unchanged; this is an explicitly selected diagnostic numerical option.
    reuse_birth_survival_partition constructs transport-only birth-energy
    boundaries once per depth/output mapping instead of once per PDF overlap.
    It preserves the reference GL4 segments and local survival cache; no CDF
    subtraction, quadrature change or cache across mappings is introduced.
    """

    relative_tolerance: float = 1.0e-3
    max_response_evaluations: int = 4096
    max_refinement_depth: int = 6
    max_unproven_zero_refinement_depth: int = 2
    partition_strategy: Literal[
        "global_union", "per_output_bin", "per_output_bin_active_cells"
    ] = "global_union"
    prequadrature_screening: bool = False
    reuse_birth_survival_partition: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.reuse_birth_survival_partition, bool):
            raise ValueError("Birth-survival partition reuse must be a boolean.")
        if not isinstance(self.prequadrature_screening, bool):
            raise ValueError("Pre-quadrature screening must be a boolean.")
        if self.prequadrature_screening and self.partition_strategy != "per_output_bin_active_cells":
            raise ValueError("Pre-quadrature screening requires per_output_bin_active_cells.")
        if self.partition_strategy not in (
            "global_union", "per_output_bin", "per_output_bin_active_cells"
        ):
            raise ValueError("Unknown incident quadrature partition strategy.")
        if not isfinite(self.relative_tolerance) or not 0 < self.relative_tolerance < 1:
            raise ValueError("Incident quadrature tolerance must lie strictly in (0, 1).")
        if (
            isinstance(self.max_response_evaluations, bool)
            or not isinstance(self.max_response_evaluations, int)
            or self.max_response_evaluations < 4
        ):
            raise ValueError("Incident quadrature requires at least four evaluations.")
        if (
            isinstance(self.max_refinement_depth, bool)
            or not isinstance(self.max_refinement_depth, int)
            or self.max_refinement_depth < 0
        ):
            raise ValueError("Incident quadrature refinement depth must be non-negative.")
        if (
            isinstance(self.max_unproven_zero_refinement_depth, bool)
            or not isinstance(self.max_unproven_zero_refinement_depth, int)
            or self.max_unproven_zero_refinement_depth < 0
        ):
            raise ValueError("Unproven-zero refinement depth must be non-negative.")


_REQUIRED_YIELD_COLUMNS = {
    "mt",
    "product_index",
    "incident_energy_mev",
    "yield",
}

_REQUIRED_PDF_COLUMNS = {
    "mt",
    "product_index",
    "incident_energy_mev",
    "emitted_energy_mev",
    "pdf_per_mev",
}

_REQUIRED_XS_COLUMNS = {
    "mt",
    "energy_mev",
    "sigma_barn",
}


class PdfIncidentAvailability(StrEnum):
    """Availability of an outgoing-energy PDF at one incident energy."""

    TABULATED = "tabulated"
    INTERPOLATABLE = "interpolatable"
    UNAVAILABLE = "unavailable"
    OUTSIDE_EVALUATED_SUPPORT = "outside_evaluated_support"


@dataclass(frozen=True)
class Al27CrossSectionTable:
    """One p+Al27 MF=3 cross-section table."""

    mt: int
    energy_mev: tuple[float, ...]
    sigma_barn: tuple[float, ...]
    discontinuities: tuple[Al27CrossSectionDiscontinuity, ...] = ()
    incident_energy_support: TendlIncidentEnergySupport | None = None

    def __post_init__(self) -> None:
        if self.mt <= 0:
            msg = "Cross-section MT must be positive."
            raise ValueError(msg)

        if len(self.energy_mev) != len(self.sigma_barn):
            msg = "Cross-section energy and sigma arrays must have equal length."
            raise ValueError(msg)

        if len(self.energy_mev) < 2:
            msg = "Cross-section table must contain at least two points."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.energy_mev):
            msg = "Cross-section energies must be finite."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.sigma_barn):
            msg = "Cross-section values must be finite."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.sigma_barn):
            msg = "Cross-section values must be non-negative."
            raise ValueError(msg)

        if any(right <= left for left, right in zip(self.energy_mev, self.energy_mev[1:])):
            msg = "Cross-section energies must be strictly increasing."
            raise ValueError(msg)

        validate_al27_cross_section_discontinuities(
            energy_mev=self.energy_mev,
            sigma_barn=self.sigma_barn,
            discontinuities=self.discontinuities,
        )

        if self.incident_energy_support is None:
            object.__setattr__(
                self,
                "incident_energy_support",
                TendlIncidentEnergySupport(
                    minimum_mev=0.0,
                    maximum_mev=self.energy_mev[-1],
                    dataset_ref="tendl/p_al27_mf3_xs.csv",
                ),
            )
        else:
            support = self.incident_energy_support
            assert support is not None
            if support.maximum_mev > self.energy_mev[-1]:
                msg = "Cross-section support maximum exceeds the tabulated domain."
                raise ValueError(msg)

    def sigma_at_energy(
        self,
        energy_mev: float,
        *,
        limit: IncidentEnergyLimit = IncidentEnergyLimit.RIGHT,
    ) -> float:
        """Return cross section in barns at proton kinetic energy."""

        support = self.incident_energy_support
        assert support is not None
        support.require_contains(
            energy_mev,
            operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR,
        )
        if energy_mev < self.energy_mev[0]:
            return 0.0
        return interpolate_al27_cross_section(
            energy_mev=energy_mev,
            xp=self.energy_mev,
            fp=self.sigma_barn,
            discontinuities=self.discontinuities,
            limit=limit,
            quantity=f"Al27 MF=3/MT={self.mt} cross section",
        )


@dataclass(frozen=True)
class SecondaryProtonProductDistribution:
    """MF=6 yield and outgoing-energy density for one emitted-proton product.

    Yield and PDF incident-energy grids are independent.  The emitted-energy
    ordinates follow the ENDF histogram-left convention: ordinate ``i`` is
    valid on ``[E[i], E[i + 1])``.  Incident-energy interpolation is performed
    only when the typed interpolation metadata declares it.
    """

    mt: int
    product_index: int
    incident_energy_mev: tuple[float, ...]
    yield_value: tuple[float, ...]
    emitted_energy_mev_by_incident: tuple[tuple[float, ...], ...]
    pdf_per_mev_by_incident: tuple[tuple[float, ...], ...]
    pdf_incident_energy_mev: tuple[float, ...] = ()
    interpolation: TendlMf6InterpolationMetadata = DEFAULT_TENDL_MF6_INTERPOLATION
    incident_energy_support: TendlIncidentEnergySupport | None = None

    def __post_init__(self) -> None:
        if self.mt <= 0:
            msg = "MF=6 product MT must be positive."
            raise ValueError(msg)

        if self.product_index < 0:
            msg = "MF=6 product index must be non-negative."
            raise ValueError(msg)

        if len(self.incident_energy_mev) != len(self.yield_value):
            msg = "MF=6 yield energy and value arrays must have equal length."
            raise ValueError(msg)

        if len(self.incident_energy_mev) < 2:
            msg = "MF=6 product distribution must contain at least two incident energies."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.incident_energy_mev):
            msg = "MF=6 yield incident energies must be finite."
            raise ValueError(msg)

        if any(not isfinite(value) or value < 0.0 for value in self.yield_value):
            msg = "MF=6 yield values must be finite and non-negative."
            raise ValueError(msg)

        if any(
            right <= left
            for left, right in zip(self.incident_energy_mev, self.incident_energy_mev[1:])
        ):
            msg = "MF=6 yield incident energies must be strictly increasing."
            raise ValueError(msg)

        if not self.pdf_incident_energy_mev and self.emitted_energy_mev_by_incident:
            if len(self.incident_energy_mev) != len(self.emitted_energy_mev_by_incident):
                msg = "MF=6 PDF incident-energy grid is required when it differs from yield grid."
                raise ValueError(msg)

            object.__setattr__(self, "pdf_incident_energy_mev", self.incident_energy_mev)

        if len(self.pdf_incident_energy_mev) != len(self.emitted_energy_mev_by_incident):
            msg = "MF=6 PDF incident and emitted-energy grids must have equal length."
            raise ValueError(msg)

        if len(self.pdf_incident_energy_mev) != len(self.pdf_per_mev_by_incident):
            msg = "MF=6 PDF incident-energy and density arrays must have equal length."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.pdf_incident_energy_mev):
            msg = "MF=6 PDF incident energies must be finite."
            raise ValueError(msg)

        if any(
            right <= left
            for left, right in zip(
                self.pdf_incident_energy_mev,
                self.pdf_incident_energy_mev[1:],
            )
        ):
            msg = "MF=6 PDF incident energies must be strictly increasing."
            raise ValueError(msg)

        for emitted_grid, pdf_grid in zip(
            self.emitted_energy_mev_by_incident,
            self.pdf_per_mev_by_incident,
            strict=True,
        ):
            if len(emitted_grid) != len(pdf_grid):
                msg = "MF=6 emitted-energy and PDF arrays must have equal length."
                raise ValueError(msg)

            if len(emitted_grid) < 2:
                msg = "MF=6 histogram PDF must contain at least two emitted-energy edges."
                raise ValueError(msg)

            if any(not isfinite(value) for value in emitted_grid):
                msg = "MF=6 emitted energies must be finite."
                raise ValueError(msg)

            if any(not isfinite(value) or value < 0.0 for value in pdf_grid):
                msg = "MF=6 PDF values must be finite and non-negative."
                raise ValueError(msg)

            if any(right <= left for left, right in zip(emitted_grid, emitted_grid[1:])):
                msg = "MF=6 emitted energies must be strictly increasing."
                raise ValueError(msg)

        if self.interpolation.yield_incident_energy is not EndfInterpolationLaw.LINEAR_LINEAR:
            msg = "Only linear-linear MF=6 yield interpolation is supported."
            raise ValueError(msg)

        if self.interpolation.pdf_incident_energy is not EndfInterpolationLaw.LINEAR_LINEAR:
            msg = "Only linear-linear MF=6 PDF incident interpolation is supported."
            raise ValueError(msg)

        if self.interpolation.pdf_outgoing_energy is not EndfInterpolationLaw.HISTOGRAM_LEFT:
            msg = "Only histogram-left MF=6 outgoing-energy PDFs are supported."
            raise ValueError(msg)

        tabulated_maximum = min(
            self.incident_energy_mev[-1],
            self.pdf_incident_energy_mev[-1],
        )
        if self.incident_energy_support is None:
            object.__setattr__(
                self,
                "incident_energy_support",
                TendlIncidentEnergySupport(
                    minimum_mev=0.0,
                    maximum_mev=tabulated_maximum,
                    dataset_ref=(
                        "tendl/p_al27_mf6_proton_yield.csv+tendl/p_al27_mf6_proton_pdf.csv"
                    ),
                ),
            )
        else:
            support = self.incident_energy_support
            assert support is not None
            if support.maximum_mev > tabulated_maximum:
                msg = "MF=6 support maximum exceeds the yield/PDF tabulated domain."
                raise ValueError(msg)

    def yield_at_incident_energy(self, incident_energy_mev: float) -> float:
        """Return product yield at incident proton energy."""

        support = self.incident_energy_support
        assert support is not None
        support.require_contains(
            incident_energy_mev,
            operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR,
        )
        return _interpolate_zero_outside(
            x=incident_energy_mev,
            xp=self.incident_energy_mev,
            fp=self.yield_value,
        )

    def pdf_at(self, *, incident_energy_mev: float, emitted_energy_mev: float) -> float:
        """Return emitted-proton PDF in 1/MeV under the declared ENDF laws."""

        availability = self.pdf_availability_at_incident_energy(incident_energy_mev)

        if availability is PdfIncidentAvailability.OUTSIDE_EVALUATED_SUPPORT:
            support = self.incident_energy_support
            assert support is not None
            support.require_contains(
                incident_energy_mev,
                operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR,
            )

        if availability is PdfIncidentAvailability.UNAVAILABLE:
            return 0.0

        exact_index = _exact_sorted_index(self.pdf_incident_energy_mev, incident_energy_mev)

        if exact_index is not None:
            return _histogram_pdf_at_index(self, exact_index, emitted_energy_mev)

        right_index = bisect_right(self.pdf_incident_energy_mev, incident_energy_mev)
        left_index = right_index - 1
        left_energy = self.pdf_incident_energy_mev[left_index]
        right_energy = self.pdf_incident_energy_mev[right_index]
        left_pdf = _histogram_pdf_at_index(self, left_index, emitted_energy_mev)
        right_pdf = _histogram_pdf_at_index(self, right_index, emitted_energy_mev)
        fraction = (incident_energy_mev - left_energy) / (right_energy - left_energy)
        return left_pdf + fraction * (right_pdf - left_pdf)

    def pdf_availability_at_incident_energy(
        self,
        incident_energy_mev: float,
    ) -> PdfIncidentAvailability:
        """Classify whether a PDF is tabulated, declared-interpolatable, or absent."""

        support = self.incident_energy_support
        assert support is not None
        if not support.contains(incident_energy_mev):
            return PdfIncidentAvailability.OUTSIDE_EVALUATED_SUPPORT

        if not self.pdf_incident_energy_mev:
            return PdfIncidentAvailability.UNAVAILABLE

        if _exact_sorted_index(self.pdf_incident_energy_mev, incident_energy_mev) is not None:
            return PdfIncidentAvailability.TABULATED

        if (
            len(self.pdf_incident_energy_mev) >= 2
            and self.pdf_incident_energy_mev[0]
            < incident_energy_mev
            < self.pdf_incident_energy_mev[-1]
            and self.interpolation.pdf_incident_energy is EndfInterpolationLaw.LINEAR_LINEAR
        ):
            return PdfIncidentAvailability.INTERPOLATABLE

        return PdfIncidentAvailability.UNAVAILABLE

    def tabulated_pdf_integral(self, index: int) -> float:
        """Integrate one tabulated histogram PDF by left rectangles."""

        emitted_grid = self.emitted_energy_mev_by_incident[index]
        pdf_grid = self.pdf_per_mev_by_incident[index]
        return sum(
            pdf * (right_energy - left_energy)
            for left_energy, right_energy, pdf in zip(
                emitted_grid[:-1],
                emitted_grid[1:],
                pdf_grid[:-1],
                strict=True,
            )
        )

    def pdf_integral(
        self,
        *,
        incident_energy_mev: float,
        emitted_energy_lower_mev: float,
        emitted_energy_upper_mev: float,
    ) -> float:
        """Integrate the interpolated ENDF histogram PDF over an energy bin.

        Incident-energy interpolation is linear between *whole* outgoing
        histograms.  Consequently, integrating the two bracketing histograms
        separately and then applying their incident-energy weights is exact;
        it also handles different outgoing grids without inventing a common
        point-sampled PDF.
        """

        if not isfinite(emitted_energy_lower_mev) or not isfinite(emitted_energy_upper_mev):
            msg = "MF=6 emitted-energy integration limits must be finite."
            raise ValueError(msg)
        if emitted_energy_upper_mev < emitted_energy_lower_mev:
            msg = "MF=6 emitted-energy upper limit must not be below the lower limit."
            raise ValueError(msg)
        if emitted_energy_upper_mev == emitted_energy_lower_mev:
            return 0.0

        return sum(
            weight
            * _tabulated_histogram_integral(
                self,
                index=index,
                lower_mev=emitted_energy_lower_mev,
                upper_mev=emitted_energy_upper_mev,
            )
            for index, weight in _pdf_incident_components(
                self,
                incident_energy_mev=incident_energy_mev,
            )
        )


@dataclass(frozen=True)
class SecondaryProtonKernel:
    """Secondary emitted-proton kernel for p+Al27."""

    products: tuple[SecondaryProtonProductDistribution, ...]
    cross_sections_by_mt: Mapping[int, Al27CrossSectionTable]
    nonelastic_cross_section: Al27NonelasticCrossSectionTable | None = None
    incident_energy_support: TendlIncidentEnergySupport | None = None

    def __post_init__(self) -> None:
        if not self.products:
            if self.cross_sections_by_mt:
                msg = (
                    "Empty diagnostic secondary-proton kernel must not contain "
                    "MF=3 production cross sections."
                )
                raise ValueError(msg)
            if self.nonelastic_cross_section is not None:
                msg = (
                    "Empty diagnostic secondary-proton kernel must not contain "
                    "an MF=3/MT=3 attenuation table."
                )
                raise ValueError(msg)
            if self.incident_energy_support is not None:
                msg = (
                    "Empty diagnostic secondary-proton kernel must not declare "
                    "incident-energy support."
                )
                raise ValueError(msg)
            return

        supports = []
        for product in self.products:
            product_support = product.incident_energy_support
            assert product_support is not None
            supports.append(product_support)
            cross_section = self.cross_sections_by_mt.get(product.mt)
            if cross_section is None:
                msg = f"Secondary-proton kernel has no MF=3 table for MT={product.mt}."
                raise ValueError(msg)
            cross_section_support = cross_section.incident_energy_support
            assert cross_section_support is not None
            supports.append(cross_section_support)

        supported_maximum = min(support.maximum_mev for support in supports)
        if self.incident_energy_support is None:
            object.__setattr__(
                self,
                "incident_energy_support",
                TendlIncidentEnergySupport(
                    minimum_mev=max(support.minimum_mev for support in supports),
                    maximum_mev=supported_maximum,
                    dataset_ref="TENDL-2025 p+Al-27 MF=3/MF=6",
                ),
            )
        else:
            support = self.incident_energy_support
            assert support is not None
            if support.maximum_mev > supported_maximum:
                msg = "Kernel support maximum exceeds a retained MF=3/MF=6 domain."
                raise ValueError(msg)

        if self.nonelastic_cross_section is not None and self.nonelastic_cross_section.mt != 3:
            msg = "Secondary-proton attenuation requires the MF=3/MT=3 table."
            raise ValueError(msg)

    def source_density_per_g_cm2(
        self,
        *,
        incident_energy_mev: float,
        emitted_energy_mev: float,
    ) -> float:
        """Return emitted proton yield density per g/cm² and MeV.

        This is sum_MT N_A/A * sigma_MT(E) * yield_MT(E) * pdf_MT(Eout|E).
        """

        if not self.products:
            msg = "Empty diagnostic secondary-proton kernel cannot evaluate a source."
            raise ValueError(msg)
        support = self.incident_energy_support
        assert support is not None
        support.require_contains(
            incident_energy_mev,
            operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR,
        )
        number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
        density = 0.0

        for product in self.products:
            cross_section = self.cross_sections_by_mt.get(product.mt)

            if cross_section is None:
                continue

            sigma_cm2 = cross_section.sigma_at_energy(incident_energy_mev) * BARN_CM2
            yield_value = product.yield_at_incident_energy(incident_energy_mev)
            pdf = product.pdf_at(
                incident_energy_mev=incident_energy_mev,
                emitted_energy_mev=emitted_energy_mev,
            )
            density += number_per_gram * sigma_cm2 * yield_value * pdf

        return density


def _exact_sorted_index(values: tuple[float, ...], target: float) -> int | None:
    index = bisect_left(values, target)

    if index < len(values) and values[index] == target:
        return index

    return None


def _histogram_pdf_at_index(
    product: SecondaryProtonProductDistribution,
    index: int,
    emitted_energy_mev: float,
) -> float:
    emitted_grid = product.emitted_energy_mev_by_incident[index]

    if not isfinite(emitted_energy_mev):
        return 0.0

    if emitted_energy_mev < emitted_grid[0] or emitted_energy_mev >= emitted_grid[-1]:
        return 0.0

    emitted_index = bisect_right(emitted_grid, emitted_energy_mev) - 1
    return product.pdf_per_mev_by_incident[index][emitted_index]


def _pdf_incident_components(
    product: SecondaryProtonProductDistribution,
    *,
    incident_energy_mev: float,
) -> tuple[tuple[int, float], ...]:
    """Return tabulated histograms and weights for ENDF lin-lin interpolation."""

    availability = product.pdf_availability_at_incident_energy(incident_energy_mev)
    if availability is PdfIncidentAvailability.OUTSIDE_EVALUATED_SUPPORT:
        support = product.incident_energy_support
        assert support is not None
        support.require_contains(
            incident_energy_mev,
            operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR,
        )
    if availability is PdfIncidentAvailability.UNAVAILABLE:
        return ()

    exact_index = _exact_sorted_index(
        product.pdf_incident_energy_mev,
        incident_energy_mev,
    )
    if exact_index is not None:
        return ((exact_index, 1.0),)

    right_index = bisect_right(
        product.pdf_incident_energy_mev,
        incident_energy_mev,
    )
    left_index = right_index - 1
    left_energy = product.pdf_incident_energy_mev[left_index]
    right_energy = product.pdf_incident_energy_mev[right_index]
    right_weight = (incident_energy_mev - left_energy) / (right_energy - left_energy)
    return ((left_index, 1.0 - right_weight), (right_index, right_weight))


def _tabulated_histogram_integral(
    product: SecondaryProtonProductDistribution,
    *,
    index: int,
    lower_mev: float,
    upper_mev: float,
) -> float:
    """Return an exact left-histogram integral over one emitted-energy interval."""

    emitted_grid = product.emitted_energy_mev_by_incident[index]
    pdf_grid = product.pdf_per_mev_by_incident[index]
    integral = 0.0
    first_index = max(0, bisect_right(emitted_grid, lower_mev) - 1)
    stop_index = min(len(emitted_grid) - 1, bisect_left(emitted_grid, upper_mev))
    for histogram_index in range(first_index, stop_index):
        bin_lower = emitted_grid[histogram_index]
        bin_upper = emitted_grid[histogram_index + 1]
        density = pdf_grid[histogram_index]
        overlap_lower = max(lower_mev, bin_lower)
        overlap_upper = min(upper_mev, bin_upper)
        if overlap_upper > overlap_lower:
            integral += density * (overlap_upper - overlap_lower)
    return integral


def _required_text(row: Mapping[str, str | None], column: str, line_number: int) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def _require_columns(fieldnames: set[str], required: set[str], name: str) -> None:
    missing_columns = required - fieldnames

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"{name} CSV is missing columns: {missing}."
        raise ValueError(msg)


def _merge_duplicate_energy_rows_if_identical(
    rows: list[tuple[float, float]],
    *,
    context: str,
) -> list[tuple[float, float]]:
    """Deduplicate exact coordinates without hiding conflicting values."""

    merged: dict[float, float] = {}

    for energy, value in rows:
        previous = merged.get(energy)

        if previous is None:
            merged[energy] = value
        elif previous != value:
            msg = (
                f"Conflicting duplicate {context} at energy {energy!r}: {previous!r} != {value!r}."
            )
            raise ValueError(msg)

    return sorted(merged.items(), key=lambda row: row[0])


def _optional_interpolation_law(
    row: Mapping[str, str | None],
    column: str,
    *,
    default: EndfInterpolationLaw,
    line_number: int,
) -> EndfInterpolationLaw:
    text = row.get(column)

    if text is None or not text.strip():
        return default

    try:
        return EndfInterpolationLaw(text.strip())
    except ValueError as error:
        msg = f"Unsupported interpolation law {text.strip()!r} at CSV line {line_number}."
        raise ValueError(msg) from error


def _record_consistent_interpolation(
    values: dict[tuple[int, int], _InterpolationValue],
    *,
    key: tuple[int, int],
    value: _InterpolationValue,
    source: str,
) -> None:
    previous = values.get(key)

    if previous is not None and previous != value:
        msg = f"Inconsistent {source} interpolation metadata for MT/product {key}."
        raise ValueError(msg)

    values[key] = value


def load_al27_cross_section_tables(
    path: str | Path,
    *,
    mts: set[int] | None = None,
) -> dict[int, Al27CrossSectionTable]:
    """Load p+Al27 MF=3 cross-section tables from compact CSV."""

    rows_by_mt: dict[int, list[tuple[float, float]]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(set(reader.fieldnames or ()), _REQUIRED_XS_COLUMNS, "MF=3")

        for line_number, row in enumerate(reader, start=2):
            mt = int(_required_text(row, "mt", line_number))

            if mts is not None and mt not in mts:
                continue

            rows_by_mt.setdefault(mt, []).append(
                (
                    float(_required_text(row, "energy_mev", line_number)),
                    float(_required_text(row, "sigma_barn", line_number)),
                )
            )

    tables: dict[int, Al27CrossSectionTable] = {}

    for mt, rows in rows_by_mt.items():
        canonical = canonicalize_al27_cross_section_rows(
            rows,
            context=f"p+Al27 MF=3/MT={mt}",
        )
        tables[mt] = Al27CrossSectionTable(
            mt=mt,
            energy_mev=canonical.energy_mev,
            sigma_barn=canonical.sigma_barn,
            discontinuities=canonical.discontinuities,
        )

    return tables


def load_secondary_proton_products(
    *,
    yield_path: str | Path,
    pdf_path: str | Path,
) -> tuple[SecondaryProtonProductDistribution, ...]:
    """Load MF=6 emitted-proton product distributions from compact CSV."""

    yield_by_key: dict[tuple[int, int], list[tuple[float, float]]] = {}
    yield_law_by_key: dict[tuple[int, int], EndfInterpolationLaw] = {}

    with Path(yield_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(set(reader.fieldnames or ()), _REQUIRED_YIELD_COLUMNS, "MF=6 yield")

        for line_number, row in enumerate(reader, start=2):
            mt = int(_required_text(row, "mt", line_number))
            product_index = int(_required_text(row, "product_index", line_number))
            incident_energy = float(_required_text(row, "incident_energy_mev", line_number))
            yield_value = float(_required_text(row, "yield", line_number))
            key = (mt, product_index)
            interpolation_law = _optional_interpolation_law(
                row,
                "yield_interpolation",
                default=DEFAULT_TENDL_MF6_INTERPOLATION.yield_incident_energy,
                line_number=line_number,
            )
            _record_consistent_interpolation(
                yield_law_by_key,
                key=key,
                value=interpolation_law,
                source="MF=6 yield",
            )
            yield_by_key.setdefault(key, []).append((incident_energy, yield_value))

    pdf_by_key_energy: dict[tuple[int, int], dict[float, list[tuple[float, float]]]] = {}
    pdf_laws_by_key: dict[
        tuple[int, int],
        tuple[EndfInterpolationLaw, EndfInterpolationLaw],
    ] = {}

    with Path(pdf_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(set(reader.fieldnames or ()), _REQUIRED_PDF_COLUMNS, "MF=6 PDF")

        for line_number, row in enumerate(reader, start=2):
            mt = int(_required_text(row, "mt", line_number))
            product_index = int(_required_text(row, "product_index", line_number))
            incident_energy = float(_required_text(row, "incident_energy_mev", line_number))
            emitted_energy = float(_required_text(row, "emitted_energy_mev", line_number))
            pdf = float(_required_text(row, "pdf_per_mev", line_number))
            key = (mt, product_index)
            incident_interpolation_law = _optional_interpolation_law(
                row,
                "pdf_incident_interpolation",
                default=DEFAULT_TENDL_MF6_INTERPOLATION.pdf_incident_energy,
                line_number=line_number,
            )
            outgoing_interpolation_law = _optional_interpolation_law(
                row,
                "pdf_outgoing_interpolation",
                default=DEFAULT_TENDL_MF6_INTERPOLATION.pdf_outgoing_energy,
                line_number=line_number,
            )
            _record_consistent_interpolation(
                pdf_laws_by_key,
                key=key,
                value=(incident_interpolation_law, outgoing_interpolation_law),
                source="MF=6 PDF",
            )
            pdf_by_key_energy.setdefault(key, {}).setdefault(incident_energy, []).append(
                (emitted_energy, pdf)
            )

    products: list[SecondaryProtonProductDistribution] = []

    for key, yield_rows in yield_by_key.items():
        mt, product_index = key
        merged_yield_rows = _merge_duplicate_energy_rows_if_identical(
            yield_rows,
            context=f"MF=6 yield for MT={mt}, product={product_index}",
        )
        incident_energies = tuple(row[0] for row in merged_yield_rows)

        emitted_grids: list[tuple[float, ...]] = []
        pdf_grids: list[tuple[float, ...]] = []
        pdf_incident_energies: list[float] = []

        for incident_energy, pdf_rows in sorted(pdf_by_key_energy.get(key, {}).items()):
            merged_pdf_rows = _merge_duplicate_energy_rows_if_identical(
                pdf_rows,
                context=(
                    f"MF=6 PDF for MT={mt}, product={product_index}, "
                    f"incident_energy={incident_energy}"
                ),
            )
            pdf_incident_energies.append(incident_energy)
            emitted_grids.append(tuple(row[0] for row in merged_pdf_rows))
            pdf_grids.append(tuple(row[1] for row in merged_pdf_rows))

        pdf_incident_law, pdf_outgoing_law = pdf_laws_by_key.get(
            key,
            (
                DEFAULT_TENDL_MF6_INTERPOLATION.pdf_incident_energy,
                DEFAULT_TENDL_MF6_INTERPOLATION.pdf_outgoing_energy,
            ),
        )
        products.append(
            SecondaryProtonProductDistribution(
                mt=mt,
                product_index=product_index,
                incident_energy_mev=incident_energies,
                yield_value=tuple(row[1] for row in merged_yield_rows),
                pdf_incident_energy_mev=tuple(pdf_incident_energies),
                emitted_energy_mev_by_incident=tuple(emitted_grids),
                pdf_per_mev_by_incident=tuple(pdf_grids),
                interpolation=TendlMf6InterpolationMetadata(
                    yield_incident_energy=yield_law_by_key[key],
                    pdf_incident_energy=pdf_incident_law,
                    pdf_outgoing_energy=pdf_outgoing_law,
                ),
            )
        )

    return tuple(sorted(products, key=lambda product: (product.mt, product.product_index)))


def load_secondary_proton_kernel(
    *,
    xs_path: str | Path,
    yield_path: str | Path,
    pdf_path: str | Path,
) -> SecondaryProtonKernel:
    """Load p+Al27 secondary-proton kernel from compact CSV tables."""

    products = load_secondary_proton_products(yield_path=yield_path, pdf_path=pdf_path)
    if not products:
        msg = "Production secondary-proton loader found no MF=6 proton products."
        raise ValueError(msg)
    mts = {product.mt for product in products}
    cross_sections = load_al27_cross_section_tables(xs_path, mts=mts)
    try:
        nonelastic_cross_section = load_al27_nonelastic_cross_section_table(xs_path)
    except ValueError as error:
        if "contains no rows for MT=3" not in str(error):
            raise
        nonelastic_cross_section = None

    return SecondaryProtonKernel(
        products=products,
        cross_sections_by_mt=cross_sections,
        nonelastic_cross_section=nonelastic_cross_section,
    )


def _birth_energy_for_output(
    *,
    output_energy_mev: float,
    residual_thickness_g_cm2: float,
    table: ProtonAlRangeEnergyTable,
) -> float | None:
    output_range = table.range_at_energy(output_energy_mev)
    birth_range = output_range + residual_thickness_g_cm2

    if birth_range > table.max_range_g_cm2:
        return None

    return table.energy_at_range(birth_range)


def _nonelastic_survival_or_unity(
    *,
    output_energy_mev: float,
    input_energy_mev: float,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable | None,
    integration_steps: int,
    optical_depth_table: _NonelasticOpticalDepthTable | None = None,
) -> tuple[float, bool]:
    """Return MT3 survival, or unity plus ``False`` when it is unsupported."""

    if input_energy_mev == output_energy_mev:
        return 1.0, True
    if cross_section_table is None:
        return 1.0, False

    support = cross_section_table.incident_energy_support
    assert support is not None
    if not support.contains(output_energy_mev) or not support.contains(input_energy_mev):
        return 1.0, False

    if optical_depth_table is not None:
        return (
            optical_depth_table.survival_factor(
                output_energy_mev=output_energy_mev,
                input_energy_mev=input_energy_mev,
            ),
            True,
        )

    return (
        proton_nonelastic_survival_factor(
            output_energy_mev=output_energy_mev,
            input_energy_mev=input_energy_mev,
            stopping_table=stopping_table,
            cross_section_table=cross_section_table,
            integration_steps=integration_steps,
        ),
        True,
    )


def _integral_of_linear_ratio(
    *,
    numerator_left: float,
    numerator_right: float,
    denominator_left: float,
    denominator_right: float,
    width: float,
) -> float:
    """Integrate a linear numerator divided by a linear denominator exactly."""

    numerator_delta = numerator_right - numerator_left
    denominator_delta = denominator_right - denominator_left
    if abs(denominator_delta) <= 1.0e-12 * max(
        denominator_left,
        denominator_right,
    ):
        return width * 0.5 * (numerator_left + numerator_right) / denominator_left

    constant_term = numerator_delta / denominator_delta
    reciprocal_term = numerator_left - constant_term * denominator_left
    return width * (
        constant_term
        + reciprocal_term / denominator_delta * log1p(denominator_delta / denominator_left)
    )


@dataclass(frozen=True)
class _NonelasticOpticalDepthTable:
    """Cumulative exact optical depth for retained linear MF3/stopping tables."""

    energy_mev: tuple[float, ...]
    cumulative_optical_depth: tuple[float, ...]
    sigma_left_barn: tuple[float, ...]
    sigma_right_barn: tuple[float, ...]
    stopping_left_mev_cm2_g: tuple[float, ...]
    stopping_right_mev_cm2_g: tuple[float, ...]

    def optical_depth_at(self, energy_mev: float) -> float:
        """Return cumulative dimensionless nonelastic optical depth at energy."""

        if energy_mev < self.energy_mev[0] or energy_mev > self.energy_mev[-1]:
            msg = "Secondary-proton attenuation energy is outside optical-depth support."
            raise ValueError(msg)
        segment_index = bisect_right(self.energy_mev, energy_mev) - 1
        if segment_index == len(self.energy_mev) - 1:
            return self.cumulative_optical_depth[-1]

        left_energy = self.energy_mev[segment_index]
        right_energy = self.energy_mev[segment_index + 1]
        partial_width = energy_mev - left_energy
        if partial_width == 0.0:
            return self.cumulative_optical_depth[segment_index]
        fraction = partial_width / (right_energy - left_energy)
        sigma_at_energy = self.sigma_left_barn[segment_index] + fraction * (
            self.sigma_right_barn[segment_index] - self.sigma_left_barn[segment_index]
        )
        stopping_at_energy = self.stopping_left_mev_cm2_g[segment_index] + fraction * (
            self.stopping_right_mev_cm2_g[segment_index]
            - self.stopping_left_mev_cm2_g[segment_index]
        )
        partial_integral = _integral_of_linear_ratio(
            numerator_left=self.sigma_left_barn[segment_index],
            numerator_right=sigma_at_energy,
            denominator_left=self.stopping_left_mev_cm2_g[segment_index],
            denominator_right=stopping_at_energy,
            width=partial_width,
        )
        number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
        return self.cumulative_optical_depth[segment_index] + (
            number_per_gram * BARN_CM2 * partial_integral
        )

    def survival_factor(
        self,
        *,
        output_energy_mev: float,
        input_energy_mev: float,
    ) -> float:
        """Return survival between two energies from cumulative optical depth."""

        optical_depth = self.optical_depth_at(input_energy_mev) - self.optical_depth_at(
            output_energy_mev
        )
        return exp(-max(0.0, optical_depth))


def _build_nonelastic_optical_depth_table(
    *,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable,
) -> _NonelasticOpticalDepthTable:
    """Build cumulative optical depth split at every interpolation/jump knot."""

    support = cross_section_table.incident_energy_support
    assert support is not None
    minimum_energy = max(stopping_table.energy_mev[0], support.minimum_mev)
    maximum_energy = min(stopping_table.energy_mev[-1], support.maximum_mev)
    energy_knots = {minimum_energy, maximum_energy}
    energy_knots.update(
        energy for energy in stopping_table.energy_mev if minimum_energy < energy < maximum_energy
    )
    energy_knots.update(
        energy
        for energy in cross_section_table.energy_mev
        if minimum_energy < energy < maximum_energy
    )
    ordered_knots = tuple(sorted(energy_knots))
    sigma_left: list[float] = []
    sigma_right: list[float] = []
    stopping_left: list[float] = []
    stopping_right: list[float] = []
    cumulative = [0.0]
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    for left_energy, right_energy in zip(ordered_knots, ordered_knots[1:]):
        left_sigma = cross_section_table.sigma_at_energy(
            left_energy,
            limit=IncidentEnergyLimit.RIGHT,
        )
        right_sigma = cross_section_table.sigma_at_energy(
            right_energy,
            limit=IncidentEnergyLimit.LEFT,
        )
        left_stopping = stopping_table.stopping_at_energy(left_energy)
        right_stopping = stopping_table.stopping_at_energy(right_energy)
        sigma_left.append(left_sigma)
        sigma_right.append(right_sigma)
        stopping_left.append(left_stopping)
        stopping_right.append(right_stopping)
        segment_integral = _integral_of_linear_ratio(
            numerator_left=left_sigma,
            numerator_right=right_sigma,
            denominator_left=left_stopping,
            denominator_right=right_stopping,
            width=right_energy - left_energy,
        )
        cumulative.append(cumulative[-1] + number_per_gram * BARN_CM2 * segment_integral)

    return _NonelasticOpticalDepthTable(
        energy_mev=ordered_knots,
        cumulative_optical_depth=tuple(cumulative),
        sigma_left_barn=tuple(sigma_left),
        sigma_right_barn=tuple(sigma_right),
        stopping_left_mev_cm2_g=tuple(stopping_left),
        stopping_right_mev_cm2_g=tuple(stopping_right),
    )


def _nonelastic_survival_piecewise_linear(
    *,
    output_energy_mev: float,
    input_energy_mev: float,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable,
) -> float:
    """Return MT3 survival using exact retained piecewise-linear segments."""

    optical_depth_table = _build_nonelastic_optical_depth_table(
        stopping_table=stopping_table,
        cross_section_table=cross_section_table,
    )
    return optical_depth_table.survival_factor(
        output_energy_mev=output_energy_mev,
        input_energy_mev=input_energy_mev,
    )


@dataclass(frozen=True)
class _SecondaryExitMapping:
    output_index: int
    output_width_mev: float
    output_lower_mev: float
    output_upper_mev: float
    birth_lower_mev: float
    birth_upper_mev: float
    residual_thickness_g_cm2: float
    attenuation_supported: bool
    attenuation_energies_mev: tuple[float, ...]


@dataclass(frozen=True)
class _SecondaryExitMappingResult:
    mappings: tuple[_SecondaryExitMapping, ...]
    range_domain_bins: int


def _secondary_exit_mappings(
    *,
    output_centers_mev: tuple[float, ...],
    output_edges_mev: tuple[float, ...],
    residual_thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable | None,
) -> _SecondaryExitMappingResult:
    """Precompute complete CSDA birth intervals for one production depth."""

    mappings: list[_SecondaryExitMapping] = []
    range_domain_bins = 0
    for output_index, output_energy in enumerate(output_centers_mev):
        output_lo = output_edges_mev[output_index]
        output_hi = output_edges_mev[output_index + 1]
        # A bin-average result always describes its complete declared bin.
        # Never clip a bin to the stopping-table support while retaining the
        # original edges: that would give the ordinate a false denominator.
        if output_lo < stopping_table.energy_mev[0] or output_hi > stopping_table.energy_mev[-1]:
            continue

        birth_lo = _birth_energy_for_output(
            output_energy_mev=output_lo,
            residual_thickness_g_cm2=residual_thickness_g_cm2,
            table=stopping_table,
        )
        birth_hi = _birth_energy_for_output(
            output_energy_mev=output_hi,
            residual_thickness_g_cm2=residual_thickness_g_cm2,
            table=stopping_table,
        )
        if birth_lo is None or birth_hi is None:
            range_domain_bins += 1
            continue

        attenuation_energies = (
            output_lo,
            output_energy,
            output_hi,
            birth_lo,
            birth_hi,
        )
        attenuation_supported = cross_section_table is not None
        if cross_section_table is not None:
            attenuation_support = cross_section_table.incident_energy_support
            assert attenuation_support is not None
            attenuation_supported = all(
                attenuation_support.contains(energy) for energy in attenuation_energies
            )
        mappings.append(
            _SecondaryExitMapping(
                output_index=output_index,
                output_width_mev=output_hi - output_lo,
                output_lower_mev=output_lo,
                output_upper_mev=output_hi,
                birth_lower_mev=birth_lo,
                birth_upper_mev=birth_hi,
                residual_thickness_g_cm2=residual_thickness_g_cm2,
                attenuation_supported=attenuation_supported,
                attenuation_energies_mev=attenuation_energies,
            )
        )

    return _SecondaryExitMappingResult(
        mappings=tuple(mappings),
        range_domain_bins=range_domain_bins,
    )


def _gauss_legendre_4_samples(
    lower: float,
    upper: float,
) -> tuple[tuple[float, float], ...]:
    """Return coordinate/weight pairs for one four-point Gaussian interval."""

    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return tuple(
        (midpoint + half_width * node, half_width * weight)
        for node, weight in zip(
            _GAUSS_LEGENDRE_4_NODES,
            _GAUSS_LEGENDRE_4_WEIGHTS,
            strict=True,
        )
    )


def _exit_energy_for_birth(
    *,
    birth_energy_mev: float,
    residual_thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
) -> float:
    """Map one emitted-proton birth energy to shield-exit energy by CSDA."""

    output_range = stopping_table.range_at_energy(birth_energy_mev) - residual_thickness_g_cm2
    return stopping_table.energy_at_range(output_range)


def _integrate_exit_survival_over_birth_segment(
    *,
    birth_lower_mev: float,
    birth_upper_mev: float,
    mapping: _SecondaryExitMapping,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable | None,
    optical_depth_table: _NonelasticOpticalDepthTable | None,
) -> float:
    """Integrate secondary MT3 survival over one smooth birth-energy segment."""

    if not mapping.attenuation_supported or cross_section_table is None:
        return birth_upper_mev - birth_lower_mev

    integral = 0.0
    for birth_energy, weight in _gauss_legendre_4_samples(
        birth_lower_mev,
        birth_upper_mev,
    ):
        output_energy = _exit_energy_for_birth(
            birth_energy_mev=birth_energy,
            residual_thickness_g_cm2=mapping.residual_thickness_g_cm2,
            stopping_table=stopping_table,
        )
        assert optical_depth_table is not None
        survival = optical_depth_table.survival_factor(
            output_energy_mev=output_energy,
            input_energy_mev=birth_energy,
        )
        integral += weight * survival
    return integral


def _integrate_tabulated_pdf_with_exit_survival(
    product: SecondaryProtonProductDistribution,
    *,
    pdf_index: int,
    mapping: _SecondaryExitMapping,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable | None,
    optical_depth_table: _NonelasticOpticalDepthTable | None,
    survival_integral_cache: dict[tuple[float, float], float],
    birth_survival_knots: tuple[float, ...] | None = None,
) -> float:
    """Integrate one histogram PDF with survival inside every overlap segment."""

    emitted_grid = product.emitted_energy_mev_by_incident[pdf_index]
    pdf_grid = product.pdf_per_mev_by_incident[pdf_index]
    integral = 0.0
    first_index = max(
        0,
        bisect_right(emitted_grid, mapping.birth_lower_mev) - 1,
    )
    stop_index = min(
        len(emitted_grid) - 1,
        bisect_left(emitted_grid, mapping.birth_upper_mev),
    )
    for histogram_index in range(first_index, stop_index):
        histogram_lower = emitted_grid[histogram_index]
        histogram_upper = emitted_grid[histogram_index + 1]
        density = pdf_grid[histogram_index]
        if density <= 0.0:
            continue
        overlap_lower = max(mapping.birth_lower_mev, histogram_lower)
        overlap_upper = min(mapping.birth_upper_mev, histogram_upper)
        if overlap_upper <= overlap_lower:
            continue
        survival_integral = 0.0
        if birth_survival_knots is None:
            segment_boundaries = _birth_survival_segment_boundaries(
                lower_mev=overlap_lower,
                upper_mev=overlap_upper,
                mapping=mapping,
                stopping_table=stopping_table,
                optical_depth_table=optical_depth_table,
            )
        else:
            # The overlap is contained in this mapping's full birth interval.
            # Restrict the SAME transport knots with strict bounds, leaving
            # PDF edges and every GL4 segment/accumulation order unchanged.
            start = bisect_right(birth_survival_knots, overlap_lower)
            stop = bisect_left(birth_survival_knots, overlap_upper)
            segment_boundaries = (
                overlap_lower, *birth_survival_knots[start:stop], overlap_upper,
            )
        for segment_lower, segment_upper in zip(
            segment_boundaries,
            segment_boundaries[1:],
        ):
            cache_key = (segment_lower, segment_upper)
            segment_integral = survival_integral_cache.get(cache_key)
            if segment_integral is None:
                segment_integral = _integrate_exit_survival_over_birth_segment(
                    birth_lower_mev=segment_lower,
                    birth_upper_mev=segment_upper,
                    mapping=mapping,
                    stopping_table=stopping_table,
                    cross_section_table=cross_section_table,
                    optical_depth_table=optical_depth_table,
                )
                survival_integral_cache[cache_key] = segment_integral
            survival_integral += segment_integral
        integral += density * survival_integral
    return integral


def _birth_survival_segment_boundaries(
    *,
    lower_mev: float,
    upper_mev: float,
    mapping: _SecondaryExitMapping,
    stopping_table: ProtonAlRangeEnergyTable,
    optical_depth_table: _NonelasticOpticalDepthTable | None,
) -> tuple[float, ...]:
    """Split birth energy at direct and mapped attenuation/stopping knots."""

    boundaries = {lower_mev, upper_mev}
    if optical_depth_table is None or not mapping.attenuation_supported:
        return tuple(sorted(boundaries))

    attenuation_knots = optical_depth_table.energy_mev
    direct_start = bisect_right(attenuation_knots, lower_mev)
    direct_stop = bisect_left(attenuation_knots, upper_mev)
    boundaries.update(attenuation_knots[direct_start:direct_stop])
    output_start = bisect_right(attenuation_knots, mapping.output_lower_mev)
    output_stop = bisect_left(attenuation_knots, mapping.output_upper_mev)
    for output_energy in attenuation_knots[output_start:output_stop]:
        mapped_birth_energy = _birth_energy_for_output(
            output_energy_mev=output_energy,
            residual_thickness_g_cm2=mapping.residual_thickness_g_cm2,
            table=stopping_table,
        )
        if mapped_birth_energy is not None and lower_mev < mapped_birth_energy < upper_mev:
            boundaries.add(mapped_birth_energy)
    return tuple(sorted(boundaries))


@dataclass(frozen=True)
class _ActiveSecondaryProduct:
    distribution: SecondaryProtonProductDistribution
    coefficient_per_g_cm2: float
    pdf_components: tuple[tuple[int, float], ...]
    emitted_minimum_mev: float
    emitted_maximum_mev: float


def _active_secondary_products_at_incident_energy(
    *,
    kernel: SecondaryProtonKernel,
    incident_energy_mev: float,
) -> tuple[_ActiveSecondaryProduct, ...]:
    """Evaluate incident-energy factors once for all output bins at one depth."""

    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    active_products: list[_ActiveSecondaryProduct] = []
    for product in kernel.products:
        cross_section = kernel.cross_sections_by_mt[product.mt]
        sigma_cm2 = cross_section.sigma_at_energy(incident_energy_mev) * BARN_CM2
        yield_value = product.yield_at_incident_energy(incident_energy_mev)
        coefficient = number_per_gram * sigma_cm2 * yield_value
        if coefficient <= 0.0:
            continue
        pdf_components = _pdf_incident_components(
            product,
            incident_energy_mev=incident_energy_mev,
        )
        if pdf_components:
            component_indices = tuple(index for index, _weight in pdf_components)
            active_products.append(
                _ActiveSecondaryProduct(
                    distribution=product,
                    coefficient_per_g_cm2=coefficient,
                    pdf_components=pdf_components,
                    emitted_minimum_mev=min(
                        product.emitted_energy_mev_by_incident[index][0]
                        for index in component_indices
                    ),
                    emitted_maximum_mev=max(
                        product.emitted_energy_mev_by_incident[index][-1]
                        for index in component_indices
                    ),
                )
            )
    return tuple(active_products)


def _integrated_secondary_source_to_exit_per_g_cm2(
    *,
    active_products: tuple[_ActiveSecondaryProduct, ...],
    nonelastic_cross_section: Al27NonelasticCrossSectionTable | None,
    optical_depth_table: _NonelasticOpticalDepthTable | None,
    mapping: _SecondaryExitMapping,
    stopping_table: ProtonAlRangeEnergyTable,
    reuse_birth_survival_partition: bool = False,
) -> float:
    """Integrate the MF3/MF6 source over a complete mapped output bin."""

    density_integral = 0.0
    # These transport-only knots depend on the complete mapping (in
    # particular its residual thickness), not on product, PDF or histogram.
    # Reuse is strictly local: never share this tuple or the survival cache
    # with another depth, output bin, stopping table or MT3 dataset.
    birth_survival_knots: tuple[float, ...] | None = None
    survival_integral_cache: dict[tuple[float, float], float] = {}
    for active_product in active_products:
        if (
            mapping.birth_upper_mev <= active_product.emitted_minimum_mev
            or mapping.birth_lower_mev >= active_product.emitted_maximum_mev
        ):
            continue
        product = active_product.distribution
        if reuse_birth_survival_partition and birth_survival_knots is None:
            birth_survival_knots = _birth_survival_segment_boundaries(
                lower_mev=mapping.birth_lower_mev,
                upper_mev=mapping.birth_upper_mev,
                mapping=mapping,
                stopping_table=stopping_table,
                optical_depth_table=optical_depth_table,
            )
        pdf_survival_integral = sum(
            weight
            * _integrate_tabulated_pdf_with_exit_survival(
                product,
                pdf_index=pdf_index,
                mapping=mapping,
                stopping_table=stopping_table,
                cross_section_table=nonelastic_cross_section,
                optical_depth_table=optical_depth_table,
                survival_integral_cache=survival_integral_cache,
                birth_survival_knots=birth_survival_knots,
            )
            for pdf_index, weight in active_product.pdf_components
            if weight > 0.0
        )
        density_integral += active_product.coefficient_per_g_cm2 * pdf_survival_integral
    return density_integral


def _secondary_incident_energy_knots(
    *,
    kernel: SecondaryProtonKernel,
    stopping_table: ProtonAlRangeEnergyTable,
) -> tuple[float, ...]:
    """Return all retained incident-energy knots that may change the integrand."""

    knots = set(stopping_table.energy_mev)
    support = kernel.incident_energy_support
    assert support is not None
    knots.update((support.minimum_mev, support.maximum_mev))
    for product in kernel.products:
        knots.update(product.incident_energy_mev)
        knots.update(product.pdf_incident_energy_mev)
        cross_section = kernel.cross_sections_by_mt[product.mt]
        knots.update(cross_section.energy_mev)
        knots.update(item.energy_mev for item in cross_section.discontinuities)
    if kernel.nonelastic_cross_section is not None:
        knots.update(kernel.nonelastic_cross_section.energy_mev)
        knots.update(item.energy_mev for item in kernel.nonelastic_cross_section.discontinuities)
        attenuation_support = kernel.nonelastic_cross_section.incident_energy_support
        assert attenuation_support is not None
        knots.update((attenuation_support.minimum_mev, attenuation_support.maximum_mev))
    return tuple(
        sorted(
            energy
            for energy in knots
            if stopping_table.energy_mev[0] <= energy <= stopping_table.energy_mev[-1]
        )
    )


class _CrossingPartitionBudgetExceeded(Exception):
    """A complete event partition exceeds an explicit resource budget."""


def _secondary_exit_depth_knots(
    *,
    output_edges_mev: tuple[float, ...],
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    max_knots: int | None = None,
    pdf_birth_knots_mev: tuple[float, ...] | None = None,
    include_transport_birth_knots: bool = True,
) -> tuple[float, ...]:
    """All supported PDF/range/attenuation crossings of moving exit bounds.

    CSDA gives R(E_birth)=R(E_exit)+t-d. Thus every crossing is
    d=t-[R(E_birth_knot)-R(E_exit_knot)], restricted to 0<d<t.
    The union includes all incident PDF slices (no sampled PDF support proof),
    stopping/MT3 knots and the maximum supported range where a full exit-bin
    mapping first becomes available. Out-of-domain ranges are never evaluated.
    """
    minimum, maximum = stopping_table.energy_mev[0], stopping_table.energy_mev[-1]
    supported_bins = tuple(
        (lower, upper)
        for lower, upper in zip(output_edges_mev, output_edges_mev[1:])
        if minimum <= lower < upper <= maximum
    )
    if not supported_bins:
        return ()
    birth_knots = set(stopping_table.energy_mev) if include_transport_birth_knots else set()
    attenuation_knots = set(stopping_table.energy_mev)
    if kernel.nonelastic_cross_section is not None:
        attenuation_knots.update(
            energy
            for energy in kernel.nonelastic_cross_section.energy_mev
            if minimum <= energy <= maximum
        )
        attenuation_support = kernel.nonelastic_cross_section.incident_energy_support
        assert attenuation_support is not None
        attenuation_knots.update(
            energy
            for energy in (attenuation_support.minimum_mev, attenuation_support.maximum_mev)
            if minimum <= energy <= maximum
        )
    if include_transport_birth_knots:
        birth_knots.update(attenuation_knots)
    if pdf_birth_knots_mev is None:
        pdf_birth_knots_mev = tuple(
            energy for product in kernel.products
            for grid in product.emitted_energy_mev_by_incident for energy in grid
        )
    birth_knots.update(energy for energy in pdf_birth_knots_mev if minimum <= energy <= maximum)
    output_knots = {edge for interval in supported_bins for edge in interval}
    output_knots.update(
        energy
        for energy in attenuation_knots
        if any(lower < energy < upper for lower, upper in supported_bins)
    )
    output_ranges = sorted(stopping_table.range_at_energy(energy) for energy in output_knots)
    events: set[float] = set()
    for birth_energy in sorted(birth_knots):
        birth_range = stopping_table.range_at_energy(birth_energy)
        start = bisect_right(output_ranges, birth_range - thickness_g_cm2)
        stop = bisect_left(output_ranges, birth_range)
        for output_range in output_ranges[start:stop]:
            depth = thickness_g_cm2 - (birth_range - output_range)
            if 0.0 < depth < thickness_g_cm2:
                events.add(depth)
                if max_knots is not None and len(events) > max_knots:
                    raise _CrossingPartitionBudgetExceeded
    return tuple(sorted(events))


@dataclass(frozen=True)
class _PrimaryEnergyDepthCrossings:
    lower_range_g_cm2: float
    upper_range_g_cm2: float
    knots_g_cm2: tuple[float, ...]


@dataclass(frozen=True)
class _PrimaryPdfCrossingCell:
    lower_range_g_cm2: float
    upper_range_g_cm2: float
    birth_knots_mev: tuple[float, ...]
    pdf_support_missing: bool


@dataclass(frozen=True)
class _ExitDepthPartition:
    knots_g_cm2: tuple[float, ...]
    budget_exceeded: bool
    active_cells: tuple[_PrimaryEnergyDepthCrossings, ...] = ()

    @property
    def unique_depth_count(self) -> int:
        return len(set(self.knots_g_cm2).union(*(cell.knots_g_cm2 for cell in self.active_cells)))


def _ranges_intersect(
    lower: float, upper: float, intervals: tuple[tuple[float, float], ...]
) -> bool:
    # Conservative one-ULP expansion only retains extra events, never changes
    # their location. All comparisons are in mass-range, not inverse-energy.
    return any(
        nextafter(lower, -inf) <= nextafter(hi, inf)
        and nextafter(upper, inf) >= nextafter(lo, -inf)
        for lo, hi in intervals
    )


def _positive_source_range_intervals(
    source: Spectrum1D, stopping_table: ProtonAlRangeEnergyTable
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for lo, hi, left, right in zip(source.x, source.x[1:], source.y, source.y[1:]):
        if left <= 0.0 and right <= 0.0:
            continue
        lo, hi = max(lo, stopping_table.energy_mev[0]), min(hi, stopping_table.energy_mev[-1])
        if lo >= hi:
            continue
        lower, upper = stopping_table.range_at_energy(lo), stopping_table.range_at_energy(hi)
        if intervals and lower <= intervals[-1][1]:
            intervals[-1] = intervals[-1][0], upper
        else:
            intervals.append((lower, upper))
    return tuple(intervals)


def _prepare_primary_pdf_crossing_cells(
    *,
    incident_energy_knots: tuple[float, ...],
    source_range_intervals: tuple[tuple[float, float], ...],
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
) -> tuple[_PrimaryPdfCrossingCell, ...]:
    reachable = tuple((lo - thickness_g_cm2, hi) for lo, hi in source_range_intervals)
    support = kernel.incident_energy_support
    assert support is not None
    cells = []
    for lower, upper in zip(incident_energy_knots, incident_energy_knots[1:]):
        if not (support.contains(lower) and support.contains(upper)):
            continue
        lower_range, upper_range = (
            stopping_table.range_at_energy(lower), stopping_table.range_at_energy(upper)
        )
        if not _ranges_intersect(lower_range, upper_range, reachable):
            continue
        births: set[float] = set()
        missing = False
        for product in kernel.products:
            cross_section = kernel.cross_sections_by_mt[product.mt]
            if _secondary_product_production_is_proven_zero(
                minimum_energy_mev=lower, maximum_energy_mev=upper,
                product=product, cross_section=cross_section,
            ):
                continue
            grid = product.pdf_incident_energy_mev
            domains = (product.incident_energy_support, cross_section.incident_energy_support)
            covered = bool(grid) and grid[0] <= lower < upper <= grid[-1] and all(
                domain is not None and domain.contains(lower) and domain.contains(upper)
                for domain in domains
            )
            if covered:
                first = max(0, bisect_right(grid, lower) - 1)
                stop = min(len(grid), bisect_left(grid, upper) + 1)
                selected = product.emitted_energy_mev_by_incident[first:stop]
            else:
                # Unknown is not zero and is not permission to discard knots.
                missing = True
                selected = product.emitted_energy_mev_by_incident
            births.update(energy for emitted in selected for energy in emitted)
        if births or missing:
            cells.append(_PrimaryPdfCrossingCell(
                lower_range, upper_range, tuple(sorted(births)), missing
            ))
    return tuple(cells)


def _active_cell_incident_energy_knots(
    *, kernel: SecondaryProtonKernel, stopping_table: ProtonAlRangeEnergyTable
) -> tuple[float, ...]:
    knots = set(_secondary_incident_energy_knots(kernel=kernel, stopping_table=stopping_table))
    for product in kernel.products:
        for support in (
            product.incident_energy_support,
            kernel.cross_sections_by_mt[product.mt].incident_energy_support,
        ):
            assert support is not None
            knots.update(
                energy for energy in (support.minimum_mev, support.maximum_mev)
                if stopping_table.energy_mev[0] <= energy <= stopping_table.energy_mev[-1]
            )
    return tuple(sorted(knots))


def _prepare_active_exit_depth_partition(
    *,
    output_edges_mev: tuple[float, ...],
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    options: DepthCrossingQuadrature,
    primary_cells: tuple[_PrimaryPdfCrossingCell, ...],
    source_range_intervals: tuple[tuple[float, float], ...],
) -> _ExitDepthPartition:
    try:
        common = _secondary_exit_depth_knots(
            output_edges_mev=output_edges_mev, thickness_g_cm2=thickness_g_cm2,
            stopping_table=stopping_table, kernel=kernel, max_knots=options.max_depth_knots,
            pdf_birth_knots_mev=(),
        )
        unique = set(common)
        records = 0
        active_cells = []
        event_cache: dict[tuple[float, ...], tuple[float, ...]] = {}
        for cell in primary_cells:
            if cell.birth_knots_mev not in event_cache:
                event_cache[cell.birth_knots_mev] = _secondary_exit_depth_knots(
                    output_edges_mev=output_edges_mev, thickness_g_cm2=thickness_g_cm2,
                    stopping_table=stopping_table, kernel=kernel,
                    pdf_birth_knots_mev=cell.birth_knots_mev,
                    include_transport_birth_knots=False,
                )
            events = tuple(
                depth for depth in event_cache[cell.birth_knots_mev]
                if depth not in common and _ranges_intersect(
                    cell.lower_range_g_cm2 + depth, cell.upper_range_g_cm2 + depth,
                    source_range_intervals,
                )
            )
            unique.update(events)
            records += len(events)
            if len(unique) > options.max_depth_knots or records > options.max_active_cell_records:
                raise _CrossingPartitionBudgetExceeded
            if events:
                active_cells.append(_PrimaryEnergyDepthCrossings(
                    cell.lower_range_g_cm2, cell.upper_range_g_cm2, events
                ))
        return _ExitDepthPartition(common, False, tuple(active_cells))
    except _CrossingPartitionBudgetExceeded:
        # Discard ALL additions, never an arbitrary prefix of active cells.
        return _ExitDepthPartition((), True)


def _prepare_exit_depth_partition(
    *,
    output_edges_mev: tuple[float, ...],
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    options: DepthCrossingQuadrature,
) -> _ExitDepthPartition:
    try:
        knots = _secondary_exit_depth_knots(
            output_edges_mev=output_edges_mev,
            thickness_g_cm2=thickness_g_cm2,
            stopping_table=stopping_table,
            kernel=kernel,
            max_knots=options.max_depth_knots,
        )
        return _ExitDepthPartition(knots, False)
    except _CrossingPartitionBudgetExceeded:
        # Never keep an arbitrary prefix and call it a complete partition.
        return _ExitDepthPartition((), True)


def _exit_depth_partition_parameters(
    partition: _ExitDepthPartition,
    options: DepthCrossingQuadrature,
) -> tuple[tuple[str, str], ...]:
    return (
        (
            "depth_output_crossing_knots",
            "budget_exceeded_legacy_partition_retained"
            if partition.budget_exceeded
            else "all_supported_pdf_range_attenuation_crossings",
        ),
        ("depth_output_crossing_count", str(len(partition.knots_g_cm2))),
        ("depth_output_crossing_max_knots", str(options.max_depth_knots)),
        ("depth_output_crossing_accuracy", "not_established_by_refinement"),
    )


def _depth_quadrature_intervals(
    *,
    entrance_energy_mev: float,
    thickness_g_cm2: float,
    depth_steps: int,
    stopping_table: ProtonAlRangeEnergyTable,
    incident_energy_knots_mev: tuple[float, ...],
    exit_depth_knots_g_cm2: tuple[float, ...] = (),
    active_exit_cells: tuple[_PrimaryEnergyDepthCrossings, ...] = (),
) -> tuple[tuple[float, float], ...]:
    """Split depth at uniform cells and every mapped physics/data knot."""

    entrance_range = stopping_table.range_at_energy(entrance_energy_mev)
    active_depth = min(
        thickness_g_cm2,
        entrance_range - stopping_table.range_g_cm2[0],
    )
    if active_depth <= 0.0:
        return ()

    boundaries = {active_depth * index / depth_steps for index in range(depth_steps + 1)}
    boundaries.update(depth for depth in exit_depth_knots_g_cm2 if 0.0 < depth < active_depth)
    for cell in active_exit_cells:
        # Expand the mass-range operands BEFORE subtraction: when d is tiny
        # compared with R0, one ULP of the difference cannot cover cancellation
        # in R0-Rcell. This can only retain an extra event; d* itself is unchanged.
        lower = nextafter(
            nextafter(entrance_range, -inf) - nextafter(cell.upper_range_g_cm2, inf), -inf
        )
        upper = nextafter(
            nextafter(entrance_range, inf) - nextafter(cell.lower_range_g_cm2, -inf), inf
        )
        start, stop = bisect_left(cell.knots_g_cm2, lower), bisect_right(cell.knots_g_cm2, upper)
        boundaries.update(depth for depth in cell.knots_g_cm2[start:stop] if 0 < depth < active_depth)
    for energy in incident_energy_knots_mev:
        if energy >= entrance_energy_mev:
            continue
        depth = entrance_range - stopping_table.range_at_energy(energy)
        if 0.0 < depth < active_depth:
            boundaries.add(depth)
    ordered = tuple(sorted(boundaries))
    return tuple((lower, upper) for lower, upper in zip(ordered, ordered[1:]) if upper > lower)


def _mapped_input_edge_at_depth(
    *,
    entrance_edge_mev: float,
    depth_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
) -> float | None:
    """Map an incident-bin edge to a depth without extrapolating range data."""

    residual_range = stopping_table.range_at_energy(entrance_edge_mev) - depth_g_cm2
    if residual_range < stopping_table.range_g_cm2[0]:
        return None
    return stopping_table.energy_at_range(residual_range)


@dataclass(frozen=True)
class _SingleIncidentResponse:
    counts: tuple[float, ...]
    reaction_energies: tuple[float, ...]
    attenuation_energies: tuple[float, ...]
    stopped_below_range_support: bool
    range_domain_mappings: int
    production_samples: int
    attenuation_missing: bool


def _single_incident_secondary_response(
    *,
    entrance_energy_mev: float,
    output_centers_mev: tuple[float, ...],
    output_edges_mev: tuple[float, ...],
    thickness_g_cm2: float,
    depth_steps: int,
    survival_integration_steps: int,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    optical_depth_table: _NonelasticOpticalDepthTable | None,
    incident_energy_knots: tuple[float, ...],
    exit_depth_knots_g_cm2: tuple[float, ...] = (),
    active_exit_cells: tuple[_PrimaryEnergyDepthCrossings, ...] = (),
    reuse_birth_survival_partition: bool = False,
) -> _SingleIncidentResponse:
    """Return exit-bin counts for one incident proton, with no source amplitude.

    The depth, emitted-histogram and attenuation operators are unchanged;
    the outer quadrature does not add absent transport physics.
    """

    counts = [0.0] * len(output_centers_mev)
    reaction_energies = [entrance_energy_mev]
    attenuation_energies = [entrance_energy_mev]
    range_domain_mappings = 0
    production_samples = 0
    attenuation_missing = False
    support = kernel.incident_energy_support
    assert support is not None
    cross_section = kernel.nonelastic_cross_section
    entrance_range = stopping_table.range_at_energy(entrance_energy_mev)
    intervals = _depth_quadrature_intervals(
        entrance_energy_mev=entrance_energy_mev,
        thickness_g_cm2=thickness_g_cm2,
        depth_steps=depth_steps,
        stopping_table=stopping_table,
        incident_energy_knots_mev=incident_energy_knots,
        exit_depth_knots_g_cm2=exit_depth_knots_g_cm2,
        active_exit_cells=active_exit_cells,
    )
    for lower, upper in intervals:
        for depth, depth_weight in _gauss_legendre_4_samples(lower, upper):
            energy = stopping_table.energy_at_range(entrance_range - depth)
            reaction_energies.append(energy)
            attenuation_energies.append(energy)
            if not support.contains(energy):
                continue
            production_samples += 1
            active_products = _active_secondary_products_at_incident_energy(
                kernel=kernel,
                incident_energy_mev=energy,
            )
            if not active_products:
                continue
            exit_mappings = _secondary_exit_mappings(
                output_centers_mev=output_centers_mev,
                output_edges_mev=output_edges_mev,
                residual_thickness_g_cm2=thickness_g_cm2 - depth,
                stopping_table=stopping_table,
                cross_section_table=cross_section,
            )
            range_domain_mappings += exit_mappings.range_domain_bins
            primary_survival, primary_supported = _nonelastic_survival_or_unity(
                output_energy_mev=energy,
                input_energy_mev=entrance_energy_mev,
                stopping_table=stopping_table,
                cross_section_table=cross_section,
                integration_steps=survival_integration_steps,
                optical_depth_table=optical_depth_table,
            )
            for mapping in exit_mappings.mappings:
                density = _integrated_secondary_source_to_exit_per_g_cm2(
                    active_products=active_products,
                    nonelastic_cross_section=cross_section,
                    optical_depth_table=optical_depth_table,
                    mapping=mapping,
                    stopping_table=stopping_table,
                    reuse_birth_survival_partition=reuse_birth_survival_partition,
                )
                if density <= 0:
                    continue
                attenuation_energies.extend(mapping.attenuation_energies_mev)
                attenuation_missing |= not primary_supported or not mapping.attenuation_supported
                counts[mapping.output_index] += primary_survival * density * depth_weight
    return _SingleIncidentResponse(
        counts=tuple(counts),
        # Coverage is an interval test, so extrema retain all coverage evidence
        # without storing each nested depth/output quadrature node in the cache.
        reaction_energies=(min(reaction_energies), max(reaction_energies)),
        attenuation_energies=(min(attenuation_energies), max(attenuation_energies)),
        stopped_below_range_support=(
            entrance_range - thickness_g_cm2 < stopping_table.range_g_cm2[0]
        ),
        range_domain_mappings=range_domain_mappings,
        production_samples=production_samples,
        attenuation_missing=attenuation_missing,
    )


def _point_source_quadrature_boundaries(
    *,
    source_nodes: tuple[float, ...],
    incident_energy_knots: tuple[float, ...],
    stopping_table: ProtonAlRangeEnergyTable,
    thickness_g_cm2: float,
    exit_depth_knots_g_cm2: tuple[float, ...] = (),
    max_coupled_knots: int | None = None,
    active_exit_cells: tuple[_PrimaryEnergyDepthCrossings, ...] = (),
    source_range_intervals: tuple[tuple[float, float], ...] | None = None,
) -> tuple[float, ...]:
    """Split the true source support at direct and full-depth mapped knots."""

    boundaries = set(source_nodes)
    for energy in incident_energy_knots:
        if source_nodes[0] < energy < source_nodes[-1]:
            boundaries.add(energy)
        mapped_range = stopping_table.range_at_energy(energy) + thickness_g_cm2
        if mapped_range <= stopping_table.max_range_g_cm2:
            mapped_energy = stopping_table.energy_at_range(mapped_range)
            if source_nodes[0] < mapped_energy < source_nodes[-1]:
                boundaries.add(mapped_energy)
    initial_count = len(boundaries)

    def add_coupled_range(mapped_range: float) -> None:
        if mapped_range > stopping_table.max_range_g_cm2:
            return
        if source_range_intervals is not None and not _ranges_intersect(
            mapped_range, mapped_range, source_range_intervals
        ):
            return
        mapped_energy = stopping_table.energy_at_range(mapped_range)
        if source_nodes[0] < mapped_energy < source_nodes[-1]:
            boundaries.add(mapped_energy)
            if max_coupled_knots is not None and len(boundaries) - initial_count > max_coupled_knots:
                raise _CrossingPartitionBudgetExceeded

    for energy in incident_energy_knots:
        incident_range = stopping_table.range_at_energy(energy)
        for depth in exit_depth_knots_g_cm2:
            add_coupled_range(incident_range + depth)
    for cell in active_exit_cells:
        # Every original cell boundary is required, even if adjacent cells
        # happen to use exactly the same PDF pair.
        for depth in cell.knots_g_cm2:
            add_coupled_range(cell.lower_range_g_cm2 + depth)
            add_coupled_range(cell.upper_range_g_cm2 + depth)
    return tuple(sorted(boundaries))


def _secondary_product_production_is_proven_zero(
    *,
    minimum_energy_mev: float,
    maximum_energy_mev: float,
    product: SecondaryProtonProductDistribution,
    cross_section: Al27CrossSectionTable,
) -> bool:
    """Use nonnegative linear factors on every knot-free incident interval."""
    if minimum_energy_mev >= maximum_energy_mev or any(
        support is None
        or not support.contains(minimum_energy_mev)
        or not support.contains(maximum_energy_mev)
        for support in (product.incident_energy_support, cross_section.incident_energy_support)
    ):
        return False
    knots = {minimum_energy_mev, maximum_energy_mev}
    knots.update(
        energy
        for energy in (*cross_section.energy_mev, *product.incident_energy_mev)
        if minimum_energy_mev < energy < maximum_energy_mev
    )
    knots.update(
        item.energy_mev
        for item in cross_section.discontinuities
        if minimum_energy_mev < item.energy_mev < maximum_energy_mev
    )
    ordered = tuple(sorted(knots))
    for lower, upper in zip(ordered, ordered[1:]):
        # A zero interior value of a nonnegative linear factor proves it is
        # identically zero on this cell, not merely zero at a sampled node.
        middle = 0.5 * (lower + upper)
        if not lower < middle < upper:
            # Adjacent representable floats have no representable midpoint.
            # Sampling a rounded endpoint cannot prove a linear factor zero
            # throughout the mathematical open interval (or across a jump).
            return False
        if (
            cross_section.sigma_at_energy(middle) > 0
            and product.yield_at_incident_energy(middle) > 0
        ):
            return False
    return True


def _secondary_production_is_proven_zero(
    *,
    entrance_lower_mev: float,
    entrance_upper_mev: float,
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
) -> bool:
    """Prove zero by MF3*yield on every reachable incident-energy segment.

    No sampled response or PDF-node absence is used as evidence of zero. This
    intentionally conservative proof does not attempt to certify geometrical
    non-overlap of moving birth and exit support.
    """

    minimum_range = stopping_table.range_at_energy(entrance_lower_mev) - thickness_g_cm2
    if minimum_range < stopping_table.range_g_cm2[0]:
        return False
    minimum_energy = stopping_table.energy_at_range(minimum_range)
    support = kernel.incident_energy_support
    assert support is not None
    if not support.contains(minimum_energy) or not support.contains(entrance_upper_mev):
        return False
    return all(
        _secondary_product_production_is_proven_zero(
            minimum_energy_mev=minimum_energy,
            maximum_energy_mev=entrance_upper_mev,
            product=product,
            cross_section=kernel.cross_sections_by_mt[product.mt],
        )
        for product in kernel.products
    )


def _secondary_bin_geometry_is_proven_zero(
    *,
    entrance_lower_mev: float,
    entrance_upper_mev: float,
    output_edges_mev: tuple[float, float],
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
) -> bool:
    """Prove non-overlap using every possible positive histogram interval.

    For 0 <= d <= t the CSDA birth interval is contained in
    [E_exit_lo, R^-1(R(E_exit_hi)+t)]. No sampled response or PDF slice is a
    support proof. Refuse this proof if any primary/birth path or required PDF
    is unsupported. Declared linear incident interpolation limits possible
    slices to all bracketing neighbours of the reachable interval, including
    every interior slice. An identically zero MF3*yield product needs no PDF.
    This bound still does not exploit depth/incident/PDF correlations.
    """
    minimum, maximum = stopping_table.energy_mev[0], stopping_table.energy_mev[-1]
    lower, upper = output_edges_mev
    if not (
        minimum <= entrance_lower_mev < entrance_upper_mev <= maximum
        and minimum <= lower < upper <= maximum
    ):
        return False
    primary_minimum_range = stopping_table.range_at_energy(entrance_lower_mev) - thickness_g_cm2
    birth_maximum_range = stopping_table.range_at_energy(upper) + thickness_g_cm2
    if (
        primary_minimum_range < stopping_table.range_g_cm2[0]
        or birth_maximum_range > stopping_table.max_range_g_cm2
    ):
        return False
    primary_minimum = stopping_table.energy_at_range(primary_minimum_range)
    birth_maximum = stopping_table.energy_at_range(birth_maximum_range)
    support = kernel.incident_energy_support
    if support is None or not (
        support.contains(primary_minimum) and support.contains(entrance_upper_mev)
    ):
        return False
    for product in kernel.products:
        cross_section = kernel.cross_sections_by_mt[product.mt]
        if any(
            domain is None
            or not domain.contains(primary_minimum)
            or not domain.contains(entrance_upper_mev)
            for domain in (product.incident_energy_support, cross_section.incident_energy_support)
        ):
            return False
        if _secondary_product_production_is_proven_zero(
            minimum_energy_mev=primary_minimum,
            maximum_energy_mev=entrance_upper_mev,
            product=product,
            cross_section=cross_section,
        ):
            continue
        grid = product.pdf_incident_energy_mev
        if not grid or primary_minimum < grid[0] or entrance_upper_mev > grid[-1]:
            return False
        first = max(0, bisect_right(grid, primary_minimum) - 1)
        stop = min(len(grid), bisect_left(grid, entrance_upper_mev) + 1)
        for emitted, density in zip(
            product.emitted_energy_mev_by_incident[first:stop],
            product.pdf_per_mev_by_incident[first:stop],
            strict=True,
        ):
            if any(
                value > 0.0 and max(lower, lo) < min(birth_maximum, hi)
                for lo, hi, value in zip(emitted, emitted[1:], density)
            ):
                return False
    return True


_ScreeningClassification = Literal["source_zero", "proven_zero", "omitted_unsupported", "evaluate"]


@dataclass(frozen=True)
class _ScreenedIncidentInterval:
    lower_mev: float
    upper_mev: float
    classification: _ScreeningClassification
    basis: str
    source_integral: float
    path_energies: tuple[float, ...] = ()
    range_domain_missing: bool = False


@dataclass(frozen=True)
class _SecondaryScreeningPlan:
    boundaries_mev: tuple[float, ...]
    intervals: tuple[_ScreenedIncidentInterval, ...]
    evaluation_range_intervals_g_cm2: tuple[tuple[float, float], ...]

    def for_interval(self, lower: float, upper: float) -> _ScreenedIncidentInterval:
        index = bisect_right(self.boundaries_mev, lower) - 1
        if 0 <= index < len(self.intervals):
            item = self.intervals[index]
            if item.lower_mev <= lower < upper <= item.upper_mev:
                return item
        # A proof on one cell does not apply to a larger, crossing interval.
        return _ScreenedIncidentInterval(lower, upper, "evaluate", "not_contained_in_base_cell", 0.0)


def _screen_secondary_incident_interval(
    *,
    lower_mev: float,
    upper_mev: float,
    output_edges_mev: tuple[float, float],
    thickness_g_cm2: float,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    source_integral: float = 0.0,
) -> _ScreenedIncidentInterval:
    """Classify a whole positive-source cell without sampling its response.

    Unsupported is explicitly NOT a physical zero. Only the two analytic
    full-support proofs may certify zero. Partial paths remain evaluable.
    Coverage extrema are retained even if no costly response is ever called.
    """
    minimum, maximum = stopping_table.energy_mev[0], stopping_table.energy_mev[-1]
    if not minimum <= lower_mev < upper_mev <= maximum:
        return _ScreenedIncidentInterval(
            lower_mev, upper_mev, "omitted_unsupported", "incident_range_domain",
            source_integral, (lower_mev, upper_mev), True,
        )
    rmin, rmax = stopping_table.range_g_cm2[0], stopping_table.max_range_g_cm2
    entrance_lower_range = stopping_table.range_at_energy(lower_mev)
    entrance_upper_range = stopping_table.range_at_energy(upper_mev)
    raw_minimum_range = entrance_lower_range - thickness_g_cm2
    minimum_path_range = max(rmin, raw_minimum_range)
    minimum_path_energy = stopping_table.energy_at_range(minimum_path_range)
    path_energies = (minimum_path_energy, upper_mev)
    stopped = raw_minimum_range < rmin

    def decision(
        classification: _ScreeningClassification, basis: str, *, range_missing: bool = False
    ) -> _ScreenedIncidentInterval:
        return _ScreenedIncidentInterval(
            lower_mev, upper_mev, classification, basis, source_integral,
            path_energies, stopped or range_missing,
        )

    # A complete output-bin average cannot be formed by silently clipping its
    # declared edges to the range table. Preserve the omission explicitly.
    output_lower, output_upper = output_edges_mev
    if not minimum <= output_lower < output_upper <= maximum:
        return decision("omitted_unsupported", "output_range_domain", range_missing=True)

    support = kernel.incident_energy_support
    assert support is not None
    # Expand operands before subtraction to avoid inventing an unsupported
    # region through cancellation near a support boundary.
    conservative_raw_minimum_range = nextafter(
        nextafter(entrance_lower_range, -inf) - nextafter(thickness_g_cm2, inf), -inf
    )
    lower_path_bound = max(rmin, conservative_raw_minimum_range)
    lower_energy_bound = stopping_table.energy_at_range(lower_path_bound)
    if (
        nextafter(upper_mev, inf) < nextafter(support.minimum_mev, -inf)
        or nextafter(lower_energy_bound, -inf) > nextafter(support.maximum_mev, inf)
    ):
        return decision("omitted_unsupported", "all_primary_depths_outside_tendl_support")

    # Last possible primary depth gives the smallest remaining secondary path.
    # If even there the FULL output bin needs birth range > Rmax, no represented
    # mapping exists. Equality or a narrow supported strip MUST be evaluated.
    maximum_depth_bound = min(thickness_g_cm2, nextafter(
        nextafter(entrance_upper_range, inf) - nextafter(rmin, -inf), inf
    ))
    residual_lower_bound = max(0.0, nextafter(
        nextafter(thickness_g_cm2, -inf) - nextafter(maximum_depth_bound, inf), -inf
    ))
    output_upper_range = stopping_table.range_at_energy(output_upper)
    birth_lower_bound = nextafter(
        nextafter(output_upper_range, -inf) + nextafter(residual_lower_bound, -inf), -inf
    )
    if birth_lower_bound > nextafter(rmax, inf):
        return decision("omitted_unsupported", "no_supported_full_birth_mapping", range_missing=True)

    # These guards preserve domain evidence which the old sampled path gathered
    # even for a zero response. No missing PDF is needed for a reaction proven
    # inactive; otherwise the geometry proof checks all necessary PDF support.
    full_birth_bound = nextafter(
        nextafter(output_upper_range, inf) + nextafter(thickness_g_cm2, inf), inf
    )
    if (
        conservative_raw_minimum_range < rmin or stopped or full_birth_bound > rmax
        or not support.contains(lower_energy_bound) or not support.contains(upper_mev)
    ):
        return decision("evaluate", "partial_or_boundary_range_path")
    if _secondary_production_is_proven_zero(
        entrance_lower_mev=lower_mev, entrance_upper_mev=upper_mev,
        thickness_g_cm2=thickness_g_cm2, stopping_table=stopping_table, kernel=kernel,
    ):
        return decision("proven_zero", "identically_zero_supported_reaction")
    if _secondary_bin_geometry_is_proven_zero(
        entrance_lower_mev=lower_mev, entrance_upper_mev=upper_mev,
        output_edges_mev=output_edges_mev, thickness_g_cm2=thickness_g_cm2,
        stopping_table=stopping_table, kernel=kernel,
    ):
        return decision("proven_zero", "supported_positive_pdf_segments_do_not_overlap")
    return decision("evaluate", "no_full_interval_zero_proof")


def _prepare_secondary_screening_plan(
    *,
    incident_spectrum: Spectrum1D,
    output_edges_mev: tuple[float, float],
    incident_energy_knots: tuple[float, ...],
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    thickness_g_cm2: float,
) -> _SecondaryScreeningPlan:
    boundaries = _point_source_quadrature_boundaries(
        source_nodes=incident_spectrum.x, incident_energy_knots=incident_energy_knots,
        stopping_table=stopping_table, thickness_g_cm2=thickness_g_cm2,
    )

    def source_value(energy: float) -> float:
        index = min(bisect_right(incident_spectrum.x, energy) - 1, len(incident_spectrum.x) - 2)
        lo, hi = incident_spectrum.x[index:index + 2]
        fraction = (energy - lo) / (hi - lo)
        return (1.0 - fraction) * incident_spectrum.y[index] + fraction * incident_spectrum.y[index + 1]

    intervals = []
    ranges: list[tuple[float, float]] = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        left, right = source_value(lower), source_value(upper)
        if left <= 0.0 and right <= 0.0:
            intervals.append(_ScreenedIncidentInterval(lower, upper, "source_zero", "zero_linear_source", 0.0))
            continue
        item = _screen_secondary_incident_interval(
            lower_mev=lower, upper_mev=upper, output_edges_mev=output_edges_mev,
            thickness_g_cm2=thickness_g_cm2, stopping_table=stopping_table, kernel=kernel,
            source_integral=(0.5 * left + 0.5 * right) * (upper - lower),
        )
        intervals.append(item)
        if item.classification == "evaluate":
            lo, hi = stopping_table.range_at_energy(lower), stopping_table.range_at_energy(upper)
            if ranges and lo <= ranges[-1][1]:
                ranges[-1] = ranges[-1][0], hi
            else:
                ranges.append((lo, hi))
    return _SecondaryScreeningPlan(boundaries, tuple(intervals), tuple(ranges))


@dataclass(frozen=True)
class _IncidentOutputPartition:
    output_indices: tuple[int, ...]
    centers_mev: tuple[float, ...]
    edges_mev: tuple[float, ...]
    depth: _ExitDepthPartition
    boundaries_mev: tuple[float, ...]
    coupled_budget_exceeded: bool
    screening: _SecondaryScreeningPlan | None = None


def _prepare_incident_output_partition(
    *,
    output_indices: tuple[int, ...],
    source_nodes: tuple[float, ...],
    output_edges: tuple[float, ...],
    incident_energy_knots: tuple[float, ...],
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    thickness_g_cm2: float,
    depth_options: DepthCrossingQuadrature,
    primary_pdf_cells: tuple[_PrimaryPdfCrossingCell, ...] | None = None,
    source_range_intervals: tuple[tuple[float, float], ...] | None = None,
    screening_source: Spectrum1D | None = None,
) -> _IncidentOutputPartition:
    # Keep the ORIGINAL full-grid bin boundaries, not edges reconstructed from
    # a singleton centre. Only the vector reference or one bin is supported.
    first, last = output_indices[0], output_indices[-1]
    centers = source_nodes[first : last + 1]
    edges = output_edges[first : last + 2]
    screening = None
    if screening_source is not None:
        if len(output_indices) != 1 or primary_pdf_cells is None:
            raise ValueError("Screening requires an active-cell scalar-bin partition.")
        screening = _prepare_secondary_screening_plan(
            incident_spectrum=screening_source, output_edges_mev=(edges[0], edges[1]),
            incident_energy_knots=incident_energy_knots, stopping_table=stopping_table,
            kernel=kernel, thickness_g_cm2=thickness_g_cm2,
        )
        source_range_intervals = screening.evaluation_range_intervals_g_cm2
        if not source_range_intervals:
            return _IncidentOutputPartition(
                output_indices, centers, edges, _ExitDepthPartition((), False),
                screening.boundaries_mev, False, screening,
            )
    if primary_pdf_cells is None:
        depth = _prepare_exit_depth_partition(
            output_edges_mev=edges,
            thickness_g_cm2=thickness_g_cm2,
            stopping_table=stopping_table,
            kernel=kernel,
            options=depth_options,
        )
    else:
        assert source_range_intervals is not None
        depth = _prepare_active_exit_depth_partition(
            output_edges_mev=edges, thickness_g_cm2=thickness_g_cm2,
            stopping_table=stopping_table, kernel=kernel, options=depth_options,
            primary_cells=primary_pdf_cells, source_range_intervals=source_range_intervals,
        )
    coupled_budget_exceeded = False
    try:
        boundaries = _point_source_quadrature_boundaries(
            source_nodes=source_nodes,
            incident_energy_knots=incident_energy_knots,
            stopping_table=stopping_table,
            thickness_g_cm2=thickness_g_cm2,
            exit_depth_knots_g_cm2=depth.knots_g_cm2,
            max_coupled_knots=depth_options.max_coupled_incident_knots,
            active_exit_cells=depth.active_cells,
            source_range_intervals=source_range_intervals,
        )
    except _CrossingPartitionBudgetExceeded:
        coupled_budget_exceeded = True
        boundaries = _point_source_quadrature_boundaries(
            source_nodes=source_nodes,
            incident_energy_knots=incident_energy_knots,
            stopping_table=stopping_table,
            thickness_g_cm2=thickness_g_cm2,
        )
    return _IncidentOutputPartition(
        output_indices, centers, edges, depth, boundaries, coupled_budget_exceeded, screening
    )


def _calculate_secondary_with_incident_quadrature(
    *,
    incident_spectrum: Spectrum1D,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    thickness_g_cm2: float,
    depth_steps: int,
    survival_integration_steps: int,
    options: IncidentEnergyQuadrature,
    depth_options: DepthCrossingQuadrature,
    model: str | None,
) -> Spectrum1D:
    """Integrate a point source with a shared global response-work budget."""

    output_edges = log_bin_edges_from_centers(incident_spectrum.x)
    zero = tuple(0.0 for _ in incident_spectrum.x)
    support = kernel.incident_energy_support
    assert support is not None
    cross_section = kernel.nonelastic_cross_section
    optical_depth = (
        None
        if cross_section is None
        else _build_nonelastic_optical_depth_table(
            stopping_table=stopping_table,
            cross_section_table=cross_section,
        )
    )
    active_pruning = options.partition_strategy == "per_output_bin_active_cells"
    knots = (
        _active_cell_incident_energy_knots(kernel=kernel, stopping_table=stopping_table)
        if active_pruning
        else _secondary_incident_energy_knots(kernel=kernel, stopping_table=stopping_table)
    )
    source_range_intervals = (
        _positive_source_range_intervals(incident_spectrum, stopping_table) if active_pruning else None
    )
    primary_pdf_cells = (
        _prepare_primary_pdf_crossing_cells(
            incident_energy_knots=knots, source_range_intervals=source_range_intervals,
            thickness_g_cm2=thickness_g_cm2, stopping_table=stopping_table, kernel=kernel,
        ) if source_range_intervals is not None else None
    )
    pdf_support_missing = bool(primary_pdf_cells) and any(
        cell.pdf_support_missing for cell in primary_pdf_cells or ()
    )
    bin_local = options.partition_strategy != "global_union"
    output_groups = (
        tuple((index,) for index in range(len(zero)))
        if bin_local
        else (tuple(range(len(zero))),)
    )
    partitions = tuple(
        _prepare_incident_output_partition(
            output_indices=indices,
            source_nodes=incident_spectrum.x,
            output_edges=output_edges,
            incident_energy_knots=knots,
            stopping_table=stopping_table,
            kernel=kernel,
            thickness_g_cm2=thickness_g_cm2,
            depth_options=depth_options,
            primary_pdf_cells=primary_pdf_cells,
            source_range_intervals=source_range_intervals,
            screening_source=incident_spectrum if options.prequadrature_screening else None,
        )
        for indices in output_groups
    )
    screening_records = tuple(
        item for partition in partitions if partition.screening is not None
        for item in partition.screening.intervals
    )
    screened_omitted = sum(item.classification == "omitted_unsupported" for item in screening_records)
    screened_range_missing = any(item.range_domain_missing for item in screening_records)
    screened_coverage_energies = tuple(energy for item in screening_records for energy in item.path_energies)
    screened_reaction_domain_missing = any(
        not support.contains(energy) for energy in screened_coverage_energies
    )
    depth_budget_exceeded = any(partition.depth.budget_exceeded for partition in partitions)
    coupled_budget_exceeded = any(partition.coupled_budget_exceeded for partition in partitions)
    partition_incomplete = depth_budget_exceeded or coupled_budget_exceeded
    # The partition identity matters: K_j(E0) is not K_k(E0). All caches share
    # ONE budget, not max_response_evaluations times the number of exit bins.
    response_cache: dict[tuple[int, float], _SingleIncidentResponse] = {}

    def source_value(energy: float) -> float:
        index = min(bisect_right(incident_spectrum.x, energy) - 1, len(zero) - 2)
        lower, upper = incident_spectrum.x[index : index + 2]
        fraction = (energy - lower) / (upper - lower)
        return (1 - fraction) * incident_spectrum.y[index] + fraction * incident_spectrum.y[
            index + 1
        ]

    def rule(partition_index: int, lower: float, upper: float) -> tuple[float, ...] | None:
        partition = partitions[partition_index]
        samples = _gauss_legendre_4_samples(lower, upper)
        required = sum((partition_index, energy) not in response_cache for energy, _ in samples)
        if len(response_cache) + required > options.max_response_evaluations:
            return None
        counts = [0.0] * len(partition.output_indices)
        for energy, weight in samples:
            key = (partition_index, energy)
            if key not in response_cache:
                response_cache[key] = _single_incident_secondary_response(
                    entrance_energy_mev=energy,
                    output_centers_mev=partition.centers_mev,
                    output_edges_mev=partition.edges_mev,
                    thickness_g_cm2=thickness_g_cm2,
                    depth_steps=depth_steps,
                    survival_integration_steps=survival_integration_steps,
                    stopping_table=stopping_table,
                    kernel=kernel,
                    optical_depth_table=optical_depth,
                    incident_energy_knots=knots,
                    exit_depth_knots_g_cm2=partition.depth.knots_g_cm2,
                    active_exit_cells=partition.depth.active_cells,
                    reuse_birth_survival_partition=options.reuse_birth_survival_partition,
                )
            factor = weight * source_value(energy)
            for index, response in enumerate(response_cache[key].counts):
                counts[index] += factor * response
        return tuple(counts)

    def add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(a + b for a, b in zip(left, right, strict=True))

    def refine(
        partition_index: int,
        lower: float,
        upper: float,
        coarse: tuple[float, ...],
        level: int,
    ) -> tuple[tuple[float, ...], float | None, int]:
        middle = 0.5 * (lower + upper)
        left = rule(partition_index, lower, middle)
        right = rule(partition_index, middle, upper)
        if left is None or right is None:
            return coarse, None, 1
        fine = add(left, right)
        error = sum(abs(a - b) for a, b in zip(fine, coarse, strict=True))
        unproven_zero = not any(fine) and not any(coarse)
        if unproven_zero and _secondary_production_is_proven_zero(
            entrance_lower_mev=lower,
            entrance_upper_mev=upper,
            thickness_g_cm2=thickness_g_cm2,
            stopping_table=stopping_table,
            kernel=kernel,
        ):
            return fine, 0.0, 0
        if unproven_zero and bin_local and _secondary_bin_geometry_is_proven_zero(
            entrance_lower_mev=lower,
            entrance_upper_mev=upper,
            output_edges_mev=(
                partitions[partition_index].edges_mev[0],
                partitions[partition_index].edges_mev[1],
            ),
            thickness_g_cm2=thickness_g_cm2,
            stopping_table=stopping_table,
            kernel=kernel,
        ):
            # Samples and their domain evidence have already been retained.
            # Only this supported-bin proof can avoid searching further zeros.
            return fine, 0.0, 0
        if unproven_zero and level >= min(
            options.max_refinement_depth,
            options.max_unproven_zero_refinement_depth,
        ):
            # Two zero quadrature estimates do not exclude a narrow positive
            # response between all sampled nodes. Its error is unknown, not 0.
            return fine, None, 1
        if not unproven_zero and error <= options.relative_tolerance * sum(fine):
            return fine, error, 0
        if level >= options.max_refinement_depth:
            return fine, error, 1
        left_value, left_error, left_unresolved = refine(
            partition_index, lower, middle, left, level + 1
        )
        right_value, right_error, right_unresolved = refine(
            partition_index, middle, upper, right, level + 1
        )
        total_error = (
            None if left_error is None or right_error is None else left_error + right_error
        )
        return add(left_value, right_value), total_error, left_unresolved + right_unresolved

    coarse_intervals: list[tuple[int, float, float, tuple[float, ...]]] = []
    missing_range_intervals = 0
    unresolved = 0
    unresolved_by_partition = [0] * len(partitions)
    relevant_source_energies: list[float] = []
    # First take coarse samples across all partitions, before spending the
    # remaining global budget on refinement of any one exit bin.
    for partition_index, partition in enumerate(partitions):
        for lower, upper in zip(partition.boundaries_mev, partition.boundaries_mev[1:]):
            # Positive-to-zero intervals are retained; skip only a zero linear segment.
            if source_value(lower) <= 0 and source_value(upper) <= 0:
                continue
            relevant_source_energies.extend((lower, upper))
            if lower < stopping_table.energy_mev[0] or upper > stopping_table.energy_mev[-1]:
                missing_range_intervals += 1
                continue
            if partition.screening is not None:
                decision = partition.screening.for_interval(lower, upper)
                if decision.classification != "evaluate":
                    continue
            coarse = rule(partition_index, lower, upper)
            if coarse is None:
                unresolved += 1
                unresolved_by_partition[partition_index] += 1
            else:
                coarse_intervals.append((partition_index, lower, upper, coarse))
    total = [0.0] * len(zero)
    estimated_error: float | None = 0.0 if unresolved == 0 else None
    for partition_index, lower, upper, coarse in coarse_intervals:
        value, error, failures = refine(partition_index, lower, upper, coarse, 0)
        for output_index, count in zip(partitions[partition_index].output_indices, value):
            total[output_index] += count
        unresolved += failures
        unresolved_by_partition[partition_index] += failures
        estimated_error = (
            None if estimated_error is None or error is None else estimated_error + error
        )
    response_values = tuple(response_cache.values())
    evaluations_by_partition = [0] * len(partitions)
    for partition_index, _energy in response_cache:
        evaluations_by_partition[partition_index] += 1
    estimated_relative_error = (
        "unavailable"
        if estimated_error is None
        else (
            f"{estimated_error / sum(total):.17g}"
            if sum(total) > 0
            else ("0" if estimated_error == 0 else "undefined_zero_result")
        )
    )
    # These estimators only describe retained numerical contributions. Missing
    # physical data can never acquire a zero full-result error by being skipped.
    full_error_unknown = options.prequadrature_screening and bool(
        screened_omitted or pdf_support_missing or missing_range_intervals or screened_range_missing
        or screened_reaction_domain_missing
        or any(response.attenuation_missing or response.range_domain_mappings for response in response_values)
    )
    screening_counts = {
        category: sum(item.classification == category for item in screening_records)
        for category in ("source_zero", "proven_zero", "omitted_unsupported", "evaluate")
    }
    screening_integrals = {
        category: [
            sum(item.source_integral for item in partition.screening.intervals
                if item.classification == category) if partition.screening is not None else 0.0
            for partition in partitions
        ]
        for category in screening_counts
    }
    supported_error_text = "unavailable" if estimated_error is None else f"{estimated_error:.17g}"
    supported_status = (
        "incomplete" if unresolved or partition_incomplete
        else "empty_supported_domain" if screened_omitted and not (
            screening_counts["evaluate"] or screening_counts["proven_zero"]
        ) else "converged"
    )
    parameters = (
        (
            "birth_survival_partition",
            "per_mapping_exact_reuse" if options.reuse_birth_survival_partition
            else "per_pdf_histogram_overlap",
        ),
        ("birth_survival_reuse_scope", "one_mapping_only_no_cross_depth_or_dataset_cache"),
        ("prequadrature_screening", "enabled" if options.prequadrature_screening else "disabled"),
        *( (f"prequadrature_{category}_base_intervals", str(count))
           for category, count in screening_counts.items() ),
        ("prequadrature_source_integrals_by_output_bin", json.dumps(screening_integrals, sort_keys=True)),
        ("prequadrature_integral_scope", "per_output_bin_source_weights_not_summed_across_bins"),
        ("prequadrature_unknown_contribution_policy", "omission_is_unknown_not_zero"),
        ("prequadrature_reaction_domain_missing", str(screened_reaction_domain_missing).lower()),
        (
            "crossing_partition_scope",
            "evaluable_screened_intervals_only" if options.prequadrature_screening else "all_positive_source_intervals",
        ),
        ("incident_quadrature_supported_estimated_absolute_error", supported_error_text),
        ("incident_quadrature_supported_estimated_relative_error", estimated_relative_error),
        ("incident_quadrature_supported_status", supported_status),
        ("incident_quadrature_supported_error_scope", "retained_contributions_only_excludes_unknown_physical_domains"),
        (
            "depth_pdf_pruning",
            "exact_reachable_primary_cells_with_all_interpolation_neighbours"
            if active_pruning else "none_all_retained_pdf_slices",
        ),
        ("depth_active_pdf_support_missing", str(pdf_support_missing).lower()),
        ("depth_active_primary_cells", str(len(primary_pdf_cells or ()))),
        (
            "depth_output_crossing_active_cell_records",
            str(sum(len(cell.knots_g_cm2) for p in partitions for cell in p.depth.active_cells)),
        ),
        ("depth_output_crossing_max_active_cell_records", str(depth_options.max_active_cell_records)),
        (
            "incident_bin_operator",
            "integral_linear_point_source_times_scalar_bin_response"
            if bin_local
            else "integral_linear_point_source_times_vector_response",
        ),
        ("incident_quadrature_partition_strategy", options.partition_strategy),
        (
            "incident_quadrature_response_evaluation_unit",
            "scalar_exit_bin_response" if bin_local else "vector_exit_spectrum_response",
        ),
        ("incident_quadrature_budget_scope", "global_across_all_exit_partitions"),
        (
            "incident_quadrature_response_bin_evaluations",
            str(sum(len(partitions[index].output_indices) for index, _ in response_cache)),
        ),
        ("crossing_knot_budget_scope", "per_exit_bin" if bin_local else "whole_exit_grid"),
        (
            "incident_source_support_mev",
            f"{incident_spectrum.x[0]:.17g}..{incident_spectrum.x[-1]:.17g}",
        ),
        ("incident_source_extrapolation", "none"),
        ("incident_integrated_domain", "true_source_intersected_with_range_table_support"),
        ("incident_omitted_range_intervals", str(missing_range_intervals)),
        ("incident_quadrature", "open_gl4_adaptive_direct_mapped_and_coupled_crossing_knots"),
        (
            "incident_crossing_partition",
            "incomplete_depth_partition"
            if depth_budget_exceeded
            else "budget_exceeded_legacy_partition_retained"
            if coupled_budget_exceeded
            else "all_coupled_depth_events_included",
        ),
        ("incident_crossing_max_knots", str(depth_options.max_coupled_incident_knots)),
        (
            "incident_partition_intervals",
            str(sum(len(partition.boundaries_mev) - 1 for partition in partitions)),
        ),
        (
            "incident_partition_intervals_by_partition",
            ",".join(str(len(partition.boundaries_mev) - 1) for partition in partitions),
        ),
        ("incident_quadrature_relative_tolerance", f"{options.relative_tolerance:.17g}"),
        (
            "incident_quadrature_error_measure",
            "L1_exit_bin_counts_coarse_fine_difference_outer_integral_only",
        ),
        (
            "incident_quadrature_estimated_absolute_error",
            "unavailable" if full_error_unknown else supported_error_text,
        ),
        ("incident_quadrature_estimated_relative_error", "unavailable" if full_error_unknown else estimated_relative_error),
        ("incident_quadrature_response_evaluations", str(len(response_cache))),
        (
            "incident_quadrature_response_evaluations_by_partition",
            ",".join(str(count) for count in evaluations_by_partition),
        ),
        (
            "incident_quadrature_supported_depth_samples",
            str(sum(response.production_samples for response in response_values)),
        ),
        ("incident_quadrature_max_response_evaluations", str(options.max_response_evaluations)),
        ("incident_quadrature_max_refinement_depth", str(options.max_refinement_depth)),
        (
            "incident_quadrature_max_unproven_zero_refinement_depth",
            str(options.max_unproven_zero_refinement_depth),
        ),
        (
            "incident_quadrature_zero_policy",
            "reaction_or_supported_pdf_envelope_zero_proof_else_bounded_unknown_error"
            if bin_local
            else "reaction_zero_proof_or_bounded_refinement_with_unknown_error",
        ),
        ("incident_quadrature_unresolved_intervals", str(unresolved)),
        (
            "incident_quadrature_unresolved_intervals_by_partition",
            ",".join(str(count) for count in unresolved_by_partition),
        ),
        (
            "incident_quadrature_status",
            "incomplete"
            if unresolved or partition_incomplete or pdf_support_missing or full_error_unknown
            else ("converged_on_supported_domain" if missing_range_intervals else "converged"),
        ),
        ("depth_steps", str(depth_steps)),
        ("depth_quadrature", "gauss_legendre_4_split_at_transport_knots"),
        (
            "depth_output_crossing_knots",
            "budget_exceeded_legacy_partition_retained"
            if depth_budget_exceeded
            else "all_reachable_active_pdf_and_transport_crossings"
            if active_pruning
            else "all_supported_pdf_range_attenuation_crossings",
        ),
        (
            "depth_output_crossing_count",
            str(sum(partition.depth.unique_depth_count for partition in partitions)),
        ),
        (
            "depth_output_crossing_counts_by_partition",
            ",".join(str(partition.depth.unique_depth_count) for partition in partitions),
        ),
        (
            "depth_output_crossing_budget_exceeded_by_partition",
            ",".join(str(partition.depth.budget_exceeded).lower() for partition in partitions),
        ),
        (
            "incident_crossing_budget_exceeded_by_partition",
            ",".join(str(partition.coupled_budget_exceeded).lower() for partition in partitions),
        ),
        ("depth_output_crossing_max_knots", str(depth_options.max_depth_knots)),
        ("depth_output_crossing_accuracy", "not_established_by_refinement"),
        ("survival_integration_steps", str(survival_integration_steps)),
        ("thickness_g_cm2", f"{thickness_g_cm2:.17g}"),
        ("primary_survival_to_depth", "TENDL_MF3_MT3"),
        ("secondary_survival_to_exit", "TENDL_MF3_MT3"),
        ("secondary_pdf_integration", "exact_endf_histogram_segments"),
        ("mf6_reference_frame", "centre_of_mass_lct_3"),
        ("mf6_cm_to_lab_transform", "not_included"),
        ("secondary_angular_transport", "not_included"),
        ("secondary_path_length", "radial_residual_thickness_diagnostic"),
        ("secondary_generation", "single_collision_generation"),
        ("higher_generation_cascade", "not_included"),
    )
    output = derive_spectrum(
        incident_spectrum,
        y=tuple(value / (hi - lo) for value, lo, hi in zip(total, output_edges, output_edges[1:])),
        particle=Particle.PROTON,
        model=model or f"{incident_spectrum.model}+{PROTON_AL_SECONDARY_MODEL}",
        location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor(material="aluminium", areal_density_g_cm2=thickness_g_cm2),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=output_edges,
        transformation=TransformationStep(
            name=PROTON_AL_SECONDARY_MODEL,
            version=(
                "9_diagnostic_birth_survival_partition_reuse"
                if options.reuse_birth_survival_partition else
                "8_diagnostic_prequadrature_screening"
                if options.prequadrature_screening else
                "7_diagnostic_active_pdf_cell_crossings"
                if active_pruning else
                "6_diagnostic_per_output_bin_crossings"
                if bin_local
                else "5_diagnostic_incident_and_depth_crossings"
            ),
            parameters=parameters,
        ),
        additional_dataset_refs=(
            "stopping/proton_al_range.csv",
            "tendl/p_al27_mf3_xs.csv",
            "tendl/p_al27_mf6_proton_yield.csv",
            "tendl/p_al27_mf6_proton_pdf.csv",
        ),
    )
    reasons = [
        PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
        PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
        PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
        PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    ]
    if pdf_support_missing:
        reasons.append(PROTON_AL_SECONDARY_ACTIVE_PDF_SUPPORT_REASON)
    if screened_omitted:
        reasons.append(PROTON_AL_SECONDARY_SCREENED_DOMAIN_REASON)
    if depth_budget_exceeded:
        reasons.append(PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON)
    if partition_incomplete:
        reasons.append(PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON)
    if unresolved or partition_incomplete:
        reasons.append(PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON)
    reaction_coverage = assess_tendl_incident_energy_coverage(
        relevant_energies_mev=(
            *relevant_source_energies,
            *screened_coverage_energies,
            *(energy for response in response_values for energy in response.reaction_energies),
        ),
        support=support,
    ).incomplete_reason(operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR)
    if reaction_coverage is not None:
        reasons.append(reaction_coverage)
    if cross_section is None:
        if any(response.attenuation_missing for response in response_values):
            reasons.append(PROTON_AL_SECONDARY_ATTENUATION_MISSING_REASON)
    else:
        attenuation_support = cross_section.incident_energy_support
        assert attenuation_support is not None
        attenuation_coverage = assess_tendl_incident_energy_coverage(
            relevant_energies_mev=(
                *relevant_source_energies,
                *screened_coverage_energies,
                *(
                    energy
                    for response in response_values
                    for energy in response.attenuation_energies
                ),
            ),
            support=attenuation_support,
        ).incomplete_reason(operator=PROTON_AL_SECONDARY_ATTENUATION_OPERATOR)
        if attenuation_coverage is not None:
            reasons.append(attenuation_coverage)
    output_range_missing = any(
        lo < stopping_table.energy_mev[0] or hi > stopping_table.energy_mev[-1]
        for lo, hi in zip(output_edges, output_edges[1:])
    ) and any(response.production_samples for response in response_values)
    if (
        missing_range_intervals
        or screened_range_missing
        or output_range_missing
        or any(
            response.stopped_below_range_support or response.range_domain_mappings
            for response in response_values
        )
    ):
        reasons.append(
            f"{PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX}: supported "
            "contributions retained; some incident intervals, depth paths or exit "
            "bins exceed the aluminium range-table support"
        )
    return mark_spectrum_incomplete(output, *reasons)


def calculate_secondary_proton_spectrum_through_al(
    *,
    incident_spectrum: Spectrum1D,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    thickness_g_cm2: float,
    depth_steps: int = 24,
    survival_integration_steps: int = 64,
    incident_quadrature: IncidentEnergyQuadrature | None = None,
    depth_quadrature: DepthCrossingQuadrature = DepthCrossingQuadrature(),
    model: str | None = None,
) -> Spectrum1D:
    """Return diagnostic single-generation secondary-proton transport in Al.

    Every retained first-generation source contribution includes MF=3/MT=3
    survival of the incident primary to its production depth and survival of
    the emitted proton from that depth to the shield exit. If MT3 support is
    unavailable, the diagnostic spectrum retains the unattenuated contribution
    and is marked ``INCOMPLETE``. Higher-generation reactions of those emitted
    protons are intentionally not synthesized and are reported separately.

    ``incident_quadrature`` enables the diagnostic adaptive outer integral for
    POINT_SAMPLE inputs. It is opt-in pending broad runtime and isolated-
    secondary validation; the default retains its existing quadrature blocker.
    ``depth_quadrature`` adds exact supported birth/exit crossing partitions
    within explicit knot budgets. Exceeding a budget preserves the earlier
    diagnostic partition with a separate INCOMPLETE reason, never a silent
    prefix of the required events. Complete partitions do not by themselves
    establish general numerical convergence or the absent transport physics.
    """

    if not isfinite(thickness_g_cm2) or thickness_g_cm2 < 0.0:
        msg = "Al shielding thickness must be non-negative and finite."
        raise ValueError(msg)

    if depth_steps < 1:
        msg = "Secondary-proton depth integration must use at least one step."
        raise ValueError(msg)

    if survival_integration_steps < 1:
        msg = "Secondary-proton survival integration must use at least one step."
        raise ValueError(msg)

    if incident_spectrum.particle is not Particle.PROTON:
        msg = "Secondary-proton Al kernel requires an incident proton spectrum."
        raise ValueError(msg)

    if incident_spectrum.source not in (
        RadiationSource.SEP,
        RadiationSource.GCR,
        RadiationSource.ERB,
    ):
        msg = "Secondary-proton Al kernel supports SEP, GCR and ERB spectra only."
        raise ValueError(msg)

    if thickness_g_cm2 == 0.0:
        return derive_spectrum(
            incident_spectrum,
            y=tuple(0.0 for _ in incident_spectrum.x),
            particle=Particle.PROTON,
            model=model or f"{incident_spectrum.model}+{PROTON_AL_SECONDARY_MODEL}",
            location=SpectrumLocation.BEHIND_SHIELD,
            shielding=ShieldingDescriptor(
                material="aluminium",
                areal_density_g_cm2=thickness_g_cm2,
            ),
            transformation=TransformationStep(
                name=PROTON_AL_SECONDARY_MODEL,
                version="3",
                parameters=(("thickness_g_cm2", "0"),),
            ),
        )

    if not kernel.products:
        msg = (
            "Positive-thickness secondary-proton transport requires a populated "
            "MF=3/MF=6 production kernel."
        )
        raise ValueError(msg)

    if (
        incident_quadrature is not None
        and incident_spectrum.ordinate_basis is not SpectrumOrdinateBasis.POINT_SAMPLE
    ):
        raise ValueError("Incident-energy quadrature currently requires POINT_SAMPLE input.")

    if incident_quadrature is not None:
        return _calculate_secondary_with_incident_quadrature(
            incident_spectrum=incident_spectrum,
            stopping_table=stopping_table,
            kernel=kernel,
            thickness_g_cm2=thickness_g_cm2,
            depth_steps=depth_steps,
            survival_integration_steps=survival_integration_steps,
            options=incident_quadrature,
            depth_options=depth_quadrature,
            model=model,
        )

    if incident_spectrum.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE:
        assert incident_spectrum.bin_edges is not None
        input_edges = incident_spectrum.bin_edges
        incident_bin_operator = "declared_bin_average_times_bin_width"
    else:
        input_edges = log_bin_edges_from_centers(incident_spectrum.x)
        incident_bin_operator = "point_sample_center_times_derived_log_bin_width"
    output_edges = input_edges
    depth_partition = _prepare_exit_depth_partition(
        output_edges_mev=output_edges,
        thickness_g_cm2=thickness_g_cm2,
        stopping_table=stopping_table,
        kernel=kernel,
        options=depth_quadrature,
    )
    secondary_values = [0.0 for _ in incident_spectrum.x]
    support = kernel.incident_energy_support
    assert support is not None
    nonelastic_cross_section = kernel.nonelastic_cross_section
    optical_depth_table = (
        None
        if nonelastic_cross_section is None
        else _build_nonelastic_optical_depth_table(
            stopping_table=stopping_table,
            cross_section_table=nonelastic_cross_section,
        )
    )
    attenuation_relevant_energies: list[float] = []
    attenuation_missing_for_nonzero_contribution = False
    incident_table_energy_domain_bins = 0
    incident_depth_range_domain_samples = 0
    production_capable_depth_samples = 0
    range_domain_mappings = 0
    relevant_energies: list[float] = []
    for input_index, (energy, value) in enumerate(
        zip(incident_spectrum.x, incident_spectrum.y, strict=True),
    ):
        if value <= 0.0:
            continue
        complete_input_bin_energies = (
            input_edges[input_index],
            energy,
            input_edges[input_index + 1],
        )
        relevant_energies.extend(complete_input_bin_energies)
        attenuation_relevant_energies.extend(complete_input_bin_energies)
    output_table_energy_domain_bins = sum(
        1
        for output_index in range(len(incident_spectrum.x))
        if output_edges[output_index] < stopping_table.energy_mev[0]
        or output_edges[output_index + 1] > stopping_table.energy_mev[-1]
    )
    incident_energy_knots = _secondary_incident_energy_knots(
        kernel=kernel,
        stopping_table=stopping_table,
    )

    for input_index, entrance_energy in enumerate(incident_spectrum.x):
        input_width = input_edges[input_index + 1] - input_edges[input_index]
        incident_count = incident_spectrum.y[input_index] * input_width

        if incident_count <= 0.0:
            continue

        input_lo = input_edges[input_index]
        input_hi = input_edges[input_index + 1]
        if input_lo < stopping_table.energy_mev[0] or input_hi > stopping_table.energy_mev[-1]:
            incident_table_energy_domain_bins += 1
            continue

        entrance_range = stopping_table.range_at_energy(entrance_energy)
        depth_intervals = _depth_quadrature_intervals(
            entrance_energy_mev=entrance_energy,
            thickness_g_cm2=thickness_g_cm2,
            depth_steps=depth_steps,
            stopping_table=stopping_table,
            incident_energy_knots_mev=incident_energy_knots,
            exit_depth_knots_g_cm2=depth_partition.knots_g_cm2,
        )
        if entrance_range - thickness_g_cm2 < stopping_table.range_g_cm2[0]:
            incident_depth_range_domain_samples += 1

        for depth_lower, depth_upper in depth_intervals:
            for depth, depth_weight in _gauss_legendre_4_samples(
                depth_lower,
                depth_upper,
            ):
                mapped_input_edges = tuple(
                    _mapped_input_edge_at_depth(
                        entrance_edge_mev=edge,
                        depth_g_cm2=depth,
                        stopping_table=stopping_table,
                    )
                    for edge in (input_lo, input_hi)
                )
                mapped_supported_edges = tuple(
                    edge for edge in mapped_input_edges if edge is not None
                )
                relevant_energies.extend(mapped_supported_edges)
                attenuation_relevant_energies.extend(mapped_supported_edges)
                incident_depth_range_domain_samples += sum(
                    edge is None for edge in mapped_input_edges
                )

                residual_range = entrance_range - depth
                incident_energy_at_depth = stopping_table.energy_at_range(residual_range)

                relevant_energies.append(incident_energy_at_depth)
                attenuation_relevant_energies.extend((entrance_energy, incident_energy_at_depth))
                if not support.contains(incident_energy_at_depth):
                    # Unsupported reaction production is omitted only from this
                    # diagnostic value; the returned spectrum is release-blocked.
                    continue

                production_capable_depth_samples += 1

                exit_mapping_result = _secondary_exit_mappings(
                    output_centers_mev=incident_spectrum.x,
                    output_edges_mev=output_edges,
                    residual_thickness_g_cm2=thickness_g_cm2 - depth,
                    stopping_table=stopping_table,
                    cross_section_table=nonelastic_cross_section,
                )
                range_domain_mappings += exit_mapping_result.range_domain_bins

                primary_survival, primary_attenuation_supported = _nonelastic_survival_or_unity(
                    output_energy_mev=incident_energy_at_depth,
                    input_energy_mev=entrance_energy,
                    stopping_table=stopping_table,
                    cross_section_table=nonelastic_cross_section,
                    integration_steps=survival_integration_steps,
                    optical_depth_table=optical_depth_table,
                )
                active_products = _active_secondary_products_at_incident_energy(
                    kernel=kernel,
                    incident_energy_mev=incident_energy_at_depth,
                )
                if not active_products:
                    continue
                for mapping in exit_mapping_result.mappings:
                    if all(
                        mapping.birth_upper_mev <= active_product.emitted_minimum_mev
                        or mapping.birth_lower_mev >= active_product.emitted_maximum_mev
                        for active_product in active_products
                    ):
                        continue
                    source_density_integral = _integrated_secondary_source_to_exit_per_g_cm2(
                        active_products=active_products,
                        nonelastic_cross_section=nonelastic_cross_section,
                        optical_depth_table=optical_depth_table,
                        mapping=mapping,
                        stopping_table=stopping_table,
                    )
                    if source_density_integral <= 0.0:
                        continue

                    attenuation_relevant_energies.extend(mapping.attenuation_energies_mev)
                    if not primary_attenuation_supported or not mapping.attenuation_supported:
                        attenuation_missing_for_nonzero_contribution = True

                    secondary_counts = (
                        incident_count * primary_survival * source_density_integral * depth_weight
                    )
                    secondary_values[mapping.output_index] += (
                        secondary_counts / mapping.output_width_mev
                    )

    output = derive_spectrum(
        incident_spectrum,
        y=tuple(secondary_values),
        particle=Particle.PROTON,
        model=model or f"{incident_spectrum.model}+{PROTON_AL_SECONDARY_MODEL}",
        location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor(
            material="aluminium",
            areal_density_g_cm2=thickness_g_cm2,
        ),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=output_edges,
        transformation=TransformationStep(
            name=PROTON_AL_SECONDARY_MODEL,
            version="4_depth_crossing_partitions",
            parameters=(
                ("depth_steps", str(depth_steps)),
                ("depth_quadrature", "gauss_legendre_4_split_at_transport_knots"),
                *_exit_depth_partition_parameters(depth_partition, depth_quadrature),
                ("incident_bin_operator", incident_bin_operator),
                ("primary_survival_to_depth", "TENDL_MF3_MT3"),
                ("mf6_reference_frame", "centre_of_mass_lct_3"),
                ("mf6_cm_to_lab_transform", "not_included"),
                ("secondary_generation", "single_collision_generation"),
                ("higher_generation_cascade", "not_included"),
                ("secondary_angular_transport", "not_included"),
                ("secondary_pdf_integration", "exact_endf_histogram_segments"),
                ("secondary_path_length", "radial_residual_thickness_diagnostic"),
                ("secondary_survival_to_exit", "TENDL_MF3_MT3"),
                (
                    "secondary_survival_quadrature",
                    "gauss_legendre_4_split_at_pdf_and_transport_knots",
                ),
                ("survival_integration_steps", str(survival_integration_steps)),
                ("thickness_g_cm2", f"{thickness_g_cm2:.17g}"),
                (
                    "tendl_incident_energy_support_mev",
                    f"{support.minimum_mev:g}..{support.maximum_mev:g}",
                ),
            ),
        ),
        additional_dataset_refs=(
            "stopping/proton_al_range.csv",
            "tendl/p_al27_mf3_xs.csv",
            "tendl/p_al27_mf6_proton_yield.csv",
            "tendl/p_al27_mf6_proton_pdf.csv",
        ),
    )
    coverage = assess_tendl_incident_energy_coverage(
        relevant_energies_mev=relevant_energies,
        support=support,
    )
    reason = coverage.incomplete_reason(
        operator=PROTON_AL_SECONDARY_SUPPORT_OPERATOR,
    )
    if reason is not None:
        output = mark_spectrum_incomplete(output, reason)
    if nonelastic_cross_section is None and attenuation_missing_for_nonzero_contribution:
        output = mark_spectrum_incomplete(
            output,
            PROTON_AL_SECONDARY_ATTENUATION_MISSING_REASON,
        )
    elif nonelastic_cross_section is not None:
        attenuation_support = nonelastic_cross_section.incident_energy_support
        assert attenuation_support is not None
        attenuation_coverage = assess_tendl_incident_energy_coverage(
            relevant_energies_mev=attenuation_relevant_energies,
            support=attenuation_support,
        )
        attenuation_reason = attenuation_coverage.incomplete_reason(
            operator=PROTON_AL_SECONDARY_ATTENUATION_OPERATOR,
        )
        if attenuation_reason is not None:
            output = mark_spectrum_incomplete(output, attenuation_reason)
    transport_domain_reasons: list[str] = []
    if incident_table_energy_domain_bins:
        transport_domain_reasons.append(
            f"{PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX}: "
            f"{incident_table_energy_domain_bins} nonzero incident bin(s) exceed "
            "the aluminium range-table energy support "
            f"{stopping_table.energy_mev[0]:.17g}.."
            f"{stopping_table.energy_mev[-1]:.17g} MeV"
        )
    if incident_depth_range_domain_samples:
        transport_domain_reasons.append(
            f"{PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX}: "
            f"{incident_depth_range_domain_samples} nonzero incident depth sample(s) "
            "map below the aluminium range-table minimum range "
            f"{stopping_table.range_g_cm2[0]:.17g} g/cm^2"
        )
    if output_table_energy_domain_bins and production_capable_depth_samples:
        transport_domain_reasons.append(
            f"{PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX}: "
            f"{output_table_energy_domain_bins} output bin(s) exceed the aluminium "
            "range-table energy support "
            f"{stopping_table.energy_mev[0]:.17g}.."
            f"{stopping_table.energy_mev[-1]:.17g} MeV"
        )
    if range_domain_mappings:
        transport_domain_reasons.append(
            f"{PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX}: "
            f"{range_domain_mappings} output-bin/depth mapping(s) require an "
            "aluminium range above "
            f"{stopping_table.max_range_g_cm2:.17g} g/cm^2 at "
            f"{thickness_g_cm2:.17g} g/cm^2 shielding"
        )
    if transport_domain_reasons:
        output = mark_spectrum_incomplete(output, *transport_domain_reasons)
    output = mark_spectrum_incomplete(
        output,
        PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
        PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
        PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
        PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    )
    if depth_partition.budget_exceeded:
        output = mark_spectrum_incomplete(
            output,
            PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON,
            PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON,
        )
    if incident_spectrum.ordinate_basis is SpectrumOrdinateBasis.POINT_SAMPLE:
        output = mark_spectrum_incomplete(
            output,
            PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON,
        )
    return output


def combine_primary_and_secondary_proton_spectra(
    *,
    primary: Spectrum1D,
    secondary: Spectrum1D,
    model: str | None = None,
) -> Spectrum1D:
    """Return total proton spectrum behind shield."""

    if primary.x != secondary.x:
        msg = "Primary and secondary proton spectra must share the same energy grid."
        raise ValueError(msg)

    if primary.x_unit is not secondary.x_unit or primary.y_unit is not secondary.y_unit:
        msg = "Primary and secondary proton spectra must share units."
        raise ValueError(msg)

    if primary.quantity is not secondary.quantity:
        msg = "Primary and secondary proton spectra must share the same quantity."
        raise ValueError(msg)

    return add_spectra(
        primary,
        secondary,
        model=model or f"{primary.model}+{secondary.model}+{PROTON_AL_TOTAL_MODEL}",
    )
