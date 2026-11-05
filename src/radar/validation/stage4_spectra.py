"""Run the diagnostic Stage-4 OST spectrum reference matrix.

The generated report contains calculated numbers, but it is deliberately not
a numerical gold baseline.  Its purpose is to preserve inputs, contracts,
completeness and release blockers until an independently traceable COSRAD
baseline is available.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import json
from math import isclose
from pathlib import Path
from typing import cast

from radar.core.products import SpectrumProduct, select_products_for_assessment_mode
from radar.core.profiles import MethodologyProfile
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
)
from radar.core.types import (
    AssessmentMode,
    LetSpectrumRole,
    DoseQuantity,
    OrbitType,
    Particle,
    RadiationProductKind,
    RadiationQuantityMode,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.data.audit import (
    DataAuditReport,
    DataAuditScope,
    audit_normative_resources,
)
from radar.erb.shielding import (
    calculate_erb_electron_shielding_spectrum,
    calculate_erb_proton_shielding_spectrum,
)
from radar.gcr.pipeline_shielding import (
    calculate_gcr_shielding_let_products_for_thickness,
)
from radar.gcr.parameters import GCR_ELEMENT_SYMBOLS
from radar.pipelines.erb import ErbPipelineResult, calculate_erb_pipeline
from radar.pipelines.execution import build_pipeline_source_models
from radar.pipelines.gcr import GcrPipelineResult, calculate_gcr_pipeline
from radar.pipelines.sep import SepPipelineResult, calculate_sep_pipeline
from radar.sep.pipeline_shielding import (
    calculate_sep_shielding_let_products_for_thickness,
)
from radar.sep.model import ost_sep_required_hze_ions
from radar.shielding.resources import ShieldingNormativeTables, load_normative_shielding_tables
from radar.version import __version__

STAGE4_REFERENCE_REPORT_SCHEMA = "radar.stage4.spectrum-reference.v4"
STAGE4_REFERENCE_SHIELD_THICKNESSES_G_CM2 = (0.01, 0.1, 1.0, 3.0, 10.0)
STAGE4_REFERENCE_MODES = (AssessmentMode.PEAK, AssessmentMode.CUMULATIVE)
_SOURCE_ORDER = (RadiationSource.SEP, RadiationSource.GCR, RadiationSource.ERB)
_ZERO_IDENTITY_REL_TOLERANCE = 1.0e-12
_ZERO_IDENTITY_ABS_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class Stage4ReferenceScenario:
    """One orbit in the fixed Stage-4 validation matrix."""

    name: str
    orbit: OrbitConfig

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "Stage-4 reference scenario name must not be empty."
            raise ValueError(msg)


class Stage4SpeciesScope(StrEnum):
    """Typed species shape used by a Stage-4 product signature."""

    NON_AGGREGATE = "non_aggregate"
    AGGREGATE = "aggregate"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class Stage4ProductSignature:
    """Minimum typed identity required for one spectral stage product."""

    product_kind: RadiationProductKind
    quantity: SpectrumQuantity
    particle: Particle
    axis_kind: SpectrumAxisKind | None
    temporal_basis: TemporalBasis | None
    species_scope: Stage4SpeciesScope
    let_role: LetSpectrumRole | None = None

    @classmethod
    def from_product(cls, product: SpectrumProduct) -> Stage4ProductSignature:
        """Build an observed signature without interpreting display strings."""

        species = product.spectrum.species
        if species is None:
            scope = Stage4SpeciesScope.UNSPECIFIED
        elif species.is_aggregate:
            scope = Stage4SpeciesScope.AGGREGATE
        else:
            scope = Stage4SpeciesScope.NON_AGGREGATE
        return cls(
            product_kind=product.kind,
            quantity=product.spectrum.quantity,
            particle=product.spectrum.particle,
            axis_kind=product.spectrum.axis_kind,
            temporal_basis=product.spectrum.temporal_basis,
            species_scope=scope,
            let_role=product.spectrum.let_role,
        )

    def matches(self, product: SpectrumProduct) -> bool:
        """Return whether a product satisfies this fully typed signature."""

        return self == self.from_product(product)

    @property
    def semantic_role(self) -> str:
        """Return the explicit report role implied by typed metadata."""

        if self.let_role is not None:
            return self.let_role.value
        return "energy_species"

    def to_summary(self) -> dict[str, str]:
        """Return the stable JSON representation of this signature."""

        return {
            "product_kind": self.product_kind.value,
            "quantity": self.quantity.value,
            "particle": self.particle.value,
            "axis_kind": (
                "unspecified" if self.axis_kind is None else self.axis_kind.value
            ),
            "temporal_basis": (
                "unspecified"
                if self.temporal_basis is None
                else self.temporal_basis.value
            ),
            "species_scope": self.species_scope.value,
            "semantic_role": self.semantic_role,
        }


STAGE4_REFERENCE_SCENARIOS = (
    Stage4ReferenceScenario(
        name="LEO",
        orbit=OrbitConfig.circular(altitude_km=400.0, inclination_deg=51.6),
    ),
    Stage4ReferenceScenario(
        name="GEO",
        orbit=OrbitConfig.circular(altitude_km=35_786.0, inclination_deg=0.0),
    ),
    Stage4ReferenceScenario(
        name="HEO",
        orbit=OrbitConfig(
            orbit_type=OrbitType.ELLIPTIC,
            perigee_altitude_km=500.0,
            apogee_altitude_km=39_700.0,
            inclination_deg=63.4,
        ),
    ),
)


@dataclass(frozen=True)
class Stage4ShieldExecution:
    """Calculated products and explicit gaps at one positive shield depth."""

    thickness_g_cm2: float
    products: tuple[SpectrumProduct, ...]
    incomplete_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage4ZeroShieldExecution:
    """Low-level zero-shield transport output, outside ``ShieldingConfig``."""

    products: tuple[SpectrumProduct, ...]
    incomplete_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage4SourceExecution:
    """Normalized raw execution result for one radiation source."""

    source: RadiationSource
    model_name: str
    model_version: str
    model_document: str
    on_orbit_products: tuple[SpectrumProduct, ...]
    shielded_by_thickness: tuple[Stage4ShieldExecution, ...]
    zero_shield: Stage4ZeroShieldExecution
    release_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage4CaseExecution:
    """Raw execution result for one orbit/mode combination."""

    sources: tuple[Stage4SourceExecution, ...]

    def __post_init__(self) -> None:
        observed = tuple(item.source for item in self.sources)
        if observed != _SOURCE_ORDER:
            msg = "Stage-4 case execution must contain SEP, GCR and ERB in canonical order."
            raise ValueError(msg)


Stage4CaseExecutor = Callable[[CalculationConfig], Stage4CaseExecution]
Stage4AuditProvider = Callable[[], DataAuditReport]


def build_stage4_reference_config(
    scenario: Stage4ReferenceScenario,
    mode: AssessmentMode,
) -> CalculationConfig:
    """Build one fixed production-profile input from the approved matrix."""

    dose_quantity = (
        DoseQuantity.ACCUMULATED_DOSE
        if mode is AssessmentMode.CUMULATIVE
        else DoseQuantity.DOSE_RATE
    )
    dose_unit = Unit.RAD if mode is AssessmentMode.CUMULATIVE else Unit.RAD_PER_SECOND
    return CalculationConfig(
        mission=MissionConfig(
            launch_year=2026,
            lifetime_years=5,
            sep_exceedance_probability=0.1,
        ),
        orbit=scenario.orbit,
        shielding=ShieldingConfig(
            thicknesses_g_cm2=STAGE4_REFERENCE_SHIELD_THICKNESSES_G_CM2,
        ),
        methodology=MethodologyConfig(profile=MethodologyProfile.OST_134_1044_2007),
        assessment_mode=mode,
        dose_quantity=dose_quantity,
        dose_unit=dose_unit,
    )


def _selected_products(
    products: tuple[SpectrumProduct, ...],
    mode: AssessmentMode,
) -> tuple[SpectrumProduct, ...]:
    return select_products_for_assessment_mode(products, mode)


def _release_blocker(release_boundary: Callable[[], object]) -> tuple[str, ...]:
    try:
        release_boundary()
    except ValueError as exc:
        return (str(exc),)
    return ()


def _positive_sep_shields(
    result: SepPipelineResult,
    mode: AssessmentMode,
) -> tuple[Stage4ShieldExecution, ...]:
    return tuple(
        Stage4ShieldExecution(
            thickness_g_cm2=item.thickness_g_cm2,
            products=_selected_products(item.products, mode),
            incomplete_reasons=item.incomplete_reasons,
        )
        for item in result.shielding_let_by_thickness
    )


def _positive_gcr_shields(
    result: GcrPipelineResult,
    mode: AssessmentMode,
) -> tuple[Stage4ShieldExecution, ...]:
    return tuple(
        Stage4ShieldExecution(
            thickness_g_cm2=item.thickness_g_cm2,
            products=_selected_products(item.products, mode),
            incomplete_reasons=item.incomplete_reasons,
        )
        for item in result.shielding_let_by_thickness
    )


def _positive_erb_shields(
    result: ErbPipelineResult,
    config: CalculationConfig,
) -> tuple[Stage4ShieldExecution, ...]:
    return tuple(
        Stage4ShieldExecution(
            thickness_g_cm2=thickness,
            products=_selected_products(
                tuple(
                    item.shielded_product
                    for item in result.shielding_by_thickness
                    if item.thickness_g_cm2 == thickness
                ),
                config.assessment_mode,
            ),
        )
        for thickness in config.shielding.thicknesses_g_cm2
    )


def _zero_sep_shield(
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
) -> Stage4ZeroShieldExecution:
    result = calculate_sep_shielding_let_products_for_thickness(
        products=products,
        tables=tables,
        thickness_g_cm2=0.0,
    )
    return Stage4ZeroShieldExecution(
        products=result.shielded_products,
        incomplete_reasons=result.incomplete_reasons,
    )


def _zero_gcr_shield(
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
) -> Stage4ZeroShieldExecution:
    result = calculate_gcr_shielding_let_products_for_thickness(
        products=products,
        tables=tables,
        thickness_g_cm2=0.0,
    )
    return Stage4ZeroShieldExecution(
        products=result.shielded_products,
        incomplete_reasons=result.incomplete_reasons,
    )


def _zero_erb_shield(
    products: tuple[SpectrumProduct, ...],
    tables: ShieldingNormativeTables,
) -> Stage4ZeroShieldExecution:
    shielded: list[SpectrumProduct] = []
    for product in products:
        if product.spectrum.particle is Particle.PROTON:
            spectrum = calculate_erb_proton_shielding_spectrum(
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=0.0,
            ).total
        elif product.spectrum.particle is Particle.ELECTRON:
            spectrum = calculate_erb_electron_shielding_spectrum(
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=0.0,
            ).primary
        else:
            continue
        shielded.append(SpectrumProduct(kind=product.kind, spectrum=spectrum))
    return Stage4ZeroShieldExecution(products=tuple(shielded))


def execute_stage4_reference_case(config: CalculationConfig) -> Stage4CaseExecution:
    """Run the three real OST source/orbit/shielding pipelines for one case."""

    if config.methodology.profile is not MethodologyProfile.OST_134_1044_2007:
        msg = "Stage-4 reference execution requires the OST 134-1044-2007 profile."
        raise ValueError(msg)

    models = build_pipeline_source_models(config)
    sep_result = calculate_sep_pipeline(config=config, sep_model=models.sep_model)
    gcr_result = calculate_gcr_pipeline(config=config, gcr_model=models.gcr_model)
    erb_result = calculate_erb_pipeline(config=config, erb_model=models.erb_model)
    tables = load_normative_shielding_tables()

    sep_on_orbit = _selected_products(sep_result.on_orbit_products, config.assessment_mode)
    gcr_on_orbit = _selected_products(gcr_result.on_orbit_products, config.assessment_mode)
    erb_on_orbit = _selected_products(erb_result.on_orbit_products, config.assessment_mode)

    return Stage4CaseExecution(
        sources=(
            Stage4SourceExecution(
                source=RadiationSource.SEP,
                model_name=models.sep_model.metadata.name,
                model_version=models.sep_model.metadata.version,
                model_document=models.sep_model.metadata.document,
                on_orbit_products=sep_on_orbit,
                shielded_by_thickness=_positive_sep_shields(
                    sep_result,
                    config.assessment_mode,
                ),
                zero_shield=_zero_sep_shield(sep_on_orbit, tables),
                release_blockers=_release_blocker(sep_result.production_release_products),
            ),
            Stage4SourceExecution(
                source=RadiationSource.GCR,
                model_name=models.gcr_model.metadata.name,
                model_version=models.gcr_model.metadata.version,
                model_document=models.gcr_model.metadata.document,
                on_orbit_products=gcr_on_orbit,
                shielded_by_thickness=_positive_gcr_shields(
                    gcr_result,
                    config.assessment_mode,
                ),
                zero_shield=_zero_gcr_shield(gcr_on_orbit, tables),
                release_blockers=_release_blocker(gcr_result.production_release_products),
            ),
            Stage4SourceExecution(
                source=RadiationSource.ERB,
                model_name=models.erb_model.metadata.name,
                model_version=models.erb_model.metadata.version,
                model_document=models.erb_model.metadata.document,
                on_orbit_products=erb_on_orbit,
                shielded_by_thickness=_positive_erb_shields(erb_result, config),
                zero_shield=_zero_erb_shield(erb_on_orbit, tables),
                release_blockers=_release_blocker(erb_result.production_release_products),
            ),
        )
    )


def _required_product_signatures(
    source: RadiationSource,
    mode: AssessmentMode,
    *,
    behind_shield: bool,
) -> tuple[Stage4ProductSignature, ...]:
    energy_kind = {
        (RadiationSource.SEP, AssessmentMode.PEAK): RadiationProductKind.PEAK_FLUX,
        (RadiationSource.SEP, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_FLUENCE
        ),
        (RadiationSource.GCR, AssessmentMode.PEAK): RadiationProductKind.MAXIMUM_FLUX,
        (RadiationSource.GCR, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_FLUENCE
        ),
        (RadiationSource.ERB, AssessmentMode.PEAK): RadiationProductKind.MAXIMUM_FLUX,
        (RadiationSource.ERB, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_FLUENCE
        ),
    }[(source, mode)]
    energy_particles = (
        (Particle.PROTON, Particle.ELECTRON)
        if source is RadiationSource.ERB
        else (Particle.PROTON, Particle.HZE)
    )
    energy_quantity = (
        SpectrumQuantity.DIFFERENTIAL_FLUENCE
        if mode is AssessmentMode.CUMULATIVE
        else {
            RadiationSource.SEP: SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            RadiationSource.GCR: SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            RadiationSource.ERB: SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        }[source]
    )
    temporal_basis = (
        TemporalBasis.MISSION_TOTAL
        if mode is AssessmentMode.CUMULATIVE
        else {
            RadiationSource.SEP: TemporalBasis.SEP_PEAK,
            RadiationSource.GCR: TemporalBasis.SOURCE_MAXIMUM,
            RadiationSource.ERB: TemporalBasis.SOURCE_MAXIMUM,
        }[source]
    )
    required = [
        Stage4ProductSignature(
            product_kind=energy_kind,
            quantity=energy_quantity,
            particle=particle,
            axis_kind=(
                SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
                if particle is Particle.HZE
                else SpectrumAxisKind.KINETIC_ENERGY
            ),
            temporal_basis=temporal_basis,
            species_scope=Stage4SpeciesScope.NON_AGGREGATE,
        )
        for particle in energy_particles
    ]
    if not behind_shield or source is RadiationSource.ERB:
        return tuple(required)
    let_kind = {
        (RadiationSource.SEP, AssessmentMode.PEAK): RadiationProductKind.PEAK_LET_FLUX,
        (RadiationSource.SEP, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_LET_FLUENCE
        ),
        (RadiationSource.GCR, AssessmentMode.PEAK): (
            RadiationProductKind.MAXIMUM_LET_FLUX
        ),
        (RadiationSource.GCR, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_LET_FLUENCE
        ),
    }[(source, mode)]
    required.append(
        Stage4ProductSignature(
            product_kind=let_kind,
            quantity=(
                SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
                if mode is AssessmentMode.CUMULATIVE
                else SpectrumQuantity.LET_DIFFERENTIAL_FLUX
            ),
            particle=Particle.HZE,
            axis_kind=SpectrumAxisKind.LINEAR_ENERGY_TRANSFER,
            temporal_basis=temporal_basis,
            species_scope=Stage4SpeciesScope.AGGREGATE,
            let_role=LetSpectrumRole.HZE_DIRECT_ION,
        )
    )
    return tuple(required)


def _product_sort_key(product: SpectrumProduct) -> tuple[str, str, int, int, str, str]:
    species = product.spectrum.species
    return (
        product.kind.value,
        product.spectrum.particle.value,
        -1 if species is None or species.atomic_number is None else species.atomic_number,
        -1 if species is None or species.mass_number is None else species.mass_number,
        "" if species is None else species.symbol,
        product.spectrum.quantity.value,
    )


def _integral_unit(product: SpectrumProduct) -> Unit:
    if product.radiation_quantity_mode is RadiationQuantityMode.FLUENCE:
        return Unit.INTEGRAL_FLUENCE
    return Unit.INTEGRAL_FLUX


_DIFFERENTIAL_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)


def _integral_summary(product: SpectrumProduct) -> dict[str, object] | None:
    """Return one scalar only when the ordinate is differential.

    An ``INTEGRAL_*`` spectrum already stores a threshold integral at every
    abscissa.  Integrating that curve again has no physical meaning and would
    also produce the wrong dimensions.
    """

    spectrum = product.spectrum
    if spectrum.quantity not in _DIFFERENTIAL_QUANTITIES:
        return None
    method = {
        SpectrumOrdinateBasis.POINT_SAMPLE: "trapezoidal_over_reported_grid",
        SpectrumOrdinateBasis.BIN_AVERAGE: "bin_sum_over_explicit_edges",
    }[spectrum.ordinate_basis]
    return {
        "value": spectrum.integral_conserved(),
        "unit": _integral_unit(product).value,
        "method": method,
    }


def _spectrum_summary(product: SpectrumProduct) -> dict[str, object]:
    spectrum = product.spectrum
    species = spectrum.species
    shielding = spectrum.shielding
    model_reference = spectrum.model_reference
    return {
        "product_kind": product.kind.value,
        "quantity": spectrum.quantity.value,
        "particle": spectrum.particle.value,
        "let_role": None if spectrum.let_role is None else spectrum.let_role.value,
        "ordinate_basis": spectrum.ordinate_basis.value,
        "bin_edges": None if spectrum.bin_edges is None else list(spectrum.bin_edges),
        "source": spectrum.source.value,
        "species": (
            None
            if species is None
            else {
                "symbol": species.symbol,
                "atomic_number": species.atomic_number,
                "mass_number": species.mass_number,
                "is_aggregate": species.is_aggregate,
            }
        ),
        "axis_kind": None if spectrum.axis_kind is None else spectrum.axis_kind.value,
        "x_unit": spectrum.x_unit.value,
        "y_unit": spectrum.y_unit.value,
        "temporal_basis": (
            None if spectrum.temporal_basis is None else spectrum.temporal_basis.value
        ),
        "angular_basis": (
            None if spectrum.angular_basis is None else spectrum.angular_basis.value
        ),
        "assessment_mode": (
            None if spectrum.assessment_mode is None else spectrum.assessment_mode.value
        ),
        "mission_duration_s": spectrum.mission_duration_s,
        "location": None if spectrum.location is None else spectrum.location.value,
        "shielding": (
            None
            if shielding is None
            else {
                "material": shielding.material,
                "areal_density_g_cm2": shielding.areal_density_g_cm2,
                "areal_density_unit": shielding.areal_density_unit.value,
            }
        ),
        "model_reference": (
            None
            if model_reference is None
            else {
                "name": model_reference.name,
                "version": model_reference.version,
                "document_refs": list(model_reference.document_refs),
                "dataset_refs": list(model_reference.dataset_refs),
            }
        ),
        "transformation_chain": [
            {
                "name": step.name,
                "version": step.version,
                "parameters": [
                    {"name": name, "value": value} for name, value in step.parameters
                ],
            }
            for step in spectrum.transformation_chain
        ],
        "completeness": (
            "unspecified" if spectrum.completeness is None else spectrum.completeness.value
        ),
        "incomplete_reasons": list(spectrum.incomplete_reasons),
        "point_count": len(spectrum.x),
        "x_min": spectrum.x[0],
        "x_max": spectrum.x[-1],
        "integral": _integral_summary(product),
    }


def _required_ion_roster(
    source: RadiationSource,
) -> tuple[tuple[int, str], ...]:
    if source is RadiationSource.SEP:
        return ost_sep_required_hze_ions()
    if source is RadiationSource.GCR:
        return tuple(enumerate(GCR_ELEMENT_SYMBOLS, start=1))
    return ()


def _ion_roster_summary(
    *,
    source: RadiationSource,
    mode: AssessmentMode,
    products: tuple[SpectrumProduct, ...],
) -> dict[str, object]:
    """Compare energy-species products with the exact OST ion roster."""

    required = _required_ion_roster(source)
    if not required:
        return {
            "required": [],
            "observed": [],
            "missing": [],
            "unexpected": [],
            "duplicates": [],
            "complete": True,
            "incomplete_reasons": [],
        }

    energy_kind = {
        (RadiationSource.SEP, AssessmentMode.PEAK): RadiationProductKind.PEAK_FLUX,
        (RadiationSource.SEP, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_FLUENCE
        ),
        (RadiationSource.GCR, AssessmentMode.PEAK): RadiationProductKind.MAXIMUM_FLUX,
        (RadiationSource.GCR, AssessmentMode.CUMULATIVE): (
            RadiationProductKind.MISSION_FLUENCE
        ),
    }[(source, mode)]
    observed: list[tuple[int, str]] = []
    for product in products:
        spectrum = product.spectrum
        species = spectrum.species
        if (
            product.kind is not energy_kind
            or spectrum.axis_kind
            not in (
                SpectrumAxisKind.KINETIC_ENERGY,
                SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
            )
            or species is None
            or species.is_aggregate
            or species.atomic_number is None
        ):
            continue
        if source is RadiationSource.SEP and spectrum.particle is not Particle.HZE:
            continue
        if source is RadiationSource.GCR and spectrum.particle not in (
            Particle.PROTON,
            Particle.HZE,
        ):
            continue
        observed.append((species.atomic_number, species.symbol))

    observed_tuple = tuple(observed)
    observed_set = set(observed_tuple)
    required_set = set(required)
    missing = tuple(ion for ion in required if ion not in observed_set)
    unexpected = tuple(ion for ion in observed_tuple if ion not in required_set)
    duplicates = tuple(
        ion
        for ion in dict.fromkeys(observed_tuple)
        if observed_tuple.count(ion) > 1
    )

    def entries(ions: tuple[tuple[int, str], ...]) -> list[dict[str, object]]:
        return [
            {"atomic_number": atomic_number, "symbol": symbol}
            for atomic_number, symbol in ions
        ]

    def render(ions: tuple[tuple[int, str], ...]) -> str:
        return ", ".join(
            f"{symbol}(Z={atomic_number})" for atomic_number, symbol in ions
        )

    prefix = f"Stage-4 OST {source.value.upper()} normative ion roster incomplete"
    reasons: list[str] = []
    if missing:
        reasons.append(f"{prefix}: missing required ions {render(missing)}")
    if unexpected:
        reasons.append(f"{prefix}: unexpected ions {render(unexpected)}")
    if duplicates:
        reasons.append(f"{prefix}: duplicate ions {render(duplicates)}")
    return {
        "required": entries(required),
        "observed": entries(tuple(dict.fromkeys(observed_tuple))),
        "missing": entries(missing),
        "unexpected": entries(unexpected),
        "duplicates": entries(duplicates),
        "complete": not reasons,
        "incomplete_reasons": reasons,
    }


def _stage_summary(
    *,
    source: RadiationSource,
    mode: AssessmentMode,
    products: tuple[SpectrumProduct, ...],
    behind_shield: bool,
    thickness_g_cm2: float | None = None,
    incomplete_reasons: tuple[str, ...] = (),
) -> dict[str, object]:
    required_signatures = _required_product_signatures(
        source,
        mode,
        behind_shield=behind_shield,
    )
    observed_signatures = tuple(
        dict.fromkeys(Stage4ProductSignature.from_product(product) for product in products)
    )
    missing_signatures = tuple(
        signature
        for signature in required_signatures
        if not any(signature.matches(product) for product in products)
    )
    required_kinds = tuple(
        dict.fromkeys(signature.product_kind for signature in required_signatures)
    )
    observed_kinds = tuple(dict.fromkeys(product.kind for product in products))
    missing_kinds = tuple(kind for kind in required_kinds if kind not in observed_kinds)
    ion_roster = _ion_roster_summary(
        source=source,
        mode=mode,
        products=products,
    )
    reasons = list(incomplete_reasons)
    reasons.extend(cast(list[str], ion_roster["incomplete_reasons"]))
    if any(count > 1 for count in Counter(_identity_key(p) for p in products).values()):
        reasons.append("spectral stage contains duplicate product identities")
    for product in products:
        spectrum = product.spectrum
        reasons.extend(product.production_contract_violations())
        if spectrum.source is not source:
            reasons.append("spectrum source does not match its reported source stage")
        if spectrum.assessment_mode is not mode:
            reasons.append("spectrum assessment mode does not match the configured mode")
        expected_location = (
            SpectrumLocation.BEHIND_SHIELD if behind_shield else SpectrumLocation.ON_ORBIT
        )
        if spectrum.location is not expected_location:
            reasons.append("spectrum location does not match its reported stage")
        if behind_shield:
            shield = spectrum.shielding
            if (
                shield is None
                or shield.material.casefold() != "aluminium"
                or shield.areal_density_unit is not Unit.THICKNESS
                or shield.areal_density_g_cm2 != thickness_g_cm2
            ):
                reasons.append("spectrum aluminium depth does not match its reported shield depth")
        reasons.extend(spectrum.incomplete_reasons)
        if spectrum.completeness is None:
            reasons.append("spectrum completeness metadata is missing")
        elif spectrum.completeness is SpectrumCompleteness.INCOMPLETE:
            reasons.append("spectrum is explicitly incomplete")
    reasons.extend(
        "required product signature is missing: "
        f"kind={signature.product_kind.value}, "
        f"quantity={signature.quantity.value}, "
        f"particle={signature.particle.value}, "
        "axis_kind="
        f"{signature.axis_kind.value if signature.axis_kind is not None else 'unspecified'}, "
        "temporal_basis="
        f"{signature.temporal_basis.value if signature.temporal_basis is not None else 'unspecified'}, "
        f"species_scope={signature.species_scope.value}"
        for signature in missing_signatures
    )
    unique_reasons = tuple(dict.fromkeys(reasons))
    return {
        "required_product_kinds": [kind.value for kind in required_kinds],
        "observed_product_kinds": [kind.value for kind in observed_kinds],
        "missing_product_kinds": [kind.value for kind in missing_kinds],
        "required_product_signatures": [
            signature.to_summary() for signature in required_signatures
        ],
        "observed_product_signatures": [
            signature.to_summary() for signature in observed_signatures
        ],
        "missing_product_signatures": [
            signature.to_summary() for signature in missing_signatures
        ],
        "normative_ion_roster": ion_roster,
        "complete": not unique_reasons,
        "incomplete_reasons": list(unique_reasons),
        "spectra": [
            _spectrum_summary(product)
            for product in sorted(products, key=_product_sort_key)
        ],
    }


def _identity_key(product: SpectrumProduct) -> tuple[object, ...]:
    species = product.spectrum.species
    return (
        product.kind,
        product.spectrum.particle,
        None if species is None else species.symbol,
        None if species is None else species.atomic_number,
        None if species is None else species.mass_number,
        None if species is None else species.is_aggregate,
    )


def _identity_key_sort_key(identity: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in identity)


def _identity_key_summary(identity: tuple[object, ...]) -> dict[str, object]:
    product_kind, particle, symbol, atomic_number, mass_number, is_aggregate = identity
    assert isinstance(product_kind, RadiationProductKind)
    assert isinstance(particle, Particle)
    return {
        "product_kind": product_kind.value,
        "particle": particle.value,
        "species": symbol,
        "atomic_number": atomic_number,
        "mass_number": mass_number,
        "is_aggregate": is_aggregate,
    }


def _expanded_identity_counter(
    counts: Counter[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        identity
        for identity in sorted(counts, key=_identity_key_sort_key)
        for _ in range(counts[identity])
    )


def _zero_product_multiset_summary(
    execution: Stage4SourceExecution,
) -> dict[str, object]:
    expected_keys = tuple(_identity_key(product) for product in execution.on_orbit_products)
    observed_keys = tuple(
        _identity_key(product) for product in execution.zero_shield.products
    )
    expected_counts = Counter(expected_keys)
    observed_counts = Counter(observed_keys)
    missing = _expanded_identity_counter(expected_counts - observed_counts)
    extra = _expanded_identity_counter(observed_counts - expected_counts)
    excess_duplicates = tuple(
        identity
        for identity in sorted(observed_counts, key=_identity_key_sort_key)
        if observed_counts[identity] > expected_counts[identity]
        and observed_counts[identity] > 1
    )
    cardinality_matches = len(expected_keys) == len(observed_keys)
    key_multiset_matches = expected_counts == observed_counts
    return {
        "expected_count": len(expected_keys),
        "observed_count": len(observed_keys),
        "cardinality_matches": cardinality_matches,
        "key_multiset_matches": key_multiset_matches,
        "missing": [_identity_key_summary(identity) for identity in missing],
        "extra": [_identity_key_summary(identity) for identity in extra],
        "duplicates": [
            _identity_key_summary(identity) for identity in excess_duplicates
        ],
        "complete": cardinality_matches and key_multiset_matches,
    }


def _zero_identity_summary(
    execution: Stage4SourceExecution,
    mode: AssessmentMode,
) -> dict[str, object]:
    source_products = sorted(execution.on_orbit_products, key=_product_sort_key)
    zero_products = sorted(execution.zero_shield.products, key=_product_sort_key)
    matched_zero_indices: set[int] = set()
    checks: list[dict[str, object]] = []
    for source_product in source_products:
        identity = _identity_key(source_product)
        matched_index = next(
            (
                index
                for index, product in enumerate(zero_products)
                if index not in matched_zero_indices
                and _identity_key(product) == identity
            ),
            None,
        )
        if matched_index is None:
            checks.append(
                {
                    "product_kind": source_product.kind.value,
                    "particle": source_product.spectrum.particle.value,
                    "species": identity[2],
                    "identity_passed": False,
                    "reason": "zero-shield product is missing",
                }
            )
            continue

        matched_zero_indices.add(matched_index)
        shielded_product = zero_products[matched_index]
        source_spectrum = source_product.spectrum
        shielded_spectrum = shielded_product.spectrum
        same_grid = source_spectrum.x == shielded_spectrum.x
        same_length = len(source_spectrum.y) == len(shielded_spectrum.y)
        max_absolute_difference = 0.0
        max_relative_difference = 0.0
        values_close = same_length
        if same_length:
            for source_value, shielded_value in zip(
                source_spectrum.y,
                shielded_spectrum.y,
                strict=True,
            ):
                difference = abs(source_value - shielded_value)
                scale = max(abs(source_value), abs(shielded_value), 1.0e-300)
                max_absolute_difference = max(max_absolute_difference, difference)
                max_relative_difference = max(max_relative_difference, difference / scale)
                values_close = values_close and isclose(
                    source_value,
                    shielded_value,
                    rel_tol=_ZERO_IDENTITY_REL_TOLERANCE,
                    abs_tol=_ZERO_IDENTITY_ABS_TOLERANCE,
                )
        source_reference = source_spectrum.model_reference
        shielded_reference = shielded_spectrum.model_reference
        same_document_refs = (
            None if source_reference is None else source_reference.document_refs
        ) == (
            None if shielded_reference is None else shielded_reference.document_refs
        )
        same_dataset_refs = (
            None if source_reference is None else source_reference.dataset_refs
        ) == (
            None if shielded_reference is None else shielded_reference.dataset_refs
        )
        contract_checks = {
            "x_unit": source_spectrum.x_unit is shielded_spectrum.x_unit,
            "y_unit": source_spectrum.y_unit is shielded_spectrum.y_unit,
            "quantity": source_spectrum.quantity is shielded_spectrum.quantity,
            "particle": source_spectrum.particle is shielded_spectrum.particle,
            "source": source_spectrum.source is shielded_spectrum.source,
            "axis_kind": source_spectrum.axis_kind is shielded_spectrum.axis_kind,
            "angular_basis": (
                source_spectrum.angular_basis is shielded_spectrum.angular_basis
            ),
            "temporal_basis": (
                source_spectrum.temporal_basis is shielded_spectrum.temporal_basis
            ),
            "species": source_spectrum.species == shielded_spectrum.species,
            "assessment_mode": (
                source_spectrum.assessment_mode is shielded_spectrum.assessment_mode
            ),
            "mission_duration_s": (
                source_spectrum.mission_duration_s
                == shielded_spectrum.mission_duration_s
            ),
            "let_role": source_spectrum.let_role is shielded_spectrum.let_role,
            "ordinate_basis": (
                source_spectrum.ordinate_basis is shielded_spectrum.ordinate_basis
            ),
            "bin_edges": source_spectrum.bin_edges == shielded_spectrum.bin_edges,
            "completeness": (
                source_spectrum.completeness is shielded_spectrum.completeness
            ),
            "incomplete_reasons": (
                source_spectrum.incomplete_reasons
                == shielded_spectrum.incomplete_reasons
            ),
            "document_refs": same_document_refs,
            "dataset_refs": same_dataset_refs,
            "behind_zero_aluminium": (
                shielded_spectrum.location is SpectrumLocation.BEHIND_SHIELD
                and shielded_spectrum.shielding is not None
                and shielded_spectrum.shielding.material.casefold() == "aluminium"
                and shielded_spectrum.shielding.areal_density_g_cm2 == 0.0
                and shielded_spectrum.shielding.areal_density_unit is Unit.THICKNESS
            ),
        }
        exact_values_equal = source_spectrum.y == shielded_spectrum.y
        identity_passed = (
            same_grid
            and exact_values_equal
            and all(contract_checks.values())
        )
        checks.append(
            {
                "product_kind": source_product.kind.value,
                "particle": source_spectrum.particle.value,
                "species": identity[2],
                "identity_passed": identity_passed,
                "same_grid": same_grid,
                "exact_values_equal": exact_values_equal,
                "values_within_tolerance": values_close,
                "contract_checks": contract_checks,
                "max_absolute_difference": max_absolute_difference,
                "max_relative_difference": max_relative_difference,
            }
        )

    for index, shielded_product in enumerate(zero_products):
        if index in matched_zero_indices:
            continue
        identity = _identity_key(shielded_product)
        checks.append(
            {
                "product_kind": shielded_product.kind.value,
                "particle": shielded_product.spectrum.particle.value,
                "species": identity[2],
                "identity_passed": False,
                "reason": "zero-shield product is unexpected",
            }
        )

    ion_roster = _ion_roster_summary(
        source=execution.source,
        mode=mode,
        products=execution.zero_shield.products,
    )
    product_multiset = _zero_product_multiset_summary(execution)
    multiset_reasons = (
        ()
        if bool(product_multiset["complete"])
        else ("zero-shield product key multiset does not match on-orbit products",)
    )
    reasons = tuple(
        dict.fromkeys(
            (
                *execution.zero_shield.incomplete_reasons,
                *cast(list[str], ion_roster["incomplete_reasons"]),
                *multiset_reasons,
            )
        )
    )
    identity_passed = bool(checks) and all(
        bool(item["identity_passed"]) for item in checks
    )
    release_ready = (
        identity_passed
        and not reasons
        and bool(product_multiset["complete"])
        and bool(ion_roster["complete"])
        and all(
            not product.production_contract_violations()
            for product in execution.zero_shield.products
        )
    )
    return {
        "checked_at_low_level": True,
        "checked_thickness_g_cm2": 0.0,
        "public_shielding_config_contains_zero": False,
        "relative_tolerance": _ZERO_IDENTITY_REL_TOLERANCE,
        "absolute_tolerance": _ZERO_IDENTITY_ABS_TOLERANCE,
        "identity_passed": identity_passed,
        "release_ready": release_ready,
        "product_multiset": product_multiset,
        "normative_ion_roster": ion_roster,
        "incomplete_reasons": list(reasons),
        "products": checks,
    }


def _positive_shield_thickness_summary(
    execution: Stage4SourceExecution,
    config: CalculationConfig,
) -> dict[str, object]:
    expected = tuple(config.shielding.thicknesses_g_cm2)
    observed = tuple(item.thickness_g_cm2 for item in execution.shielded_by_thickness)
    expected_counts = Counter(expected)
    observed_counts = Counter(observed)
    missing = tuple(sorted((expected_counts - observed_counts).elements()))
    extra = tuple(sorted((observed_counts - expected_counts).elements()))
    duplicates = tuple(
        sorted(
            thickness
            for thickness, count in observed_counts.items()
            if count > 1
        )
    )
    reasons: list[str] = []
    if missing:
        reasons.append(
            "positive shield depths are missing: "
            + ",".join(f"{value:g}" for value in missing)
        )
    if extra:
        reasons.append(
            "unexpected positive shield depths are present: "
            + ",".join(f"{value:g}" for value in extra)
        )
    if duplicates:
        reasons.append(
            "positive shield depths are duplicated: "
            + ",".join(f"{value:g}" for value in duplicates)
        )
    return {
        "expected_g_cm2": list(expected),
        "observed_g_cm2": list(observed),
        "missing_g_cm2": list(missing),
        "extra_g_cm2": list(extra),
        "duplicates_g_cm2": list(duplicates),
        "complete": not reasons,
        "incomplete_reasons": reasons,
    }


def _source_summary(
    execution: Stage4SourceExecution,
    config: CalculationConfig,
) -> dict[str, object]:
    on_orbit = _stage_summary(
        source=execution.source,
        mode=config.assessment_mode,
        products=execution.on_orbit_products,
        behind_shield=False,
    )
    shields = [
        {
            "thickness_g_cm2": item.thickness_g_cm2,
            **_stage_summary(
                source=execution.source,
                mode=config.assessment_mode,
                products=item.products,
                behind_shield=True,
                thickness_g_cm2=item.thickness_g_cm2,
                incomplete_reasons=item.incomplete_reasons,
            ),
        }
        for item in execution.shielded_by_thickness
    ]
    positive_shield_thicknesses = _positive_shield_thickness_summary(
        execution,
        config,
    )
    zero_identity = _zero_identity_summary(execution, config.assessment_mode)
    complete = (
        bool(on_orbit["complete"])
        and bool(positive_shield_thicknesses["complete"])
        and all(bool(item["complete"]) for item in shields)
    )
    return {
        "source": execution.source.value,
        "model": {
            "name": execution.model_name,
            "version": execution.model_version,
            "document": execution.model_document,
        },
        "on_orbit": on_orbit,
        "positive_shield_thicknesses": positive_shield_thicknesses,
        "behind_aluminium_shield": shields,
        "zero_shield_identity": zero_identity,
        "complete": complete,
        "release_blockers": list(execution.release_blockers),
    }


def _config_summary(
    scenario: Stage4ReferenceScenario,
    config: CalculationConfig,
) -> dict[str, object]:
    return {
        "scenario": scenario.name,
        "assessment_mode": config.assessment_mode.value,
        "methodology_profile": config.methodology.profile.value,
        "dose_quantity_contract": config.dose_quantity.value,
        "dose_unit_contract": config.dose_unit.value,
        "launch_year": config.mission.launch_year,
        "mission_lifetime_years": config.mission.lifetime_years,
        "sep_exceedance_probability": config.mission.sep_exceedance_probability,
        "kp": config.kp,
        "orbit": {
            "type": config.orbit.orbit_type.value,
            "perigee_altitude_km": config.orbit.perigee_altitude_km,
            "apogee_altitude_km": config.orbit.apogee_altitude_km,
            "inclination_deg": config.orbit.inclination_deg,
            "argument_of_perigee_deg": config.orbit.argument_of_perigee_deg,
        },
        "shielding": {
            "material": "aluminium",
            "geometry": config.shielding.geometry.value,
            "thicknesses_g_cm2": list(config.shielding.thicknesses_g_cm2),
        },
        "solar_activity_contract": {
            "reference_start_year": config.methodology.solar_cycle_reference_start_year,
            "reference_cycle_number": config.methodology.solar_cycle_reference_number,
            "wolf_number_version": config.methodology.solar_activity_wolf_version,
            "monthly_policy": config.methodology.solar_activity_monthly_policy,
        },
    }


def _default_audit_provider() -> DataAuditReport:
    return audit_normative_resources(
        strict=True,
        scope=DataAuditScope.for_profile(MethodologyProfile.OST_134_1044_2007),
    )


def run_stage4_reference_validation(
    *,
    scenarios: Sequence[Stage4ReferenceScenario] = STAGE4_REFERENCE_SCENARIOS,
    modes: Sequence[AssessmentMode] = STAGE4_REFERENCE_MODES,
    case_executor: Stage4CaseExecutor = execute_stage4_reference_case,
    audit_provider: Stage4AuditProvider = _default_audit_provider,
) -> dict[str, object]:
    """Run a selectable matrix and return a deterministic JSON-ready report."""

    if not scenarios or not modes:
        raise ValueError("Stage-4 reference validation requires at least one scenario and mode.")

    audit = audit_provider()
    cases: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = [
        {
            "scope": "report",
            "code": "diagnostic_not_numerical_gold",
            "message": (
                "This calculated report is diagnostic and has no independently "
                "traceable COSRAD numerical gold baseline."
            ),
        }
    ]
    for finding in audit.findings:
        blockers.append(
            {
                "scope": "normative_data",
                "code": finding.code,
                "severity": finding.severity.value,
                "path": finding.path,
                "message": finding.message,
            }
        )

    for scenario in scenarios:
        for mode in modes:
            config = build_stage4_reference_config(scenario, mode)
            execution = case_executor(config)
            source_summaries = [
                _source_summary(source_execution, config)
                for source_execution in execution.sources
            ]
            case_id = f"{scenario.name.lower()}_{mode.value}"
            case_complete = all(bool(item["complete"]) for item in source_summaries)
            zero_identities_pass = all(
                bool(
                    cast(dict[str, object], item["zero_shield_identity"])[
                        "identity_passed"
                    ]
                )
                for item in source_summaries
            )
            zero_shields_release_ready = all(
                bool(
                    cast(dict[str, object], item["zero_shield_identity"])[
                        "release_ready"
                    ]
                )
                for item in source_summaries
            )
            for item in source_summaries:
                source = str(item["source"])
                for message in cast(list[object], item["release_blockers"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "code": "pipeline_release_gate",
                            "message": str(message),
                        }
                    )
                thicknesses = cast(
                    dict[str, object],
                    item["positive_shield_thicknesses"],
                )
                if not bool(thicknesses["complete"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "stage": "behind_aluminium_shield",
                            "code": "positive_shield_thickness_multiset_mismatch",
                            "message": (
                                "The calculated positive shield-depth multiset does "
                                "not exactly match the configured shield depths."
                            ),
                            "expected_g_cm2": thicknesses["expected_g_cm2"],
                            "observed_g_cm2": thicknesses["observed_g_cm2"],
                            "missing_g_cm2": thicknesses["missing_g_cm2"],
                            "extra_g_cm2": thicknesses["extra_g_cm2"],
                            "duplicates_g_cm2": thicknesses["duplicates_g_cm2"],
                        }
                    )
                if not bool(item["complete"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "code": "incomplete_spectral_products",
                            "message": "One or more mandatory spectral products are incomplete.",
                        }
                    )
                roster_stages = [
                    (
                        "on_orbit",
                        None,
                        cast(dict[str, object], item["on_orbit"]),
                    ),
                    *(
                        (
                            "behind_aluminium_shield",
                            shield["thickness_g_cm2"],
                            shield,
                        )
                        for shield in cast(
                            list[dict[str, object]],
                            item["behind_aluminium_shield"],
                        )
                    ),
                ]
                for stage_name, thickness, stage in roster_stages:
                    roster = cast(
                        dict[str, object],
                        stage["normative_ion_roster"],
                    )
                    if bool(roster["complete"]):
                        continue
                    blocker: dict[str, object] = {
                        "scope": "calculation_case",
                        "case_id": case_id,
                        "source": source,
                        "stage": stage_name,
                        "code": "normative_ion_roster_incomplete",
                        "message": (
                            "The OST-required ion roster is not exactly covered."
                        ),
                        "missing_ions": roster["missing"],
                        "unexpected_ions": roster["unexpected"],
                        "duplicate_ions": roster["duplicates"],
                    }
                    if thickness is not None:
                        blocker["thickness_g_cm2"] = thickness
                    blockers.append(blocker)
                zero_identity = cast(dict[str, object], item["zero_shield_identity"])
                zero_roster = cast(
                    dict[str, object],
                    zero_identity["normative_ion_roster"],
                )
                if not bool(zero_roster["complete"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "stage": "zero_shield",
                            "thickness_g_cm2": 0.0,
                            "code": "normative_ion_roster_incomplete",
                            "message": (
                                "The OST-required zero-shield ion roster is not "
                                "exactly covered."
                            ),
                            "missing_ions": zero_roster["missing"],
                            "unexpected_ions": zero_roster["unexpected"],
                            "duplicate_ions": zero_roster["duplicates"],
                        }
                    )
                zero_product_multiset = cast(
                    dict[str, object],
                    zero_identity["product_multiset"],
                )
                if not bool(zero_product_multiset["complete"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "stage": "zero_shield",
                            "thickness_g_cm2": 0.0,
                            "code": "zero_shield_product_multiset_mismatch",
                            "message": (
                                "The zero-shield product key multiset does not "
                                "exactly match the on-orbit product key multiset."
                            ),
                            "expected_count": zero_product_multiset[
                                "expected_count"
                            ],
                            "observed_count": zero_product_multiset[
                                "observed_count"
                            ],
                            "missing_products": zero_product_multiset["missing"],
                            "extra_products": zero_product_multiset["extra"],
                            "duplicate_products": zero_product_multiset[
                                "duplicates"
                            ],
                        }
                    )
                if not bool(zero_identity["identity_passed"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "code": "zero_shield_identity_failed",
                            "message": "The low-level zero-shield energy identity did not pass.",
                        }
                    )
                if not bool(zero_identity["release_ready"]):
                    blockers.append(
                        {
                            "scope": "calculation_case",
                            "case_id": case_id,
                            "source": source,
                            "code": "zero_shield_release_not_ready",
                            "message": (
                                "The zero-shield diagnostic is not release-ready."
                            ),
                            "reasons": cast(
                                list[object],
                                zero_identity["incomplete_reasons"],
                            ),
                        }
                    )
            cases.append(
                {
                    "case_id": case_id,
                    "inputs": _config_summary(scenario, config),
                    "complete": case_complete,
                    "zero_shield_identities_pass": zero_identities_pass,
                    "zero_shields_release_ready": zero_shields_release_ready,
                    "sources": source_summaries,
                }
            )

    unique_blockers = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in blockers
        }.values()
    )
    calculations_release_ready = (
        audit.release_ready
        and all(bool(item["complete"]) for item in cases)
        and all(bool(item["zero_shield_identities_pass"]) for item in cases)
        and all(bool(item["zero_shields_release_ready"]) for item in cases)
        and not any(
            source["release_blockers"]
            for case in cases
            for source in cast(list[dict[str, object]], case["sources"])
        )
    )
    full_matrix_requested = (
        tuple(scenarios) == STAGE4_REFERENCE_SCENARIOS
        and tuple(modes) == STAGE4_REFERENCE_MODES
    )
    return {
        "schema": STAGE4_REFERENCE_REPORT_SCHEMA,
        "program_version": __version__,
        "report_kind": "stage4_spectrum_reference_diagnostic",
        "intended_use": {
            "diagnostic_only": True,
            "numerical_gold": False,
            "normative_export_allowed": False,
        },
        "matrix": {
            "full_reference_matrix_requested": full_matrix_requested,
            "scenario_count": len(scenarios),
            "mode_count": len(modes),
            "case_count": len(cases),
        },
        "normative_data_audit": {
            "release_id": audit.release_id,
            "profile": MethodologyProfile.OST_134_1044_2007.value,
            "strict": True,
            "release_ready": audit.release_ready,
            "error_count": audit.error_count,
            "warning_count": audit.warning_count,
        },
        "cases": cases,
        "release_gate": {
            "calculation_products_release_ready": calculations_release_ready,
            "report_eligible_for_normative_export": False,
            "blockers": unique_blockers,
        },
    }


def stage4_reference_validation_json(report: dict[str, object]) -> str:
    """Serialize a validation report deterministically."""

    return json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _selected_scenarios(names: Sequence[str]) -> tuple[Stage4ReferenceScenario, ...]:
    if not names:
        return STAGE4_REFERENCE_SCENARIOS
    requested = set(names)
    return tuple(item for item in STAGE4_REFERENCE_SCENARIOS if item.name in requested)


def _selected_modes(values: Sequence[str]) -> tuple[AssessmentMode, ...]:
    if not values:
        return STAGE4_REFERENCE_MODES
    requested = {AssessmentMode(value) for value in values}
    return tuple(mode for mode in STAGE4_REFERENCE_MODES if mode in requested)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage-4 reference matrix and write its deterministic JSON summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(item.name for item in STAGE4_REFERENCE_SCENARIOS),
        default=[],
        help="Run only this scenario; repeat to select more than one.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=tuple(mode.value for mode in STAGE4_REFERENCE_MODES),
        default=[],
        help="Run only this assessment mode; repeat to select both.",
    )
    parser.add_argument(
        "--fail-on-release-blockers",
        action="store_true",
        help="Return exit code 1 when the diagnostic report contains release blockers.",
    )
    arguments = parser.parse_args(argv)
    report = run_stage4_reference_validation(
        scenarios=_selected_scenarios(arguments.scenario),
        modes=_selected_modes(arguments.mode),
    )
    rendered = stage4_reference_validation_json(report)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    release_gate = cast(dict[str, object], report["release_gate"])
    if arguments.fail_on_release_blockers and release_gate["blockers"]:
        return 1
    return 0


__all__ = [
    "STAGE4_REFERENCE_MODES",
    "STAGE4_REFERENCE_REPORT_SCHEMA",
    "STAGE4_REFERENCE_SCENARIOS",
    "STAGE4_REFERENCE_SHIELD_THICKNESSES_G_CM2",
    "Stage4CaseExecution",
    "Stage4ProductSignature",
    "Stage4ReferenceScenario",
    "Stage4ShieldExecution",
    "Stage4SourceExecution",
    "Stage4SpeciesScope",
    "Stage4ZeroShieldExecution",
    "build_stage4_reference_config",
    "execute_stage4_reference_case",
    "main",
    "run_stage4_reference_validation",
    "stage4_reference_validation_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
