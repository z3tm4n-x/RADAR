"""Radiation product metadata and validation."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.dose_units import (
    default_dose_unit_for_radiation_quantity_mode,
    dose_quantity_for_radiation_quantity_mode,
)
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    AssessmentMode,
    DoseQuantity,
    RadiationProductKind,
    RadiationQuantityMode,
    SpectrumAxisKind,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit

FLUENCE_PRODUCT_KINDS = (
    RadiationProductKind.MISSION_FLUENCE,
    RadiationProductKind.MISSION_LET_FLUENCE,
)

FLUX_PRODUCT_KINDS = (
    RadiationProductKind.ORBIT_AVERAGED_FLUX,
    RadiationProductKind.MEAN_FLUX,
    RadiationProductKind.MAXIMUM_FLUX,
    RadiationProductKind.PEAK_FLUX,
    RadiationProductKind.MEAN_LET_FLUX,
    RadiationProductKind.MAXIMUM_LET_FLUX,
    RadiationProductKind.PEAK_LET_FLUX,
)

PRODUCT_KIND_SPECTRUM_QUANTITIES: dict[
    RadiationProductKind,
    tuple[SpectrumQuantity, ...],
] = {
    RadiationProductKind.MISSION_FLUENCE: (
        SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        SpectrumQuantity.INTEGRAL_FLUENCE,
    ),
    RadiationProductKind.ORBIT_AVERAGED_FLUX: (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.INTEGRAL_FLUX,
    ),
    RadiationProductKind.MEAN_FLUX: (
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_INTEGRAL_FLUX,
    ),
    RadiationProductKind.MAXIMUM_FLUX: (
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
    ),
    RadiationProductKind.PEAK_FLUX: (
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        SpectrumQuantity.PEAK_INTEGRAL_FLUX,
    ),
    RadiationProductKind.MISSION_LET_FLUENCE: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    ),
    RadiationProductKind.MEAN_LET_FLUX: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.MAXIMUM_LET_FLUX: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ),
    RadiationProductKind.PEAK_LET_FLUX: (
        SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    ),
}

SPECTRUM_QUANTITY_UNITS: dict[SpectrumQuantity, Unit] = {
    SpectrumQuantity.DIFFERENTIAL_FLUENCE: Unit.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.INTEGRAL_FLUENCE: Unit.INTEGRAL_FLUENCE,
    SpectrumQuantity.DIFFERENTIAL_FLUX: Unit.DIFFERENTIAL_FLUX,
    SpectrumQuantity.INTEGRAL_FLUX: Unit.INTEGRAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX: Unit.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_INTEGRAL_FLUX: Unit.INTEGRAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX: Unit.DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX: Unit.INTEGRAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX: Unit.DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_INTEGRAL_FLUX: Unit.INTEGRAL_FLUX,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE: Unit.DIFFERENTIAL_LET_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX: Unit.DIFFERENTIAL_LET_FLUX,
}

PER_NUCLEON_DIFFERENTIAL_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)

PRODUCT_KIND_TEMPORAL_BASES: dict[RadiationProductKind, TemporalBasis] = {
    RadiationProductKind.MISSION_FLUENCE: TemporalBasis.MISSION_TOTAL,
    RadiationProductKind.ORBIT_AVERAGED_FLUX: TemporalBasis.ORBIT_AVERAGED,
    RadiationProductKind.MEAN_FLUX: TemporalBasis.MEAN_INTERNAL,
    RadiationProductKind.MAXIMUM_FLUX: TemporalBasis.SOURCE_MAXIMUM,
    RadiationProductKind.PEAK_FLUX: TemporalBasis.SEP_PEAK,
    RadiationProductKind.MISSION_LET_FLUENCE: TemporalBasis.MISSION_TOTAL,
    RadiationProductKind.MEAN_LET_FLUX: TemporalBasis.MEAN_INTERNAL,
    RadiationProductKind.MAXIMUM_LET_FLUX: TemporalBasis.SOURCE_MAXIMUM,
    RadiationProductKind.PEAK_LET_FLUX: TemporalBasis.SEP_PEAK,
}

PRODUCT_KIND_ASSESSMENT_MODES: dict[
    RadiationProductKind,
    AssessmentMode | None,
] = {
    RadiationProductKind.MISSION_FLUENCE: AssessmentMode.CUMULATIVE,
    RadiationProductKind.ORBIT_AVERAGED_FLUX: None,
    RadiationProductKind.MEAN_FLUX: None,
    RadiationProductKind.MAXIMUM_FLUX: AssessmentMode.PEAK,
    RadiationProductKind.PEAK_FLUX: AssessmentMode.PEAK,
    RadiationProductKind.MISSION_LET_FLUENCE: AssessmentMode.CUMULATIVE,
    RadiationProductKind.MEAN_LET_FLUX: None,
    RadiationProductKind.MAXIMUM_LET_FLUX: AssessmentMode.PEAK,
    RadiationProductKind.PEAK_LET_FLUX: AssessmentMode.PEAK,
}

ASSESSMENT_MODE_PRODUCT_KINDS: dict[
    AssessmentMode,
    tuple[RadiationProductKind, ...],
] = {
    AssessmentMode.CUMULATIVE: (
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MISSION_LET_FLUENCE,
    ),
    AssessmentMode.PEAK: (
        RadiationProductKind.PEAK_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.PEAK_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
    ),
}

CANONICAL_DIFFERENTIAL_PRODUCT_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)


def radiation_quantity_mode_for_product_kind(
    product_kind: RadiationProductKind,
) -> RadiationQuantityMode:
    """Return Flux/Fluence class for a named radiation product."""

    if product_kind in FLUENCE_PRODUCT_KINDS:
        return RadiationQuantityMode.FLUENCE

    if product_kind in FLUX_PRODUCT_KINDS:
        return RadiationQuantityMode.FLUX

    msg = f"Unsupported radiation product kind: {product_kind}"
    raise ValueError(msg)


def dose_quantity_for_product_kind(product_kind: RadiationProductKind) -> DoseQuantity:
    """Return dose quantity implied by a radiation product kind."""

    mode = radiation_quantity_mode_for_product_kind(product_kind)

    return dose_quantity_for_radiation_quantity_mode(mode)


def default_dose_unit_for_product_kind(product_kind: RadiationProductKind) -> Unit:
    """Return default dose unit implied by a radiation product kind."""

    mode = radiation_quantity_mode_for_product_kind(product_kind)

    return default_dose_unit_for_radiation_quantity_mode(mode)


def allowed_spectrum_quantities_for_product_kind(
    product_kind: RadiationProductKind,
) -> tuple[SpectrumQuantity, ...]:
    """Return allowed spectrum quantities for a radiation product kind."""

    try:
        return PRODUCT_KIND_SPECTRUM_QUANTITIES[product_kind]
    except KeyError as exc:
        msg = f"Unsupported radiation product kind: {product_kind}"
        raise ValueError(msg) from exc


def expected_spectrum_unit_for_quantity(
    quantity: SpectrumQuantity,
) -> Unit:
    """Return required model spectrum y-unit for a spectrum quantity."""

    try:
        return SPECTRUM_QUANTITY_UNITS[quantity]
    except KeyError as exc:
        msg = f"Unsupported spectrum quantity: {quantity}"
        raise ValueError(msg) from exc


def allowed_spectrum_units_for_quantity(
    quantity: SpectrumQuantity,
    *,
    axis_kind: SpectrumAxisKind | None = None,
) -> tuple[Unit, ...]:
    """Return units compatible with a quantity and an explicit axis kind.

    Legacy spectra have no ``axis_kind`` and retain their historical MeV-based
    unit.  A production MeV/nucleon differential spectrum must use the matching
    differential-per-MeV/nucleon ordinate unit.
    """

    default_unit = expected_spectrum_unit_for_quantity(quantity)
    if (
        axis_kind is SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
        and quantity in PER_NUCLEON_DIFFERENTIAL_QUANTITIES
    ):
        if quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
            return (Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,)
        return (Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,)

    return (default_unit,)


def temporal_basis_for_product_kind(
    product_kind: RadiationProductKind,
) -> TemporalBasis:
    """Return the mandatory time semantics of a named product."""

    try:
        return PRODUCT_KIND_TEMPORAL_BASES[product_kind]
    except KeyError as exc:
        msg = f"Unsupported radiation product kind: {product_kind}"
        raise ValueError(msg) from exc


def assessment_mode_for_product_kind(
    product_kind: RadiationProductKind,
) -> AssessmentMode | None:
    """Return the public mode, or ``None`` for internal mean products."""

    try:
        return PRODUCT_KIND_ASSESSMENT_MODES[product_kind]
    except KeyError as exc:
        msg = f"Unsupported radiation product kind: {product_kind}"
        raise ValueError(msg) from exc


def product_kinds_for_assessment_mode(
    assessment_mode: AssessmentMode,
) -> tuple[RadiationProductKind, ...]:
    """Return the explicit public product matrix for one calculation mode.

    ``PEAK`` selects SEP peak products together with GCR/ERB maximum products.
    ``CUMULATIVE`` selects only quantities accumulated over the mission.  Mean
    spectra are intentionally internal and never selected as public results.
    """

    try:
        return ASSESSMENT_MODE_PRODUCT_KINDS[assessment_mode]
    except KeyError as exc:
        msg = f"Unsupported assessment mode: {assessment_mode}"
        raise ValueError(msg) from exc


def select_products_for_assessment_mode(
    products: tuple[SpectrumProduct, ...],
    assessment_mode: AssessmentMode,
    *,
    validate_for_release: bool = False,
    allow_untyped_legacy_diagnostics: bool = False,
) -> tuple[SpectrumProduct, ...]:
    """Select public products for peak or cumulative calculation output.

    The opt-in legacy fallback exists only for static test/integration models
    that predate typed temporal metadata. It is never used by a production
    release boundary.
    """

    allowed_kinds = product_kinds_for_assessment_mode(assessment_mode)
    selected = tuple(
        product
        for product in products
        if product.kind in allowed_kinds
        and product.spectrum.quantity in CANONICAL_DIFFERENTIAL_PRODUCT_QUANTITIES
    )

    if (
        not selected
        and allow_untyped_legacy_diagnostics
        and products
        and all(product.spectrum.axis_kind is None for product in products)
    ):
        selected = products

    if validate_for_release:
        from radar.data.integrity import require_release_eligible_normative_resource

        for product in selected:
            product.validate_production_contract()
            reference = product.spectrum.model_reference
            assert reference is not None
            for dataset_reference in reference.dataset_refs:
                path, separator, declared_digest = dataset_reference.partition(
                    "@sha256:"
                )
                entry = require_release_eligible_normative_resource(path)
                if separator and declared_digest != entry.sha256:
                    msg = (
                        "Spectrum dataset reference digest does not match the "
                        f"integrity manifest: {dataset_reference}."
                    )
                    raise ValueError(msg)

    return selected


def validate_spectrum_matches_product_kind(
    spectrum: Spectrum1D,
    product_kind: RadiationProductKind,
) -> None:
    """Validate that spectrum quantity and unit match a named radiation product."""

    allowed_quantities = allowed_spectrum_quantities_for_product_kind(product_kind)

    if spectrum.quantity not in allowed_quantities:
        allowed_values = ", ".join(quantity.value for quantity in allowed_quantities)
        msg = (
            f"Spectrum quantity {spectrum.quantity.value} does not match "
            f"radiation product {product_kind.value}. "
            f"Allowed quantities: {allowed_values}."
        )
        raise ValueError(msg)

    expected_units = allowed_spectrum_units_for_quantity(
        spectrum.quantity,
        axis_kind=spectrum.axis_kind,
    )

    if spectrum.y_unit not in expected_units:
        expected_unit_values = ", ".join(unit.value for unit in expected_units)
        msg = (
            f"Spectrum unit {spectrum.y_unit.value} does not match "
            f"spectrum quantity {spectrum.quantity.value}. "
            f"Expected unit: {expected_unit_values}."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class SpectrumProduct:
    """A named radiation product with explicit product semantics."""

    kind: RadiationProductKind
    spectrum: Spectrum1D
    label: str = ""

    def __post_init__(self) -> None:
        validate_spectrum_matches_product_kind(
            spectrum=self.spectrum,
            product_kind=self.kind,
        )

    @property
    def radiation_quantity_mode(self) -> RadiationQuantityMode:
        """Flux/Fluence class of this product."""

        return radiation_quantity_mode_for_product_kind(self.kind)

    @property
    def dose_quantity(self) -> DoseQuantity:
        """Dose quantity implied by this product."""

        return dose_quantity_for_product_kind(self.kind)

    @property
    def default_dose_unit(self) -> Unit:
        """Default dose unit implied by this product."""

        return default_dose_unit_for_product_kind(self.kind)

    def production_contract_violations(self) -> tuple[str, ...]:
        """Return spectrum and product-semantics production violations."""

        violations = list(self.spectrum.production_contract_violations())

        expected_temporal_basis = temporal_basis_for_product_kind(self.kind)
        if self.spectrum.temporal_basis is not expected_temporal_basis:
            violations.append(
                f"product {self.kind.value} requires temporal_basis "
                f"{expected_temporal_basis.value}"
            )

        expected_assessment_mode = assessment_mode_for_product_kind(self.kind)
        if self.spectrum.assessment_mode is not expected_assessment_mode:
            expected_value = (
                "none"
                if expected_assessment_mode is None
                else expected_assessment_mode.value
            )
            violations.append(
                f"product {self.kind.value} requires assessment_mode {expected_value}"
            )

        return tuple(dict.fromkeys(violations))

    def validate_production_contract(self) -> None:
        """Raise if the named product is not production-ready."""

        violations = self.production_contract_violations()
        if violations:
            details = "; ".join(violations)
            msg = f"Spectrum product does not satisfy the production contract: {details}."
            raise ValueError(msg)

    def validate_calculation_contract(self) -> None:
        """Validate all metadata while allowing an explicit release quarantine."""

        release_quarantine = "incomplete spectra cannot be released as production results"
        violations = tuple(
            violation
            for violation in self.production_contract_violations()
            if violation != release_quarantine
        )
        if violations:
            details = "; ".join(violations)
            msg = f"Spectrum product has an invalid calculation contract: {details}."
            raise ValueError(msg)
