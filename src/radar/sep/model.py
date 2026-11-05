"""Solar energetic particle model interface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
import math
from typing import Protocol

from radar.core.constants import (
    SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON,
    SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON,
)
from radar.core.products import (
    SpectrumProduct,
    assessment_mode_for_product_kind,
    temporal_basis_for_product_kind,
)
from radar.core.profiles import (
    GOST_SEP_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.project import MissionConfig
from radar.core.spectra import (
    ModelReference,
    ParticleSpecies,
    Spectrum1D,
    TransformationStep,
)
from radar.core.spectrum_ops import mark_spectrum_incomplete, scale_spectrum
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    AngularBasis,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.data.integrity import validate_normative_resource_integrity
from radar.sep.event_count import SepEventCountPolicy, calculate_sep_expected_events
from radar.sep.hze_spectrum import (
    OST_SEP_HZE_B9_RESOURCE,
    OST_SEP_HZE_GAMMA2_CORRECTED_MULTIPLIER,
    OST_SEP_HZE_GAMMA2_CORRECTION_POLICY,
    OstSepHzeIonRecord,
    evaluate_ost_sep_hze_spectrum,
    load_ost_sep_hze_ion_records,
    ost_sep_hze_coefficients_from_proton,
    OST_SEP_HZE_GAMMA2_MULTIPLIER,
)
from radar.sep.proton_spectrum import (
    SepProtonCoefficientRecord,
    SepProtonSpectrumCoefficients,
    SepProtonSpectrumProduct,
    directional_flux_to_omnidirectional_flux,
    evaluate_sep_proton_spectrum,
    load_sep_proton_coefficient_records,
    lookup_sep_proton_coefficients_interpolated,
)


SEP_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.HZE,
)

SEP_ALLOWED_ENERGY_X_UNITS = (
    Unit.MEV,
    Unit.MEV_PER_NUCLEON,
    Unit.GEV_PER_NUCLEON,
)

SEP_ALLOWED_ENERGY_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)

SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
OST_SEP_PROTON_COEFFICIENT_MODEL = "ost_134_1044_2007"
OST_SEP_PROTON_COEFFICIENT_RESOURCE = "sep_protons/ost_134_1044_2007_coefficients.csv"
GOST_SEP_PROTON_COEFFICIENT_MODEL = "gost_r_25645_165_2025"
GOST_SEP_PROTON_COEFFICIENT_RESOURCE = (
    "sep_protons/gost_r_25645_165_2025_coefficients.csv"
)
OST_SEP_NORMATIVE_ION_ROSTER_REASON_PREFIX = (
    "OST SEP Appendix B.9 positive-abundance ion roster incomplete"
)
SEP_CALLER_COEFFICIENT_RECORDS_DATASET_REF = (
    "diagnostic://caller-supplied/sep-proton-coefficient-records"
)
SEP_CALLER_HZE_ION_RECORDS_DATASET_REF = (
    "diagnostic://caller-supplied/sep-hze-ion-records"
)
SEP_CALLER_COEFFICIENT_RECORDS_INCOMPLETE_REASON = (
    "SEP diagnostic override: caller-supplied proton coefficient records were "
    "used instead of the declared normative coefficient dataset"
)
SEP_CALLER_HZE_ION_RECORDS_INCOMPLETE_REASON = (
    "SEP diagnostic override: caller-supplied HZE ion records were used instead "
    "of the official OST Appendix B.9 dataset"
)
SEP_NONDEFAULT_HZE_GAMMA2_INCOMPLETE_REASON_PREFIX = (
    "SEP diagnostic override: HZE gamma2 multiplier differs from the normative "
    "1.68 value"
)


@dataclass(frozen=True)
class _SepInputProvenance:
    dataset_refs: tuple[str, ...]
    normative_dataset_refs: tuple[str, ...]
    version: str
    incomplete_reasons: tuple[str, ...] = ()
    transformation_chain: tuple[TransformationStep, ...] = ()


def validate_sep_energy_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a SEP energy spectrum before shielding."""

    if spectrum.source is not RadiationSource.SEP:
        msg = "SEP model spectrum source must be SEP."
        raise ValueError(msg)

    if spectrum.particle not in SEP_ALLOWED_PARTICLES:
        msg = "SEP model spectrum must describe protons or HZE particles."
        raise ValueError(msg)

    if spectrum.quantity not in SEP_ALLOWED_ENERGY_QUANTITIES:
        msg = "SEP model spectrum quantity must be an energy fluence or flux."
        raise ValueError(msg)

    if spectrum.x_unit not in SEP_ALLOWED_ENERGY_X_UNITS:
        msg = "SEP model energy grid must be in MeV or GeV/nucleon."
        raise ValueError(msg)

    if spectrum.particle is Particle.HZE and spectrum.axis_kind is not None:
        if (
            spectrum.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
            or spectrum.x_unit is not Unit.MEV_PER_NUCLEON
        ):
            msg = "Typed SEP HZE energy spectra must use MeV/nucleon."
            raise ValueError(msg)

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        expected_fluence_units = (Unit.DIFFERENTIAL_FLUENCE,)
        if spectrum.x_unit is Unit.MEV_PER_NUCLEON:
            expected_fluence_units = (
                Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
            )
        if spectrum.y_unit not in expected_fluence_units:
            msg = "SEP differential fluence spectrum values must use differential fluence units."
            raise ValueError(msg)

    if spectrum.quantity in (
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    ):
        expected_flux_units = (Unit.DIFFERENTIAL_FLUX,)
        if spectrum.x_unit is Unit.MEV_PER_NUCLEON:
            expected_flux_units = (Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,)
        if spectrum.y_unit not in expected_flux_units:
            msg = "SEP differential flux spectrum values must use differential flux units."
            raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "SEP model energy grid values must be positive."
        raise ValueError(msg)


def validate_sep_proton_fluence_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a SEP proton differential fluence spectrum."""

    if spectrum.particle is not Particle.PROTON:
        msg = "SEP model spectrum must describe protons."
        raise ValueError(msg)

    if spectrum.source is not RadiationSource.SEP:
        msg = "SEP model spectrum source must be SEP."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        msg = "SEP model spectrum quantity must be differential fluence."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "SEP model energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
        msg = "SEP model spectrum values must use differential fluence units."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "SEP model energy grid values must be positive."
        raise ValueError(msg)


def _mission_fluence_product(spectrum: Spectrum1D) -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=spectrum,
        label="SEP proton mission fluence",
    )


def _sep_source_product(
    *,
    kind: RadiationProductKind,
    spectrum: Spectrum1D,
    label: str,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=spectrum,
        label=label,
    )


def _mission_duration_seconds(lifetime_years: int) -> float:
    if lifetime_years < 1:
        msg = "Mission lifetime must be at least one year."
        raise ValueError(msg)

    return float(lifetime_years * SECONDS_PER_YEAR)



def _validate_monthly_wolf_number_count(
    *,
    monthly_smoothed_wolf_numbers: tuple[float, ...],
    lifetime_years: int,
) -> None:
    expected_months = 12 * lifetime_years

    if len(monthly_smoothed_wolf_numbers) != expected_months:
        msg = (
            "OST SEP monthly Wolf number series length must match "
            f"12 * mission lifetime years: expected {expected_months}, "
            f"got {len(monthly_smoothed_wolf_numbers)}."
        )
        raise ValueError(msg)


def _load_sep_proton_coefficient_records(
    resource_path: str,
) -> tuple[SepProtonCoefficientRecord, ...]:
    """Load one explicitly selected normative SEP coefficient dataset."""

    validate_normative_resource_integrity(resource_path)
    resource = resources.files("radar.data.normative").joinpath(
        resource_path,
    )

    with resources.as_file(resource) as path:
        return load_sep_proton_coefficient_records(path)


@lru_cache(maxsize=1)
def _load_ost_sep_hze_ion_records() -> tuple[OstSepHzeIonRecord, ...]:
    validate_normative_resource_integrity(OST_SEP_HZE_B9_RESOURCE)
    resource = resources.files("radar.data.normative").joinpath(
        OST_SEP_HZE_B9_RESOURCE,
    )

    with resources.as_file(resource) as path:
        return load_ost_sep_hze_ion_records(path)


def ost_sep_required_hze_ions() -> tuple[tuple[int, str], ...]:
    """Return the exact positive-abundance ion roster from OST table B.9."""

    return tuple(
        (record.z, record.symbol)
        for record in _load_ost_sep_hze_ion_records()
        if record.relative_abundance > 0.0
    )


def ost_sep_normative_ion_roster_reasons(
    ion_records: tuple[OstSepHzeIonRecord, ...],
) -> tuple[str, ...]:
    """Return explicit release blockers for a non-exact OST B.9 roster."""

    required = ost_sep_required_hze_ions()
    observed = tuple(
        (record.z, record.symbol)
        for record in ion_records
        if record.relative_abundance > 0.0
    )
    observed_set = set(observed)
    required_set = set(required)
    missing = tuple(ion for ion in required if ion not in observed_set)
    unexpected = tuple(ion for ion in observed if ion not in required_set)
    duplicates = tuple(
        ion for ion in dict.fromkeys(observed) if observed.count(ion) > 1
    )
    reasons: list[str] = []

    def render(ions: tuple[tuple[int, str], ...]) -> str:
        return ", ".join(f"{symbol}(Z={z})" for z, symbol in ions)

    if missing:
        reasons.append(
            f"{OST_SEP_NORMATIVE_ION_ROSTER_REASON_PREFIX}: missing required ions "
            f"{render(missing)}"
        )
    if unexpected:
        reasons.append(
            f"{OST_SEP_NORMATIVE_ION_ROSTER_REASON_PREFIX}: unexpected ions "
            f"{render(unexpected)}"
        )
    if duplicates:
        reasons.append(
            f"{OST_SEP_NORMATIVE_ION_ROSTER_REASON_PREFIX}: duplicate ions "
            f"{render(duplicates)}"
        )
    return tuple(reasons)


def _mark_products_incomplete(
    products: tuple[SpectrumProduct, ...],
    reasons: tuple[str, ...],
) -> tuple[SpectrumProduct, ...]:
    if not reasons:
        return products
    return tuple(
        replace(
            product,
            spectrum=mark_spectrum_incomplete(product.spectrum, *reasons),
        )
        for product in products
    )


def _diagnostic_version(version: str, tags: tuple[str, ...]) -> str:
    if not tags:
        return version
    return f"{version}+diagnostic.{'.'.join(tags)}"


def _diagnostic_override_step(
    parameters: tuple[tuple[str, str], ...],
) -> tuple[TransformationStep, ...]:
    if not parameters:
        return ()
    return (
        TransformationStep(
            name="sep_diagnostic_input_override",
            version="1",
            parameters=parameters,
        ),
    )


def _source_status_with_overrides(
    *,
    normative_dataset_refs: tuple[str, ...],
    energy_grid: tuple[float, ...],
    override_reasons: tuple[str, ...],
) -> tuple[SpectrumCompleteness, tuple[str, ...]]:
    completeness, reasons = _source_contract_status(
        dataset_refs=normative_dataset_refs,
        energy_grid=energy_grid,
    )
    merged_reasons = tuple(dict.fromkeys((*reasons, *override_reasons)))
    if merged_reasons:
        return SpectrumCompleteness.INCOMPLETE, merged_reasons
    return completeness, ()


def _make_sep_proton_spectrum(
    *,
    energy_grid_mev: tuple[float, ...],
    values: tuple[float, ...],
    quantity: SpectrumQuantity,
    y_unit: Unit,
    model: str,
    product_kind: RadiationProductKind,
    mission_duration_seconds: float,
    document_refs: tuple[str, ...],
    dataset_refs: tuple[str, ...],
    normative_dataset_refs: tuple[str, ...],
    override_reasons: tuple[str, ...],
    version: str,
    transformation_chain: tuple[TransformationStep, ...],
) -> Spectrum1D:
    completeness, incomplete_reasons = _source_status_with_overrides(
        normative_dataset_refs=normative_dataset_refs,
        energy_grid=energy_grid_mev,
        override_reasons=override_reasons,
    )
    return Spectrum1D(
        x=energy_grid_mev,
        y=values,
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model=model,
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=temporal_basis_for_product_kind(product_kind),
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies(symbol="p", atomic_number=1, mass_number=1),
        assessment_mode=assessment_mode_for_product_kind(product_kind),
        mission_duration_s=(
            mission_duration_seconds
            if assessment_mode_for_product_kind(product_kind)
            is not None
            and product_kind is RadiationProductKind.MISSION_FLUENCE
            else None
        ),
        model_reference=ModelReference(
            name=model,
            version=version,
            document_refs=document_refs,
            dataset_refs=dataset_refs,
        ),
        transformation_chain=transformation_chain,
        completeness=completeness,
        incomplete_reasons=incomplete_reasons,
    )


def _make_sep_hze_spectrum(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    values: tuple[float, ...],
    quantity: SpectrumQuantity,
    y_unit: Unit,
    model: str,
    product_kind: RadiationProductKind,
    ion: OstSepHzeIonRecord,
    mission_duration_seconds: float,
    document_refs: tuple[str, ...],
    dataset_refs: tuple[str, ...],
    normative_dataset_refs: tuple[str, ...],
    override_reasons: tuple[str, ...],
    version: str,
    transformation_chain: tuple[TransformationStep, ...],
) -> Spectrum1D:
    completeness, incomplete_reasons = _source_status_with_overrides(
        normative_dataset_refs=normative_dataset_refs,
        energy_grid=energy_grid_mev_per_nucleon,
        override_reasons=override_reasons,
    )
    return Spectrum1D(
        x=energy_grid_mev_per_nucleon,
        y=values,
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model=model,
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=temporal_basis_for_product_kind(product_kind),
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies(
            symbol=ion.symbol,
            atomic_number=ion.z,
            mass_number=int(round(ion.mass_number)),
        ),
        assessment_mode=assessment_mode_for_product_kind(product_kind),
        mission_duration_s=(
            mission_duration_seconds
            if assessment_mode_for_product_kind(product_kind)
            is not None
            and product_kind is RadiationProductKind.MISSION_FLUENCE
            else None
        ),
        model_reference=ModelReference(
            name=model,
            version=version,
            document_refs=document_refs,
            dataset_refs=dataset_refs,
        ),
        transformation_chain=transformation_chain,
        completeness=completeness,
        incomplete_reasons=incomplete_reasons,
    )


def _sep_source_evaluation_step(
    *,
    version: str,
    coefficient_model: str,
    product: SepProtonSpectrumProduct,
    expected_events: float,
    exceedance_probability: float,
    particle: str,
) -> TransformationStep:
    return TransformationStep(
        name="sep_normative_source_spectrum",
        version=version,
        parameters=(
            ("coefficient_model", coefficient_model),
            ("table_product", product.value),
            ("expected_events", f"{expected_events:.17g}"),
            ("exceedance_probability", f"{exceedance_probability:.17g}"),
            ("particle", particle),
        ),
    )


def _omnidirectional_peak_step() -> TransformationStep:
    return TransformationStep(
        name="directional_to_omnidirectional_4pi",
        version="1",
        parameters=(("solid_angle_sr", f"{4.0 * math.pi:.17g}"),),
    )


def _mission_mean_step(mission_duration_seconds: float) -> TransformationStep:
    return TransformationStep(
        name="mission_fluence_to_mean_flux",
        version="1",
        parameters=(("mission_duration_s", f"{mission_duration_seconds:.17g}"),),
    )


def _hze_scaling_step(
    *,
    ion: OstSepHzeIonRecord,
    gamma2_multiplier: float,
) -> TransformationStep:
    if math.isclose(
        gamma2_multiplier,
        OST_SEP_HZE_GAMMA2_CORRECTED_MULTIPLIER,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        gamma2_policy = OST_SEP_HZE_GAMMA2_CORRECTION_POLICY
    else:
        gamma2_policy = "explicit_nondefault_comparison_value"

    return TransformationStep(
        name="ost_sep_hze_scaling",
        version="2",
        parameters=(
            ("z", str(ion.z)),
            ("mass_number", f"{ion.mass_number:.17g}"),
            ("source_tabulated_mass_parameter", f"{ion.mass_number:.17g}"),
            ("species_mass_label_policy", "legacy_round_not_isotope_identification"),
            ("transport_mass_policy", "not_established"),
            ("mass_to_charge", f"{ion.mass_to_charge:.17g}"),
            ("relative_abundance", f"{ion.relative_abundance:.17g}"),
            ("gamma2_multiplier", f"{gamma2_multiplier:.17g}"),
            ("gamma2_multiplier_policy", gamma2_policy),
        ),
    )


@lru_cache(maxsize=None)
def _dataset_contract_status(
    dataset_refs: tuple[str, ...],
) -> tuple[SpectrumCompleteness, tuple[str, ...]]:
    reasons: list[str] = []
    for dataset_ref in dataset_refs:
        entry = validate_normative_resource_integrity(dataset_ref)
        if not entry.release_eligible:
            reasons.append(
                f"dataset {dataset_ref} is {entry.status.value} and is not release eligible"
            )
    if reasons:
        return SpectrumCompleteness.INCOMPLETE, tuple(reasons)
    return SpectrumCompleteness.COMPLETE, ()


def _source_contract_status(
    *,
    dataset_refs: tuple[str, ...],
    energy_grid: tuple[float, ...],
) -> tuple[SpectrumCompleteness, tuple[str, ...]]:
    """Combine resource eligibility with the COSRAD-aligned SEP domain."""

    _, dataset_reasons = _dataset_contract_status(dataset_refs)
    reasons = list(dataset_reasons)

    first_center = energy_grid[0]
    if first_center > SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON:
        reasons.append(
            "SEP source energy domain incomplete: first center "
            f"{first_center:.17g} MeV exceeds the COSRAD-aligned "
            f"{SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON:.17g} MeV center"
        )

    last_center = energy_grid[-1]
    if last_center < SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON:
        reasons.append(
            "SEP source energy domain incomplete: last center "
            f"{last_center:.17g} MeV is below the COSRAD-aligned "
            f"{SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON:.17g} MeV center"
        )

    if reasons:
        return SpectrumCompleteness.INCOMPLETE, tuple(reasons)
    return SpectrumCompleteness.COMPLETE, ()


def _is_sep_proton_mission_fluence_product(product: SpectrumProduct) -> bool:
    return (
        product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.source is RadiationSource.SEP
        and product.spectrum.particle is Particle.PROTON
        and product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    )


def _sep_proton_mission_fluence_products(
    products: tuple[SpectrumProduct, ...],
    spectrum: Spectrum1D,
) -> tuple[SpectrumProduct, ...]:
    return tuple(
        product
        for product in products
        if _is_sep_proton_mission_fluence_product(product)
        and product.spectrum == spectrum
    )


@dataclass(frozen=True)
class SepModelInput:
    """Input parameters passed to a SEP model."""

    mission: MissionConfig

    @property
    def lifetime_years(self) -> int:
        """Return mission lifetime in integer years."""

        return self.mission.lifetime_years

    @property
    def exceedance_probability(self) -> float:
        """Return SEP exceedance probability."""

        return self.mission.sep_exceedance_probability


@dataclass(frozen=True)
class SepModelResult:
    """Result returned by a SEP model."""

    spectrum: Spectrum1D
    lifetime_years: int
    exceedance_probability: float
    model: str
    document: str
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        validate_sep_proton_fluence_spectrum(self.spectrum)

        products = self.products or (_mission_fluence_product(self.spectrum),)

        if not products:
            msg = "SEP model result must contain at least one radiation product."
            raise ValueError(msg)

        spectra = tuple(product.spectrum for product in products)

        for spectrum in spectra:
            validate_sep_energy_spectrum(spectrum)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=spectra,
            source=RadiationSource.SEP,
            mismatch_message="SEP model product spectra must match result spectra.",
        )

        carries_production_metadata = any(
            product.spectrum.axis_kind is not None for product in products
        )
        if carries_production_metadata:
            for product in products:
                product.validate_calculation_contract()

        mission_fluence_products = _sep_proton_mission_fluence_products(
            products=products,
            spectrum=self.spectrum,
        )

        if len(mission_fluence_products) != 1:
            msg = (
                "SEP model product spectrum set must contain exactly one proton "
                "mission fluence product matching result spectrum."
            )
            raise ValueError(msg)

        object.__setattr__(self, "products", products)

        if not isinstance(self.lifetime_years, int):
            msg = "SEP result lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < 1:
            msg = "SEP result lifetime must be at least one year."
            raise ValueError(msg)

        if not 0.0 < self.exceedance_probability < 1.0:
            msg = "SEP exceedance probability must be between 0 and 1."
            raise ValueError(msg)

        if not self.model:
            msg = "SEP model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "SEP source document must not be empty."
            raise ValueError(msg)

    @property
    def spectra(self) -> tuple[Spectrum1D, ...]:
        """Return spectra carried by SEP radiation products."""

        return tuple(product.spectrum for product in self.products)

    @property
    def product(self) -> SpectrumProduct:
        """Return the SEP proton mission fluence product."""

        mission_fluence_products = _sep_proton_mission_fluence_products(
            products=self.products,
            spectrum=self.spectrum,
        )

        if len(mission_fluence_products) != 1:
            msg = "SEP model result does not contain exactly one proton mission fluence product."
            raise ValueError(msg)

        return mission_fluence_products[0]


class SepModelProtocol(Protocol):
    """Protocol implemented by concrete SEP models."""

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""
        ...

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Calculate mission SEP radiation products."""


@dataclass(frozen=True)
class StaticSepModel:
    """Simple deterministic SEP model used for tests and integration plumbing.

    The input spectrum is interpreted as annual differential fluence.
    The model scales it by integer mission lifetime.

    Normative OST/GOST SEP models must be implemented separately.
    """

    annual_fluence_spectrum: Spectrum1D
    model: str = "static_sep_model"
    document: str = "test"
    model_family: SourceModelFamily = SourceModelFamily.CUSTOM

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=self.model_family,
            name=self.model,
            document=self.document,
        )

    def __post_init__(self) -> None:
        validate_sep_proton_fluence_spectrum(self.annual_fluence_spectrum)

        if not self.model:
            msg = "SEP model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "SEP source document must not be empty."
            raise ValueError(msg)

        _ = self.metadata

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Return annual test fluence scaled by integer mission lifetime."""

        mission_spectrum = scale_spectrum(
            spectrum=self.annual_fluence_spectrum,
            factor=float(model_input.lifetime_years),
            model=f"{self.annual_fluence_spectrum.model}*{model_input.lifetime_years}years",
        )

        return SepModelResult(
            spectrum=mission_spectrum,
            lifetime_years=model_input.lifetime_years,
            exceedance_probability=model_input.exceedance_probability,
            model=self.model,
            document=self.document,
        )

@dataclass(frozen=True)
class OstSepModel:
    """Normative OST SEP proton source model before geomagnetic penetration.

    The model calculates proton fluence, peak flux, and mean flux spectra in
    interplanetary space using OST 134-1044-2007 Appendix B proton tables.

    Optional HZE source spectra use Appendix B.6/B.9. Geomagnetic penetration,
    shielding, LET conversion, dose, and single-event effects are calculated later.
    """

    energy_grid_mev: tuple[float, ...] = ()
    monthly_smoothed_wolf_numbers: tuple[float, ...] = ()
    coefficient_records: tuple[SepProtonCoefficientRecord, ...] = ()
    hze_energy_grid_mev_per_nucleon: tuple[float, ...] = ()
    hze_ion_records: tuple[OstSepHzeIonRecord, ...] = ()
    model: str = "ost_sep_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "ost_134_1044_2007_change_2_appendix_b_v1"
    coefficient_model: str = OST_SEP_PROTON_COEFFICIENT_MODEL
    coefficient_resource: str = OST_SEP_PROTON_COEFFICIENT_RESOURCE
    event_count_policy: SepEventCountPolicy = SepEventCountPolicy.OST_134_1044_2007
    hze_gamma2_multiplier: float = OST_SEP_HZE_GAMMA2_MULTIPLIER

    def _metadata_override_tags(self) -> tuple[str, ...]:
        tags: list[str] = []
        if self.coefficient_records:
            tags.append("caller_coefficients")
        if self.hze_ion_records:
            tags.append("caller_hze_ions")
        if not math.isclose(
            self.hze_gamma2_multiplier,
            OST_SEP_HZE_GAMMA2_CORRECTED_MULTIPLIER,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            tags.append("nondefault_gamma2")
        return tuple(tags)

    def _proton_input_provenance(self) -> _SepInputProvenance:
        if not self.coefficient_records:
            return _SepInputProvenance(
                dataset_refs=(self.coefficient_resource,),
                normative_dataset_refs=(self.coefficient_resource,),
                version=self.version,
            )
        return _SepInputProvenance(
            dataset_refs=(SEP_CALLER_COEFFICIENT_RECORDS_DATASET_REF,),
            normative_dataset_refs=(),
            version=_diagnostic_version(
                self.version,
                ("caller_coefficients",),
            ),
            incomplete_reasons=(
                SEP_CALLER_COEFFICIENT_RECORDS_INCOMPLETE_REASON,
            ),
            transformation_chain=_diagnostic_override_step(
                (
                    ("caller_supplied_coefficient_records", "true"),
                    ("coefficient_record_count", str(len(self.coefficient_records))),
                    ("declared_normative_dataset_used", "false"),
                )
            ),
        )

    def _hze_input_provenance(self) -> _SepInputProvenance:
        coefficient_override = bool(self.coefficient_records)
        ion_override = bool(self.hze_ion_records)
        gamma2_override = not math.isclose(
            self.hze_gamma2_multiplier,
            OST_SEP_HZE_GAMMA2_CORRECTED_MULTIPLIER,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        dataset_refs = (
            SEP_CALLER_COEFFICIENT_RECORDS_DATASET_REF
            if coefficient_override
            else self.coefficient_resource,
            SEP_CALLER_HZE_ION_RECORDS_DATASET_REF
            if ion_override
            else OST_SEP_HZE_B9_RESOURCE,
        )
        normative_dataset_refs = tuple(
            reference
            for reference, is_override in (
                (self.coefficient_resource, coefficient_override),
                (OST_SEP_HZE_B9_RESOURCE, ion_override),
            )
            if not is_override
        )
        tags: list[str] = []
        reasons: list[str] = []
        parameters: list[tuple[str, str]] = []
        if coefficient_override:
            tags.append("caller_coefficients")
            reasons.append(SEP_CALLER_COEFFICIENT_RECORDS_INCOMPLETE_REASON)
            parameters.extend(
                (
                    ("caller_supplied_coefficient_records", "true"),
                    ("coefficient_record_count", str(len(self.coefficient_records))),
                )
            )
        if ion_override:
            tags.append("caller_hze_ions")
            reasons.append(SEP_CALLER_HZE_ION_RECORDS_INCOMPLETE_REASON)
            parameters.extend(
                (
                    ("caller_supplied_hze_ion_records", "true"),
                    ("hze_ion_record_count", str(len(self.hze_ion_records))),
                )
            )
        if gamma2_override:
            tags.append("nondefault_gamma2")
            reasons.append(
                f"{SEP_NONDEFAULT_HZE_GAMMA2_INCOMPLETE_REASON_PREFIX}: "
                f"configured={self.hze_gamma2_multiplier:.17g}"
            )
            parameters.extend(
                (
                    (
                        "configured_hze_gamma2_multiplier",
                        f"{self.hze_gamma2_multiplier:.17g}",
                    ),
                    (
                        "normative_hze_gamma2_multiplier",
                        f"{OST_SEP_HZE_GAMMA2_CORRECTED_MULTIPLIER:.17g}",
                    ),
                )
            )
        return _SepInputProvenance(
            dataset_refs=tuple(dict.fromkeys(dataset_refs)),
            normative_dataset_refs=normative_dataset_refs,
            version=_diagnostic_version(self.version, tuple(tags)),
            incomplete_reasons=tuple(reasons),
            transformation_chain=_diagnostic_override_step(tuple(parameters)),
        )

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.OST_134_1044_2007,
            name=self.model,
            document=self.document,
            version=_diagnostic_version(self.version, self._metadata_override_tags()),
        )

    def __post_init__(self) -> None:
        if not self.model:
            msg = "SEP model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "SEP source document must not be empty."
            raise ValueError(msg)

        if not self.coefficient_model:
            msg = "SEP coefficient model name must not be empty."
            raise ValueError(msg)

        if not self.coefficient_resource:
            msg = "SEP coefficient resource path must not be empty."
            raise ValueError(msg)

        if (
            not math.isfinite(self.hze_gamma2_multiplier)
            or self.hze_gamma2_multiplier <= 0.0
        ):
            msg = "SEP HZE gamma2 multiplier must be positive and finite."
            raise ValueError(msg)

        if bool(self.energy_grid_mev) != bool(self.monthly_smoothed_wolf_numbers):
            msg = "OST SEP model requires both energy grid and Wolf number series."
            raise ValueError(msg)

        if self.energy_grid_mev:
            if any(not math.isfinite(energy) for energy in self.energy_grid_mev):
                msg = "OST SEP proton energy grid values must be finite."
                raise ValueError(msg)

            if any(energy <= 0.0 for energy in self.energy_grid_mev):
                msg = "OST SEP proton energy grid values must be positive."
                raise ValueError(msg)

            if any(
                right <= left
                for left, right in zip(self.energy_grid_mev, self.energy_grid_mev[1:])
            ):
                msg = "OST SEP proton energy grid must be strictly increasing."
                raise ValueError(msg)

        if self.hze_energy_grid_mev_per_nucleon:
            if not self.energy_grid_mev or not self.monthly_smoothed_wolf_numbers:
                msg = "OST SEP HZE grid requires proton source configuration."
                raise ValueError(msg)

            if any(
                not math.isfinite(energy)
                for energy in self.hze_energy_grid_mev_per_nucleon
            ):
                msg = "OST SEP HZE energy grid values must be finite."
                raise ValueError(msg)

            if any(energy <= 0.0 for energy in self.hze_energy_grid_mev_per_nucleon):
                msg = "OST SEP HZE energy grid values must be positive."
                raise ValueError(msg)

            if any(
                right <= left
                for left, right in zip(
                    self.hze_energy_grid_mev_per_nucleon,
                    self.hze_energy_grid_mev_per_nucleon[1:],
                )
            ):
                msg = "OST SEP HZE energy grid must be strictly increasing."
                raise ValueError(msg)

        if self.monthly_smoothed_wolf_numbers:
            if any(not math.isfinite(value) for value in self.monthly_smoothed_wolf_numbers):
                msg = "OST SEP Wolf numbers must be finite."
                raise ValueError(msg)

            if any(value < 0.0 for value in self.monthly_smoothed_wolf_numbers):
                msg = "OST SEP Wolf numbers must be non-negative."
                raise ValueError(msg)

        _ = self.metadata

    def _coefficient_records(self) -> tuple[SepProtonCoefficientRecord, ...]:
        if self.coefficient_records:
            return self.coefficient_records

        return _load_sep_proton_coefficient_records(self.coefficient_resource)

    def _hze_ion_records(self) -> tuple[OstSepHzeIonRecord, ...]:
        if self.hze_ion_records:
            return self.hze_ion_records

        return _load_ost_sep_hze_ion_records()

    def _hze_source_products(
        self,
        *,
        ion_records: tuple[OstSepHzeIonRecord, ...],
        provenance: _SepInputProvenance,
        fluence_coefficients: SepProtonSpectrumCoefficients,
        peak_flux_coefficients: SepProtonSpectrumCoefficients,
        mission_duration_seconds: float,
        expected_events: float,
        exceedance_probability: float,
    ) -> tuple[SpectrumProduct, ...]:
        if not self.hze_energy_grid_mev_per_nucleon:
            return ()

        products: list[SpectrumProduct] = []
        document_refs = tuple(
            dict.fromkeys((self.document, OST_134_1044_2007_DOCUMENT))
        )

        for ion in ion_records:
            if ion.relative_abundance == 0.0:
                continue

            fluence_hze_coefficients = ost_sep_hze_coefficients_from_proton(
                proton_coefficients=fluence_coefficients,
                ion=ion,
                gamma2_multiplier=self.hze_gamma2_multiplier,
            )
            peak_flux_hze_coefficients = ost_sep_hze_coefficients_from_proton(
                proton_coefficients=peak_flux_coefficients,
                ion=ion,
                gamma2_multiplier=self.hze_gamma2_multiplier,
            )

            mission_fluence_values = evaluate_ost_sep_hze_spectrum(
                self.hze_energy_grid_mev_per_nucleon,
                fluence_hze_coefficients,
            )
            raw_peak_directional_flux_values = evaluate_ost_sep_hze_spectrum(
                self.hze_energy_grid_mev_per_nucleon,
                peak_flux_hze_coefficients,
            )
            peak_flux_values = directional_flux_to_omnidirectional_flux(
                raw_peak_directional_flux_values,
            )
            mean_flux_values = tuple(
                value / mission_duration_seconds
                for value in mission_fluence_values
            )

            ion_tag = f"z{ion.z:02d}_{ion.symbol.lower()}"
            ion_label = f"{ion.symbol} Z={ion.z}"

            mission_fluence_spectrum = _make_sep_hze_spectrum(
                energy_grid_mev_per_nucleon=self.hze_energy_grid_mev_per_nucleon,
                values=mission_fluence_values,
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
                model=f"{self.model}:hze:{ion_tag}:mission_fluence",
                product_kind=RadiationProductKind.MISSION_FLUENCE,
                ion=ion,
                mission_duration_seconds=mission_duration_seconds,
                document_refs=document_refs,
                dataset_refs=provenance.dataset_refs,
                normative_dataset_refs=provenance.normative_dataset_refs,
                override_reasons=provenance.incomplete_reasons,
                version=provenance.version,
                transformation_chain=(
                    *provenance.transformation_chain,
                    _sep_source_evaluation_step(
                        version=self.version,
                        coefficient_model=self.coefficient_model,
                        product=SepProtonSpectrumProduct.FLUENCE,
                        expected_events=expected_events,
                        exceedance_probability=exceedance_probability,
                        particle=f"Z={ion.z},A={int(round(ion.mass_number))}",
                    ),
                    _hze_scaling_step(
                        ion=ion,
                        gamma2_multiplier=self.hze_gamma2_multiplier,
                    ),
                ),
            )
            peak_flux_spectrum = _make_sep_hze_spectrum(
                energy_grid_mev_per_nucleon=self.hze_energy_grid_mev_per_nucleon,
                values=peak_flux_values,
                quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,
                model=f"{self.model}:hze:{ion_tag}:peak_flux",
                product_kind=RadiationProductKind.PEAK_FLUX,
                ion=ion,
                mission_duration_seconds=mission_duration_seconds,
                document_refs=document_refs,
                dataset_refs=provenance.dataset_refs,
                normative_dataset_refs=provenance.normative_dataset_refs,
                override_reasons=provenance.incomplete_reasons,
                version=provenance.version,
                transformation_chain=(
                    *provenance.transformation_chain,
                    _sep_source_evaluation_step(
                        version=self.version,
                        coefficient_model=self.coefficient_model,
                        product=SepProtonSpectrumProduct.PEAK_FLUX,
                        expected_events=expected_events,
                        exceedance_probability=exceedance_probability,
                        particle=f"Z={ion.z},A={int(round(ion.mass_number))}",
                    ),
                    _hze_scaling_step(
                        ion=ion,
                        gamma2_multiplier=self.hze_gamma2_multiplier,
                    ),
                    _omnidirectional_peak_step(),
                ),
            )
            mean_flux_spectrum = _make_sep_hze_spectrum(
                energy_grid_mev_per_nucleon=self.hze_energy_grid_mev_per_nucleon,
                values=mean_flux_values,
                quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,
                model=f"{self.model}:hze:{ion_tag}:mean_flux",
                product_kind=RadiationProductKind.MEAN_FLUX,
                ion=ion,
                mission_duration_seconds=mission_duration_seconds,
                document_refs=document_refs,
                dataset_refs=provenance.dataset_refs,
                normative_dataset_refs=provenance.normative_dataset_refs,
                override_reasons=provenance.incomplete_reasons,
                version=provenance.version,
                transformation_chain=(
                    *provenance.transformation_chain,
                    _sep_source_evaluation_step(
                        version=self.version,
                        coefficient_model=self.coefficient_model,
                        product=SepProtonSpectrumProduct.FLUENCE,
                        expected_events=expected_events,
                        exceedance_probability=exceedance_probability,
                        particle=f"Z={ion.z},A={int(round(ion.mass_number))}",
                    ),
                    _hze_scaling_step(
                        ion=ion,
                        gamma2_multiplier=self.hze_gamma2_multiplier,
                    ),
                    _mission_mean_step(mission_duration_seconds),
                ),
            )

            products.extend(
                (
                    _sep_source_product(
                        kind=RadiationProductKind.MISSION_FLUENCE,
                        spectrum=mission_fluence_spectrum,
                        label=(
                            "SEP HZE mission fluence by OST 134-1044-2007 "
                            f"Appendix B for {ion_label}"
                        ),
                    ),
                    _sep_source_product(
                        kind=RadiationProductKind.PEAK_FLUX,
                        spectrum=peak_flux_spectrum,
                        label=(
                            "SEP HZE omnidirectional peak flux by OST 134-1044-2007 "
                            f"Appendix B for {ion_label}; derived as 4π times "
                            "directional per-steradian peak flux"
                        ),
                    ),
                    _sep_source_product(
                        kind=RadiationProductKind.MEAN_FLUX,
                        spectrum=mean_flux_spectrum,
                        label=(
                            "SEP HZE mean flux derived from OST 134-1044-2007 "
                            f"Appendix B mission fluence for {ion_label} divided "
                            "by mission duration"
                        ),
                    ),
                )
            )

        return tuple(products)

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Calculate OST SEP proton source spectra before geomagnetic penetration."""

        if not self.energy_grid_mev or not self.monthly_smoothed_wolf_numbers:
            msg = (
                "SEP proton source model is not implemented without configured energy grid "
                "and monthly Wolf number series."
            )
            raise NotImplementedError(msg)

        _validate_monthly_wolf_number_count(
            monthly_smoothed_wolf_numbers=self.monthly_smoothed_wolf_numbers,
            lifetime_years=model_input.lifetime_years,
        )

        event_count = calculate_sep_expected_events(
            monthly_smoothed_wolf_numbers=self.monthly_smoothed_wolf_numbers,
            policy=self.event_count_policy,
        )
        records = self._coefficient_records()
        proton_provenance = self._proton_input_provenance()

        fluence_coefficients = lookup_sep_proton_coefficients_interpolated(
            records,
            model=self.coefficient_model,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=event_count.expected_events,
            probability=model_input.exceedance_probability,
        )
        peak_flux_coefficients = lookup_sep_proton_coefficients_interpolated(
            records,
            model=self.coefficient_model,
            product=SepProtonSpectrumProduct.PEAK_FLUX,
            event_count=event_count.expected_events,
            probability=model_input.exceedance_probability,
        )

        mission_fluence_values = evaluate_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            coefficients=fluence_coefficients,
        )
        raw_peak_directional_flux_values = evaluate_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            coefficients=peak_flux_coefficients,
        )
        peak_flux_values = directional_flux_to_omnidirectional_flux(
            raw_peak_directional_flux_values,
        )

        mission_duration_seconds = _mission_duration_seconds(model_input.lifetime_years)
        mean_flux_values = tuple(
            value / mission_duration_seconds
            for value in mission_fluence_values
        )

        mission_fluence_spectrum = _make_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            values=mission_fluence_values,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            model=f"{self.model}:mission_fluence",
            product_kind=RadiationProductKind.MISSION_FLUENCE,
            mission_duration_seconds=mission_duration_seconds,
            document_refs=(self.document,),
            dataset_refs=proton_provenance.dataset_refs,
            normative_dataset_refs=proton_provenance.normative_dataset_refs,
            override_reasons=proton_provenance.incomplete_reasons,
            version=proton_provenance.version,
            transformation_chain=(
                *proton_provenance.transformation_chain,
                _sep_source_evaluation_step(
                    version=self.version,
                    coefficient_model=self.coefficient_model,
                    product=SepProtonSpectrumProduct.FLUENCE,
                    expected_events=event_count.expected_events,
                    exceedance_probability=model_input.exceedance_probability,
                    particle="Z=1,A=1",
                ),
            ),
        )
        peak_flux_spectrum = _make_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            values=peak_flux_values,
            quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            model=f"{self.model}:peak_flux",
            product_kind=RadiationProductKind.PEAK_FLUX,
            mission_duration_seconds=mission_duration_seconds,
            document_refs=(self.document,),
            dataset_refs=proton_provenance.dataset_refs,
            normative_dataset_refs=proton_provenance.normative_dataset_refs,
            override_reasons=proton_provenance.incomplete_reasons,
            version=proton_provenance.version,
            transformation_chain=(
                *proton_provenance.transformation_chain,
                _sep_source_evaluation_step(
                    version=self.version,
                    coefficient_model=self.coefficient_model,
                    product=SepProtonSpectrumProduct.PEAK_FLUX,
                    expected_events=event_count.expected_events,
                    exceedance_probability=model_input.exceedance_probability,
                    particle="Z=1,A=1",
                ),
                _omnidirectional_peak_step(),
            ),
        )
        mean_flux_spectrum = _make_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            values=mean_flux_values,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            model=f"{self.model}:mean_flux",
            product_kind=RadiationProductKind.MEAN_FLUX,
            mission_duration_seconds=mission_duration_seconds,
            document_refs=(self.document,),
            dataset_refs=proton_provenance.dataset_refs,
            normative_dataset_refs=proton_provenance.normative_dataset_refs,
            override_reasons=proton_provenance.incomplete_reasons,
            version=proton_provenance.version,
            transformation_chain=(
                *proton_provenance.transformation_chain,
                _sep_source_evaluation_step(
                    version=self.version,
                    coefficient_model=self.coefficient_model,
                    product=SepProtonSpectrumProduct.FLUENCE,
                    expected_events=event_count.expected_events,
                    exceedance_probability=model_input.exceedance_probability,
                    particle="Z=1,A=1",
                ),
                _mission_mean_step(mission_duration_seconds),
            ),
        )

        hze_ion_records = (
            self._hze_ion_records()
            if self.hze_energy_grid_mev_per_nucleon
            else ()
        )
        hze_products = self._hze_source_products(
            ion_records=hze_ion_records,
            provenance=self._hze_input_provenance(),
            fluence_coefficients=fluence_coefficients,
            peak_flux_coefficients=peak_flux_coefficients,
            mission_duration_seconds=mission_duration_seconds,
            expected_events=event_count.expected_events,
            exceedance_probability=model_input.exceedance_probability,
        )

        mission_fluence_product = _sep_source_product(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=mission_fluence_spectrum,
            label=f"SEP proton mission fluence by {self.document}",
        )
        peak_flux_product = _sep_source_product(
            kind=RadiationProductKind.PEAK_FLUX,
            spectrum=peak_flux_spectrum,
            label=(
                "SEP proton omnidirectional peak flux by "
                f"{self.document}; derived as 4π times directional per-steradian peak flux"
            ),
        )
        mean_flux_product = _sep_source_product(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=mean_flux_spectrum,
            label=(
                "SEP proton mean flux derived from "
                f"{self.document} mission fluence divided by mission duration"
            ),
        )

        products = (
            mission_fluence_product,
            peak_flux_product,
            mean_flux_product,
            *hze_products,
        )
        if self.metadata.model_family is SourceModelFamily.OST_134_1044_2007:
            products = _mark_products_incomplete(
                products,
                ost_sep_normative_ion_roster_reasons(hze_ion_records),
            )

        return SepModelResult(
            spectrum=products[0].spectrum,
            lifetime_years=model_input.lifetime_years,
            exceedance_probability=model_input.exceedance_probability,
            model=self.model,
            document=self.document,
            products=products,
        )


@dataclass(frozen=True)
class GostSepModel(OstSepModel):
    """GOST R 25645.165-2025 SEP proton source model using Wolf numbers 1.0.

    OST and GOST use the same interpolation algorithm but different transferred
    datasets.  They are selected explicitly because at least one printed control
    value differs (table B.2 in OST versus table 2 in GOST).  The GOST event-count
    policy uses W1.0 Wolf numbers with coefficient 0.0130; the optional W2.0 policy
    remains available through the lower-level event-count API.
    """

    model: str = "gost_sep_model"
    document: str = GOST_SEP_DOCUMENT
    version: str = "gost_r_25645_165_2025_draft_v1"
    coefficient_model: str = GOST_SEP_PROTON_COEFFICIENT_MODEL
    coefficient_resource: str = GOST_SEP_PROTON_COEFFICIENT_RESOURCE
    event_count_policy: SepEventCountPolicy = SepEventCountPolicy.GOST_R_25645_165_2025_W1_0

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.GOST_SEP,
            name=self.model,
            document=self.document,
            version=_diagnostic_version(self.version, self._metadata_override_tags()),
        )
