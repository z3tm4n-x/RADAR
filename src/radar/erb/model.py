"""Earth radiation belt model interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol

from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    AE8_AP8_RADBELT_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.project import CalculationConfig, MissionConfig
from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SolarActivityLevel,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.erb.appendix_e import (
    OST_ERB_DIFFERENTIAL_ERRATA_STATUS,
    OST_ERB_ELECTRON_DIFFERENTIAL_POLICY,
    OstErbApproximationFitQuality,
    OstErbElectronIntegralApproximation,
    OstErbProtonIntegralApproximation,
    fit_ost_erb_electron_integral_approximation,
    fit_ost_erb_proton_integral_approximation,
    integrate_differential_flux_tail_power_law,
)
from radar.erb.constants import DEFAULT_IGRF_EPOCH, EARTH_REFERENCE_RADIUS_KM, ERB_SECONDS_PER_YEAR
from radar.erb.field import (
    ErbFieldLineSample,
    ost_l_shell_b_over_b0_for_orbit,
    trace_l_shell_b_over_b0_for_orbit,
)
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import sample_erb_orbit_grid
from radar.erb.ost_tables import (
    OstErbFluxTable,
    OstErbInterpolationMode,
    OstErbTableSet,
    load_ost_erb_appendix_a_tables,
)
from radar.erb.radbelt import RADBELT_LOG_BIN_RATIO, RadbeltAscMap
from radar.erb.resources import load_ae8_ap8_radbelt_maps, load_igrf14_coefficients
from radar.solar_activity.model import (
    MissionSolarActivity,
    build_calculation_solar_activity,
    build_mission_solar_activity,
    mission_solar_activity_monthly_wolf_numbers,
)
from radar.solar_activity.ost import ost_wolf_number_cycle_table

OST_ERB_APPENDIX_A_REFERENCE = "OST 134-1044-2007 Appendix A tables A.1.1/A.1.2/A.2.1/A.2.2"
OST_ERB_APPENDIX_D_REFERENCE = "OST 134-1044-2007 Appendix D L-B coordinate calculation"
OST_ERB_APPENDIX_E_REFERENCE = "OST 134-1044-2007 Appendix E orbit-averaged ERB spectra"
AE8_AP8_RADBELT_REFERENCE = AE8_AP8_RADBELT_DOCUMENT
AE8_AP8_PROTON_ENERGIES_MEV = tuple(0.1 * 10.0 ** (index / 10.0) for index in range(38))
AE8_AP8_ELECTRON_ENERGIES_MEV = tuple(0.05 * 10.0 ** (index / 10.0) for index in range(24))

ERB_IGRF_DATASET_REF = "erb/igrf14coeffs.txt"
ERB_WOLF_DATASET_REF = "ost_134_1044_2007/table_g_1_wolf_numbers.csv"
OST_ERB_TABLE_DATASET_REF = "erb/ost_appendix_a_tables.json"
AE8_AP8_MAP_DATASET_REFS = {
    Particle.PROTON: ("erb/ap8min.asc", "erb/ap8max.asc"),
    Particle.ELECTRON: ("erb/ae8min.asc", "erb/ae8max.asc"),
}
AE8_AP8_PROVISIONAL_REASON = (
    "AE8/AP8 RADBELT maps have no retained immutable upstream provenance"
)
OST_ERB_ORBIT_SAMPLING_CONVERGENCE_REASON = (
    "OST ERB orbit sampling convergence below 1% has not been established "
    "for every reported energy node"
)
OST_ERB_ORBIT_TRAJECTORY_REASON = (
    "OST ERB independent mean-anomaly/RAAN ensemble has not been validated "
    "against the time-propagated orbit trajectory required by Appendix E.1"
)

OST_ERB_MAGNETIC_REGION_SELECTION_REASON = (
    "OST D.2.6 region preclassification by shifted-dipole L is an implementation "
    "convention pending boundary validation"
)

ERB_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.ELECTRON,
)

ERB_ALLOWED_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)


def validate_erb_energy_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a supported ERB energy flux spectrum."""

    if spectrum.source is not RadiationSource.ERB:
        msg = "ERB model spectrum source must be ERB."
        raise ValueError(msg)

    if spectrum.particle not in ERB_ALLOWED_PARTICLES:
        msg = "ERB model spectrum must describe protons or electrons."
        raise ValueError(msg)

    if spectrum.quantity not in ERB_ALLOWED_QUANTITIES:
        msg = "ERB model spectrum quantity must be a supported differential flux quantity."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "ERB model energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
            msg = "ERB model fluence spectrum values must use differential fluence units."
            raise ValueError(msg)
    elif spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
        msg = "ERB model flux spectrum values must use differential flux units."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "ERB model energy grid values must be positive."
        raise ValueError(msg)


def _default_erb_product_kind(spectrum: Spectrum1D) -> RadiationProductKind:
    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUX:
        return RadiationProductKind.ORBIT_AVERAGED_FLUX

    if spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MEAN_FLUX

    if spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MAXIMUM_FLUX

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        return RadiationProductKind.MISSION_FLUENCE

    msg = f"Unsupported ERB spectrum quantity: {spectrum.quantity}"
    raise ValueError(msg)


def _default_erb_products(spectra: tuple[Spectrum1D, ...]) -> tuple[SpectrumProduct, ...]:
    return tuple(
        SpectrumProduct(
            kind=_default_erb_product_kind(spectrum),
            spectrum=spectrum,
            label=f"ERB {spectrum.particle.value} {spectrum.quantity.value}",
        )
        for spectrum in spectra
    )


@dataclass(frozen=True)
class ErbModelInput:
    """Input parameters passed to an ERB model."""

    config: CalculationConfig
    solar_activity: MissionSolarActivity | None = None

    def __post_init__(self) -> None:
        if self.solar_activity is None:
            return
        self.solar_activity.validate_for_mission(self.config.mission)

    @property
    def lifetime_years(self) -> int:
        """Return mission lifetime in integer years."""

        return self.config.mission.lifetime_years

    @property
    def kp(self) -> int:
        """Return geomagnetic Kp index."""

        return self.config.kp


@dataclass(frozen=True)
class ErbIntegralSpectrum:
    """Appendix E.4 integral ERB flux spectrum Phi(>E)."""

    energies_mev: tuple[float, ...]
    integral_flux_gt_e: tuple[float, ...]
    particle: Particle
    model: str
    document: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.particle not in ERB_ALLOWED_PARTICLES:
            msg = "ERB integral spectrum must describe protons or electrons."
            raise ValueError(msg)

        if not self.energies_mev:
            msg = "ERB integral spectrum energy grid must not be empty."
            raise ValueError(msg)

        if len(self.energies_mev) != len(self.integral_flux_gt_e):
            msg = "ERB integral spectrum energy and flux grids must have equal length."
            raise ValueError(msg)

        if any(not math.isfinite(energy) or energy <= 0.0 for energy in self.energies_mev):
            msg = "ERB integral spectrum energy grid values must be finite and positive."
            raise ValueError(msg)

        if tuple(sorted(self.energies_mev)) != self.energies_mev:
            msg = "ERB integral spectrum energy grid must be sorted."
            raise ValueError(msg)

        if len(set(self.energies_mev)) != len(self.energies_mev):
            msg = "ERB integral spectrum energy grid values must be unique."
            raise ValueError(msg)

        if any(not math.isfinite(value) or value < 0.0 for value in self.integral_flux_gt_e):
            msg = "ERB integral spectrum flux values must be finite and non-negative."
            raise ValueError(msg)

        if not self.model:
            msg = "ERB integral spectrum model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "ERB integral spectrum document must not be empty."
            raise ValueError(msg)


@dataclass(frozen=True)
class ErbIntegralSpectrumApproximationFit:
    """Appendix E.8 fit attached to an ERB integral spectrum."""

    particle: Particle
    model: str
    document: str
    approximation: OstErbProtonIntegralApproximation | OstErbElectronIntegralApproximation
    quality: OstErbApproximationFitQuality
    label: str = ""

    def __post_init__(self) -> None:
        if self.particle not in ERB_ALLOWED_PARTICLES:
            msg = "ERB integral approximation fit must describe protons or electrons."
            raise ValueError(msg)

        if self.particle is Particle.PROTON and not isinstance(
            self.approximation,
            OstErbProtonIntegralApproximation,
        ):
            msg = "ERB proton integral approximation fit must use a proton approximation."
            raise ValueError(msg)

        if self.particle is Particle.ELECTRON and not isinstance(
            self.approximation,
            OstErbElectronIntegralApproximation,
        ):
            msg = "ERB electron integral approximation fit must use an electron approximation."
            raise ValueError(msg)

        if not self.model:
            msg = "ERB integral approximation fit model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "ERB integral approximation fit document must not be empty."
            raise ValueError(msg)


@dataclass(frozen=True)
class ErbModelResult:
    """Result returned by an ERB model."""

    spectra: tuple[Spectrum1D, ...]
    lifetime_years: int
    kp: int
    model: str
    document: str
    products: tuple[SpectrumProduct, ...] = ()
    integral_spectra: tuple[ErbIntegralSpectrum, ...] = ()
    integral_approximation_fits: tuple[ErbIntegralSpectrumApproximationFit, ...] = ()
    method_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "ERB model result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

        products = self.products or _default_erb_products(self.spectra)

        if not products:
            msg = "ERB model result must contain at least one radiation product."
            raise ValueError(msg)

        for product in products:
            validate_product_allowed_for_source(
                product=product,
                source=RadiationSource.ERB,
            )

        if tuple(product.spectrum for product in products) != self.spectra:
            msg = "ERB model product spectra must match result spectra."
            raise ValueError(msg)

        integral_particles = {spectrum.particle for spectrum in self.integral_spectra}

        for integral_spectrum in self.integral_spectra:
            if integral_spectrum.document != self.document:
                msg = "ERB integral spectrum document must match result document."
                raise ValueError(msg)

        for approximation_fit in self.integral_approximation_fits:
            if approximation_fit.document != self.document:
                msg = "ERB integral approximation fit document must match result document."
                raise ValueError(msg)

            if approximation_fit.particle not in integral_particles:
                msg = "ERB integral approximation fit particle must match an integral spectrum."
                raise ValueError(msg)

        for key, value in self.method_metadata:
            if not key:
                msg = "ERB model result method metadata key must not be empty."
                raise ValueError(msg)

            if not value:
                msg = "ERB model result method metadata value must not be empty."
                raise ValueError(msg)

        object.__setattr__(self, "products", products)

        if not isinstance(self.lifetime_years, int):
            msg = "ERB result lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < 1:
            msg = "ERB result lifetime must be at least one year."
            raise ValueError(msg)

        if not isinstance(self.kp, int):
            msg = "ERB result Kp must be an integer."
            raise ValueError(msg)

        if not 0 <= self.kp <= 9:
            msg = "ERB result Kp must be in the range 0..9."
            raise ValueError(msg)

        if not self.model:
            msg = "ERB model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "ERB source document must not be empty."
            raise ValueError(msg)


class ErbModelProtocol(Protocol):
    """Protocol implemented by concrete ERB models."""

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""
        ...

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Calculate mission ERB spectra."""


@dataclass(frozen=True)
class StaticErbModel:
    """Simple deterministic ERB model used for tests and integration plumbing.

    Normative ERB models must be implemented separately.
    Dose and bremsstrahlung calculations must be implemented separately.
    """

    spectra: tuple[Spectrum1D, ...]
    model: str = "static_erb_model"
    document: str = "test"
    model_family: SourceModelFamily = SourceModelFamily.CUSTOM

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.ERB,
            model_family=self.model_family,
            name=self.model,
            document=self.document,
        )

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "Static ERB model must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

        if not self.model:
            msg = "ERB model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "ERB source document must not be empty."
            raise ValueError(msg)

        _ = self.metadata

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Return configured test ERB spectra."""

        return ErbModelResult(
            spectra=self.spectra,
            lifetime_years=model_input.lifetime_years,
            kp=model_input.kp,
            model=self.model,
            document=self.document,
        )


OstErbPeakState = Literal["worst", "cycle", "maxofstates"]
Ae8Ap8SolarMode = Literal["cosrad", "ost"]


def _clamp_unit_interval(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _ost_erb_solar_max_fraction(
    *,
    mission: MissionConfig,
    reference_start_year: int,
    solar_activity: MissionSolarActivity | None = None,
) -> float:
    cycle_table = ost_wolf_number_cycle_table(mission.solar_activity_level)
    if solar_activity is None:
        solar_activity = build_mission_solar_activity(
            mission=mission,
            cycle_table=cycle_table,
            reference_start_year=reference_start_year,
        )
    monthly_wolf_numbers = mission_solar_activity_monthly_wolf_numbers(solar_activity)

    mean_wolf_number = sum(monthly_wolf_numbers) / len(monthly_wolf_numbers)
    minimum = cycle_table.wolf_numbers[0]
    maximum = max(cycle_table.wolf_numbers)

    if maximum <= minimum:
        return 0.0

    return _clamp_unit_interval((mean_wolf_number - minimum) / (maximum - minimum))


def _ae8_ap8_solar_max_fraction(
    *,
    mission: MissionConfig,
    solar_mode: Ae8Ap8SolarMode,
    reference_start_year: int,
    solar_activity: MissionSolarActivity | None = None,
) -> float:
    if solar_mode == "ost":
        return _ost_erb_solar_max_fraction(
            mission=mission,
            reference_start_year=reference_start_year,
            solar_activity=solar_activity,
        )

    if mission.solar_activity_level is SolarActivityLevel.MINIMUM:
        return 0.0

    if mission.solar_activity_level is SolarActivityLevel.MAXIMUM:
        return 1.0

    return 0.5


def _erb_solar_activity(
    *,
    model_input: ErbModelInput,
    configured: MissionSolarActivity | None,
    reference_start_year_override: int | None,
) -> MissionSolarActivity:
    """Resolve the shared timeline, retaining the legacy explicit-anchor hook."""

    if reference_start_year_override is not None:
        methodology = model_input.config.methodology
        return build_mission_solar_activity(
            mission=model_input.config.mission,
            cycle_table=ost_wolf_number_cycle_table(
                model_input.config.mission.solar_activity_level,
            ),
            reference_start_year=reference_start_year_override,
            reference_cycle_number=methodology.solar_cycle_reference_number,
            wolf_number_version=methodology.solar_activity_wolf_version,
            monthly_policy=methodology.solar_activity_monthly_policy,
            anchor_basis=("explicit ERB model override; not specified by OST 134-1044-2007"),
        )

    solar_activity = configured or model_input.solar_activity
    if solar_activity is None:
        solar_activity = build_calculation_solar_activity(model_input.config)

    solar_activity.validate_for_mission(model_input.config.mission)
    return solar_activity


def _field_line_coverage_metadata(
    *,
    prefix: str,
    field_line_samples: tuple[ErbFieldLineSample, ...],
) -> tuple[tuple[str, str], ...]:
    total_samples = len(field_line_samples)
    valid_samples = sum(1 for sample in field_line_samples if sample.valid)
    invalid_samples = total_samples - valid_samples

    return (
        (f"{prefix}_total_samples", str(total_samples)),
        (f"{prefix}_valid_samples", str(valid_samples)),
        (f"{prefix}_invalid_samples", str(invalid_samples)),
        (
            f"{prefix}_valid_sample_fraction",
            _sample_fraction(count=valid_samples, total=total_samples),
        ),
    )


def _weighted_vector(
    *,
    minimum: tuple[float, ...],
    maximum: tuple[float, ...],
    fraction_max: float,
) -> tuple[float, ...]:
    return tuple(
        (1.0 - fraction_max) * min_value + fraction_max * max_value
        for min_value, max_value in zip(minimum, maximum, strict=True)
    )


def _max_vector(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        max(left_value, right_value) for left_value, right_value in zip(left, right, strict=True)
    )


def _add_to_vector(
    accumulator: list[float],
    values: tuple[float, ...],
) -> None:
    for index, value in enumerate(values):
        accumulator[index] += value


def _divide_vector(
    values: list[float],
    divisor: int,
) -> tuple[float, ...]:
    if divisor < 1:
        return tuple(0.0 for _ in values)

    return tuple(value / divisor for value in values)


def _multiply_vector(
    values: tuple[float, ...],
    factor: float,
) -> tuple[float, ...]:
    return tuple(value * factor for value in values)


def _sample_fraction(
    *,
    count: int,
    total: int,
) -> str:
    if total < 1:
        return "0"

    return f"{count / total:.12g}"


@dataclass(frozen=True)
class _OstErbTableCoverage:
    total_samples: int
    valid_samples: int
    invalid_samples: int
    l_below_table: int
    l_above_table: int
    bb0_above_nearest_shell: int
    bb0_above_interpolation_stencil: int
    bb0_above_interpolation_stencil_shells: int

    @property
    def unsupported_valid_samples(self) -> int:
        return (
            self.l_below_table
            + self.l_above_table
            + self.bb0_above_interpolation_stencil
        )

    @property
    def support_complete(self) -> bool:
        return self.invalid_samples == 0 and self.unsupported_valid_samples == 0


def _table_coverage(
    *,
    table: OstErbFluxTable,
    field_line_samples: tuple[ErbFieldLineSample, ...],
) -> _OstErbTableCoverage:
    total_samples = len(field_line_samples)
    valid_samples = sum(1 for sample in field_line_samples if sample.valid)
    invalid_samples = total_samples - valid_samples
    l_below_table = 0
    l_above_table = 0
    bb0_above_nearest_shell = 0
    bb0_above_interpolation_stencil = 0
    bb0_above_interpolation_stencil_shells = 0

    for sample in field_line_samples:
        if not sample.valid:
            continue

        support = table.interpolation_support(
            l_shell=sample.l_shell,
            b_over_b0=sample.b_over_b0,
        )
        if support.l_below_table:
            l_below_table += 1
            continue
        if support.l_above_table:
            l_above_table += 1
            continue

        nearest_shell = min(
            table.shells,
            key=lambda shell: abs(shell.l_shell - sample.l_shell),
        )
        if sample.b_over_b0 > nearest_shell.b_over_b0[-1] * (1.0 + 1.0e-12):
            bb0_above_nearest_shell += 1

        if support.b_over_b0_above_shell_indices:
            bb0_above_interpolation_stencil += 1
            bb0_above_interpolation_stencil_shells += len(
                support.b_over_b0_above_shell_indices
            )

    return _OstErbTableCoverage(
        total_samples=total_samples,
        valid_samples=valid_samples,
        invalid_samples=invalid_samples,
        l_below_table=l_below_table,
        l_above_table=l_above_table,
        bb0_above_nearest_shell=bb0_above_nearest_shell,
        bb0_above_interpolation_stencil=bb0_above_interpolation_stencil,
        bb0_above_interpolation_stencil_shells=(
            bb0_above_interpolation_stencil_shells
        ),
    )


def _table_coverage_metadata(
    *,
    prefix: str,
    table: OstErbFluxTable,
    field_line_samples: tuple[ErbFieldLineSample, ...],
) -> tuple[tuple[str, str], ...]:
    coverage = _table_coverage(table=table, field_line_samples=field_line_samples)
    valid_divisor = max(coverage.valid_samples, 1)

    return (
        (f"{prefix}_total_samples", str(coverage.total_samples)),
        (f"{prefix}_valid_samples", str(coverage.valid_samples)),
        (f"{prefix}_invalid_samples", str(coverage.invalid_samples)),
        (f"{prefix}_l_below_table_samples", str(coverage.l_below_table)),
        (f"{prefix}_l_above_table_samples", str(coverage.l_above_table)),
        (
            f"{prefix}_bb0_above_nearest_shell_samples",
            str(coverage.bb0_above_nearest_shell),
        ),
        (
            f"{prefix}_bb0_above_interpolation_stencil_samples",
            str(coverage.bb0_above_interpolation_stencil),
        ),
        (
            f"{prefix}_bb0_above_interpolation_stencil_shells",
            str(coverage.bb0_above_interpolation_stencil_shells),
        ),
        (
            f"{prefix}_unsupported_valid_samples",
            str(coverage.unsupported_valid_samples),
        ),
        (
            f"{prefix}_valid_sample_fraction",
            _sample_fraction(
                count=coverage.valid_samples,
                total=coverage.total_samples,
            ),
        ),
        (
            f"{prefix}_l_below_table_fraction",
            _sample_fraction(count=coverage.l_below_table, total=valid_divisor),
        ),
        (
            f"{prefix}_l_above_table_fraction",
            _sample_fraction(count=coverage.l_above_table, total=valid_divisor),
        ),
        (
            f"{prefix}_bb0_above_nearest_shell_fraction",
            _sample_fraction(
                count=coverage.bb0_above_nearest_shell,
                total=valid_divisor,
            ),
        ),
        (
            f"{prefix}_bb0_above_interpolation_stencil_fraction",
            _sample_fraction(
                count=coverage.bb0_above_interpolation_stencil,
                total=valid_divisor,
            ),
        ),
        (f"{prefix}_support_complete", str(coverage.support_complete).lower()),
        (f"{prefix}_table_l_min", f"{table.l_grid[0]:.12g}"),
        (f"{prefix}_table_l_max", f"{table.l_grid[-1]:.12g}"),
    )


def _table_support_reason(
    *,
    table: OstErbFluxTable,
    coverage: _OstErbTableCoverage,
) -> str | None:
    if coverage.unsupported_valid_samples == 0:
        return None

    return (
        f"OST ERB Appendix A table {table.name} interpolation support is "
        f"incomplete for {coverage.unsupported_valid_samples}/"
        f"{coverage.valid_samples} valid orbit samples: "
        f"L below={coverage.l_below_table}, L above={coverage.l_above_table}, "
        "B/B0 above at least one selected interpolation shell="
        f"{coverage.bb0_above_interpolation_stencil}"
    )


def _ost_erb_product_incomplete_reasons(
    *,
    tables: tuple[OstErbFluxTable, ...],
    field_line_samples: tuple[ErbFieldLineSample, ...],
    common_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reasons = [
        *common_reasons,
        OST_ERB_ORBIT_SAMPLING_CONVERGENCE_REASON,
        OST_ERB_ORBIT_TRAJECTORY_REASON,
        OST_ERB_MAGNETIC_REGION_SELECTION_REASON,
    ]
    reasons.extend(reason for sample in field_line_samples for reason in sample.incomplete_reasons)
    invalid_samples = sum(1 for sample in field_line_samples if not sample.valid)
    if invalid_samples:
        reasons.append(
            "OST ERB magnetic-coordinate tracing is incomplete for "
            f"{invalid_samples}/{len(field_line_samples)} orbit samples"
        )

    for table in dict.fromkeys(tables):
        reason = _table_support_reason(
            table=table,
            coverage=_table_coverage(
                table=table,
                field_line_samples=field_line_samples,
            ),
        )
        if reason is not None:
            reasons.append(reason)

    return tuple(dict.fromkeys(reasons))


def _erb_spectrum(
    *,
    energies_mev: tuple[float, ...],
    values: tuple[float, ...],
    quantity: SpectrumQuantity,
    y_unit: Unit,
    particle: Particle,
    model: str,
    product_kind: RadiationProductKind,
    mission_seconds: float,
    version: str,
    document_refs: tuple[str, ...],
    dataset_refs: tuple[str, ...],
    transformation_name: str,
    completeness: SpectrumCompleteness,
    incomplete_reasons: tuple[str, ...],
) -> Spectrum1D:
    if product_kind is RadiationProductKind.MEAN_FLUX:
        temporal_basis = TemporalBasis.MEAN_INTERNAL
        assessment_mode = None
        mission_duration_s = None
    elif product_kind is RadiationProductKind.MAXIMUM_FLUX:
        temporal_basis = TemporalBasis.SOURCE_MAXIMUM
        assessment_mode = AssessmentMode.PEAK
        mission_duration_s = None
    elif product_kind is RadiationProductKind.MISSION_FLUENCE:
        temporal_basis = TemporalBasis.MISSION_TOTAL
        assessment_mode = AssessmentMode.CUMULATIVE
        mission_duration_s = mission_seconds
    else:
        msg = f"Unsupported ERB product kind for typed spectrum: {product_kind.value}."
        raise ValueError(msg)

    species = (
        ParticleSpecies(symbol="p", atomic_number=1, mass_number=1)
        if particle is Particle.PROTON
        else ParticleSpecies(symbol="e-")
    )
    return Spectrum1D(
        x=energies_mev,
        y=values,
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.ERB,
        model=model,
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=temporal_basis,
        location=SpectrumLocation.ON_ORBIT,
        species=species,
        assessment_mode=assessment_mode,
        mission_duration_s=mission_duration_s,
        model_reference=ModelReference(
            name=model,
            version=version,
            document_refs=document_refs,
            dataset_refs=dataset_refs,
        ),
        transformation_chain=(
            TransformationStep(
                name=transformation_name,
                version=version,
                parameters=(
                    ("particle", particle.value),
                    ("product_kind", product_kind.value),
                    ("angular_basis", "omnidirectional_4pi"),
                ),
            ),
        ),
        completeness=completeness,
        incomplete_reasons=incomplete_reasons,
    )


def _erb_product(
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


def _positive_integral_fit_grid(
    integral_spectrum: ErbIntegralSpectrum,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    positive_pairs = tuple(
        (energy, value)
        for energy, value in zip(
            integral_spectrum.energies_mev,
            integral_spectrum.integral_flux_gt_e,
            strict=True,
        )
        if value > 0.0
    )

    return (
        tuple(energy for energy, _ in positive_pairs),
        tuple(value for _, value in positive_pairs),
    )


def _particle_integral_approximation_fits(
    *,
    particle: Particle,
    integral_spectrum: ErbIntegralSpectrum,
    model: str,
    document: str,
    label: str,
) -> tuple[ErbIntegralSpectrumApproximationFit, ...]:
    energies_mev, integral_flux_gt_e = _positive_integral_fit_grid(integral_spectrum)

    if len(energies_mev) < 2:
        return ()

    if particle is Particle.PROTON:
        proton_fit = fit_ost_erb_proton_integral_approximation(
            energies_mev=energies_mev,
            integral_flux_gt_e=integral_flux_gt_e,
        )

        return (
            ErbIntegralSpectrumApproximationFit(
                particle=particle,
                model=model,
                document=document,
                approximation=proton_fit.approximation,
                quality=proton_fit.quality,
                label=label,
            ),
        )

    if particle is Particle.ELECTRON:
        electron_fit = fit_ost_erb_electron_integral_approximation(
            energies_mev=energies_mev,
            integral_flux_gt_e=integral_flux_gt_e,
        )

        return (
            ErbIntegralSpectrumApproximationFit(
                particle=particle,
                model=model,
                document=document,
                approximation=electron_fit.approximation,
                quality=electron_fit.quality,
                label=label,
            ),
        )

    msg = "ERB integral approximation fit particle must be proton or electron."
    raise ValueError(msg)


def _appendix_e_fit_quality_metadata(
    *,
    prefix: str,
    integral_spectrum: ErbIntegralSpectrum,
    approximation_fits: tuple[ErbIntegralSpectrumApproximationFit, ...],
) -> tuple[tuple[str, str], ...]:
    positive_sample_count = sum(1 for value in integral_spectrum.integral_flux_gt_e if value > 0.0)

    if not approximation_fits:
        return (
            (
                f"{prefix}_appendix_e_fit_status",
                "skipped_insufficient_positive_integral_flux_values",
            ),
            (f"{prefix}_appendix_e_fit_positive_samples", str(positive_sample_count)),
        )

    quality = approximation_fits[0].quality
    status = "success" if quality.success else "solver_reported_failure"

    return (
        (f"{prefix}_appendix_e_fit_status", status),
        (f"{prefix}_appendix_e_fit_positive_samples", str(positive_sample_count)),
        (f"{prefix}_appendix_e_fit_sample_count", str(quality.sample_count)),
        (f"{prefix}_appendix_e_fit_objective", f"{quality.objective:.12g}"),
        (
            f"{prefix}_appendix_e_fit_rms_relative_error",
            f"{quality.rms_relative_error:.12g}",
        ),
        (
            f"{prefix}_appendix_e_fit_max_abs_relative_error",
            f"{quality.max_abs_relative_error:.12g}",
        ),
        (f"{prefix}_appendix_e_fit_success", str(quality.success).lower()),
    )


@dataclass(frozen=True)
class Ae8Ap8ErbModel:
    """AE8/AP8 RADBELT ERB source model using bundled NASA/NSSDC ASC maps."""

    model: str = "ae8_ap8_radbelt_model"
    document: str = AE8_AP8_RADBELT_REFERENCE
    version: str = "ae8_ap8_radbelt_asc_v2"
    solar_mode: Ae8Ap8SolarMode = "cosrad"
    peak_state: OstErbPeakState = "worst"
    anomaly_samples: int = 48
    node_samples: int = 36
    igrf_epoch: float = DEFAULT_IGRF_EPOCH
    solar_reference_start_year: int | None = None
    solar_activity: MissionSolarActivity | None = None
    maps: dict[str, RadbeltAscMap] | None = None
    igrf_coefficients: IgrfCoefficients | None = None

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.ERB,
            model_family=SourceModelFamily.AE8_AP8,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        if not self.model:
            msg = "AE8/AP8 ERB model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "AE8/AP8 ERB model document must not be empty."
            raise ValueError(msg)

        if not self.version:
            msg = "AE8/AP8 ERB model version must not be empty."
            raise ValueError(msg)

        if self.solar_mode not in ("cosrad", "ost"):
            msg = "AE8/AP8 ERB solar_mode must be cosrad or ost."
            raise ValueError(msg)

        if self.peak_state not in ("worst", "cycle", "maxofstates"):
            msg = "AE8/AP8 ERB peak_state must be worst, cycle, or maxofstates."
            raise ValueError(msg)

        if self.anomaly_samples < 1:
            msg = "AE8/AP8 ERB anomaly sample count must be positive."
            raise ValueError(msg)

        if self.node_samples < 1:
            msg = "AE8/AP8 ERB node sample count must be positive."
            raise ValueError(msg)

        if not math.isfinite(self.igrf_epoch):
            msg = "AE8/AP8 ERB IGRF epoch must be finite."
            raise ValueError(msg)

        if self.solar_reference_start_year is not None and not isinstance(
            self.solar_reference_start_year,
            int,
        ):
            msg = "AE8/AP8 ERB solar reference start year must be an integer."
            raise ValueError(msg)

        if self.maps is not None:
            expected_maps = {"AE8MAX", "AE8MIN", "AP8MAX", "AP8MIN"}
            if set(self.maps) != expected_maps:
                msg = "AE8/AP8 ERB maps must contain AE8MAX, AE8MIN, AP8MAX, and AP8MIN."
                raise ValueError(msg)

        _ = self.metadata

    def _maps(self) -> dict[str, RadbeltAscMap]:
        return self.maps or load_ae8_ap8_radbelt_maps()

    def _igrf_coefficients(self) -> IgrfCoefficients:
        return self.igrf_coefficients or load_igrf14_coefficients(epoch=self.igrf_epoch)

    def _solar_reference_start_year(self, model_input: ErbModelInput) -> int:
        if self.solar_reference_start_year is not None:
            return self.solar_reference_start_year

        return model_input.config.mission.launch_year

    def _field_line_samples(
        self,
        model_input: ErbModelInput,
    ) -> tuple[ErbFieldLineSample, ...]:
        orbit = model_input.config.orbit
        grid = sample_erb_orbit_grid(
            perigee_km=orbit.perigee_altitude_km,
            apogee_km=orbit.apogee_altitude_km,
            inclination_deg=orbit.inclination_deg,
            argument_of_perigee_deg=orbit.argument_of_perigee_deg,
            anomaly_samples=self.anomaly_samples,
            node_samples=self.node_samples,
        )

        return trace_l_shell_b_over_b0_for_orbit(
            grid.samples,
            self._igrf_coefficients(),
        )

    def _particle_products(
        self,
        *,
        particle: Particle,
        energies_mev: tuple[float, ...],
        minimum_map: RadbeltAscMap,
        maximum_map: RadbeltAscMap,
        worst_map: RadbeltAscMap,
        field_line_samples: tuple[ErbFieldLineSample, ...],
        fraction_max: float,
        mission_seconds: float,
    ) -> tuple[tuple[SpectrumProduct, ...], tuple[ErbIntegralSpectrum, ...]]:
        energy_count = len(energies_mev)
        mean_accumulator = [0.0 for _ in range(energy_count)]
        maximum_flux = tuple(0.0 for _ in range(energy_count))
        valid_count = 0

        for field_line_sample in field_line_samples:
            if not field_line_sample.valid:
                continue

            minimum_flux = minimum_map.differential_flux_bins(
                l_shell=field_line_sample.l_shell,
                b_over_b0=field_line_sample.b_over_b0,
                lower_energies_mev=energies_mev,
            )
            maximum_state_flux = maximum_map.differential_flux_bins(
                l_shell=field_line_sample.l_shell,
                b_over_b0=field_line_sample.b_over_b0,
                lower_energies_mev=energies_mev,
            )
            cycle_flux = _weighted_vector(
                minimum=minimum_flux,
                maximum=maximum_state_flux,
                fraction_max=fraction_max,
            )

            _add_to_vector(mean_accumulator, cycle_flux)

            if self.peak_state == "cycle":
                peak_flux = cycle_flux
            elif self.peak_state == "worst":
                peak_flux = worst_map.differential_flux_bins(
                    l_shell=field_line_sample.l_shell,
                    b_over_b0=field_line_sample.b_over_b0,
                    lower_energies_mev=energies_mev,
                )
            else:
                peak_flux = _max_vector(minimum_flux, maximum_state_flux)

            maximum_flux = _max_vector(maximum_flux, peak_flux)
            valid_count += 1

        mean_flux = _divide_vector(mean_accumulator, valid_count)
        fluence = _multiply_vector(mean_flux, mission_seconds)

        particle_label = particle.value
        model_prefix = f"{self.model}:{particle_label}"
        incomplete_reasons = [AE8_AP8_PROVISIONAL_REASON]
        if self.maps is not None:
            incomplete_reasons.append(
                "caller-supplied RADBELT maps are not tied to the bundled integrity manifest"
            )
        if self.igrf_coefficients is not None:
            incomplete_reasons.append(
                "caller-supplied IGRF coefficients are not tied to the bundled integrity manifest"
            )
        dataset_refs = (
            *AE8_AP8_MAP_DATASET_REFS[particle],
            ERB_IGRF_DATASET_REF,
            ERB_WOLF_DATASET_REF,
        )

        mean_spectrum = _erb_spectrum(
            energies_mev=energies_mev,
            values=mean_flux,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            particle=particle,
            model=f"{model_prefix}:mean_flux",
            product_kind=RadiationProductKind.MEAN_FLUX,
            mission_seconds=mission_seconds,
            version=self.version,
            document_refs=(self.document,),
            dataset_refs=dataset_refs,
            transformation_name="ae8_ap8_orbit_spectrum",
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=tuple(incomplete_reasons),
        )
        maximum_spectrum = _erb_spectrum(
            energies_mev=energies_mev,
            values=maximum_flux,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            particle=particle,
            model=f"{model_prefix}:maximum_flux",
            product_kind=RadiationProductKind.MAXIMUM_FLUX,
            mission_seconds=mission_seconds,
            version=self.version,
            document_refs=(self.document,),
            dataset_refs=dataset_refs,
            transformation_name="ae8_ap8_orbit_spectrum",
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=tuple(incomplete_reasons),
        )
        fluence_spectrum = _erb_spectrum(
            energies_mev=energies_mev,
            values=fluence,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            particle=particle,
            model=f"{model_prefix}:mission_fluence",
            product_kind=RadiationProductKind.MISSION_FLUENCE,
            mission_seconds=mission_seconds,
            version=self.version,
            document_refs=(self.document,),
            dataset_refs=dataset_refs,
            transformation_name="ae8_ap8_orbit_spectrum",
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=tuple(incomplete_reasons),
        )
        mean_integral_spectrum = ErbIntegralSpectrum(
            energies_mev=energies_mev,
            integral_flux_gt_e=integrate_differential_flux_tail_power_law(
                energies_mev=energies_mev,
                differential_flux=mean_flux,
            ),
            particle=particle,
            model=f"{model_prefix}:mean_integral_flux",
            document=self.document,
            label=f"AE8/AP8 RADBELT {particle_label} mean integral flux",
        )

        return (
            (
                _erb_product(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=mean_spectrum,
                    label=f"AE8/AP8 RADBELT {particle_label} mean flux",
                ),
                _erb_product(
                    kind=RadiationProductKind.MAXIMUM_FLUX,
                    spectrum=maximum_spectrum,
                    label=f"AE8/AP8 RADBELT {particle_label} maximum flux",
                ),
                _erb_product(
                    kind=RadiationProductKind.MISSION_FLUENCE,
                    spectrum=fluence_spectrum,
                    label=f"AE8/AP8 RADBELT {particle_label} mission fluence",
                ),
            ),
            (mean_integral_spectrum,),
        )

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Calculate AE8/AP8 RADBELT proton and electron ERB spectra."""

        maps = self._maps()
        field_line_samples = self._field_line_samples(model_input)

        solar_activity = _erb_solar_activity(
            model_input=model_input,
            configured=self.solar_activity,
            reference_start_year_override=self.solar_reference_start_year,
        )
        solar_reference_start_year = solar_activity.reference_start_year
        fraction_max = _ae8_ap8_solar_max_fraction(
            mission=model_input.config.mission,
            solar_mode=self.solar_mode,
            reference_start_year=solar_reference_start_year,
            solar_activity=solar_activity,
        )
        mission_seconds = model_input.config.mission.lifetime_years * ERB_SECONDS_PER_YEAR

        proton_products, proton_integral_spectra = self._particle_products(
            particle=Particle.PROTON,
            energies_mev=AE8_AP8_PROTON_ENERGIES_MEV,
            minimum_map=maps["AP8MIN"],
            maximum_map=maps["AP8MAX"],
            worst_map=maps["AP8MIN"],
            field_line_samples=field_line_samples,
            fraction_max=fraction_max,
            mission_seconds=mission_seconds,
        )
        electron_products, electron_integral_spectra = self._particle_products(
            particle=Particle.ELECTRON,
            energies_mev=AE8_AP8_ELECTRON_ENERGIES_MEV,
            minimum_map=maps["AE8MIN"],
            maximum_map=maps["AE8MAX"],
            worst_map=maps["AE8MAX"],
            field_line_samples=field_line_samples,
            fraction_max=fraction_max,
            mission_seconds=mission_seconds,
        )

        products = (
            *proton_products,
            *electron_products,
        )
        integral_spectra = (
            *proton_integral_spectra,
            *electron_integral_spectra,
        )
        spectra = tuple(product.spectrum for product in products)
        field_line_metadata = _field_line_coverage_metadata(
            prefix="radbelt",
            field_line_samples=field_line_samples,
        )

        return ErbModelResult(
            spectra=spectra,
            lifetime_years=model_input.lifetime_years,
            kp=model_input.kp,
            model=self.model,
            document=self.document,
            products=products,
            integral_spectra=integral_spectra,
            method_metadata=(
                ("ae8_ap8_reference", AE8_AP8_RADBELT_REFERENCE),
                ("radbelt_maps", "AP8MIN,AP8MAX,AE8MIN,AE8MAX"),
                (
                    "radbelt_differential_flux",
                    "integral_bin_difference_over_energy_width",
                ),
                ("radbelt_energy_grid", "lower_bin_edges_10_per_decade"),
                ("magnetic_coordinate_solver", "igrf_rk4_refined_minimum_b_polar_limit_v2"),
                ("solar_mode", self.solar_mode),
                ("solar_reference_start_year", str(solar_reference_start_year)),
                ("solar_fraction_max", f"{fraction_max:.12g}"),
                ("anomaly_samples", str(self.anomaly_samples)),
                ("node_samples", str(self.node_samples)),
                ("igrf_epoch", f"{self._igrf_coefficients().epoch:g}"),
                ("igrf_configured_epoch", f"{self.igrf_epoch:g}"),
                ("peak_state", self.peak_state),
                ("radbelt_log_bin_ratio", f"{RADBELT_LOG_BIN_RATIO:.12g}"),
                *solar_activity.metadata,
                *field_line_metadata,
            ),
        )


@dataclass(frozen=True)
class OstErbModel:
    """Normative OST 134-1044-2007 Appendix A/D/E ERB source model."""

    model: str = "ost_erb_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "ost_appendix_a_v3"
    interpolation: OstErbInterpolationMode = OstErbInterpolationMode.STRICT
    peak_state: OstErbPeakState = "worst"
    anomaly_samples: int = 48
    node_samples: int = 36
    igrf_epoch: float = DEFAULT_IGRF_EPOCH
    solar_reference_start_year: int | None = None
    solar_activity: MissionSolarActivity | None = None
    tables: OstErbTableSet | None = None
    igrf_coefficients: IgrfCoefficients | None = None

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.ERB,
            model_family=SourceModelFamily.OST_134_1044_2007,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        if not self.model:
            msg = "OST ERB model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "OST ERB model document must not be empty."
            raise ValueError(msg)

        if not self.version:
            msg = "OST ERB model version must not be empty."
            raise ValueError(msg)

        if self.peak_state not in ("worst", "cycle", "maxofstates"):
            msg = "OST ERB peak_state must be worst, cycle, or maxofstates."
            raise ValueError(msg)

        if self.anomaly_samples < 1:
            msg = "OST ERB anomaly sample count must be positive."
            raise ValueError(msg)

        if self.node_samples < 1:
            msg = "OST ERB node sample count must be positive."
            raise ValueError(msg)

        if not math.isfinite(self.igrf_epoch):
            msg = "OST ERB IGRF epoch must be finite."
            raise ValueError(msg)

        if self.solar_reference_start_year is not None and not isinstance(
            self.solar_reference_start_year,
            int,
        ):
            msg = "OST ERB solar reference start year must be an integer."
            raise ValueError(msg)

        _ = self.metadata

    def _tables(self) -> OstErbTableSet:
        return self.tables or load_ost_erb_appendix_a_tables()

    def _igrf_coefficients(self) -> IgrfCoefficients:
        return self.igrf_coefficients or load_igrf14_coefficients(epoch=self.igrf_epoch)

    def _solar_reference_start_year(self, model_input: ErbModelInput) -> int:
        if self.solar_reference_start_year is not None:
            return self.solar_reference_start_year

        return model_input.config.mission.launch_year

    def _field_line_samples(
        self,
        model_input: ErbModelInput,
    ) -> tuple[ErbFieldLineSample, ...]:
        orbit = model_input.config.orbit
        grid = sample_erb_orbit_grid(
            perigee_km=orbit.perigee_altitude_km,
            apogee_km=orbit.apogee_altitude_km,
            inclination_deg=orbit.inclination_deg,
            argument_of_perigee_deg=orbit.argument_of_perigee_deg,
            anomaly_samples=self.anomaly_samples,
            node_samples=self.node_samples,
        )

        return ost_l_shell_b_over_b0_for_orbit(
            grid.samples,
            self._igrf_coefficients(),
        )

    def _particle_products(
        self,
        *,
        particle: Particle,
        minimum_table: OstErbFluxTable,
        maximum_table: OstErbFluxTable,
        worst_table: OstErbFluxTable,
        field_line_samples: tuple[ErbFieldLineSample, ...],
        fraction_max: float,
        mission_seconds: float,
    ) -> tuple[
        tuple[SpectrumProduct, ...],
        tuple[ErbIntegralSpectrum, ...],
        tuple[ErbIntegralSpectrumApproximationFit, ...],
    ]:
        if not field_line_samples:
            msg = "OST Appendix E.3 orbit average requires a nonempty time grid."
            raise ValueError(msg)
        energy_count = len(minimum_table.energies_mev)
        mean_accumulator = [0.0 for _ in range(energy_count)]
        maximum_flux = tuple(0.0 for _ in range(energy_count))

        for field_line_sample in field_line_samples:
            if not field_line_sample.valid:
                continue

            minimum_flux = minimum_table.vector(
                l_shell=field_line_sample.l_shell,
                b_over_b0=field_line_sample.b_over_b0,
                mode=self.interpolation,
            )
            maximum_state_flux = maximum_table.vector(
                l_shell=field_line_sample.l_shell,
                b_over_b0=field_line_sample.b_over_b0,
                mode=self.interpolation,
            )
            cycle_flux = _weighted_vector(
                minimum=minimum_flux,
                maximum=maximum_state_flux,
                fraction_max=fraction_max,
            )

            _add_to_vector(mean_accumulator, cycle_flux)

            if self.peak_state == "cycle":
                peak_flux = cycle_flux
            elif self.peak_state == "worst":
                peak_flux = worst_table.vector(
                    l_shell=field_line_sample.l_shell,
                    b_over_b0=field_line_sample.b_over_b0,
                    mode=self.interpolation,
                )
            else:
                peak_flux = _max_vector(minimum_flux, maximum_state_flux)

            maximum_flux = _max_vector(maximum_flux, peak_flux)

        # E.3 divides by the whole sampled duration.  Failed magnetic traces
        # leave an unknown contribution, not a shorter mission.  Retain only
        # the computed contribution over that full duration and mark every
        # product INCOMPLETE below; never renormalize to the valid subset.
        mean_flux = _divide_vector(mean_accumulator, len(field_line_samples))
        fluence = _multiply_vector(mean_flux, mission_seconds)

        particle_label = particle.value
        model_prefix = f"{self.model}:{particle_label}"
        common_incomplete_reasons: list[str] = []
        if self.tables is not None:
            common_incomplete_reasons.append(
                "caller-supplied OST ERB tables are not tied to the bundled integrity manifest"
            )
        if self.igrf_coefficients is not None:
            common_incomplete_reasons.append(
                "caller-supplied IGRF coefficients are not tied to the bundled integrity manifest"
            )
        common_reasons = tuple(common_incomplete_reasons)
        mean_incomplete_reasons = _ost_erb_product_incomplete_reasons(
            tables=(minimum_table, maximum_table),
            field_line_samples=field_line_samples,
            common_reasons=common_reasons,
        )
        peak_tables: tuple[OstErbFluxTable, ...]
        if self.peak_state == "worst":
            peak_tables = (worst_table,)
        else:
            peak_tables = (minimum_table, maximum_table)
        maximum_incomplete_reasons = _ost_erb_product_incomplete_reasons(
            tables=peak_tables,
            field_line_samples=field_line_samples,
            common_reasons=common_reasons,
        )
        dataset_refs = (
            OST_ERB_TABLE_DATASET_REF,
            ERB_IGRF_DATASET_REF,
            ERB_WOLF_DATASET_REF,
        )

        mean_spectrum = _erb_spectrum(
            energies_mev=minimum_table.energies_mev,
            values=mean_flux,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            particle=particle,
            model=f"{model_prefix}:mean_flux",
            product_kind=RadiationProductKind.MEAN_FLUX,
            mission_seconds=mission_seconds,
            version=self.version,
            document_refs=(self.document,),
            dataset_refs=dataset_refs,
            transformation_name="ost_erb_appendix_a_orbit_spectrum",
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=mean_incomplete_reasons,
        )
        maximum_spectrum = _erb_spectrum(
            energies_mev=minimum_table.energies_mev,
            values=maximum_flux,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            particle=particle,
            model=f"{model_prefix}:maximum_flux",
            product_kind=RadiationProductKind.MAXIMUM_FLUX,
            mission_seconds=mission_seconds,
            version=self.version,
            document_refs=(self.document,),
            dataset_refs=dataset_refs,
            transformation_name="ost_erb_appendix_a_orbit_spectrum",
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=maximum_incomplete_reasons,
        )
        fluence_spectrum = _erb_spectrum(
            energies_mev=minimum_table.energies_mev,
            values=fluence,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            particle=particle,
            model=f"{model_prefix}:mission_fluence",
            product_kind=RadiationProductKind.MISSION_FLUENCE,
            mission_seconds=mission_seconds,
            version=self.version,
            document_refs=(self.document,),
            dataset_refs=dataset_refs,
            transformation_name="ost_erb_appendix_a_orbit_spectrum",
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=mean_incomplete_reasons,
        )
        mean_integral_spectrum = ErbIntegralSpectrum(
            energies_mev=minimum_table.energies_mev,
            integral_flux_gt_e=integrate_differential_flux_tail_power_law(
                energies_mev=minimum_table.energies_mev,
                differential_flux=mean_flux,
            ),
            particle=particle,
            model=f"{model_prefix}:mean_integral_flux",
            document=self.document,
            label=f"OST ERB {particle_label} mean integral flux",
        )
        integral_approximation_fits = _particle_integral_approximation_fits(
            particle=particle,
            integral_spectrum=mean_integral_spectrum,
            model=f"{model_prefix}:mean_integral_flux_fit",
            document=self.document,
            label=f"OST ERB {particle_label} mean integral flux approximation fit",
        )

        return (
            (
                _erb_product(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=mean_spectrum,
                    label=f"OST ERB {particle_label} mean flux",
                ),
                _erb_product(
                    kind=RadiationProductKind.MAXIMUM_FLUX,
                    spectrum=maximum_spectrum,
                    label=f"OST ERB {particle_label} maximum flux",
                ),
                _erb_product(
                    kind=RadiationProductKind.MISSION_FLUENCE,
                    spectrum=fluence_spectrum,
                    label=f"OST ERB {particle_label} mission fluence",
                ),
            ),
            (mean_integral_spectrum,),
            integral_approximation_fits,
        )

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Calculate OST Appendix A proton and electron ERB spectra."""

        tables = self._tables()
        field_line_samples = self._field_line_samples(model_input)

        solar_activity = _erb_solar_activity(
            model_input=model_input,
            configured=self.solar_activity,
            reference_start_year_override=self.solar_reference_start_year,
        )
        solar_reference_start_year = solar_activity.reference_start_year
        fraction_max = _ost_erb_solar_max_fraction(
            mission=model_input.config.mission,
            reference_start_year=solar_reference_start_year,
            solar_activity=solar_activity,
        )
        mission_seconds = model_input.config.mission.lifetime_years * ERB_SECONDS_PER_YEAR

        (
            proton_products,
            proton_integral_spectra,
            proton_integral_approximation_fits,
        ) = self._particle_products(
            particle=Particle.PROTON,
            minimum_table=tables.proton_min,
            maximum_table=tables.proton_max,
            worst_table=tables.proton_min,
            field_line_samples=field_line_samples,
            fraction_max=fraction_max,
            mission_seconds=mission_seconds,
        )
        (
            electron_products,
            electron_integral_spectra,
            electron_integral_approximation_fits,
        ) = self._particle_products(
            particle=Particle.ELECTRON,
            minimum_table=tables.electron_min,
            maximum_table=tables.electron_max,
            worst_table=tables.electron_max,
            field_line_samples=field_line_samples,
            fraction_max=fraction_max,
            mission_seconds=mission_seconds,
        )

        products = (
            *proton_products,
            *electron_products,
        )
        integral_spectra = (
            *proton_integral_spectra,
            *electron_integral_spectra,
        )
        integral_approximation_fits = (
            *proton_integral_approximation_fits,
            *electron_integral_approximation_fits,
        )
        spectra = tuple(product.spectrum for product in products)
        proton_coverage_metadata = _table_coverage_metadata(
            prefix="proton",
            table=tables.proton_min,
            field_line_samples=field_line_samples,
        )
        electron_coverage_metadata = _table_coverage_metadata(
            prefix="electron",
            table=tables.electron_min,
            field_line_samples=field_line_samples,
        )
        proton_max_coverage_metadata = _table_coverage_metadata(
            prefix="proton_max",
            table=tables.proton_max,
            field_line_samples=field_line_samples,
        )
        electron_max_coverage_metadata = _table_coverage_metadata(
            prefix="electron_max",
            table=tables.electron_max,
            field_line_samples=field_line_samples,
        )
        proton_fit_metadata = _appendix_e_fit_quality_metadata(
            prefix="proton",
            integral_spectrum=proton_integral_spectra[0],
            approximation_fits=proton_integral_approximation_fits,
        )
        electron_fit_metadata = _appendix_e_fit_quality_metadata(
            prefix="electron",
            integral_spectrum=electron_integral_spectra[0],
            approximation_fits=electron_integral_approximation_fits,
        )

        return ErbModelResult(
            spectra=spectra,
            lifetime_years=model_input.lifetime_years,
            kp=model_input.kp,
            model=self.model,
            document=self.document,
            products=products,
            integral_spectra=integral_spectra,
            integral_approximation_fits=integral_approximation_fits,
            method_metadata=(
                ("ost_appendices", "A,D,E"),
                ("appendix_a_reference", OST_ERB_APPENDIX_A_REFERENCE),
                ("appendix_d_reference", OST_ERB_APPENDIX_D_REFERENCE),
                ("appendix_d_inner_field", "igrf_rk4_minimum_b_then_d32"),
                ("magnetic_coordinate_solver", "ost_two_region_igrf_rk4_shifted_dipole_v3"),
                ("appendix_d_outer_field", "shifted_dipole_d21_d24"),
                ("appendix_d_region_selection", "L_shifted_dipole_gt_5.5_outer_else_inner"),
                ("appendix_d_boundary_policy_status", "implementation_convention_pending_validation"),
                ("appendix_d_outer_igrf_crosscheck_band", "5_le_L_shifted_dipole_le_6"),
                ("appendix_d_boundary_smoothing", "none"),
                ("appendix_d_outer_moment_scale", "1_reference_convention_cancels_in_B_over_B0"),
                ("appendix_d_igrf_synthesis", "negative_gradient_of_IGRF_potential"),
                ("appendix_d_igrf_formula_status", "printed_d25_d27_inconsistency_not_official_errata"),
                ("appendix_d_trace_step_km", f"{0.06 * EARTH_REFERENCE_RADIUS_KM:g}"),
                ("appendix_d_inner_samples", str(sum(
                    sample.method == "igrf_rk4_minimum_b" for sample in field_line_samples
                ))),
                ("appendix_d_outer_samples", str(sum(
                    sample.method == "ost_shifted_dipole_d21_d24" for sample in field_line_samples
                ))),
                ("appendix_d_diagnostic_samples", str(sum(
                    bool(sample.incomplete_reasons) for sample in field_line_samples
                ))),
                ("appendix_d_candidate_comparisons", str(sum(
                    sample.valid and sample.alternative_l_shell is not None
                    for sample in field_line_samples
                ))),
                ("appendix_d_max_candidate_l_difference", "{:.12g}".format(max((
                    abs(sample.l_shell - sample.alternative_l_shell)
                    for sample in field_line_samples
                    if sample.valid and sample.alternative_l_shell is not None
                ), default=0.0))),
                *_field_line_coverage_metadata(prefix="magnetic", field_line_samples=field_line_samples),
                ("appendix_e_reference", OST_ERB_APPENDIX_E_REFERENCE),
                ("appendix_e_integral_spectrum", "tail_power_law_numeric_quadrature"),
                (
                    "appendix_e_approximation_use",
                    "diagnostic_fit_not_source_for_production_spectra",
                ),
                (
                    "appendix_e_differential_definition",
                    "negative_derivative_of_integral_flux_above_threshold",
                ),
                (
                    "appendix_e_electron_differential_policy",
                    OST_ERB_ELECTRON_DIFFERENTIAL_POLICY,
                ),
                ("appendix_e_formula_errata_status", OST_ERB_DIFFERENTIAL_ERRATA_STATUS),
                (
                    "appendix_e_approximation_fit_objective",
                    "sum_squared_relative_integral_flux_residuals",
                ),
                ("appendix_e_interpolation", "three_point_lagrange_with_log_positive_flux_values"),
                (
                    "appendix_e_zero_node_interpolation_policy",
                    "lagrange_on_linear_flux_then_clamp_negative_to_zero",
                ),
                ("appendix_e_averaging", "equal_weight_mean_anomaly_samples"),
                (
                    "appendix_e_trajectory_assumption",
                    "independent_mean_anomaly_raan_ensemble_not_time_propagated_track",
                ),
                ("appendix_e_average_denominator", "all_orbit_time_samples"),
                (
                    "appendix_e_failed_trace_policy",
                    "computed_contribution_over_full_duration_incomplete_if_any_trace_failed",
                ),
                (
                    "appendix_e_time_weighting_status",
                    "approximated_by_uniform_mean_anomaly_sampling",
                ),
                ("solar_reference_start_year", str(solar_reference_start_year)),
                ("solar_fraction_max", f"{fraction_max:.12g}"),
                ("solar_interpolation", "linear_ost_appendix_e_5"),
                (
                    "solar_time_aggregation",
                    "mean_of_calendar_month_fractions_equivalent_by_linearity",
                ),
                ("anomaly_samples", str(self.anomaly_samples)),
                ("node_samples", str(self.node_samples)),
                (
                    "orbit_sampling_convergence_status",
                    "not_established_below_1_percent_for_every_energy_node",
                ),
                ("orbit_sampling_convergence_release_eligible", "false"),
                ("igrf_epoch", f"{self._igrf_coefficients().epoch:g}"),
                ("igrf_configured_epoch", f"{self.igrf_epoch:g}"),
                ("interpolation", self.interpolation.value),
                ("peak_state", self.peak_state),
                *solar_activity.metadata,
                *proton_coverage_metadata,
                *electron_coverage_metadata,
                *proton_max_coverage_metadata,
                *electron_max_coverage_metadata,
                *proton_fit_metadata,
                *electron_fit_metadata,
            ),
        )
