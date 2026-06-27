"""Earth radiation belt model interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol

from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.project import CalculationConfig, MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.erb.appendix_e import (
    OstErbApproximationFitQuality,
    OstErbElectronIntegralApproximation,
    OstErbProtonIntegralApproximation,
    fit_ost_erb_electron_integral_approximation,
    fit_ost_erb_proton_integral_approximation,
    integrate_differential_flux_tail_power_law,
)
from radar.erb.constants import DEFAULT_IGRF_EPOCH, ERB_SECONDS_PER_YEAR
from radar.erb.field import ErbFieldLineSample, trace_l_shell_b_over_b0_for_orbit
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import sample_erb_orbit_grid
from radar.erb.ost_tables import (
    OstErbFluxTable,
    OstErbInterpolationMode,
    OstErbTableSet,
    load_ost_erb_appendix_a_tables,
)
from radar.erb.resources import load_igrf14_coefficients
from radar.solar_activity.model import (
    build_mission_solar_activity,
    mission_solar_activity_monthly_wolf_numbers,
)
from radar.solar_activity.ost import ost_wolf_number_cycle_table

OST_ERB_APPENDIX_A_REFERENCE = "OST 134-1044-2007 Appendix A tables A.1.1/A.1.2/A.2.1/A.2.2"
OST_ERB_APPENDIX_D_REFERENCE = "OST 134-1044-2007 Appendix D L-B coordinate calculation"
OST_ERB_APPENDIX_E_REFERENCE = "OST 134-1044-2007 Appendix E orbit-averaged ERB spectra"

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

        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.integral_flux_gt_e
        ):
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

def _clamp_unit_interval(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _ost_erb_solar_max_fraction(
    *,
    mission: MissionConfig,
    reference_start_year: int,
) -> float:
    cycle_table = ost_wolf_number_cycle_table(mission.solar_activity_level)
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
    return tuple(max(left_value, right_value) for left_value, right_value in zip(left, right, strict=True))


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


def _table_coverage_metadata(
    *,
    prefix: str,
    table: OstErbFluxTable,
    field_line_samples: tuple[ErbFieldLineSample, ...],
) -> tuple[tuple[str, str], ...]:
    total_samples = len(field_line_samples)
    valid_samples = sum(1 for sample in field_line_samples if sample.valid)
    invalid_samples = total_samples - valid_samples
    l_below_table = 0
    l_above_table = 0
    bb0_above_nearest_shell = 0

    for sample in field_line_samples:
        if not sample.valid:
            continue

        if sample.l_shell < table.l_grid[0]:
            l_below_table += 1
            continue

        if sample.l_shell > table.l_grid[-1]:
            l_above_table += 1
            continue

        nearest_shell = min(
            table.shells,
            key=lambda shell: abs(shell.l_shell - sample.l_shell),
        )

        if sample.b_over_b0 > nearest_shell.b_over_b0[-1] * (1.0 + 1.0e-12):
            bb0_above_nearest_shell += 1

    valid_divisor = max(valid_samples, 1)

    return (
        (f"{prefix}_total_samples", str(total_samples)),
        (f"{prefix}_valid_samples", str(valid_samples)),
        (f"{prefix}_invalid_samples", str(invalid_samples)),
        (f"{prefix}_l_below_table_samples", str(l_below_table)),
        (f"{prefix}_l_above_table_samples", str(l_above_table)),
        (
            f"{prefix}_bb0_above_nearest_shell_samples",
            str(bb0_above_nearest_shell),
        ),
        (
            f"{prefix}_valid_sample_fraction",
            _sample_fraction(count=valid_samples, total=total_samples),
        ),
        (
            f"{prefix}_l_below_table_fraction",
            _sample_fraction(count=l_below_table, total=valid_divisor),
        ),
        (
            f"{prefix}_l_above_table_fraction",
            _sample_fraction(count=l_above_table, total=valid_divisor),
        ),
        (
            f"{prefix}_bb0_above_nearest_shell_fraction",
            _sample_fraction(count=bb0_above_nearest_shell, total=valid_divisor),
        ),
        (f"{prefix}_table_l_min", f"{table.l_grid[0]:.12g}"),
        (f"{prefix}_table_l_max", f"{table.l_grid[-1]:.12g}"),
    )


def _erb_spectrum(
    *,
    energies_mev: tuple[float, ...],
    values: tuple[float, ...],
    quantity: SpectrumQuantity,
    y_unit: Unit,
    particle: Particle,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=energies_mev,
        y=values,
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.ERB,
        model=model,
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
    positive_sample_count = sum(
        1
        for value in integral_spectrum.integral_flux_gt_e
        if value > 0.0
    )

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
class OstErbModel:
    """Normative OST 134-1044-2007 Appendix A/D/E ERB source model."""

    model: str = "ost_erb_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "ost_appendix_a_v1"
    interpolation: OstErbInterpolationMode = OstErbInterpolationMode.STRICT
    peak_state: OstErbPeakState = "worst"
    anomaly_samples: int = 48
    node_samples: int = 36
    igrf_epoch: float = DEFAULT_IGRF_EPOCH
    solar_reference_start_year: int | None = None
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

        return trace_l_shell_b_over_b0_for_orbit(
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
        energy_count = len(minimum_table.energies_mev)
        mean_accumulator = [0.0 for _ in range(energy_count)]
        maximum_flux = tuple(0.0 for _ in range(energy_count))
        valid_count = 0

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
            valid_count += 1

        mean_flux = _divide_vector(mean_accumulator, valid_count)
        fluence = _multiply_vector(mean_flux, mission_seconds)

        particle_label = particle.value
        model_prefix = f"{self.model}:{particle_label}"

        mean_spectrum = _erb_spectrum(
            energies_mev=minimum_table.energies_mev,
            values=mean_flux,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            particle=particle,
            model=f"{model_prefix}:mean_flux",
        )
        maximum_spectrum = _erb_spectrum(
            energies_mev=minimum_table.energies_mev,
            values=maximum_flux,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            particle=particle,
            model=f"{model_prefix}:maximum_flux",
        )
        fluence_spectrum = _erb_spectrum(
            energies_mev=minimum_table.energies_mev,
            values=fluence,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            particle=particle,
            model=f"{model_prefix}:mission_fluence",
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

        solar_reference_start_year = self._solar_reference_start_year(model_input)
        fraction_max = _ost_erb_solar_max_fraction(
            mission=model_input.config.mission,
            reference_start_year=solar_reference_start_year,
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
                ("appendix_e_reference", OST_ERB_APPENDIX_E_REFERENCE),
                ("appendix_e_integral_spectrum", "tail_power_law_numeric_quadrature"),
                (
                    "appendix_e_approximation_fit_objective",
                    "sum_squared_relative_integral_flux_residuals",
                ),
                ("appendix_e_interpolation", "three_point_lagrange_with_log_positive_flux_values"),
                ("appendix_e_averaging", "equal_weight_mean_anomaly_samples"),
                (
                    "appendix_e_time_weighting_status",
                    "approximated_by_uniform_mean_anomaly_sampling",
                ),
                ("solar_reference_start_year", str(solar_reference_start_year)),
                ("solar_fraction_max", f"{fraction_max:.12g}"),
                ("anomaly_samples", str(self.anomaly_samples)),
                ("node_samples", str(self.node_samples)),
                ("igrf_epoch", f"{self.igrf_epoch:g}"),
                ("interpolation", self.interpolation.value),
                ("peak_state", self.peak_state),
                *proton_coverage_metadata,
                *electron_coverage_metadata,
                *proton_fit_metadata,
                *electron_fit_metadata,
            ),
        )
