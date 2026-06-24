"""Solar energetic particle model interface."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import math
from typing import Protocol

from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    GOST_SEP_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import scale_spectrum
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.sep.event_count import SepEventCountPolicy, calculate_sep_expected_events
from radar.sep.hze_spectrum import (
    OST_SEP_HZE_B9_RESOURCE,
    OstSepHzeIonRecord,
    evaluate_ost_sep_hze_spectrum,
    load_ost_sep_hze_ion_records,
    ost_sep_hze_coefficients_from_proton,
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

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
            msg = "SEP differential fluence spectrum values must use differential fluence units."
            raise ValueError(msg)

    if spectrum.quantity in (
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    ):
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
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


def _load_ost_sep_proton_coefficient_records() -> tuple[SepProtonCoefficientRecord, ...]:
    resource = resources.files("radar.data.normative").joinpath(
        OST_SEP_PROTON_COEFFICIENT_RESOURCE,
    )

    with resources.as_file(resource) as path:
        return load_sep_proton_coefficient_records(path)


def _load_ost_sep_hze_ion_records() -> tuple[OstSepHzeIonRecord, ...]:
    resource = resources.files("radar.data.normative").joinpath(
        OST_SEP_HZE_B9_RESOURCE,
    )

    with resources.as_file(resource) as path:
        return load_ost_sep_hze_ion_records(path)


def _make_sep_proton_spectrum(
    *,
    energy_grid_mev: tuple[float, ...],
    values: tuple[float, ...],
    quantity: SpectrumQuantity,
    y_unit: Unit,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=energy_grid_mev,
        y=values,
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model=model,
    )


def _make_sep_hze_spectrum(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    values: tuple[float, ...],
    quantity: SpectrumQuantity,
    y_unit: Unit,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=energy_grid_mev_per_nucleon,
        y=values,
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model=model,
    )


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

    HZE SEP spectra, geomagnetic penetration, shielding, LET conversion, dose, and
    single-event-effect calculations are intentionally outside this source model.
    """

    energy_grid_mev: tuple[float, ...] = ()
    monthly_smoothed_wolf_numbers: tuple[float, ...] = ()
    coefficient_records: tuple[SepProtonCoefficientRecord, ...] = ()
    hze_energy_grid_mev_per_nucleon: tuple[float, ...] = ()
    hze_ion_records: tuple[OstSepHzeIonRecord, ...] = ()
    model: str = "ost_sep_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "not_implemented"
    coefficient_model: str = OST_SEP_PROTON_COEFFICIENT_MODEL

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.OST_134_1044_2007,
            name=self.model,
            document=self.document,
            version=self.version,
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

        return _load_ost_sep_proton_coefficient_records()

    def _hze_ion_records(self) -> tuple[OstSepHzeIonRecord, ...]:
        if self.hze_ion_records:
            return self.hze_ion_records

        return _load_ost_sep_hze_ion_records()

    def _hze_source_products(
        self,
        *,
        fluence_coefficients: SepProtonSpectrumCoefficients,
        peak_flux_coefficients: SepProtonSpectrumCoefficients,
        mission_duration_seconds: float,
    ) -> tuple[SpectrumProduct, ...]:
        if not self.hze_energy_grid_mev_per_nucleon:
            return ()

        products: list[SpectrumProduct] = []

        for ion in self._hze_ion_records():
            if ion.relative_abundance == 0.0:
                continue

            fluence_hze_coefficients = ost_sep_hze_coefficients_from_proton(
                proton_coefficients=fluence_coefficients,
                ion=ion,
            )
            peak_flux_hze_coefficients = ost_sep_hze_coefficients_from_proton(
                proton_coefficients=peak_flux_coefficients,
                ion=ion,
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
                y_unit=Unit.DIFFERENTIAL_FLUENCE,
                model=f"{self.model}:hze:{ion_tag}:mission_fluence",
            )
            peak_flux_spectrum = _make_sep_hze_spectrum(
                energy_grid_mev_per_nucleon=self.hze_energy_grid_mev_per_nucleon,
                values=peak_flux_values,
                quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
                model=f"{self.model}:hze:{ion_tag}:peak_flux",
            )
            mean_flux_spectrum = _make_sep_hze_spectrum(
                energy_grid_mev_per_nucleon=self.hze_energy_grid_mev_per_nucleon,
                values=mean_flux_values,
                quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
                model=f"{self.model}:hze:{ion_tag}:mean_flux",
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
                "OST SEP model is not implemented without configured energy grid "
                "and monthly Wolf number series."
            )
            raise NotImplementedError(msg)

        _validate_monthly_wolf_number_count(
            monthly_smoothed_wolf_numbers=self.monthly_smoothed_wolf_numbers,
            lifetime_years=model_input.lifetime_years,
        )

        event_count = calculate_sep_expected_events(
            monthly_smoothed_wolf_numbers=self.monthly_smoothed_wolf_numbers,
            policy=SepEventCountPolicy.OST_134_1044_2007,
        )
        records = self._coefficient_records()

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
        )
        peak_flux_spectrum = _make_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            values=peak_flux_values,
            quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            model=f"{self.model}:peak_flux",
        )
        mean_flux_spectrum = _make_sep_proton_spectrum(
            energy_grid_mev=self.energy_grid_mev,
            values=mean_flux_values,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
            model=f"{self.model}:mean_flux",
        )

        hze_products = self._hze_source_products(
            fluence_coefficients=fluence_coefficients,
            peak_flux_coefficients=peak_flux_coefficients,
            mission_duration_seconds=mission_duration_seconds,
        )

        mission_fluence_product = _sep_source_product(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=mission_fluence_spectrum,
            label="SEP proton mission fluence by OST 134-1044-2007 Appendix B",
        )
        peak_flux_product = _sep_source_product(
            kind=RadiationProductKind.PEAK_FLUX,
            spectrum=peak_flux_spectrum,
            label=(
                "SEP proton omnidirectional peak flux by OST 134-1044-2007 "
                "Appendix B; derived as 4π times directional per-steradian peak flux"
            ),
        )
        mean_flux_product = _sep_source_product(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=mean_flux_spectrum,
            label=(
                "SEP proton mean flux derived from OST 134-1044-2007 "
                "Appendix B mission fluence divided by mission duration"
            ),
        )

        return SepModelResult(
            spectrum=mission_fluence_spectrum,
            lifetime_years=model_input.lifetime_years,
            exceedance_probability=model_input.exceedance_probability,
            model=self.model,
            document=self.document,
            products=(
                mission_fluence_product,
                peak_flux_product,
                mean_flux_product,
                *hze_products,
            ),
        )


@dataclass(frozen=True)
class GostSepModel:
    """Placeholder for normative GOST SEP model.

    The class declares metadata and profile compatibility only.
    Numerical GOST SEP equations are not implemented yet.
    """

    model: str = "gost_sep_model"
    document: str = GOST_SEP_DOCUMENT
    version: str = "not_implemented"

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.GOST_SEP,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        _ = self.metadata

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Raise until the normative GOST SEP model is implemented."""

        raise NotImplementedError("GOST SEP model is not implemented yet.")
