"""Galactic cosmic ray model interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from radar.core.products import SpectrumProduct
from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit

GCR_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.HZE,
)

GCR_ALLOWED_X_UNITS = (
    Unit.MEV,
    Unit.GEV_PER_NUCLEON,
)

GCR_ALLOWED_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
)


def validate_gcr_energy_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a supported GCR energy spectrum."""

    if spectrum.source is not RadiationSource.GCR:
        msg = "GCR model spectrum source must be GCR."
        raise ValueError(msg)

    if spectrum.particle is Particle.ELECTRON:
        msg = "GCR electrons are not calculated."
        raise ValueError(msg)

    if spectrum.particle not in GCR_ALLOWED_PARTICLES:
        msg = "GCR model spectrum must describe protons or HZE particles."
        raise ValueError(msg)

    if spectrum.quantity not in GCR_ALLOWED_QUANTITIES:
        msg = "GCR model spectrum quantity must be an energy flux or fluence quantity."
        raise ValueError(msg)

    if spectrum.x_unit not in GCR_ALLOWED_X_UNITS:
        msg = "GCR model energy grid must be in MeV or GeV/nucleon."
        raise ValueError(msg)

    if spectrum.quantity in (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    ):
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
            msg = "GCR differential flux spectrum values must use differential flux units."
            raise ValueError(msg)

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
            msg = "GCR differential fluence spectrum values must use differential fluence units."
            raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "GCR model energy grid values must be positive."
        raise ValueError(msg)


def _default_gcr_product_kind(spectrum: Spectrum1D) -> RadiationProductKind:
    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        return RadiationProductKind.MISSION_FLUENCE

    if spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MEAN_FLUX

    if spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MAXIMUM_FLUX

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUX:
        msg = (
            "GCR differential flux cannot be converted to a radiation product "
            "without specifying mean or maximum flux."
        )
        raise ValueError(msg)

    msg = f"Unsupported GCR spectrum quantity: {spectrum.quantity}"
    raise ValueError(msg)


def _default_gcr_products(spectra: tuple[Spectrum1D, ...]) -> tuple[SpectrumProduct, ...]:
    return tuple(
        SpectrumProduct(
            kind=_default_gcr_product_kind(spectrum),
            spectrum=spectrum,
            label=f"GCR {spectrum.particle.value} {spectrum.quantity.value}",
        )
        for spectrum in spectra
    )


@dataclass(frozen=True)
class GcrModelInput:
    """Input parameters passed to a GCR model."""

    mission: MissionConfig

    @property
    def launch_year(self) -> int:
        """Return mission launch year."""

        return self.mission.launch_year

    @property
    def lifetime_years(self) -> int:
        """Return mission lifetime in integer years."""

        return self.mission.lifetime_years


@dataclass(frozen=True)
class GcrModelResult:
    """Result returned by a GCR model."""

    spectra: tuple[Spectrum1D, ...]
    lifetime_years: int
    model: str
    document: str
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "GCR model result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_gcr_energy_spectrum(spectrum)

        products = self.products or _default_gcr_products(self.spectra)

        if not products:
            msg = "GCR model result must contain at least one radiation product."
            raise ValueError(msg)

        if tuple(product.spectrum for product in products) != self.spectra:
            msg = "GCR model product spectra must match result spectra."
            raise ValueError(msg)

        object.__setattr__(self, "products", products)

        if not isinstance(self.lifetime_years, int):
            msg = "GCR result lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < 1:
            msg = "GCR result lifetime must be at least one year."
            raise ValueError(msg)

        if not self.model:
            msg = "GCR model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "GCR source document must not be empty."
            raise ValueError(msg)


class GcrModelProtocol(Protocol):
    """Protocol implemented by concrete GCR models."""

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Calculate mission GCR spectra."""


@dataclass(frozen=True)
class StaticGcrModel:
    """Simple deterministic GCR model used for tests and integration plumbing.

    Normative OST/GOST GCR models must be implemented separately.
    """

    spectra: tuple[Spectrum1D, ...]
    model: str = "static_gcr_model"
    document: str = "test"

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "Static GCR model must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_gcr_energy_spectrum(spectrum)

        if not self.model:
            msg = "GCR model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "GCR source document must not be empty."
            raise ValueError(msg)

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Return configured test GCR spectra."""

        return GcrModelResult(
            spectra=self.spectra,
            lifetime_years=model_input.lifetime_years,
            model=self.model,
            document=self.document,
        )
