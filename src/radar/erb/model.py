"""Earth radiation belt model interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from radar.core.project import CalculationConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit

ERB_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.ELECTRON,
)

ERB_ALLOWED_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
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

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
        msg = "ERB model spectrum values must use differential flux units."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "ERB model energy grid values must be positive."
        raise ValueError(msg)


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
class ErbModelResult:
    """Result returned by an ERB model."""

    spectra: tuple[Spectrum1D, ...]
    lifetime_years: int
    kp: int
    model: str
    document: str

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "ERB model result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

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

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Return configured test ERB spectra."""

        return ErbModelResult(
            spectra=self.spectra,
            lifetime_years=model_input.lifetime_years,
            kp=model_input.kp,
            model=self.model,
            document=self.document,
        )
