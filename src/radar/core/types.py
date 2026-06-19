"""Common enumerations for the RADAR calculation core."""

from enum import StrEnum


class Particle(StrEnum):
    """Particle groups used by RADAR."""

    PROTON = "proton"
    ELECTRON = "electron"
    HZE = "hze"


class RadiationSource(StrEnum):
    """Natural radiation sources used by RADAR."""

    SEP = "sep"
    GCR = "gcr"
    ERB = "erb"


class RadiationQuantityMode(StrEnum):
    """High-level calculation mode inherited from the source spectrum meaning."""

    FLUENCE = "fluence"
    FLUX = "flux"


class SpectrumQuantity(StrEnum):
    """Supported spectrum quantity types."""

    DIFFERENTIAL_FLUENCE = "differential_fluence"
    DIFFERENTIAL_FLUX = "differential_flux"
    PEAK_DIFFERENTIAL_FLUX = "peak_differential_flux"
    MAXIMUM_DIFFERENTIAL_FLUX = "maximum_differential_flux"
    MEAN_DIFFERENTIAL_FLUX = "mean_differential_flux"
    LET_DIFFERENTIAL_FLUENCE = "let_differential_fluence"
    LET_DIFFERENTIAL_FLUX = "let_differential_flux"


class DoseQuantity(StrEnum):
    """Supported dose result quantity types."""

    ACCUMULATED_DOSE = "accumulated_dose"
    DOSE_RATE = "dose_rate"


class ShieldGeometry(StrEnum):
    """Shield geometry options."""

    SPHERE = "sphere"


class SolarActivityLevel(StrEnum):
    """Solar activity levels used in the non-expert mode."""

    MINIMUM = "minimum"
    MEAN = "mean"
    MAXIMUM = "maximum"


class OrbitType(StrEnum):
    """Supported orbit types."""

    CIRCULAR = "circular"
    ELLIPTIC = "elliptic"
