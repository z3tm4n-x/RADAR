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
    """Quantity class of a radiation product."""

    FLUENCE = "fluence"
    FLUX = "flux"


class RadiationProductKind(StrEnum):
    """Named radiation products required by OST/GOST calculation chains."""

    MISSION_FLUENCE = "mission_fluence"
    MODEL_FLUX = "model_flux"
    ORBIT_AVERAGED_FLUX = "orbit_averaged_flux"
    MEAN_FLUX = "mean_flux"
    MAXIMUM_FLUX = "maximum_flux"
    PEAK_FLUX = "peak_flux"

    MISSION_LET_FLUENCE = "mission_let_fluence"
    MEAN_LET_FLUX = "mean_let_flux"
    MAXIMUM_LET_FLUX = "maximum_let_flux"
    PEAK_LET_FLUX = "peak_let_flux"


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
