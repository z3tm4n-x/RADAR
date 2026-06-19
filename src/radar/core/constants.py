"""Project-wide constants for RADAR."""

from radar.core.types import DoseQuantity, ShieldGeometry
from radar.core.units import Unit

DEFAULT_KP = 3
DEFAULT_SEP_EXCEEDANCE_PROBABILITY = 0.1

DEFAULT_SHIELD_THICKNESSES_G_CM2 = (
    0.01,
    0.1,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    10.0,
)

DEFAULT_DOSE_QUANTITY = DoseQuantity.ACCUMULATED_DOSE
DEFAULT_DOSE_UNIT = Unit.RAD
DEFAULT_SHIELD_GEOMETRY = ShieldGeometry.SPHERE

MIN_LIFETIME_YEARS = 1
MAX_KP = 9
MIN_KP = 0
