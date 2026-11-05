"""Project-wide constants for RADAR."""

from radar.core.types import DoseQuantity, ShieldGeometry
from radar.core.units import Unit

DEFAULT_KP = 3
DEFAULT_SEP_EXCEEDANCE_PROBABILITY = 0.1

# Default computational grids are evaluation grids, not normative table nodes.
# Geometric subdivision retains every control point while making the numerical
# integral reproducible under the stage-4 double-density convergence check.
def _geometrically_subdivide(
    grid: tuple[float, ...],
    *,
    intervals_per_segment: int,
) -> tuple[float, ...]:
    if intervals_per_segment < 1:
        msg = "Grid subdivision must use at least one interval per segment."
        raise ValueError(msg)

    values: list[float] = []
    for lower, upper in zip(grid, grid[1:]):
        ratio = (upper / lower) ** (1.0 / intervals_per_segment)
        values.extend(lower * ratio**index for index in range(intervals_per_segment))
    values.append(grid[-1])
    return tuple(values)


# OST 134-1044-2007 Appendix V defines the analytical GCR spectrum and refers
# to GOST 25645.150.  It does not state a 10 MeV/nucleon lower truncation, and
# the installed COSRAD 4 GFN/GFX/GFXM source grids start at 1 MeV/nucleon.  The
# first OST production release therefore follows that native lower centre.
#
# The successor GOST GCR draft states an upper model boundary of
# 100 GeV/nucleon.  COSRAD 4 writes source spectra on a 0.1-decade grid through
# 79.4 GeV/nucleon; its next bin centre would be the 100 GeV/nucleon boundary.
# Keeping the final centre at 10**4.9 MeV/nucleon also keeps the logarithmic
# upper bin edge below the 100 GeV proton-range limit.  Integral products close
# the remaining finite interval explicitly at 100 GeV/nucleon.
GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON = 1.0
GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON = 100_000.0
GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON = 10.0**4.9

_GCR_PIPELINE_CONTROL_GRID_MEV_PER_NUCLEON = (
    GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON,
)
DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON = _geometrically_subdivide(
    _GCR_PIPELINE_CONTROL_GRID_MEV_PER_NUCLEON,
    # Eight subdivisions are required by the piecewise-linear proton source
    # reconstruction used by conservative CSDA remapping.  Four subdivisions
    # already change its unshielded integral by 1.025 % on density doubling;
    # eight reduce that change to about 0.26 % (2026, five-year OST reference).
    intervals_per_segment=8,
)


# Default computational grids for production SEP pipeline execution.  Appendix
# B of OST 134-1044-2007 defines an analytical spectrum and does not introduce
# the former 500/1000 MeV truncations.  The installed COSRAD 4 SFN/SFX/SFXM
# files use logarithmic centres from 1 MeV/nucleon through 10**4.1
# MeV/nucleon.  RADAR follows that observable COSRAD domain while retaining the
# distinction between this computational grid and normative coefficients.
#
# Proton transport contains a comparatively expensive secondary-particle
# quadrature, for which 0.05 decade is the coarsest grid that meets the stage-4
# <1 % transport-convergence criterion at 10 g/cm^2.  HZE CSDA transport needs
# 0.025 decade to meet the same criterion for both He and Fe.
SEP_COSRAD_FIRST_CENTER_MEV_PER_NUCLEON = 1.0
SEP_COSRAD_LAST_CENTER_MEV_PER_NUCLEON = 10.0**4.1


def _logarithmic_centers(
    *,
    first_exponent: float,
    last_exponent: float,
    exponent_step: float,
) -> tuple[float, ...]:
    interval_count = round((last_exponent - first_exponent) / exponent_step)
    if interval_count < 1:
        msg = "Logarithmic grid must contain at least one interval."
        raise ValueError(msg)
    resolved_last_exponent = first_exponent + interval_count * exponent_step
    if abs(resolved_last_exponent - last_exponent) > 1.0e-12:
        msg = "Logarithmic grid bounds must be an integer number of steps apart."
        raise ValueError(msg)
    return tuple(
        10.0 ** (first_exponent + index * exponent_step)
        for index in range(interval_count + 1)
    )


DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV = _logarithmic_centers(
    first_exponent=0.0,
    last_exponent=4.1,
    exponent_step=0.05,
)
DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON = _logarithmic_centers(
    first_exponent=0.0,
    last_exponent=4.1,
    exponent_step=0.025,
)

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
