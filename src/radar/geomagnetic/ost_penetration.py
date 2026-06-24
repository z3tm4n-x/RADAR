"""Build OST 134-1044-2007 geomagnetic penetration function psi(R)."""

from __future__ import annotations

from radar.core.constants import DEFAULT_KP
from radar.core.project import OrbitConfig
from radar.geomagnetic.ost import ost_cutoff_rigidity_gv, ost_earth_shadow_factor
from radar.geomagnetic.ost_orbit import OstOrbitSample, sample_ost_orbit
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid

OST_GEOMAGNETIC_PENETRATION_MODEL = "ost_134_1044_2007_geomagnetic_penetration"


def _normalized_samples(
    samples: tuple[OstOrbitSample, ...],
) -> tuple[OstOrbitSample, ...]:
    if not samples:
        msg = "OST penetration samples must not be empty."
        raise ValueError(msg)

    return tuple(samples)


def ost_sample_cutoff_rigidity_gv(sample: OstOrbitSample) -> float:
    """Return OST cutoff rigidity for one sampled orbit point."""

    return ost_cutoff_rigidity_gv(
        latitude_deg=sample.latitude_deg,
        longitude_deg=sample.longitude_deg,
        altitude_km=sample.altitude_km,
    )


def ost_sample_shadow_weight(sample: OstOrbitSample) -> float:
    """Return OST Earth-shadow weighting factor for one sampled orbit point."""

    return ost_earth_shadow_factor(sample.altitude_km)


def ost_penetration_values_from_samples(
    *,
    samples: tuple[OstOrbitSample, ...],
    rigidity_grid: RigidityGrid,
) -> tuple[float, ...]:
    """Return cumulative OST penetration values on a rigidity grid.

    The value psi(R) is computed as the shadow-weighted fraction of all sampled
    orbit points with local cutoff rigidity not exceeding R:

        psi(R) = sum(mu_i for Rc_i <= R) / N

    where N is the number of orbit samples. Kp disturbance corrections are not
    applied in this base builder.
    """

    normalized_samples = _normalized_samples(samples)

    cutoff_weight_pairs = sorted(
        (
            ost_sample_cutoff_rigidity_gv(sample),
            ost_sample_shadow_weight(sample),
        )
        for sample in normalized_samples
    )

    sample_count = float(len(normalized_samples))
    cumulative_weight = 0.0
    pair_index = 0
    values: list[float] = []

    for rigidity_gv in rigidity_grid.values_gv:
        while (
            pair_index < len(cutoff_weight_pairs)
            and cutoff_weight_pairs[pair_index][0] <= rigidity_gv
        ):
            cumulative_weight += cutoff_weight_pairs[pair_index][1]
            pair_index += 1

        values.append(cumulative_weight / sample_count)

    return tuple(values)


def build_ost_penetration_function_from_samples(
    *,
    samples: tuple[OstOrbitSample, ...],
    rigidity_grid: RigidityGrid,
    model: str = OST_GEOMAGNETIC_PENETRATION_MODEL,
    kp: int = DEFAULT_KP,
) -> PenetrationFunction:
    """Build OST geomagnetic penetration function from sampled orbit points."""

    return PenetrationFunction(
        rigidity_grid=rigidity_grid,
        values=ost_penetration_values_from_samples(
            samples=samples,
            rigidity_grid=rigidity_grid,
        ),
        model=model,
        kp=kp,
    )


def build_ost_penetration_function_for_orbit(
    config: OrbitConfig,
    *,
    rigidity_grid: RigidityGrid,
    raan_deg: float = 0.0,
    min_days: float = 16.0,
    steps_per_orbit: int = 60,
    max_samples: int = 400000,
    model: str = OST_GEOMAGNETIC_PENETRATION_MODEL,
    kp: int = DEFAULT_KP,
) -> PenetrationFunction:
    """Build OST geomagnetic penetration function by sampling an orbit."""

    trace = sample_ost_orbit(
        config,
        raan_deg=raan_deg,
        min_days=min_days,
        steps_per_orbit=steps_per_orbit,
        max_samples=max_samples,
    )

    return build_ost_penetration_function_from_samples(
        samples=trace.samples,
        rigidity_grid=rigidity_grid,
        model=model,
        kp=kp,
    )
