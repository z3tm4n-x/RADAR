"""Numerical local-time geometry and trace binding, not OST calibration."""

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from radar.geomagnetic.local_time import (
    CENTERED_DIPOLE_MLT_MODEL,
    CENTERED_DIPOLE_MLT_QUALIFICATION_REASON,
    LOCAL_TIME_HISTORY_QUALIFICATION_REASON,
    LocalTimeHistory,
    SOLAR_LOCAL_TIME_MODEL,
    SOLAR_LOCAL_TIME_QUALIFICATION_REASON,
    build_centered_dipole_mlt_history,
    build_solar_local_time_history,
    centered_dipole_mlt_hours,
    local_solar_time_hours,
    orbit_samples_fingerprint,
)
from radar.geomagnetic.ost_orbit import OstOrbitSample

Vector3 = tuple[float, float, float]


def _samples() -> tuple[OstOrbitSample, ...]:
    return tuple(OstOrbitSample(
        index=index, time_s=60.0 * index, true_anomaly_deg=90.0 * index,
        radius_km=7000.0, altitude_km=629.0, latitude_deg=0.0, longitude_deg=90.0 * index,
    ) for index in range(4))


def _rotate(vector: Vector3) -> Vector3:
    # Independent fixed proper orthogonal matrix, not the production projection.
    x, y, z = vector
    return (x + 2.0 * y + 2.0 * z) / 3.0, (2.0 * x + y - 2.0 * z) / 3.0, (
        -2.0 * x + 2.0 * y - z
    ) / 3.0


@pytest.mark.parametrize(("position", "expected"), [
    ((1.0, 0.0, 0.0), 12.0), ((0.0, 1.0, 0.0), 18.0),
    ((-1.0, 0.0, 0.0), 0.0), ((0.0, -1.0, 0.0), 6.0),
])
def test_cardinal_meridians(position: Vector3, expected: float) -> None:
    assert centered_dipole_mlt_hours(
        position_ecef=position, sun_direction_ecef=(1.0, 0.0, 0.0),
        north_axis_ecef=(0.0, 0.0, 1.0),
    ) == expected
    assert local_solar_time_hours(
        position_ecef=position, sun_direction_ecef=(1.0, 0.0, 0.0),
    ) == expected


@pytest.mark.parametrize("position_angle", [-179.0, -90.0, -5.0, 0.0, 90.0, 179.0])
@pytest.mark.parametrize("sun_angle", [-155.0, 0.0, 64.0])
def test_projected_longitude_difference_is_rotation_and_latitude_invariant(
    position_angle: float, sun_angle: float,
) -> None:
    p, s = math.radians(position_angle), math.radians(sun_angle)
    position = _rotate((math.cos(p), math.sin(p), 7.5))
    sun = _rotate((math.cos(s), math.sin(s), -2.5))
    actual = centered_dipole_mlt_hours(
        position_ecef=position, sun_direction_ecef=sun, north_axis_ecef=_rotate((0.0, 0.0, 1.0)),
    )
    expected = (12.0 + (position_angle - sun_angle) / 15.0) % 24.0
    assert (actual - expected + 12.0) % 24.0 - 12.0 == pytest.approx(0.0, abs=2e-14)


@pytest.mark.parametrize("scale", [5e-324, 1e-300, 1.0, 1e300, 1.7e308])
def test_normalization_does_not_overflow_or_underflow_for_finite_vectors(scale: float) -> None:
    assert centered_dipole_mlt_hours(
        position_ecef=(scale, scale, 0.0), sun_direction_ecef=(scale, 0.0, 0.0),
        north_axis_ecef=(0.0, 0.0, scale),
    ) == pytest.approx(15.0)


@pytest.mark.parametrize("key", ["position_ecef", "sun_direction_ecef", "north_axis_ecef"])
@pytest.mark.parametrize("value", [
    (0.0, 0.0, 0.0), (True, 0.0, 1.0), (float("nan"), 1.0, 0.0),
    (0.0, float("inf"), 1.0), (0.0, 1.0, -float("inf")), (1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0), ("1", 0.0, 0.0), (10**400, 0.0, 0.0),
])
def test_invalid_vectors_raise_instead_of_returning_a_clock_value(key: str, value: object) -> None:
    arguments: dict[str, object] = {
        "position_ecef": (1.0, 0.0, 0.0), "sun_direction_ecef": (1.0, 0.0, 0.0),
        "north_axis_ecef": (0.0, 0.0, 1.0),
    }
    arguments[key] = value
    with pytest.raises(ValueError):
        centered_dipole_mlt_hours(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["position_ecef", "sun_direction_ecef"])
@pytest.mark.parametrize("vector", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1e-13, 0.0, 1.0)])
def test_magnetic_pole_and_near_pole_are_explicitly_undefined(key: str, vector: Vector3) -> None:
    arguments = {
        "position_ecef": (1.0, 0.0, 0.0), "sun_direction_ecef": (1.0, 0.0, 0.0),
        "north_axis_ecef": (0.0, 0.0, 1.0),
    }
    arguments[key] = vector
    with pytest.raises(ValueError, match="undefined or ill-conditioned"):
        centered_dipole_mlt_hours(**arguments)


def test_tilted_dipole_and_geographic_solar_time_are_distinct_conventions() -> None:
    position, sun = (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)
    magnetic = centered_dipole_mlt_hours(
        position_ecef=position, sun_direction_ecef=sun, north_axis_ecef=(0.0, 1.0, 1.0),
    )
    solar = local_solar_time_hours(position_ecef=position, sun_direction_ecef=sun)
    assert magnetic == pytest.approx(12.0)
    assert solar == pytest.approx(15.0)


def test_north_axis_orientation_is_not_discarded() -> None:
    assert centered_dipole_mlt_hours(
        position_ecef=(0.0, 1.0, 0.0), sun_direction_ecef=(1.0, 0.0, 0.0),
        north_axis_ecef=(0.0, 0.0, -1.0),
    ) == 6.0


def _history() -> LocalTimeHistory:
    return build_centered_dipole_mlt_history(
        samples=_samples(), sun_directions_ecef=((1.0, 0.0, 0.0),) * 4,
        north_axis_ecef=(0.0, 0.0, 1.0), sun_model="fixed synthetic Sun",
        time_reference="relative seconds with explicit synthetic orientation",
        parameters=(("scenario", "unit test only"),), incomplete_reasons=("Synthetic history.",),
    )


def test_factory_records_binding_and_qualification_with_canonical_hours() -> None:
    history = _history()
    assert history.model == CENTERED_DIPOLE_MLT_MODEL
    assert history.values_hours == pytest.approx((12.0, 18.0, 0.0, 6.0))
    history.validate_samples(_samples())
    assert history.sample_fingerprint == orbit_samples_fingerprint(_samples())
    metadata = dict(history.parameters)
    assert metadata["sun_model"] == "fixed synthetic Sun"
    assert metadata["north_axis_ecef_unit"] == "0,0,1"
    assert len(metadata["sun_directions_ecef_sha256"]) == 64
    assert metadata["scenario"] == "unit test only"
    assert "Synthetic history." in history.incomplete_reasons
    assert CENTERED_DIPOLE_MLT_QUALIFICATION_REASON in history.incomplete_reasons
    assert LOCAL_TIME_HISTORY_QUALIFICATION_REASON in history.incomplete_reasons
    with pytest.raises(FrozenInstanceError):
        history.model = "qualified"  # type: ignore[misc]


def test_solar_factory_records_distinct_model_and_qualification() -> None:
    history = build_solar_local_time_history(
        samples=_samples(), sun_directions_ecef=((1.0, 0.0, 0.0),) * 4,
        sun_model="fixed synthetic Sun", time_reference="relative seconds",
    )
    assert history.model == SOLAR_LOCAL_TIME_MODEL
    assert history.values_hours == pytest.approx((12.0, 18.0, 0.0, 6.0))
    metadata = dict(history.parameters)
    assert metadata["geographic_north_axis_ecef_unit"] == "0,0,1"
    assert "north_axis_ecef_unit" not in metadata
    assert SOLAR_LOCAL_TIME_QUALIFICATION_REASON in history.incomplete_reasons
    assert CENTERED_DIPOLE_MLT_QUALIFICATION_REASON not in history.incomplete_reasons
    assert LOCAL_TIME_HISTORY_QUALIFICATION_REASON in history.incomplete_reasons


@pytest.mark.parametrize(("field", "value"), [
    ("index", 42), ("time_s", 1.0), ("true_anomaly_deg", 1.0), ("radius_km", 7001.0),
    ("altitude_km", 630.0), ("latitude_deg", 1.0), ("longitude_deg", 1.0),
])
def test_binding_rejects_any_changed_sample_field(field: str, value: float) -> None:
    samples = _samples()
    changed = (replace(samples[0], **{field: value}), *samples[1:])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fingerprint"):
        _history().validate_samples(changed)


def test_binding_rejects_same_length_reordering_and_different_count() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        _history().validate_samples(tuple(reversed(_samples())))
    with pytest.raises(ValueError, match="sample count"):
        _history().validate_samples(_samples()[:3])


def test_fingerprint_is_numeric_not_python_type_or_signed_zero_dependent() -> None:
    samples = _samples()
    equivalent = tuple(replace(
        sample, time_s=int(sample.time_s), true_anomaly_deg=int(sample.true_anomaly_deg),
        radius_km=7000, altitude_km=629, latitude_deg=-0.0,
        longitude_deg=int(sample.longitude_deg),
    ) for sample in samples)
    assert orbit_samples_fingerprint(equivalent) == orbit_samples_fingerprint(samples)
    _history().validate_samples(equivalent)


@pytest.mark.parametrize("field", ["time_s", "true_anomaly_deg", "radius_km", "altitude_km"])
@pytest.mark.parametrize("value", [2**53 + 1, 2**53 + 3])
def test_fingerprint_rejects_lossy_integer_sample_fields(field: str, value: int) -> None:
    changed = (replace(_samples()[0], **{field: value}),)
    with pytest.raises(ValueError, match="exactly representable"):
        orbit_samples_fingerprint(changed)


@pytest.mark.parametrize("value", [2**53 - 1, 2**53, 2**53 + 2, 2**60])
def test_fingerprint_retains_exact_large_integer_float_equivalence(value: int) -> None:
    integer = (replace(_samples()[0], time_s=value),)
    floating = (replace(_samples()[0], time_s=float(value)),)
    assert orbit_samples_fingerprint(integer) == orbit_samples_fingerprint(floating)


def test_history_rejects_changed_time_that_would_round_to_the_same_float() -> None:
    original = (replace(_samples()[0], time_s=2**53),)
    history = LocalTimeHistory(
        values_hours=(12.0,), sample_fingerprint=orbit_samples_fingerprint(original),
        model="external diagnostic",
    )
    history.validate_samples(original)
    changed = (replace(original[0], time_s=2**53 + 1),)
    with pytest.raises(ValueError, match="exactly representable"):
        history.validate_samples(changed)


@pytest.mark.parametrize("values", [(), (0.0, 24.0), (-0.1,), (True,), (float("nan"),)])
def test_direct_history_rejects_invalid_or_noncanonical_hours(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        LocalTimeHistory(values_hours=values, sample_fingerprint="0" * 64, model="external")


@pytest.mark.parametrize("fingerprint", ["", "x" * 64, "A" * 64, "0" * 63])
def test_direct_history_requires_explicit_fingerprint(fingerprint: str) -> None:
    with pytest.raises(ValueError, match="SHA256"):
        LocalTimeHistory(values_hours=(12.0,), sample_fingerprint=fingerprint, model="external")


def test_direct_history_and_replacement_cannot_erase_mandatory_qualification() -> None:
    direct = LocalTimeHistory(values_hours=(12.0,), sample_fingerprint="0" * 64, model="external")
    assert direct.incomplete_reasons == (LOCAL_TIME_HISTORY_QUALIFICATION_REASON,)
    assert replace(_history(), incomplete_reasons=()).incomplete_reasons == (
        LOCAL_TIME_HISTORY_QUALIFICATION_REASON,
    )


@pytest.mark.parametrize("parameters", [(("sun_model", "spoofed"),), (("x", "1"), ("x", "2"))])
def test_factory_rejects_metadata_overrides_or_duplicate_keys(
    parameters: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError, match="unique"):
        build_solar_local_time_history(
            samples=_samples(), sun_directions_ecef=((1.0, 0.0, 0.0),) * 4,
            sun_model="synthetic", time_reference="relative", parameters=parameters,
        )


@pytest.mark.parametrize(("sun_model", "time_reference"), [("", "relative"), ("synthetic", " ")])
def test_factory_requires_explicit_sun_and_time_provenance(sun_model: str, time_reference: str) -> None:
    with pytest.raises(ValueError, match="nonblank"):
        build_solar_local_time_history(
            samples=_samples(), sun_directions_ecef=((1.0, 0.0, 0.0),) * 4,
            sun_model=sun_model, time_reference=time_reference,
        )


def test_factory_rejects_unmatched_sun_count_and_empty_samples() -> None:
    with pytest.raises(ValueError, match="sample count"):
        build_solar_local_time_history(
            samples=_samples(), sun_directions_ecef=((1.0, 0.0, 0.0),),
            sun_model="synthetic", time_reference="relative",
        )
    with pytest.raises(ValueError, match="empty"):
        orbit_samples_fingerprint(())


def test_sun_hash_binds_order_and_effective_normalized_directions() -> None:
    arguments = {"samples": _samples(), "sun_model": "synthetic", "time_reference": "relative"}
    sun_a: tuple[Vector3, ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) * 2
    original = build_solar_local_time_history(
        **arguments, sun_directions_ecef=sun_a,  # type: ignore[arg-type]
    )
    reverse = build_solar_local_time_history(
        **arguments, sun_directions_ecef=tuple(reversed(sun_a)),  # type: ignore[arg-type]
    )
    scaled = build_solar_local_time_history(
        **arguments, sun_directions_ecef=tuple(  # type: ignore[arg-type]
            (2.0 * x, 2.0 * y, 2.0 * z) for x, y, z in sun_a
        ),
    )
    assert dict(original.parameters)["sun_directions_ecef_sha256"] != dict(reverse.parameters)[
        "sun_directions_ecef_sha256"
    ]
    assert original == scaled
