import math

import pytest

from radar.geomagnetic.ost import (
    EARTH_RADIUS_KM,
    OST_CUTOFF_REFERENCE_ALTITUDE_KM,
    ost_cutoff_rigidity_gv,
    ost_earth_shadow_factor,
    ost_vertical_cutoff_rigidity_450km_gv,
)


def test_ost_earth_shadow_factor_matches_reference_formula() -> None:
    altitude_km = 830.0
    eta = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km)
    expected = (1.0 + math.sqrt(1.0 - eta * eta)) / 2.0

    assert ost_earth_shadow_factor(altitude_km) == pytest.approx(expected)


def test_ost_earth_shadow_factor_at_surface_is_one_half() -> None:
    assert ost_earth_shadow_factor(0.0) == pytest.approx(0.5)


def test_ost_earth_shadow_factor_rejects_negative_altitude() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ost_earth_shadow_factor(-1.0)


def test_ost_vertical_cutoff_exact_table_node() -> None:
    assert ost_vertical_cutoff_rigidity_450km_gv(
        latitude_deg=85.0,
        longitude_deg=0.0,
    ) == pytest.approx(0.004)


def test_ost_vertical_cutoff_wraps_longitude() -> None:
    assert ost_vertical_cutoff_rigidity_450km_gv(
        latitude_deg=85.0,
        longitude_deg=360.0,
    ) == pytest.approx(
        ost_vertical_cutoff_rigidity_450km_gv(
            latitude_deg=85.0,
            longitude_deg=0.0,
        )
    )


def test_ost_vertical_cutoff_bilinear_reference_point() -> None:
    # Midpoint between table nodes:
    # lat 85/80, lon 0/30 -> average of 0.004, 0.004, 0.004, 0.004.
    assert ost_vertical_cutoff_rigidity_450km_gv(
        latitude_deg=82.5,
        longitude_deg=15.0,
    ) == pytest.approx(0.004)


def test_ost_vertical_cutoff_clamps_latitude_to_table_bounds() -> None:
    assert ost_vertical_cutoff_rigidity_450km_gv(
        latitude_deg=90.0,
        longitude_deg=0.0,
    ) == pytest.approx(
        ost_vertical_cutoff_rigidity_450km_gv(
            latitude_deg=85.0,
            longitude_deg=0.0,
        )
    )


def test_ost_cutoff_rigidity_at_reference_altitude_matches_table() -> None:
    assert ost_cutoff_rigidity_gv(
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_km=OST_CUTOFF_REFERENCE_ALTITUDE_KM,
    ) == pytest.approx(
        ost_vertical_cutoff_rigidity_450km_gv(
            latitude_deg=0.0,
            longitude_deg=0.0,
        )
    )


def test_ost_cutoff_rigidity_scales_with_altitude() -> None:
    table_value = ost_vertical_cutoff_rigidity_450km_gv(
        latitude_deg=0.0,
        longitude_deg=0.0,
    )
    altitude_km = 830.0
    expected = table_value * (
        (EARTH_RADIUS_KM + OST_CUTOFF_REFERENCE_ALTITUDE_KM)
        / (EARTH_RADIUS_KM + altitude_km)
    ) ** 2

    assert ost_cutoff_rigidity_gv(
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_km=altitude_km,
    ) == pytest.approx(expected)


def test_ost_cutoff_rigidity_rejects_bad_altitude() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ost_cutoff_rigidity_gv(
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_km=-1.0,
        )
