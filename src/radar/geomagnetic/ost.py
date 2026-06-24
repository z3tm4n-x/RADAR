"""OST 134-1044-2007 geomagnetic penetration primitives.

This module implements low-level pieces used to build the geomagnetic penetration
function psi(R): the tabulated vertical cutoff rigidity field, altitude scaling,
and Earth-shadow geometry. It does not propagate an orbit and does not build
psi(R) yet.
"""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.2
OST_CUTOFF_REFERENCE_ALTITUDE_KM = 450.0

OST_CUTOFF_LATITUDES_DEG = (
    85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0,
    50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0,
    15.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0,
    -20.0, -25.0, -30.0, -35.0, -40.0, -45.0,
    -50.0, -55.0, -60.0, -65.0, -70.0, -75.0,
    -80.0, -85.0,
)
OST_CUTOFF_LONGITUDES_DEG = (
    0.0, 30.0, 60.0, 90.0, 120.0, 150.0,
    180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
)

OST_VERTICAL_CUTOFF_RIGIDITY_GV = (
    (0.004, 0.004, 0.007, 0.007, 0.010, 0.010, 0.010, 0.013, 0.000, 0.013, 0.007, 0.007),
    (0.004, 0.004, 0.004, 0.025, 0.031, 0.016, 0.004, 0.007, 0.010, 0.010, 0.007, 0.004),
    (0.040, 0.109, 0.154, 0.178, 0.196, 0.178, 0.127, 0.004, 0.007, 0.007, 0.004, 0.004),
    (0.220, 0.316, 0.373, 0.421, 0.454, 0.469, 0.352, 0.169, 0.004, 0.004, 0.004, 0.079),
    (0.486, 0.666, 0.741, 0.810, 0.888, 0.951, 0.756, 0.408, 0.144, 0.018, 0.075, 0.282),
    (0.990, 1.203, 1.330, 1.426, 1.579, 1.705, 1.408, 0.846, 0.356, 0.174, 0.264, 0.615),
    (1.778, 2.018, 2.165, 2.357, 2.588, 2.711, 2.339, 1.386, 0.745, 0.389, 0.560, 1.166),
    (2.808, 3.150, 3.351, 3.615, 3.933, 4.101, 3.540, 2.379, 1.262, 0.743, 1.028, 2.010),
    (4.223, 4.472, 4.733, 5.084, 5.471, 5.630, 4.739, 3.527, 2.059, 1.285, 1.717, 3.356),
    (6.043, 6.244, 6.697, 7.381, 7.850, 8.057, 6.640, 4.768, 3.124, 1.987, 2.641, 4.669),
    (8.234, 8.237, 9.098, 9.497, 9.944, 9.635, 8.114, 6.628, 4.387, 2.932, 3.787, 7.063),
    (9.766, 10.174, 10.981, 11.663, 12.086, 11.432, 9.955, 8.356, 5.818, 3.789, 5.136, 9.046),
    (11.197, 11.779, 12.586, 13.420, 13.324, 12.535, 11.377, 10.057, 7.927, 5.236, 7.006, 10.270),
    (12.117, 12.873, 13.678, 14.356, 14.125, 13.209, 12.108, 11.052, 9.153, 6.440, 8.664, 11.208),
    (12.634, 13.348, 14.254, 14.950, 14.638, 13.681, 12.673, 11.758, 10.284, 7.683, 10.188, 11.770),
    (12.682, 13.480, 14.497, 15.217, 14.875, 13.954, 13.078, 12.280, 11.122, 9.535, 10.840, 11.950),
    (12.427, 13.291, 14.413, 15.157, 14.836, 14.017, 13.306, 12.625, 11.731, 10.510, 11.152, 11.851),
    (11.908, 12.802, 14.017, 14.770, 14.518, 13.849, 13.339, 12.781, 12.052, 11.113, 11.248, 11.536),
    (11.140, 12.067, 13.330, 14.062, 13.903, 13.417, 13.147, 12.745, 12.154, 11.335, 11.167, 11.029),
    (10.231, 11.131, 12.379, 13.027, 12.976, 12.679, 12.691, 12.508, 12.070, 11.356, 10.936, 10.354),
    (9.111, 9.921, 10.956, 11.500, 11.194, 11.353, 11.935, 12.061, 11.824, 11.188, 10.527, 9.561),
    (7.718, 8.352, 9.051, 9.381, 9.186, 9.156, 10.374, 11.388, 11.412, 10.890, 10.050, 8.544),
    (6.337, 6.934, 7.255, 6.634, 6.619, 7.312, 8.392, 9.742, 10.843, 10.450, 9.328, 7.417),
    (5.262, 5.413, 5.058, 4.635, 4.593, 5.079, 6.682, 7.678, 10.090, 9.829, 8.533, 6.334),
    (4.246, 3.949, 3.706, 3.100, 3.004, 3.625, 4.798, 6.802, 8.396, 9.134, 7.684, 5.602),
    (3.436, 3.088, 2.539, 1.933, 1.867, 2.317, 3.541, 4.780, 6.997, 8.234, 6.706, 4.915),
    (2.777, 2.272, 1.714, 1.165, 1.000, 1.336, 2.275, 3.659, 5.264, 7.218, 6.086, 3.983),
    (2.229, 1.673, 1.100, 0.611, 0.488, 0.722, 1.424, 2.508, 3.960, 5.440, 4.854, 3.222),
    (1.718, 1.199, 0.686, 0.294, 0.188, 0.326, 0.806, 1.682, 2.843, 3.924, 3.687, 2.570),
    (1.297, 0.828, 0.405, 0.111, 0.006, 0.111, 0.405, 1.038, 1.939, 2.851, 2.854, 1.990),
    (0.948, 0.546, 0.222, 0.000, 0.006, 0.006, 0.195, 0.600, 1.257, 1.866, 1.980, 1.464),
    (0.640, 0.352, 0.100, 0.004, 0.004, 0.007, 0.046, 0.328, 0.757, 1.163, 1.268, 0.985),
    (0.415, 0.205, 0.022, 0.004, 0.004, 0.004, 0.004, 0.169, 0.424, 0.664, 0.754, 0.622),
    (0.229, 0.109, 0.000, 0.004, 0.004, 0.004, 0.004, 0.064, 0.223, 0.347, 0.389, 0.341),
    (0.106, 0.037, 0.000, 0.004, 0.004, 0.004, 0.004, 0.022, 0.088, 0.139, 0.175, 0.151),
)


def ost_earth_shadow_factor(altitude_km: float) -> float:
    """Return unshadowed sky fraction mu(H) for a spherical Earth."""

    if not math.isfinite(altitude_km):
        msg = "Altitude must be finite."
        raise ValueError(msg)

    if altitude_km < 0.0:
        msg = "Altitude must be non-negative."
        raise ValueError(msg)

    eta = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km)
    return float((1.0 + math.sqrt(max(0.0, 1.0 - eta * eta))) / 2.0)


def _latitude_fraction(latitude_deg: float) -> tuple[int, float]:
    latitude = min(
        max(latitude_deg, OST_CUTOFF_LATITUDES_DEG[-1]),
        OST_CUTOFF_LATITUDES_DEG[0],
    )
    coordinate = (OST_CUTOFF_LATITUDES_DEG[0] - latitude) / 5.0
    index = int(math.floor(coordinate))
    index = min(max(index, 0), len(OST_CUTOFF_LATITUDES_DEG) - 2)
    return index, coordinate - index


def _longitude_fraction(longitude_deg: float) -> tuple[int, int, float]:
    longitude = longitude_deg % 360.0
    coordinate = longitude / 30.0
    left_index = int(math.floor(coordinate)) % len(OST_CUTOFF_LONGITUDES_DEG)
    right_index = (left_index + 1) % len(OST_CUTOFF_LONGITUDES_DEG)
    return left_index, right_index, coordinate - math.floor(coordinate)


def ost_vertical_cutoff_rigidity_450km_gv(
    latitude_deg: float,
    longitude_deg: float,
) -> float:
    """Return bilinearly interpolated OST vertical cutoff rigidity at 450 km."""

    if not math.isfinite(latitude_deg) or not math.isfinite(longitude_deg):
        msg = "Latitude and longitude must be finite."
        raise ValueError(msg)

    lat_index, lat_fraction = _latitude_fraction(latitude_deg)
    lon_left, lon_right, lon_fraction = _longitude_fraction(longitude_deg)

    top_left = OST_VERTICAL_CUTOFF_RIGIDITY_GV[lat_index][lon_left]
    top_right = OST_VERTICAL_CUTOFF_RIGIDITY_GV[lat_index][lon_right]
    bottom_left = OST_VERTICAL_CUTOFF_RIGIDITY_GV[lat_index + 1][lon_left]
    bottom_right = OST_VERTICAL_CUTOFF_RIGIDITY_GV[lat_index + 1][lon_right]

    top = top_left + (top_right - top_left) * lon_fraction
    bottom = bottom_left + (bottom_right - bottom_left) * lon_fraction

    return float(top + (bottom - top) * lat_fraction)


def ost_cutoff_rigidity_gv(
    *,
    latitude_deg: float,
    longitude_deg: float,
    altitude_km: float,
) -> float:
    """Return OST vertical cutoff rigidity scaled from 450 km to altitude."""

    if not math.isfinite(altitude_km):
        msg = "Altitude must be finite."
        raise ValueError(msg)

    if altitude_km < 0.0:
        msg = "Altitude must be non-negative."
        raise ValueError(msg)

    cutoff_450km = ost_vertical_cutoff_rigidity_450km_gv(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
    )
    scale = (
        (EARTH_RADIUS_KM + OST_CUTOFF_REFERENCE_ALTITUDE_KM)
        / (EARTH_RADIUS_KM + altitude_km)
    ) ** 2
    return float(cutoff_450km * scale)
