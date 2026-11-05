"""Bitwise equivalence of binary bracketing to the retained legacy scan.

The expected interpolation is intentionally implemented as the former linear
scan, independently of the production bracket search. Exact interior knots
must retain left-interval arithmetic, including its floating-point rounding.
"""

from math import inf, isfinite, nextafter, pi
from random import Random

import pytest

from radar.shielding.proton_al import ProtonAlRangeEnergyTable, _interpolate_strict
from radar.shielding.resources import load_normative_proton_al_range_table


def _legacy_scan(*, x, xp, fp, quantity):
    if not isfinite(x):
        raise ValueError(f"{quantity} interpolation coordinate must be finite.")
    if x < xp[0] or x > xp[-1]:
        raise ValueError(f"{quantity} interpolation coordinate is outside the tabulated range.")
    if x == xp[0]:
        return fp[0]
    if x == xp[-1]:
        return fp[-1]
    for index in range(len(xp) - 1):
        left_x, right_x = xp[index], xp[index + 1]
        if left_x <= x <= right_x:
            fraction = (x - left_x) / (right_x - left_x)
            return fp[index] + fraction * (fp[index + 1] - fp[index])
    raise ValueError(f"Could not interpolate {quantity}.")


def _synthetic_tables() -> tuple[ProtonAlRangeEnergyTable, ...]:
    return (
        ProtonAlRangeEnergyTable(
            energy_mev=(1e-9, 1.0, 2.0, 10.0, 1e9),
            range_g_cm2=(0.0, 0.123456789, pi, 123.456, 1e12),
            stopping_mev_cm2_g=(1e16, 1.0, 1e-9, pi, 1e6),
        ),
        ProtonAlRangeEnergyTable(
            energy_mev=(1e-12, 0.1, 1.0, nextafter(1.0, inf), 2.0, 20.0, 1e10),
            range_g_cm2=(0.0, 1e-14, 0.5, nextafter(0.5, inf), 3.0, 123.456, 1e14),
            stopping_mev_cm2_g=(1e16, 1.0, 1e-300, 1e300, pi, 1e16, 1.0),
        ),
        ProtonAlRangeEnergyTable(
            energy_mev=(1.0, 1e8), range_g_cm2=(0.0, 3.0),
            stopping_mev_cm2_g=(1e16, 1.0),
        ),
    )


@pytest.fixture(params=("normative", "nonlinear", "adjacent_floats", "two_nodes"))
def table(request) -> ProtonAlRangeEnergyTable:
    if request.param == "normative":
        return load_normative_proton_al_range_table()
    index = {"nonlinear": 0, "adjacent_floats": 1, "two_nodes": 2}[request.param]
    return _synthetic_tables()[index]


@pytest.fixture(params=("range", "inverse_range", "stopping"))
def interpolation(request, table):
    if request.param == "range":
        return table.range_at_energy, table.energy_mev, table.range_g_cm2, "proton Al energy"
    if request.param == "inverse_range":
        return table.energy_at_range, table.range_g_cm2, table.energy_mev, "proton Al range"
    return table.stopping_at_energy, table.energy_mev, table.stopping_mev_cm2_g, "proton Al stopping"


def _assert_same_bits(actual: float, expected: float, coordinate: float) -> None:
    assert actual.hex() == expected.hex(), (
        f"At x={coordinate!r}: binary={actual.hex()}, legacy={expected.hex()}"
    )


def test_every_knot_and_its_supported_float_neighbors_match_legacy_bits(interpolation) -> None:
    method, xp, fp, quantity = interpolation
    coordinates = {
        value for knot in xp for value in (nextafter(knot, -inf), knot, nextafter(knot, inf))
        if xp[0] <= value <= xp[-1]
    }
    for value in sorted(coordinates):
        expected = _legacy_scan(x=value, xp=xp, fp=fp, quantity=quantity)
        _assert_same_bits(method(value), expected, value)


def test_deterministic_dense_interior_values_match_legacy_bits(interpolation) -> None:
    method, xp, fp, quantity = interpolation
    generator = Random(0xA127)
    # Per-cell points cover the low-energy part of highly nonuniform tables;
    # whole-domain points independently sample their much wider intervals.
    coordinates = [
        left + generator.random() * (right - left)
        for left, right in zip(xp, xp[1:]) for _ in range(9)
    ]
    coordinates.extend(xp[0] + generator.random() * (xp[-1] - xp[0]) for _ in range(2048))
    for value in coordinates:
        expected = _legacy_scan(x=value, xp=xp, fp=fp, quantity=quantity)
        _assert_same_bits(method(value), expected, value)


def test_domain_and_nonfinite_rejections_keep_legacy_exception_type_and_text(interpolation) -> None:
    method, xp, fp, quantity = interpolation
    for value in (nextafter(xp[0], -inf), nextafter(xp[-1], inf), -inf, inf, float("nan")):
        with pytest.raises(ValueError) as expected:
            _legacy_scan(x=value, xp=xp, fp=fp, quantity=quantity)
        with pytest.raises(type(expected.value)) as actual:
            method(value)
        assert str(actual.value) == str(expected.value)


def test_exact_interior_node_must_not_shortcut_to_stored_ordinate() -> None:
    table = ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 2.0, 3.0), range_g_cm2=(0.0, 1.0, 2.0),
        stopping_mev_cm2_g=(1e16, 1.0, 3.0),
    )
    # Legacy left-interval arithmetic rounds 1e16+(1-1e16) to 0, not 1.
    # This test protects arithmetic equivalence; it does not endorse this
    # synthetic stopping table as physically useful.
    assert table.stopping_at_energy(2.0).hex() == (0.0).hex()
    assert table.stopping_at_energy(2.0) != table.stopping_mev_cm2_g[1]


def test_domain_endpoints_keep_direct_ordinate_return_and_signed_zero() -> None:
    table = ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 2.0), range_g_cm2=(-0.0, 1.0),
        stopping_mev_cm2_g=(1e16, 1.0),
    )
    assert table.range_at_energy(1.0).hex() == (-0.0).hex()
    assert table.stopping_at_energy(2.0).hex() == (1.0).hex()
    assert table.energy_at_range(-0.0).hex() == (1.0).hex()


def test_generic_source_ordinate_interpolation_preserves_left_interval_bits() -> None:
    # The same private helper is used for source ordinates, which may decrease
    # or be exactly zero and are not constrained like the monotone range.
    xp, fp = (1.0, 2.0, 3.0, 4.0), (1e16, 1.0, 0.0, 3.0)
    for value in (1.0, 1.5, nextafter(2.0, -inf), 2.0, nextafter(2.0, inf), 3.5, 4.0):
        expected = _legacy_scan(x=value, xp=xp, fp=fp, quantity="proton spectrum")
        actual = _interpolate_strict(x=value, xp=xp, fp=fp, quantity="proton spectrum")
        _assert_same_bits(actual, expected, value)
