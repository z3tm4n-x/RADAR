import pytest

from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid, linear_rigidity_grid


def test_rigidity_grid_requires_positive_increasing_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        RigidityGrid(values_gv=(0.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        RigidityGrid(values_gv=(1.0, 1.0, 2.0))


def test_linear_rigidity_grid() -> None:
    grid = linear_rigidity_grid(
        min_rigidity_gv=1.0,
        max_rigidity_gv=5.0,
        point_count=5,
    )

    assert grid.values_gv == pytest.approx((1.0, 2.0, 3.0, 4.0, 5.0))


def test_linear_rigidity_grid_rejects_bad_count() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        linear_rigidity_grid(
            min_rigidity_gv=1.0,
            max_rigidity_gv=5.0,
            point_count=1,
        )


def test_penetration_function_defaults_to_kp_3() -> None:
    penetration = PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
        values=(0.0, 0.5, 1.0),
        model="test_penetration",
    )

    assert penetration.kp == 3


def test_penetration_function_validates_values() -> None:
    with pytest.raises(ValueError, match="0..1"):
        PenetrationFunction(
            rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
            values=(0.0, 1.2, 1.0),
            model="test_penetration",
        )


def test_penetration_function_validates_length() -> None:
    with pytest.raises(ValueError, match="length"):
        PenetrationFunction(
            rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
            values=(0.0, 1.0),
            model="test_penetration",
        )


def test_penetration_function_validates_kp() -> None:
    with pytest.raises(ValueError, match="0..9"):
        PenetrationFunction(
            rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
            values=(0.0, 0.5, 1.0),
            kp=10,
            model="test_penetration",
        )


def test_penetration_function_interpolates_values() -> None:
    penetration = PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
        values=(0.0, 0.5, 1.0),
        model="test_penetration",
    )

    assert penetration.value_at(1.0) == pytest.approx(0.0)
    assert penetration.value_at(1.5) == pytest.approx(0.25)
    assert penetration.value_at(2.0) == pytest.approx(0.5)
    assert penetration.value_at(2.5) == pytest.approx(0.75)
    assert penetration.value_at(3.0) == pytest.approx(1.0)


def test_penetration_function_clamps_outside_grid() -> None:
    penetration = PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
        values=(0.1, 0.5, 0.9),
        model="test_penetration",
    )

    assert penetration.value_at(0.5) == pytest.approx(0.1)
    assert penetration.value_at(4.0) == pytest.approx(0.9)


def test_penetration_function_rejects_non_positive_rigidity() -> None:
    penetration = PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
        values=(0.0, 0.5, 1.0),
        model="test_penetration",
    )

    with pytest.raises(ValueError, match="positive"):
        penetration.value_at(0.0)
