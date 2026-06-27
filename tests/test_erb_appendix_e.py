import pytest

from radar.erb.appendix_e import integrate_differential_flux_tail_power_law


def test_appendix_e_integrates_constant_differential_flux_without_tail() -> None:
    integral_flux = integrate_differential_flux_tail_power_law(
        energies_mev=(1.0, 2.0, 3.0),
        differential_flux=(10.0, 10.0, 10.0),
    )

    assert integral_flux == pytest.approx((20.0, 10.0, 0.0))


def test_appendix_e_integrates_power_law_with_finite_tail() -> None:
    integral_flux = integrate_differential_flux_tail_power_law(
        energies_mev=(1.0, 2.0, 4.0),
        differential_flux=(8.0, 2.0, 0.5),
    )

    assert integral_flux == pytest.approx((8.0, 4.0, 2.0))


@pytest.mark.parametrize(
    ("energies_mev", "differential_flux", "message"),
    (
        ((), (), "must not be empty"),
        ((1.0, 2.0), (1.0,), "same length"),
        ((1.0, 0.0), (1.0, 1.0), "positive"),
        ((2.0, 1.0), (1.0, 1.0), "sorted"),
        ((1.0, 1.0), (1.0, 1.0), "unique"),
        ((1.0, 2.0), (1.0, -1.0), "non-negative"),
    ),
)
def test_appendix_e_integral_flux_validates_inputs(
    energies_mev: tuple[float, ...],
    differential_flux: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        integrate_differential_flux_tail_power_law(
            energies_mev=energies_mev,
            differential_flux=differential_flux,
        )
