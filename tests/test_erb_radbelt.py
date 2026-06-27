import pytest

from radar.erb.radbelt import RADBELT_LOG_BIN_RATIO, parse_radbelt_asc_text
from radar.erb.resources import load_ae8_ap8_radbelt_maps


def _asc_line(values: tuple[int, ...]) -> str:
    return " " + "".join(f"{value:6d}" for value in values)


def test_parse_radbelt_asc_text_reads_descriptor_and_map() -> None:
    text = _asc_line((2, 4, 1970, 100, 2048, 2048, 1024, 4, 11, 22, 33, 44))

    parsed = parse_radbelt_asc_text(name="AP8TEST", text=text)

    assert parsed.name == "AP8TEST"
    assert parsed.descriptor == (2, 4, 1970, 100, 2048, 2048, 1024, 4)
    assert parsed.map_values == (11, 22, 33, 44)
    assert parsed.fistep == pytest.approx(256.0)
    assert parsed.descriptor_1based(8) == 4


def test_parse_radbelt_asc_text_rejects_map_length_mismatch() -> None:
    text = _asc_line((2, 4, 1970, 100, 2048, 2048, 1024, 4, 11, 22, 33))

    with pytest.raises(ValueError, match="mismatch"):
        parse_radbelt_asc_text(name="BAD", text=text)


def test_load_ae8_ap8_radbelt_maps_reads_all_bundled_maps() -> None:
    maps = load_ae8_ap8_radbelt_maps()

    assert set(maps) == {"AE8MAX", "AE8MIN", "AP8MAX", "AP8MIN"}
    assert maps["AE8MAX"].descriptor == (7, 4, 1970, 6400, 2100, 1024, 1024, 13548)
    assert maps["AE8MIN"].descriptor == (8, 4, 1964, 6400, 2100, 1024, 1024, 13168)
    assert maps["AP8MAX"].descriptor == (2, 4, 1970, 100, 2048, 2048, 1024, 16296)
    assert maps["AP8MIN"].descriptor == (2, 4, 1964, 100, 2048, 2048, 1024, 16584)
    assert all(
        len(radbelt_map.map_values) == radbelt_map.descriptor[7]
        for radbelt_map in maps.values()
    )


def test_ap8max_integral_flux_matches_radbelt_reference_point() -> None:
    radbelt_map = load_ae8_ap8_radbelt_maps()["AP8MAX"]

    logs = radbelt_map.log_integral_flux_gt_e(
        l_shell=3.0,
        b_over_b0=1.2,
        energies_mev=(0.1, 1.0, 10.0, 100.0),
    )

    assert logs == pytest.approx((8.3482900455298, 7.326819279562384, 3.133599839922049, 0.0))
    assert radbelt_map.integral_flux_gt_e(
        l_shell=3.0,
        b_over_b0=1.2,
        energy_mev=1.0,
    ) == pytest.approx(21223611.124318335)


def test_ap8max_differential_bins_match_radbelt_reference_point() -> None:
    radbelt_map = load_ae8_ap8_radbelt_maps()["AP8MAX"]

    differential_flux = radbelt_map.differential_flux_bins(
        l_shell=3.0,
        b_over_b0=1.2,
        lower_energies_mev=(0.1, 1.0, 10.0),
    )

    assert differential_flux == pytest.approx(
        (224187533.95374262, 2358027.8814996486, 349.7621678864863)
    )


def test_ae8max_integral_and_differential_flux_match_radbelt_reference_point() -> None:
    radbelt_map = load_ae8_ap8_radbelt_maps()["AE8MAX"]

    assert radbelt_map.log_integral_flux_gt_e(
        l_shell=3.0,
        b_over_b0=1.2,
        energies_mev=(0.05, 0.5, 5.0),
    ) == pytest.approx((7.545036343864469, 6.445426968864469, 3.1445279210129065))
    assert radbelt_map.integral_flux_gt_e(
        l_shell=3.0,
        b_over_b0=1.2,
        energy_mev=0.5,
    ) == pytest.approx(2788861.6408059546)
    assert radbelt_map.differential_flux_bins(
        l_shell=3.0,
        b_over_b0=1.2,
        lower_energies_mev=(0.05, 0.5, 5.0),
    ) == pytest.approx((71753913.64163795, 619437.0643278881, 1021.7911971817147))


def test_differential_flux_bins_reject_invalid_log_bin_ratio() -> None:
    radbelt_map = load_ae8_ap8_radbelt_maps()["AP8MAX"]

    with pytest.raises(ValueError, match="ratio"):
        radbelt_map.differential_flux_bins(
            l_shell=3.0,
            b_over_b0=1.2,
            lower_energies_mev=(0.1,),
            log_bin_ratio=1.0,
        )


def test_radbelt_log_bin_ratio_matches_cosrad_grid_step() -> None:
    assert RADBELT_LOG_BIN_RATIO == pytest.approx(10.0**0.1)
