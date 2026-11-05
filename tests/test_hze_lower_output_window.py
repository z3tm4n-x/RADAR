"""Regression witnesses for omitted, but table-supported, low-energy exits."""

from math import sqrt

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.shielding.hze_al import (
    HZE_AL_LOWER_OUTPUT_WINDOW_REASON_PREFIX,
    HzeAlRangeTable,
    shield_hze_spectrum_primary_through_al,
)


def _table(*, range_offset: float = 0.0) -> HzeAlRangeTable:
    energies = (1.0, 10.0, 30.0, 100.0, 200.0)
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=energies,
        range_g_cm2=tuple(energy + range_offset for energy in energies),
    )


def _source(*, peak: bool = False, zero: bool = False) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(0.0, 0.0, 0.0) if zero else (2.0, 2.0, 2.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(
            Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY
            if peak
            else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY
        ),
        quantity=(
            SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX
            if peak
            else SpectrumQuantity.DIFFERENTIAL_FLUENCE
        ),
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model="analytic_known_source_support",
    )


def _ledger(output: Spectrum1D) -> dict[str, str]:
    return dict(output.transformation_chain[-1].parameters)


def _lower_window_reasons(output: Spectrum1D) -> tuple[str, ...]:
    return tuple(
        reason
        for reason in output.incomplete_reasons
        if reason.startswith(HZE_AL_LOWER_OUTPUT_WINDOW_REASON_PREFIX)
    )


@pytest.mark.parametrize("peak", (False, True))
def test_known_supported_low_exit_is_flagged_without_changing_output(peak: bool) -> None:
    source = _source(peak=peak)
    output = shield_hze_spectrum_primary_through_al(
        spectrum=source, table=_table(), thickness_g_cm2=20.0,
    )
    ledger = _ledger(output)
    output_minimum = 10.0 / sqrt(3.0)
    returned = 2.0 * (80.0 - output_minimum)
    omitted = 2.0 * (output_minimum - 1.0)

    # R(E)=E gives table-supported exits 1..80 from known inputs 21..100.
    # Inputs 10..21 are unresolved, not called physically stopped.
    assert float(ledger["source_ledger_integral"]) == pytest.approx(180.0)
    assert float(ledger["supported_exit_integral_in_output_window"]) == pytest.approx(returned)
    assert float(ledger["supported_exit_integral_below_output_window"]) == pytest.approx(omitted)
    assert float(ledger["unresolved_transport_integral_on_declared_source"]) == pytest.approx(22.0)
    assert returned + omitted + 22.0 == pytest.approx(180.0)
    assert ledger["source_ledger_scope"] == "declared_point_sample_support_only"
    assert ledger["physical_stopping_classification"] == "not_established"
    assert ledger["source_ledger_integral_unit"] == (
        Unit.INTEGRAL_FLUX.value if peak else Unit.INTEGRAL_FLUENCE.value
    )
    assert output.quantity is source.quantity
    assert output.y_unit is source.y_unit
    assert output.completeness is SpectrumCompleteness.INCOMPLETE
    assert len(_lower_window_reasons(output)) == 1

    # Diagnostic-only change: keep the pre-existing output window and values.
    assert output.bin_edges == pytest.approx(
        (output_minimum, sqrt(300.0), sqrt(3000.0), 80.0, 180.0, sqrt(100000.0 / 3.0))
    )
    assert output.y == pytest.approx((2.0, 2.0, 2.0, 0.0, 0.0))
    assert output.integral_conserved() == pytest.approx(returned)


@pytest.mark.parametrize("range_offset", (-1.0, 137.0))
def test_additive_range_origin_cancels_from_supported_count_ledger(range_offset: float) -> None:
    ordinary = shield_hze_spectrum_primary_through_al(
        spectrum=_source(), table=_table(), thickness_g_cm2=20.0,
    )
    shifted = shield_hze_spectrum_primary_through_al(
        spectrum=_source(), table=_table(range_offset=range_offset), thickness_g_cm2=20.0,
    )
    keys = (
        "source_ledger_integral",
        "supported_exit_integral_in_output_window",
        "supported_exit_integral_below_output_window",
        "unresolved_transport_integral_on_declared_source",
    )
    for key in keys:
        assert float(_ledger(shifted)[key]) == pytest.approx(float(_ledger(ordinary)[key]))
    assert shifted.bin_edges == pytest.approx(ordinary.bin_edges)
    assert shifted.y == pytest.approx(ordinary.y)
    assert _ledger(shifted)["physical_stopping_classification"] == "not_established"


def test_no_lower_window_warning_when_all_supported_exits_fit() -> None:
    output = shield_hze_spectrum_primary_through_al(
        spectrum=_source(), table=_table(), thickness_g_cm2=1.0,
    )
    ledger = _ledger(output)
    # Known inputs 10..100 map to 9..99, above output lower edge 5.7735.
    assert not _lower_window_reasons(output)
    assert float(ledger["supported_exit_integral_below_output_window"]) == 0.0
    assert float(ledger["supported_exit_integral_in_output_window"]) == pytest.approx(180.0)
    assert float(ledger["unresolved_transport_integral_on_declared_source"]) == 0.0


def test_all_below_table_is_unresolved_not_stopped() -> None:
    output = shield_hze_spectrum_primary_through_al(
        spectrum=_source(), table=_table(), thickness_g_cm2=1000.0,
    )
    ledger = _ledger(output)
    assert not _lower_window_reasons(output)
    assert float(ledger["supported_exit_integral_in_output_window"]) == 0.0
    assert float(ledger["supported_exit_integral_below_output_window"]) == 0.0
    assert float(ledger["unresolved_transport_integral_on_declared_source"]) == pytest.approx(180.0)
    assert ledger["physical_stopping_classification"] == "not_established"


def test_input_outside_table_is_in_unresolved_partition_without_extrapolation() -> None:
    table = HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(20.0, 30.0, 80.0),
        range_g_cm2=(20.0, 30.0, 80.0),
    )
    output = shield_hze_spectrum_primary_through_al(
        spectrum=_source(), table=table, thickness_g_cm2=20.0,
    )
    ledger = _ledger(output)
    # Only inputs 40..80 have a supported residual energy, namely 20..60.
    assert float(ledger["source_ledger_integral"]) == pytest.approx(180.0)
    assert float(ledger["supported_exit_integral_in_output_window"]) == pytest.approx(80.0)
    assert float(ledger["supported_exit_integral_below_output_window"]) == 0.0
    assert float(ledger["unresolved_transport_integral_on_declared_source"]) == pytest.approx(100.0)
    assert output.integral_conserved() == pytest.approx(80.0)
    assert not _lower_window_reasons(output)


def test_zero_source_does_not_report_positive_omitted_count() -> None:
    output = shield_hze_spectrum_primary_through_al(
        spectrum=_source(zero=True), table=_table(), thickness_g_cm2=20.0,
    )
    assert not _lower_window_reasons(output)
    assert float(_ledger(output)["supported_exit_integral_below_output_window"]) == 0.0


def test_zero_thickness_identity_does_not_require_transport_ledger() -> None:
    source = _source()
    output = shield_hze_spectrum_primary_through_al(
        spectrum=source, table=_table(), thickness_g_cm2=0.0,
    )
    assert output.x == source.x
    assert output.y == source.y
    assert output.ordinate_basis is source.ordinate_basis
    assert "source_ledger_scope" not in _ledger(output)
    assert not _lower_window_reasons(output)
