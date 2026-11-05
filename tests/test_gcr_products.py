from dataclasses import replace
from math import inf, nan

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import ModelReference, Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.products import (
    GCR_FINITE_DOMAIN_INTEGRATION_STEP,
    GCR_MISSION_PRODUCTS_MODEL_SUFFIX,
    GcrFluxTimeBin,
    GcrMissionProducts,
    calculate_gcr_mission_products,
)


def _gcr_flux_spectrum(
    *,
    x: tuple[float, ...] = (10.0, 20.0),
    y: tuple[float, ...] = (1.0, 2.0),
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.GCR,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUX,
    model: str = "gcr_test",
) -> Spectrum1D:
    return Spectrum1D(
        x=x,
        y=y,
        x_unit=x_unit,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model=model,
    )


def test_gcr_mission_products_calculate_mean_maximum_and_fluence() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(y=(1.0, 2.0)),
                duration_seconds=10.0,
            ),
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(y=(3.0, 6.0)),
                duration_seconds=20.0,
            ),
        )
    )

    assert result.mean_flux.spectrum.y == pytest.approx((70.0 / 30.0, 140.0 / 30.0))
    assert result.mean_integral_flux.spectrum.y == pytest.approx((35.0, 0.0))
    assert result.maximum_flux.spectrum.y == pytest.approx((3.0, 6.0))
    assert result.maximum_integral_flux.spectrum.y == pytest.approx((45.0, 0.0))
    assert result.mission_fluence.spectrum.y == pytest.approx((70.0, 140.0))
    assert result.mission_integral_fluence.spectrum.y == pytest.approx((1050.0, 0.0))


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("incomplete_values", ((0.0, 0.0), (0.5, 1.0)))
def test_gcr_products_merge_all_bin_status_and_references(
    reverse: bool,
    incomplete_values: tuple[float, float],
) -> None:
    complete = replace(
        _gcr_flux_spectrum(y=(2.0, 4.0)),
        completeness=SpectrumCompleteness.COMPLETE,
        model_reference=ModelReference(
            name="gcr_test", version="test-v1",
            document_refs=("document-a", "shared-document"), dataset_refs=("dataset-a",),
        ),
    )
    incomplete = replace(
        complete,
        y=incomplete_values,
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("unconfirmed later input", "additional input reason"),
        model_reference=ModelReference(
            name="gcr_test", version="test-v1",
            document_refs=("document-b", "shared-document"), dataset_refs=("dataset-b",),
        ),
    )
    bins: tuple[GcrFluxTimeBin, ...] = (
        GcrFluxTimeBin(complete, 2.0),
        GcrFluxTimeBin(incomplete, 3.0),
    )
    if reverse:
        bins = tuple(reversed(bins))
    result = calculate_gcr_mission_products(bins)
    assert result.maximum_flux.spectrum.y == complete.y
    assert result.mission_fluence.spectrum.y == pytest.approx(
        tuple(2.0 * a + 3.0 * b for a, b in zip(complete.y, incomplete.y, strict=True))
    )
    for product in result.products:
        spectrum = product.spectrum
        assert spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        assert set(spectrum.incomplete_reasons) == set(incomplete.incomplete_reasons)
        assert spectrum.model_reference is not None
        assert set(spectrum.model_reference.document_refs) == {
            "document-a", "document-b", "shared-document"
        }
        assert set(spectrum.model_reference.dataset_refs) == {"dataset-a", "dataset-b"}
        aggregation = next(
            step for step in spectrum.transformation_chain
            if step.name == GCR_MISSION_PRODUCTS_MODEL_SUFFIX
        )
        assert aggregation.version == "3"
        assert dict(aggregation.parameters)["time_bin_count"] == "2"
        assert dict(aggregation.parameters)["input_provenance_policy"] == (
            "all_bins_including_zero_and_nonmaximum_bins"
        )


@pytest.mark.parametrize("reverse", (False, True))
def test_gcr_products_do_not_promote_unknown_input_completeness(reverse: bool) -> None:
    known = replace(_gcr_flux_spectrum(), completeness=SpectrumCompleteness.COMPLETE)
    unknown = _gcr_flux_spectrum()
    bins = (GcrFluxTimeBin(known, 1.0), GcrFluxTimeBin(unknown, 1.0))
    result = calculate_gcr_mission_products(tuple(reversed(bins)) if reverse else bins)
    assert all(product.spectrum.completeness is None for product in result.products)


def test_gcr_products_union_distinct_reasons_from_multiple_incomplete_bins() -> None:
    bins = tuple(
        GcrFluxTimeBin(
            replace(
                _gcr_flux_spectrum(y=(0.0, 0.0)),
                completeness=SpectrumCompleteness.INCOMPLETE,
                incomplete_reasons=(reason, "common issue"),
            ),
            1.0,
        )
        for reason in ("first issue", "second issue", "third issue")
    )
    result = calculate_gcr_mission_products(bins)
    for product in result.products:
        assert product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        assert product.spectrum.incomplete_reasons == (
            "first issue", "common issue", "second issue", "third issue"
        )
        assert product.spectrum.y == (0.0, 0.0)


@pytest.mark.parametrize("component", ("akl", "total"))
def test_gcr_products_preserve_later_carbon_akl_gamma_guard(component: str) -> None:
    from radar.gcr.components import OstGcrSolarState
    from radar.gcr.source_spectra import (
        OST_AKL_CONVENTIONS_INCOMPLETE_REASON,
        OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON,
        calculate_ost_gcr_source_spectra_outside_magnetosphere,
    )

    state = OstGcrSolarState(60.0, 50.0, 10.0, 100.0, 24, True)
    spectra = []
    for time in (2.0, 6.0):
        ion = calculate_ost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=(10.0, 100.0),
            solar_state=state,
            nek_wolf_number=60.0,
            akl_years_from_cycle_start=time,
            symbols=("C",),
        ).ion("C")
        spectra.append(ion.akl if component == "akl" else ion.total)
    first, later = spectra
    assert OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON not in first.incomplete_reasons
    assert OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON in later.incomplete_reasons
    if component == "akl":
        assert later.y == (0.0, 0.0)
    result = calculate_gcr_mission_products(
        (GcrFluxTimeBin(first, 2.0), GcrFluxTimeBin(later, 3.0))
    )
    assert result.maximum_flux.spectrum.y == first.y
    for product in result.products:
        assert product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        assert set(product.spectrum.incomplete_reasons) == {
            OST_AKL_CONVENTIONS_INCOMPLETE_REASON, OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON
        }


def test_gcr_integral_products_stop_at_declared_finite_energy_boundary() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(
                    x=(10.0, 20.0),
                    y=(0.01, 0.0025),
                ),
                duration_seconds=1.0,
            ),
        ),
        upper_energy_bound=40.0,
    )

    assert result.mean_integral_flux.spectrum.y == pytest.approx((0.075, 0.025))
    finite_step = result.mean_integral_flux.spectrum.transformation_chain[-1]
    assert finite_step.name == GCR_FINITE_DOMAIN_INTEGRATION_STEP
    assert dict(finite_step.parameters) == {
        "integration": "local_power_law_finite_domain",
        "upper_energy_bound": "40",
        "upper_energy_bound_unit": Unit.MEV.value,
    }


def test_gcr_mission_products_use_product_contract_quantities_and_units() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(),
                duration_seconds=10.0,
            ),
        )
    )

    assert result.mean_flux.kind is RadiationProductKind.MEAN_FLUX
    assert result.mean_flux.spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert result.mean_flux.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX

    assert result.mean_integral_flux.kind is RadiationProductKind.MEAN_FLUX
    assert result.mean_integral_flux.spectrum.quantity is SpectrumQuantity.MEAN_INTEGRAL_FLUX
    assert result.mean_integral_flux.spectrum.y_unit is Unit.INTEGRAL_FLUX

    assert result.maximum_flux.kind is RadiationProductKind.MAXIMUM_FLUX
    assert (
        result.maximum_flux.spectrum.quantity
        is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    )
    assert result.maximum_flux.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX

    assert result.maximum_integral_flux.kind is RadiationProductKind.MAXIMUM_FLUX
    assert (
        result.maximum_integral_flux.spectrum.quantity
        is SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX
    )
    assert result.maximum_integral_flux.spectrum.y_unit is Unit.INTEGRAL_FLUX

    assert result.mission_fluence.kind is RadiationProductKind.MISSION_FLUENCE
    assert result.mission_fluence.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert result.mission_fluence.spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE

    assert result.mission_integral_fluence.kind is RadiationProductKind.MISSION_FLUENCE
    assert (
        result.mission_integral_fluence.spectrum.quantity
        is SpectrumQuantity.INTEGRAL_FLUENCE
    )
    assert result.mission_integral_fluence.spectrum.y_unit is Unit.INTEGRAL_FLUENCE


def test_gcr_mission_products_preserve_grid_particle_source_and_order() -> None:
    source_spectrum = _gcr_flux_spectrum(
        x=(1.0, 10.0, 100.0),
        y=(4.0, 5.0, 6.0),
        particle=Particle.HZE,
        model="gcr_o_total",
    )

    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=source_spectrum,
                duration_seconds=2.0,
            ),
        )
    )

    assert result.products == (
        result.mean_flux,
        result.mean_integral_flux,
        result.maximum_flux,
        result.maximum_integral_flux,
        result.mission_fluence,
        result.mission_integral_fluence,
    )

    for product in result.products:
        assert product.spectrum.x == source_spectrum.x
        assert product.spectrum.x_unit is source_spectrum.x_unit
        assert product.spectrum.particle is Particle.HZE
        assert product.spectrum.source is RadiationSource.GCR
        assert product.spectrum.model.startswith(
            f"gcr_o_total:{GCR_MISSION_PRODUCTS_MODEL_SUFFIX}:"
        )


def test_gcr_mission_products_accept_gev_per_nucleon_input_grid() -> None:
    result = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(
                    x=(0.1, 1.0),
                    x_unit=Unit.GEV_PER_NUCLEON,
                ),
                duration_seconds=10.0,
            ),
        )
    )

    assert result.mean_flux.spectrum.x_unit is Unit.GEV_PER_NUCLEON
    assert result.mean_integral_flux.spectrum.x_unit is Unit.GEV_PER_NUCLEON
    assert result.mission_fluence.spectrum.x_unit is Unit.GEV_PER_NUCLEON
    assert result.mission_integral_fluence.spectrum.x_unit is Unit.GEV_PER_NUCLEON


@pytest.mark.parametrize("bad_duration", [0.0, -1.0, inf, nan])
def test_gcr_flux_time_bin_rejects_bad_duration(bad_duration: float) -> None:
    with pytest.raises(ValueError, match="finite|positive"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(),
            duration_seconds=bad_duration,
        )


def test_gcr_mission_products_reject_empty_time_bins() -> None:
    with pytest.raises(ValueError, match="at least one"):
        calculate_gcr_mission_products(())


def test_gcr_mission_products_reject_non_gcr_input() -> None:
    with pytest.raises(ValueError, match="source must be GCR"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(source=RadiationSource.SEP),
            duration_seconds=1.0,
        )


def test_gcr_mission_products_reject_non_flux_input_quantity() -> None:
    with pytest.raises(ValueError, match="quantity"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(
                y_unit=Unit.DIFFERENTIAL_FLUENCE,
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            ),
            duration_seconds=1.0,
        )


def test_gcr_mission_products_reject_wrong_flux_unit() -> None:
    with pytest.raises(ValueError, match="unit"):
        GcrFluxTimeBin(
            spectrum=_gcr_flux_spectrum(y_unit=Unit.DIFFERENTIAL_FLUENCE),
            duration_seconds=1.0,
        )


def test_gcr_mission_products_reject_mismatched_grids() -> None:
    with pytest.raises(ValueError, match="same energy grid"):
        calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x=(10.0, 20.0)),
                    duration_seconds=1.0,
                ),
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x=(10.0, 30.0)),
                    duration_seconds=1.0,
                ),
            )
        )


def test_gcr_mission_products_reject_mismatched_particles() -> None:
    with pytest.raises(ValueError, match="same particle"):
        calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(particle=Particle.PROTON),
                    duration_seconds=1.0,
                ),
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(particle=Particle.HZE),
                    duration_seconds=1.0,
                ),
            )
        )


def test_gcr_mission_products_reject_mismatched_energy_units() -> None:
    with pytest.raises(ValueError, match="same energy unit"):
        calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x_unit=Unit.MEV),
                    duration_seconds=1.0,
                ),
                GcrFluxTimeBin(
                    spectrum=_gcr_flux_spectrum(x_unit=Unit.GEV_PER_NUCLEON),
                    duration_seconds=1.0,
                ),
            )
        )


def test_gcr_mission_products_reject_invalid_result_product_kinds() -> None:
    valid = calculate_gcr_mission_products(
        (
            GcrFluxTimeBin(
                spectrum=_gcr_flux_spectrum(),
                duration_seconds=1.0,
            ),
        )
    )

    with pytest.raises(ValueError, match="MEAN_FLUX"):
        GcrMissionProducts(
            mean_flux=SpectrumProduct(
                kind=RadiationProductKind.MAXIMUM_FLUX,
                spectrum=valid.maximum_flux.spectrum,
            ),
            mean_integral_flux=valid.mean_integral_flux,
            maximum_flux=valid.maximum_flux,
            maximum_integral_flux=valid.maximum_integral_flux,
            mission_fluence=valid.mission_fluence,
            mission_integral_fluence=valid.mission_integral_fluence,
        )



def test_gcr_model_result_accepts_integral_energy_quantities() -> None:
    integral_flux = _gcr_flux_spectrum(
        y_unit=Unit.INTEGRAL_FLUX,
        quantity=SpectrumQuantity.MEAN_INTEGRAL_FLUX,
    )
    integral_fluence = _gcr_flux_spectrum(
        y_unit=Unit.INTEGRAL_FLUENCE,
        quantity=SpectrumQuantity.INTEGRAL_FLUENCE,
    )

    result = GcrMissionProducts(
        mean_flux=SpectrumProduct(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=_gcr_flux_spectrum(
                quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            ),
        ),
        mean_integral_flux=SpectrumProduct(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=integral_flux,
        ),
        maximum_flux=SpectrumProduct(
            kind=RadiationProductKind.MAXIMUM_FLUX,
            spectrum=_gcr_flux_spectrum(
                quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            ),
        ),
        maximum_integral_flux=SpectrumProduct(
            kind=RadiationProductKind.MAXIMUM_FLUX,
            spectrum=_gcr_flux_spectrum(
                y_unit=Unit.INTEGRAL_FLUX,
                quantity=SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
            ),
        ),
        mission_fluence=SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=_gcr_flux_spectrum(
                y_unit=Unit.DIFFERENTIAL_FLUENCE,
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            ),
        ),
        mission_integral_fluence=SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=integral_fluence,
        ),
    )

    assert result.mean_integral_flux.spectrum is integral_flux
    assert result.mission_integral_fluence.spectrum is integral_fluence
