from __future__ import annotations

from types import SimpleNamespace

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.electron_shieldose2 import load_normative_elbrbas2_database
from radar.pipelines.dose import (
    DOSE_PIPELINE_MODEL,
    DOSE_PIPELINE_MODEL_VERSION,
    calculate_dose_pipeline_from_pipeline_results,
    calculate_dose_pipeline_from_products,
    let_products_by_thickness_from_pipeline_results,
    shielded_products_by_thickness_from_pipeline_results,
)
from radar.shielding.proton_si_let import ProtonSiLetTable


def _proton_table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(0.5, 1.0, 4.0, 8.0, 20.0),
        let_total_mev_cm2_mg=(3.0, 3.0, 3.0, 3.0, 3.0),
        let_electronic_mev_cm2_mg=(2.0, 2.0, 2.0, 2.0, 2.0),
        let_nuclear_mev_cm2_mg=(1.0, 1.0, 1.0, 1.0, 1.0),
    )


def _proton_marker(source: RadiationSource) -> str:
    if source is RadiationSource.ERB:
        return "erb_proton_al_shielding_primary_survival_secondary"
    if source is RadiationSource.SEP:
        return "sep_proton_al_shielding_primary_survival_secondary"
    if source is RadiationSource.GCR:
        return "gcr_proton_al_shielding_primary_survival_secondary"
    raise AssertionError(source)


def _shielded_proton_product(source: RadiationSource) -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=Spectrum1D(
            x=(1.0, 4.0),
            y=(10.0, 1.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=source,
            model=f"{source.value}_source+{_proton_marker(source)}",
        ),
    )


def _erb_electron_on_orbit_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=Spectrum1D(
            x=(0.1, 1.0, 10.0),
            y=(1.0e6, 1.0e5, 1.0e4),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.ELECTRON,
            source=RadiationSource.ERB,
            model="erb_electron_on_orbit",
        ),
    )


def _let_product(source: RadiationSource) -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_LET_FLUENCE,
        spectrum=Spectrum1D(
            x=(1.0, 4.0),
            y=(10.0, 0.0),
            x_unit=Unit.LET,
            y_unit=Unit.DIFFERENTIAL_LET_FLUENCE,
            quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
            particle=Particle.HZE,
            source=source,
            model=f"{source.value}_combined_proton_hze_let",
        ),
    )


def test_calculates_dose_pipeline_components_and_totals() -> None:
    thicknesses = (0.1, 1.0)
    shielded_products_by_thickness = {
        0.1: (
            _shielded_proton_product(RadiationSource.ERB),
            _shielded_proton_product(RadiationSource.SEP),
            _shielded_proton_product(RadiationSource.GCR),
        ),
        1.0: (
            _shielded_proton_product(RadiationSource.ERB),
            _shielded_proton_product(RadiationSource.SEP),
            _shielded_proton_product(RadiationSource.GCR),
        ),
    }
    let_products_by_thickness = {
        0.1: (
            _let_product(RadiationSource.SEP),
            _let_product(RadiationSource.GCR),
        ),
        1.0: (
            _let_product(RadiationSource.SEP),
            _let_product(RadiationSource.GCR),
        ),
    }

    result = calculate_dose_pipeline_from_products(
        thicknesses_g_cm2=thicknesses,
        erb_on_orbit_products=(_erb_electron_on_orbit_product(),),
        shielded_products_by_thickness=shielded_products_by_thickness,
        let_products_by_thickness=let_products_by_thickness,
        proton_si_let_table=_proton_table(),
        electron_database=load_normative_elbrbas2_database(),
        electron_n_energy_points=101,
    )

    assert result.model == DOSE_PIPELINE_MODEL
    assert result.version == DOSE_PIPELINE_MODEL_VERSION
    assert len(result.components) == 12
    assert len(result.included_components) == 8
    assert len(result.optional_components) == 4

    assert {component.component for component in result.included_components} == {
        "erb_electron_shieldose2",
        "erb_proton",
        "sep_proton",
        "gcr_proton",
    }
    assert {component.component for component in result.optional_components} == {
        "sep_combined_let_optional",
        "gcr_combined_let_optional",
    }

    assert tuple(total.thickness_g_cm2 for total in result.totals_by_thickness) == thicknesses
    assert all(total.included_total_rad > 0.0 for total in result.totals_by_thickness)
    assert all(total.optional_total_rad > 0.0 for total in result.totals_by_thickness)
    assert all(total.component_count == 6 for total in result.totals_by_thickness)


def test_dose_pipeline_primary_total_excludes_optional_let_dose() -> None:
    thicknesses = (0.1,)
    proton = _shielded_proton_product(RadiationSource.SEP)
    let = _let_product(RadiationSource.SEP)

    result = calculate_dose_pipeline_from_products(
        thicknesses_g_cm2=thicknesses,
        erb_on_orbit_products=(),
        shielded_products_by_thickness={0.1: (proton,)},
        let_products_by_thickness={0.1: (let,)},
        proton_si_let_table=_proton_table(),
        electron_n_energy_points=101,
    )

    included = sum(component.dose_rad for component in result.included_components)
    optional = sum(component.dose_rad for component in result.optional_components)

    assert result.totals_by_thickness[0].included_total_rad == pytest.approx(included)
    assert result.totals_by_thickness[0].optional_total_rad == pytest.approx(optional)
    assert included > 0.0
    assert optional > 0.0


def test_dose_pipeline_rejects_invalid_thicknesses() -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_dose_pipeline_from_products(
            thicknesses_g_cm2=(0.0,),
            erb_on_orbit_products=(),
            shielded_products_by_thickness={},
            let_products_by_thickness={},
            electron_n_energy_points=101,
        )



def _fake_shielding_result(
    *,
    thickness_g_cm2: float,
    shielded_products: tuple[SpectrumProduct, ...] = (),
    let_products: tuple[SpectrumProduct, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        thickness_g_cm2=thickness_g_cm2,
        shielded_products=shielded_products,
        let_products=let_products,
    )


def _fake_erb_shielding_result(
    *,
    thickness_g_cm2: float,
    shielded_product: SpectrumProduct,
) -> SimpleNamespace:
    return SimpleNamespace(
        thickness_g_cm2=thickness_g_cm2,
        shielded_product=shielded_product,
    )


def test_collects_pipeline_products_by_thickness() -> None:
    sep_result = SimpleNamespace(
        shielding_let_by_thickness=(
            _fake_shielding_result(
                thickness_g_cm2=0.1,
                shielded_products=(_shielded_proton_product(RadiationSource.SEP),),
                let_products=(_let_product(RadiationSource.SEP),),
            ),
        ),
    )
    gcr_result = SimpleNamespace(
        shielding_let_by_thickness=(
            _fake_shielding_result(
                thickness_g_cm2=0.1,
                shielded_products=(_shielded_proton_product(RadiationSource.GCR),),
                let_products=(_let_product(RadiationSource.GCR),),
            ),
        ),
    )
    erb_result = SimpleNamespace(
        shielding_by_thickness=(
            _fake_erb_shielding_result(
                thickness_g_cm2=0.1,
                shielded_product=_shielded_proton_product(RadiationSource.ERB),
            ),
        ),
    )

    shielded = shielded_products_by_thickness_from_pipeline_results(
        sep_result=sep_result,
        gcr_result=gcr_result,
        erb_result=erb_result,
    )
    let_products = let_products_by_thickness_from_pipeline_results(
        sep_result=sep_result,
        gcr_result=gcr_result,
    )

    assert len(shielded[0.1]) == 3
    assert {
        product.spectrum.source
        for product in shielded[0.1]
    } == {RadiationSource.SEP, RadiationSource.GCR, RadiationSource.ERB}
    assert len(let_products[0.1]) == 2
    assert {
        product.spectrum.source
        for product in let_products[0.1]
    } == {RadiationSource.SEP, RadiationSource.GCR}


def test_calculates_dose_pipeline_from_source_pipeline_results() -> None:
    sep_result = SimpleNamespace(
        shielding_let_by_thickness=(
            _fake_shielding_result(
                thickness_g_cm2=0.1,
                shielded_products=(_shielded_proton_product(RadiationSource.SEP),),
                let_products=(_let_product(RadiationSource.SEP),),
            ),
            _fake_shielding_result(
                thickness_g_cm2=1.0,
                shielded_products=(_shielded_proton_product(RadiationSource.SEP),),
                let_products=(_let_product(RadiationSource.SEP),),
            ),
        ),
    )
    gcr_result = SimpleNamespace(
        shielding_let_by_thickness=(
            _fake_shielding_result(
                thickness_g_cm2=0.1,
                shielded_products=(_shielded_proton_product(RadiationSource.GCR),),
                let_products=(_let_product(RadiationSource.GCR),),
            ),
            _fake_shielding_result(
                thickness_g_cm2=1.0,
                shielded_products=(_shielded_proton_product(RadiationSource.GCR),),
                let_products=(_let_product(RadiationSource.GCR),),
            ),
        ),
    )
    erb_result = SimpleNamespace(
        on_orbit_products=(_erb_electron_on_orbit_product(),),
        shielding_by_thickness=(
            _fake_erb_shielding_result(
                thickness_g_cm2=0.1,
                shielded_product=_shielded_proton_product(RadiationSource.ERB),
            ),
            _fake_erb_shielding_result(
                thickness_g_cm2=1.0,
                shielded_product=_shielded_proton_product(RadiationSource.ERB),
            ),
        ),
    )
    config = SimpleNamespace(
        shielding=SimpleNamespace(thicknesses_g_cm2=(0.1, 1.0)),
    )

    result = calculate_dose_pipeline_from_pipeline_results(
        config=config,
        sep_result=sep_result,
        gcr_result=gcr_result,
        erb_result=erb_result,
        proton_si_let_table=_proton_table(),
        electron_database=load_normative_elbrbas2_database(),
        electron_n_energy_points=101,
    )

    assert len(result.components) == 12
    assert len(result.included_components) == 8
    assert len(result.optional_components) == 4
    assert all(total.included_total_rad > 0.0 for total in result.totals_by_thickness)
    assert all(total.optional_total_rad > 0.0 for total in result.totals_by_thickness)
