import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.erb.output_tables import (
    ERB_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    ERB_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
    ERB_OUTPUT_LOCATION_ON_ORBIT,
    ERB_OUTPUT_SHIELDING_NOT_APPLIED,
    ERB_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED,
    erb_product_output_tables,
)
from radar.output_tables import OutputTableKind


def _erb_spectrum(
    *,
    particle: Particle = Particle.PROTON,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    source: RadiationSource = RadiationSource.ERB,
    model: str = "erb_test_model:proton:mean_flux",
) -> Spectrum1D:
    return Spectrum1D(
        x=(0.1, 1.0, 10.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model=model,
    )


def _erb_product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=_erb_spectrum(),
        label="ERB proton mean flux",
    )


def test_erb_product_output_tables_include_deferred_effects_metadata() -> None:
    table = erb_product_output_tables((_erb_product(),))[0]

    assert table.table_id == (
        "erb_proton_mean_flux_mean_differential_flux_on_orbit"
    )
    assert table.kind is OutputTableKind.SPECTRUM
    assert table.columns[0].title == "\u042d\u043d\u0435\u0440\u0433\u0438\u044f"
    assert table.columns[0].unit == "MeV"
    assert table.columns[1].title == (
        "\u0421\u0440\u0435\u0434\u043d\u0438\u0439 "
        "\u0434\u0438\u0444\u0444\u0435\u0440\u0435\u043d\u0446\u0438\u0430\u043b\u044c\u043d\u044b\u0439 "
        "\u043f\u043e\u0442\u043e\u043a"
    )
    assert table.rows[0].cells == (0.1, 1.0)

    metadata = dict(table.metadata)

    assert metadata["location"] == ERB_OUTPUT_LOCATION_ON_ORBIT
    assert metadata["stage"] == "source_model"
    assert metadata["shielding"] == ERB_OUTPUT_SHIELDING_NOT_APPLIED
    assert metadata["product_kind"] == "mean_flux"
    assert metadata["spectrum_quantity"] == "mean_differential_flux"
    assert metadata["dose_status"] == ERB_OUTPUT_DOSE_STATUS_NOT_CALCULATED
    assert (
        metadata["single_event_effects_status"]
        == ERB_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED
    )
    assert metadata["dose_see_dependency"] == ERB_OUTPUT_DOSE_SEE_DEFERRED_REASON
    assert metadata["quantity"] == "mean_differential_flux"
    assert metadata["particle"] == "proton"
    assert metadata["source"] == "erb"
    assert metadata["model"] == "erb_test_model:proton:mean_flux"


def test_erb_product_output_tables_include_particle_and_product_tokens() -> None:
    products = (
        SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=_erb_spectrum(
                particle=Particle.ELECTRON,
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                y_unit=Unit.DIFFERENTIAL_FLUENCE,
                model="erb_test_model:electron:fluence",
            ),
        ),
    )

    table = erb_product_output_tables(products)[0]

    assert table.table_id == (
        "erb_electron_mission_fluence_differential_fluence_on_orbit"
    )
    assert "\u042d\u043b\u0435\u043a\u0442\u0440\u043e\u043d\u044b" in table.title


def test_erb_product_output_tables_reject_empty_products() -> None:
    with pytest.raises(ValueError, match="at least one product"):
        erb_product_output_tables(())


def test_erb_product_output_tables_reject_wrong_source() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=_erb_spectrum(source=RadiationSource.GCR),
    )

    with pytest.raises(ValueError, match="spectrum source"):
        erb_product_output_tables((product,))
