from radar.shielding.proton_al_survival import DEFAULT_AL27_NONELASTIC_MT
from radar.shielding.resources import (
    load_normative_al27_nonelastic_cross_section_table,
    load_normative_hze_al_range_tables,
    load_normative_hze_si_let_tables,
    load_normative_proton_al_range_table,
    load_normative_proton_si_let_table,
    load_normative_secondary_proton_kernel,
    load_normative_shielding_tables,
)


def test_normative_proton_resource_tables_load() -> None:
    proton_al = load_normative_proton_al_range_table()
    proton_si = load_normative_proton_si_let_table()

    assert len(proton_al.energy_mev) > 10
    assert len(proton_si.energy_mev) > 10
    assert proton_al.range_at_energy(10.0) > 0.0
    assert proton_si.total_let_at_energy(10.0) > 0.0


def test_normative_hze_resource_tables_load() -> None:
    hze_al = load_normative_hze_al_range_tables()
    hze_si = load_normative_hze_si_let_tables()

    assert hze_al
    assert hze_si
    assert set(hze_al) == set(hze_si)

    sample_z = max(hze_al)
    assert hze_al[sample_z].z == sample_z
    assert hze_si[sample_z].z == sample_z
    assert len(hze_al[sample_z].energy_mev_per_nucleon) > 10
    assert len(hze_si[sample_z].energy_mev_per_nucleon) > 10


def test_normative_al27_nonelastic_xs_resource_merges_duplicate_energies() -> None:
    table = load_normative_al27_nonelastic_cross_section_table()

    assert table.mt == DEFAULT_AL27_NONELASTIC_MT
    assert len(set(table.energy_mev)) == len(table.energy_mev)
    assert table.sigma_at_energy(30.0) > 0.0
    assert table.sigma_at_energy(40.0) > 0.0


def test_normative_secondary_proton_kernel_resource_loads() -> None:
    kernel = load_normative_secondary_proton_kernel()

    assert kernel.products
    assert kernel.cross_sections_by_mt

    for product in kernel.products:
        assert len(set(product.incident_energy_mev)) == len(product.incident_energy_mev)


def test_normative_shielding_tables_bundle_loads() -> None:
    tables = load_normative_shielding_tables()

    assert tables.proton_al_range.energy_mev
    assert tables.proton_si_let.energy_mev
    assert tables.hze_al_ranges_by_z
    assert tables.hze_si_let_by_z
    assert tables.al27_nonelastic_xs.mt == DEFAULT_AL27_NONELASTIC_MT
    assert tables.secondary_proton_kernel.products
