from __future__ import annotations

import numpy as np
import pytest

from radar.dose.electron_shieldose2 import (
    DEFAULT_DETECTOR,
    calculate_electron_dose,
    load_normative_elbrbas2_database,
)


def test_loads_packaged_elbrbas2_database_for_silicon() -> None:
    database = load_normative_elbrbas2_database()

    assert database.detector == DEFAULT_DETECTOR
    assert database.electron_energy_min_MeV == pytest.approx(1.0e-3)
    assert database.electron_energy_max_MeV == pytest.approx(100.0)
    assert database.response_energy_min_MeV == pytest.approx(2.0e-3)
    assert database.response_energy_max_MeV == pytest.approx(50.0)


def test_calculates_nonnegative_electron_fluence_dose() -> None:
    database = load_normative_elbrbas2_database()

    energy = np.array([0.1, 1.0, 10.0])
    fluence = np.array([1.0e6, 1.0e5, 1.0e4])
    depths = np.array([0.01, 0.1, 1.0, 10.0])

    result = calculate_electron_dose(
        energy_MeV=energy,
        spectrum=fluence,
        depths_gcm2=depths,
        database=database,
        quantity="fluence",
        n_energy_points=101,
    )

    assert result.output_unit == "rad"
    assert tuple(result.depths_gcm2) == tuple(depths)
    assert np.all(result.electron_slab >= 0.0)
    assert np.all(result.brems_slab >= 0.0)
    assert np.all(result.total_slab == result.electron_slab + result.brems_slab)
    assert np.any(result.total_slab > 0.0)
    assert result.metadata["bremsstrahlung_dose_included"] is True
    assert result.metadata["electron_output_spectrum_calculated"] is False


def test_rejects_zero_or_negative_depths() -> None:
    database = load_normative_elbrbas2_database()

    with pytest.raises(ValueError, match="finite and positive"):
        calculate_electron_dose(
            energy_MeV=[0.1, 1.0],
            spectrum=[1.0e6, 1.0e5],
            depths_gcm2=[0.0, 1.0],
            database=database,
            quantity="fluence",
            n_energy_points=51,
        )


def test_rejects_invalid_quantity() -> None:
    database = load_normative_elbrbas2_database()

    with pytest.raises(ValueError, match="quantity must be"):
        calculate_electron_dose(
            energy_MeV=[0.1, 1.0],
            spectrum=[1.0e6, 1.0e5],
            depths_gcm2=[0.1, 1.0, 10.0],
            database=database,
            quantity="bad",  # type: ignore[arg-type]
            n_energy_points=51,
        )
