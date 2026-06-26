"""Normative GCR, NEK and AKL parameter tables."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Final

OST_NEK_HE_BC_PRINTED: Final[float] = 6.6e-2
OST_NEK_HE_BC_USED: Final[float] = 6.6e-3
OST_NEK_HE_BC_CORRECTION_NOTE: Final[str] = (
    "OST Appendix V.2 prints b_c(He)=6.6e-2; RADAR uses 6.6e-3 "
    "as a probable misprint correction because the printed value inflates "
    "low-energy He NEK by orders of magnitude at high solar activity."
)


@dataclass(frozen=True)
class GcrIonParameters:
    """Main GCR ion parameters."""

    symbol: str
    z: int
    mass_number: float
    c: float
    gamma: float
    alpha: float

    @property
    def mass_to_charge(self) -> float:
        """Return A/Z."""

        return self.mass_number / self.z

    def __post_init__(self) -> None:
        if not self.symbol:
            msg = "GCR ion symbol must not be empty."
            raise ValueError(msg)

        if self.z < 1:
            msg = "GCR ion charge Z must be positive."
            raise ValueError(msg)

        if not isfinite(self.mass_number) or self.mass_number <= 0.0:
            msg = "GCR ion mass number must be positive and finite."
            raise ValueError(msg)

        for name, value in (
            ("C", self.c),
            ("gamma", self.gamma),
            ("alpha", self.alpha),
        ):
            if not isfinite(value):
                msg = f"GCR ion {name} parameter must be finite."
                raise ValueError(msg)

            if value < 0.0:
                msg = f"GCR ion {name} parameter must be non-negative."
                raise ValueError(msg)


@dataclass(frozen=True)
class NekParameters:
    """Low-energy cosmic-ray component parameters from OST Appendix V.2."""

    z: int
    ac: float
    bc: float
    ag: float
    bg: float

    def __post_init__(self) -> None:
        if self.z < 1:
            msg = "NEK ion charge Z must be positive."
            raise ValueError(msg)

        for name, value in (
            ("ac", self.ac),
            ("bc", self.bc),
            ("ag", self.ag),
            ("bg", self.bg),
        ):
            if not isfinite(value):
                msg = f"NEK {name} parameter must be finite."
                raise ValueError(msg)


@dataclass(frozen=True)
class AklParameters:
    """Anomalous cosmic-ray component parameters from OST Appendix V.3."""

    symbol: str
    z: int
    c: float

    def __post_init__(self) -> None:
        if not self.symbol:
            msg = "AKL ion symbol must not be empty."
            raise ValueError(msg)

        if self.z < 1:
            msg = "AKL ion charge Z must be positive."
            raise ValueError(msg)

        if not isfinite(self.c) or self.c < 0.0:
            msg = "AKL C parameter must be non-negative and finite."
            raise ValueError(msg)


_GCR_C_TABLE: Final[tuple[tuple[str, int, float, float], ...]] = (
    ("H", 1, 1.0, 1.98e4),
    ("He", 2, 4.0, 3.69e3),
    ("Li", 3, 6.9, 37.9),
    ("Be", 4, 9.0, 18.8),
    ("B", 5, 10.8, 53.4),
    ("C", 6, 12.0, 79.0),
    ("N", 7, 14.0, 17.6),
    ("O", 8, 16.0, 82.7),
    ("F", 9, 19.0, 1.11),
    ("Ne", 10, 20.2, 14.1),
    ("Na", 11, 23.0, 2.03),
    ("Mg", 12, 24.3, 16.4),
    ("Al", 13, 27.0, 2.60),
    ("Si", 14, 28.1, 13.6),
    ("P", 15, 31.0, 0.68),
    ("S", 16, 32.1, 3.07),
    ("Cl", 17, 35.4, 0.63),
    ("Ar", 18, 39.9, 1.96),
    ("K", 19, 39.1, 0.60),
    ("Ca", 20, 40.1, 1.70),
    ("Sc", 21, 44.9, 0.50),
    ("Ti", 22, 47.9, 2.21),
    ("V", 23, 50.9, 0.70),
    ("Cr", 24, 52.0, 1.73),
    ("Mn", 25, 54.9, 0.75),
    ("Fe", 26, 55.8, 11.2),
    ("Co", 27, 58.9, 0.06),
    ("Ni", 28, 58.7, 0.49),
    ("Cu", 29, 63.5, 6.34e-3),
    ("Zn", 30, 65.4, 8.20e-3),
    ("Ga", 31, 69.7, 6.06e-4),
    ("Ge", 32, 72.6, 1.30e-3),
    ("As", 33, 74.9, 8.30e-5),
    ("Se", 34, 79.0, 4.85e-4),
    ("Br", 35, 79.9, 9.04e-5),
    ("Kr", 36, 83.8, 2.52e-4),
    ("Rb", 37, 85.5, 8.20e-5),
    ("Sr", 38, 87.6, 2.70e-4),
    ("Y", 39, 88.9, 6.05e-5),
    ("Zr", 40, 91.2, 1.50e-4),
    ("Nb", 41, 92.9, 2.70e-5),
    ("Mo", 42, 95.9, 7.54e-5),
    ("Tc", 43, 97.0, 8.80e-6),
    ("Ru", 44, 101.0, 2.88e-5),
    ("Rh", 45, 102.9, 1.49e-5),
    ("Pd", 46, 106.4, 4.28e-5),
    ("Ag", 47, 107.9, 1.39e-5),
    ("Cd", 48, 112.4, 3.72e-5),
    ("In", 49, 114.8, 8.20e-6),
    ("Sn", 50, 118.7, 4.38e-5),
    ("Sb", 51, 121.8, 9.20e-6),
    ("Te", 52, 127.6, 5.31e-5),
    ("I", 53, 126.9, 1.02e-5),
    ("Xe", 54, 131.3, 2.51e-5),
    ("Cs", 55, 132.9, 6.00e-6),
    ("Ba", 56, 137.3, 6.24e-5),
    ("La", 57, 138.9, 5.50e-6),
    ("Ce", 58, 140.1, 1.67e-5),
    ("Pr", 59, 140.9, 4.00e-6),
    ("Nd", 60, 144.2, 1.49e-5),
    ("Pm", 61, 144.2, 1.70e-6),
    ("Sm", 62, 145.0, 1.67e-5),
    ("Eu", 63, 150.4, 2.80e-6),
    ("Gd", 64, 152.0, 1.30e-5),
    ("Tb", 65, 157.3, 3.20e-6),
    ("Dy", 66, 158.9, 1.30e-5),
    ("Ho", 67, 162.5, 4.90e-6),
    ("Er", 68, 164.9, 8.20e-6),
    ("Tm", 69, 167.3, 1.60e-6),
    ("Yb", 70, 168.9, 8.20e-6),
    ("Lu", 71, 173.0, 1.20e-6),
    ("Hf", 72, 175.0, 7.50e-6),
    ("Ta", 73, 178.5, 1.66e-5),
    ("W", 74, 180.9, 7.50e-6),
    ("Re", 75, 183.9, 2.60e-6),
    ("Os", 76, 186.2, 1.11e-5),
    ("Ir", 77, 190.2, 7.30e-6),
    ("Pt", 78, 192.2, 1.39e-5),
    ("Au", 79, 195.1, 2.60e-6),
    ("Hg", 80, 197.0, 4.50e-6),
    ("Tl", 81, 200.6, 1.30e-6),
    ("Pb", 82, 200.4, 1.30e-5),
    ("Bi", 83, 207.2, 6.80e-7),
    ("Po", 84, 209.0, 0.0),
    ("At", 85, 210.0, 0.0),
    ("Rn", 86, 222.0, 0.0),
    ("Fr", 87, 223.0, 0.0),
    ("Ra", 88, 226.0, 0.0),
    ("Ac", 89, 227.0, 0.0),
    ("Th", 90, 232.0, 7.50e-7),
    ("Pa", 91, 231.0, 0.0),
    ("U", 92, 238.0, 4.57e-7),
)

GCR_ELEMENT_SYMBOLS: Final[tuple[str, ...]] = tuple(row[0] for row in _GCR_C_TABLE)

_OST_H_TO_NI: Final[dict[str, tuple[int, float, float, float, float]]] = {
    "H": (1, 1.0, 1.32e4, 2.93, 2.85),
    "He": (2, 4.0, 1.72e3, 2.64, 3.12),
    "Li": (3, 6.9, 12.0, 2.91, 3.41),
    "Be": (4, 9.0, 6.30, 2.64, 4.30),
    "B": (5, 10.8, 32.0, 2.90, 3.93),
    "C": (6, 12.0, 63.4, 2.64, 3.18),
    "N": (7, 14.0, 31.0, 2.90, 3.77),
    "O": (8, 16.0, 55.0, 2.64, 3.11),
    "F": (9, 19.0, 1.40, 2.88, 4.05),
    "Ne": (10, 20.2, 7.70, 2.64, 3.11),
    "Na": (11, 23.0, 2.50, 2.86, 3.14),
    "Mg": (12, 24.3, 11.30, 2.64, 3.65),
    "Al": (13, 27.0, 3.10, 2.85, 3.46),
    "Si": (14, 28.1, 9.50, 2.64, 3.00),
    "P": (15, 31.0, 1.10, 2.84, 4.04),
    "S": (16, 32.1, 2.50, 2.64, 3.30),
    "Cl": (17, 35.4, 0.98, 2.83, 4.40),
    "Ar": (18, 39.9, 1.60, 2.64, 4.33),
    "K": (19, 39.1, 0.90, 2.82, 4.49),
    "Ca": (20, 40.1, 1.39, 2.64, 2.93),
    "Sc": (21, 44.9, 0.70, 2.80, 3.78),
    "Ti": (22, 47.9, 1.80, 2.64, 3.79),
    "V": (23, 50.9, 0.95, 2.79, 3.50),
    "Cr": (24, 52.0, 1.41, 2.64, 3.28),
    "Mn": (25, 54.9, 0.98, 2.78, 3.29),
    "Fe": (26, 55.8, 8.40, 2.64, 3.01),
    "Co": (27, 58.9, 0.07, 2.77, 4.25),
    "Ni": (28, 58.7, 0.40, 2.64, 3.52),
}

_AKL_TABLE: Final[tuple[tuple[str, int, float], ...]] = (
    ("He", 2, 1.2e-3),
    ("C", 6, 4.6e-6),
    ("N", 7, 4.6e-5),
    ("O", 8, 2.4e-4),
    ("Ne", 10, 2.4e-5),
    ("Ar", 18, 4.6e-7),
)


def _gost_gamma_alpha(z: int) -> tuple[float, float]:
    if z == 1:
        return 2.80, 3.60

    if z == 2:
        return 2.77, 3.12

    if 3 <= z <= 5:
        return 3.06, 4.00

    return 2.70, 3.60


def get_gost_gcr_ion_parameters() -> dict[str, GcrIonParameters]:
    """Return main GCR ion parameters from GOST 25645.150."""

    result: dict[str, GcrIonParameters] = {}

    for symbol, z, mass_number, c_value in _GCR_C_TABLE:
        gamma, alpha = _gost_gamma_alpha(z)
        result[symbol] = GcrIonParameters(
            symbol=symbol,
            z=z,
            mass_number=mass_number,
            c=c_value,
            gamma=gamma,
            alpha=alpha,
        )

    return result


def get_ost_gcr_ion_parameters() -> dict[str, GcrIonParameters]:
    """Return main GCR ion parameters from OST 134-1044-2007 Appendix V.1."""

    result: dict[str, GcrIonParameters] = {}

    for symbol, z, mass_number, c_value in _GCR_C_TABLE:
        gamma = 2.63
        alpha = 3.01

        if symbol in _OST_H_TO_NI:
            z, mass_number, c_value, gamma, alpha = _OST_H_TO_NI[symbol]

        result[symbol] = GcrIonParameters(
            symbol=symbol,
            z=z,
            mass_number=mass_number,
            c=c_value,
            gamma=gamma,
            alpha=alpha,
        )

    return result


def get_ost_nek_parameter(z: int) -> NekParameters | None:
    """Return OST NEK parameters for one ion charge."""

    if z < 1:
        msg = "NEK ion charge Z must be positive."
        raise ValueError(msg)

    if z == 1:
        return NekParameters(z=z, ac=4.1e-2, bc=2.57e-2, ag=3.50, bg=0.0)

    if z == 2:
        return NekParameters(
            z=z,
            ac=3.6e-2,
            bc=OST_NEK_HE_BC_USED,
            ag=2.45,
            bg=0.0,
        )

    if 3 <= z <= 5:
        return NekParameters(z=z, ac=0.0, bc=0.0, ag=0.0, bg=0.0)

    if 6 <= z <= 28:
        q = float(z)
        return NekParameters(
            z=z,
            ac=2.3e-4 * exp(-0.18 * q),
            bc=1.64e-2 * exp(0.05 * q),
            ag=2.47 + 0.17 * sqrt(q),
            bg=0.016 - 0.03 / sqrt(q),
        )

    return None


def get_ost_nek_parameters() -> dict[int, NekParameters]:
    """Return OST NEK parameters for Z=1..28."""

    return {
        z: parameters
        for z in range(1, 29)
        if (parameters := get_ost_nek_parameter(z)) is not None
    }


def get_ost_akl_parameters() -> dict[str, AklParameters]:
    """Return OST AKL parameters."""

    return {
        symbol: AklParameters(
            symbol=symbol,
            z=z,
            c=c_value,
        )
        for symbol, z, c_value in _AKL_TABLE
    }


__all__ = [
    "AklParameters",
    "GCR_ELEMENT_SYMBOLS",
    "GcrIonParameters",
    "NekParameters",
    "OST_NEK_HE_BC_CORRECTION_NOTE",
    "OST_NEK_HE_BC_PRINTED",
    "OST_NEK_HE_BC_USED",
    "get_gost_gcr_ion_parameters",
    "get_ost_akl_parameters",
    "get_ost_gcr_ion_parameters",
    "get_ost_nek_parameter",
    "get_ost_nek_parameters",
]
