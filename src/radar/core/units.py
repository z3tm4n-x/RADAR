"""Units used by the RADAR calculation core."""

from enum import StrEnum


class Unit(StrEnum):
    """Physical units used in typed calculation results."""

    DIMENSIONLESS = "1"

    MEV = "MeV"
    GEV_PER_NUCLEON = "GeV/nucleon"
    LET = "MeV*cm^2/mg"

    SECOND = "s"
    DAY = "day"
    YEAR = "year"

    THICKNESS = "g/cm^2"

    DIFFERENTIAL_FLUENCE = "cm^-2 MeV^-1"
    DIFFERENTIAL_FLUX = "cm^-2 s^-1 MeV^-1"
    DIFFERENTIAL_LET_FLUENCE = "cm^-2 (MeV*cm^2/mg)^-1"
    DIFFERENTIAL_LET_FLUX = "cm^-2 s^-1 (MeV*cm^2/mg)^-1"

    RAD = "rad"
    RAD_PER_SECOND = "rad/s"
    RAD_PER_DAY = "rad/day"
    RAD_PER_YEAR = "rad/year"
