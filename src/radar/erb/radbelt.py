"""RADBELT AE8/AP8 ASC map parsing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadbeltAscMap:
    """Parsed RADBELT AE8/AP8 ASC map data."""

    name: str
    descriptor: tuple[int, ...]
    map_values: tuple[int, ...]
    fistep: float

    def __post_init__(self) -> None:
        if not self.name:
            msg = "RADBELT map name must not be empty."
            raise ValueError(msg)

        if len(self.descriptor) != 8:
            msg = "RADBELT descriptor must contain exactly 8 integers."
            raise ValueError(msg)

        expected_map_length = self.descriptor[7]

        if expected_map_length < 0:
            msg = "RADBELT map length must be non-negative."
            raise ValueError(msg)

        if len(self.map_values) != expected_map_length:
            msg = (
                "RADBELT map length mismatch: "
                f"have {len(self.map_values)}, expected {expected_map_length}."
            )
            raise ValueError(msg)

        if self.descriptor[1] == 0:
            msg = "RADBELT descriptor[2] must not be zero."
            raise ValueError(msg)

    def descriptor_1based(self, index: int) -> int:
        """Return a descriptor value using the original RADBELT 1-based index."""

        if not 1 <= index <= 8:
            msg = "RADBELT descriptor index must be in the range 1..8."
            raise ValueError(msg)

        return self.descriptor[index - 1]


def parse_radbelt_asc_text(
    *,
    name: str,
    text: str,
) -> RadbeltAscMap:
    """Parse a RADBELT ASC file written with Fortran FORMAT(1X,12I6)."""

    values: list[int] = []

    for raw_line in text.splitlines():
        if not raw_line:
            continue

        line = raw_line[1:]

        for offset in range(0, len(line), 6):
            chunk = line[offset : offset + 6]

            if chunk.strip():
                values.append(int(chunk))

    if len(values) < 8:
        msg = f"{name}: invalid RADBELT ASC data."
        raise ValueError(msg)

    descriptor = tuple(values[:8])
    expected_map_length = descriptor[7]
    map_values = tuple(values[8 : 8 + expected_map_length])

    if len(map_values) != expected_map_length:
        msg = (
            f"{name}: RADBELT map length mismatch: "
            f"have {len(map_values)}, expected {expected_map_length}."
        )
        raise ValueError(msg)

    return RadbeltAscMap(
        name=name,
        descriptor=descriptor,
        map_values=map_values,
        fistep=descriptor[6] / descriptor[1],
    )


__all__ = [
    "RadbeltAscMap",
    "parse_radbelt_asc_text",
]
