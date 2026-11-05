"""Read native ASTAR 1.2 tabulations from the pinned official NIST archive.

Derived from NIST STAR 1.2 data and source. This reader is a new implementation,
not execution/reproduction of the original Fortran program and not the current
online ASTAR release. No ASTAR spline interpolation or extrapolation is done.
NIST permits redistribution/modification with a derivative/modified notice:
https://physics.nist.gov/PhysRefData/Star/Text/download.html

Crucially, MATS uses the main program's IDNO order, while FALPH/UALPH uses
AREAD's IND order. Carbon 906 occurs in different positions in these arrays.
Joining by ID rather than assuming equal positions is necessary for Al/Si.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import io
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
import re
import tarfile


ASTAR_ARCHIVE_SHA256 = "b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6"
ASTAR_ARCHIVE_URL = "https://physics.nist.gov/PhysRefData/Star/STAR.tar.gz"
DERIVATIVE_NOTICE = (
    "Derived from NIST STAR 1.2 native tabulations by a new Python reader; "
    "not compiled Fortran output, not online ASTAR, no interpolation or extrapolation."
)
_MEMBERS = tuple(f"ASTAR/{name}" for name in (
    "ASTAR.f", "ACONV.f", "ENG.ALF", "MATS", "FALPH",
))
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def _positive(value: float, name: str) -> None:
    try:
        invalid = (isinstance(value, bool) or not isinstance(value, Real)
                   or not isfinite(value) or value <= 0)
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError(f"{name} must be a positive finite real number, not bool.")


@dataclass(frozen=True)
class AstarRow:
    """One native node; energy refers to the whole alpha particle in MeV."""

    energy_mev_total: float
    stopping_electronic_mev_cm2_g: float
    stopping_nuclear_mev_cm2_g: float
    range_csda_g_cm2: float
    detour_factor: float

    def __post_init__(self) -> None:
        for name in (
            "energy_mev_total", "stopping_electronic_mev_cm2_g",
            "stopping_nuclear_mev_cm2_g", "range_csda_g_cm2", "detour_factor",
        ):
            _positive(getattr(self, name), name)
        if self.detour_factor > 1:
            raise ValueError("Detour factor cannot exceed one.")
        _positive(self.stopping_total_mev_cm2_g, "total stopping")
        _positive(self.range_projected_g_cm2, "projected range")

    @property
    def stopping_total_mev_cm2_g(self) -> float:
        return self.stopping_electronic_mev_cm2_g + self.stopping_nuclear_mev_cm2_g

    @property
    def range_projected_g_cm2(self) -> float:
        return self.range_csda_g_cm2 * self.detour_factor


@dataclass(frozen=True)
class AstarMaterial:
    id: int
    name: str
    rows: tuple[AstarRow, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not int or self.id <= 0:
            raise ValueError("ASTAR material ID must be a positive integer.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ASTAR material name must be nonempty.")
        if (not isinstance(self.rows, tuple) or len(self.rows) < 2
                or any(not isinstance(row, AstarRow) for row in self.rows)):
            raise ValueError("Rows must be an immutable tuple of at least two ASTAR nodes.")
        for left, right in zip(self.rows, self.rows[1:]):
            if (left.energy_mev_total >= right.energy_mev_total
                    or left.range_csda_g_cm2 >= right.range_csda_g_cm2):
                raise ValueError("ASTAR energies and CSDA ranges must increase strictly.")


@dataclass(frozen=True)
class AstarArchive:
    materials: tuple[AstarMaterial, ...]
    archive_sha256: str
    member_sha256: tuple[tuple[str, str], ...]
    version: str = field(default="1.2", init=False)
    derivative_notice: str = field(default=DERIVATIVE_NOTICE, init=False)
    fortran_execution_replayed: bool = field(default=False, init=False)
    online_version_equivalence_verified: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def material(self, material_id: int) -> AstarMaterial:
        if type(material_id) is not int:
            raise ValueError("Material lookup requires an integer ID, not bool.")
        matches = [material for material in self.materials if material.id == material_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one ASTAR material with ID {material_id}.")
        return matches[0]


def _text(data: bytes, name: str, *, dos_eof: bool = False) -> str:
    try:
        value = data.decode("ascii")
    except (AttributeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} must be ASCII bytes.") from exc
    value = value.rstrip()
    if dos_eof and value.endswith("\x1a"):
        value = value[:-1].rstrip()
    if any(ord(c) < 32 and c not in "\r\n\t" for c in value):
        raise ValueError(f"Unexpected control character in {name}.")
    return value


def _statements(source: str) -> tuple[str, ...]:
    """Join fixed-form Fortran continuation records, ignoring comment records."""
    result: list[str] = []
    for line in source.splitlines():
        if not line.strip() or line[0] in "Cc*!":
            continue
        if len(line) < 7:
            raise ValueError("Unexpected short fixed-form Fortran record.")
        body = line[6:72].strip()
        if line[5] not in " 0":
            if not result:
                raise ValueError("Fortran continuation lacks a preceding statement.")
            result[-1] += " " + body
        else:
            result.append(body)
    return tuple(result)


def _data_values(statements: tuple[str, ...], name: str) -> tuple[float, ...]:
    pattern = re.compile(rf"DATA\s+{re.escape(name)}\s*/([^/]+)/", re.IGNORECASE)
    matches = [m for statement in statements if (m := pattern.fullmatch(statement))]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Fortran DATA {name} declaration.")
    tokens = matches[0].group(1).split(",")
    return tuple(_numeric(token.strip(), f"DATA {name}") for token in tokens)


def _numeric(token: str, label: str) -> float:
    if _NUMBER.fullmatch(token) is None:
        raise ValueError(f"Invalid Fortran numeric token in {label}.")
    value = float(token.replace("D", "E").replace("d", "e"))
    _positive(value, label)
    return value


def _ids(statements: tuple[str, ...], name: str) -> tuple[int, ...]:
    values = _data_values(statements, name)
    if (len(values) != 74 or len(set(values)) != 74
            or any(not value.is_integer() for value in values)):
        raise ValueError(f"{name} must declare 74 distinct integer material IDs.")
    return tuple(int(value) for value in values)


def _parse_members(members: Mapping[str, bytes]) -> tuple[AstarMaterial, ...]:
    """Parse declared nodes only; this private helper makes no source-pin claim."""
    if set(members) != set(_MEMBERS):
        raise ValueError("Exactly the five ASTAR source/data members are required.")
    source = _text(members["ASTAR/ASTAR.f"], "ASTAR.f")
    astar = _statements(source)
    converter = _statements(_text(members["ASTAR/ACONV.f"], "ACONV.f"))
    if not re.search(r"^C\s+Version 1\.2\s*$", source, flags=re.MULTILINE):
        raise ValueError("Expected the ASTAR version 1.2 source declaration.")
    idno, ind = _ids(astar, "IDNO"), _ids(astar, "IND")
    if set(idno) != set(ind):
        raise ValueError("Display and data material inventories disagree.")
    if _data_values(astar, "NMAX") != (122.0,) or _data_values(converter, "NMAX") != (122.0,):
        raise ValueError("Expected 122 native ASTAR nodes.")
    required_converter = (
        "DO 10 J=1,74", "READ (7,*) (STE(N),N=1,NMAX)",
        "READ (7,*) (STN(N),N=1,NMAX)", "READ (7,*) (RG(N),N=1,NMAX)",
        "READ (7,*) (DET(N),N=1,NMAX)", "WRITE (8,REC=J) STE,STN,RG,DET",
    )
    positions = []
    for statement in required_converter:
        if converter.count(statement) != 1:
            raise ValueError("Unrecognized ACONV data-array layout.")
        positions.append(converter.index(statement))
    if positions != sorted(positions):
        raise ValueError("ACONV data-array records are in an unexpected order.")
    if "READ (7,REC=JD) STE,STN,RG,DET" not in astar:
        raise ValueError("Unrecognized AREAD data-array order.")
    if "AVPEN(L)=DETOUR(L)*RANGE(L)" not in astar:
        raise ValueError("Unrecognized projected-range definition.")
    energy_tokens = _text(members["ASTAR/ENG.ALF"], "ENG.ALF", dos_eof=True).split()
    if len(energy_tokens) != 123 or energy_tokens[0] != "122":
        raise ValueError("ENG.ALF must contain a 122-node grid with one count header.")
    energies = tuple(_numeric(token, "ENG.ALF") for token in energy_tokens[1:])
    if (energies != _data_values(astar, "E") or energies[0] != 0.001 or energies[-1] != 1000.0
            or any(a >= b for a, b in zip(energies, energies[1:]))):
        raise ValueError("ENG.ALF disagrees with APOL's strictly increasing native energy grid.")
    names = tuple(line.strip() for line in _text(members["ASTAR/MATS"], "MATS").splitlines())
    if len(names) != 74 or any(not name for name in names) or len(set(names)) != 74:
        raise ValueError("MATS must contain exactly 74 distinct nonempty material labels.")
    labels = dict(zip(idno, names, strict=True))
    if labels.get(13) != "ALUMINUM" or labels.get(14) != "SILICON":
        raise ValueError("Al/Si identifiers do not match MATS labels.")
    lines = _text(members["ASTAR/FALPH"], "FALPH").splitlines()
    arrays: list[tuple[float, ...]] = []
    current: list[float] = []
    for line in lines:
        if not line.strip():
            raise ValueError("Empty FALPH data record.")
        current.extend(_numeric(token, "FALPH") for token in line.split())
        if len(current) > 122:
            raise ValueError("FALPH record crosses a 122-value array boundary.")
        if len(current) == 122:
            arrays.append(tuple(current))
            current = []
    if current or len(arrays) != 74 * 4:
        raise ValueError("FALPH must contain exactly four 122-value arrays for each of 74 materials.")
    materials = []
    for index, material_id in enumerate(ind):
        electronic, nuclear, csda, detour = arrays[4 * index:4 * index + 4]
        materials.append(AstarMaterial(
            material_id, labels[material_id], tuple(AstarRow(e, se, sn, r, d)
                for e, se, sn, r, d in zip(energies, electronic, nuclear, csda, detour, strict=True)),
        ))
    return tuple(materials)


def _extract_members(payload: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    names: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                        or ":" in member.name or member.name in names
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("Unsafe or duplicate archive member.")
                names.add(member.name)
                if member.name not in _MEMBERS:
                    continue
                if not member.isfile() or member.size > 2_000_000:
                    raise ValueError("Invalid ASTAR data member.")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing ASTAR member content.")
                result[member.name] = stream.read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("Unreadable STAR gzip/tar archive.") from exc
    if set(result) != set(_MEMBERS):
        raise ValueError("Missing ASTAR source/data member.")
    return result


def load_astar_archive(path: Path) -> AstarArchive:
    """Read pinned archive bytes without executing code or extracting to disk."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ASTAR_ARCHIVE_SHA256:
        raise ValueError("Official STAR 1.2 archive SHA256 mismatch.")
    members = _extract_members(payload)
    materials = _parse_members(members)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("STAR archive changed during parsing.")
    return AstarArchive(materials, digest, tuple(
        (name, hashlib.sha256(members[name]).hexdigest()) for name in _MEMBERS
    ))
