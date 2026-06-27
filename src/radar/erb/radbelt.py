"""RADBELT AE8/AP8 ASC map parsing and map evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

RADBELT_LOG_BIN_RATIO = 10.0**0.1


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

    def _map_1based(self, index: int) -> int:
        """Return a map value using the original RADBELT 1-based index."""

        if index < 1:
            msg = "RADBELT map index must be positive."
            raise ValueError(msg)

        return self.map_values[index - 1]

    def _m(self, start: int, offset: int) -> int:
        return self._map_1based(start + offset - 1)

    def trara2_scaled_log_integral(self, *, start: int, il: int, ib: int) -> float:
        """Evaluate the translated RADBELT TRARA2 helper.

        The return value is the scaled log10 integral flux used internally by TRARA1.
        This is a structured Python translation of the original RADBELT/TRMFUN labels.
        """

        fistep = self.fistep
        fnl = float(il)
        fnb = float(ib)
        itime = 0
        i2 = 0
        i1 = 0
        l1 = 0

        while True:
            l2 = self._m(start, i2 + 1)
            if self._m(start, i2 + 2) > il:
                break
            i1 = i2
            l1 = l2
            i2 += l2

        if (l1 < 4) and (l2 < 4):
            return 0.0

        if not (self._m(start, i2 + 3) > self._m(start, i1 + 3)):
            i1, i2 = i2, i1
            l1, l2 = l2, l1

        fll1 = float(self._m(start, i1 + 2))
        fll2 = float(self._m(start, i2 + 2))

        if fll2 == fll1:
            return 0.0

        dfl = (fnl - fll1) / (fll2 - fll1)
        flog1 = float(self._m(start, i1 + 3))
        flog2 = float(self._m(start, i2 + 3))
        fkb1 = 0.0
        fkb2 = 0.0

        while True:
            if l1 >= 4:
                exhausted = True
                j2 = 4
                fincr2 = 0.0

                for j2 in range(4, l2 + 1):
                    fincr2 = float(self._m(start, i2 + j2))
                    if fkb2 + fincr2 > fnb:
                        exhausted = False
                        break
                    fkb2 += fincr2
                    flog2 -= fistep

                if exhausted:
                    itime += 1
                    if itime == 1:
                        i1, i2 = i2, i1
                        l1, l2 = l2, l1
                        fll1 = float(self._m(start, i1 + 2))
                        fll2 = float(self._m(start, i2 + 2))

                        if fll2 == fll1:
                            return 0.0

                        dfl = (fnl - fll1) / (fll2 - fll1)
                        flog1 = float(self._m(start, i1 + 3))
                        flog2 = float(self._m(start, i2 + 3))
                        fkb1 = 0.0
                        fkb2 = 0.0
                        continue

                    return 0.0

                if itime == 1:
                    fkb2 = 0.0
                    j2 = 4
                    fincr2 = float(self._m(start, i2 + j2))
                    flog2 = float(self._m(start, i2 + 3))
                    flog1 = float(self._m(start, i1 + 3))
                    flogm = flog1 + (flog2 - flog1) * dfl
                    fkbm = 0.0
                    fkb2 += fincr2
                    flog2 -= fistep
                    sl2 = flog2 / fkb2
                    if l1 < 4:
                        j1 = 4
                        fincr1 = 0.0
                        sl1 = -900000.0
                    else:
                        j1 = 4
                        fincr1 = float(self._m(start, i1 + j1))
                        fkb1 += fincr1
                        flog1 -= fistep
                        sl1 = flog1 / fkb1
                    break

                if j2 == 4:
                    flogm = flog1 + (flog2 - flog1) * dfl
                    fkbm = 0.0
                    fkb2 += fincr2
                    flog2 -= fistep
                    sl2 = flog2 / fkb2
                    if l1 < 4:
                        j1 = 4
                        fincr1 = 0.0
                        sl1 = -900000.0
                    else:
                        j1 = 4
                        fincr1 = float(self._m(start, i1 + j1))
                        fkb1 += fincr1
                        flog1 -= fistep
                        sl1 = flog1 / fkb1
                    break

                sl2 = flog2 / fkb2
                fkbj1: float | None = None

                for j1 in range(4, l1 + 1):
                    fincr1 = float(self._m(start, i1 + j1))
                    fkb1 += fincr1
                    flog1 -= fistep
                    fkbj1 = ((flog1 / fistep) * fincr1 + fkb1) / (
                        (fincr1 / fistep) * sl2 + 1.0
                    )
                    if fkbj1 <= fkb1:
                        break
                else:
                    if fkbj1 is None or fkbj1 > fkb2:
                        return 0.0

                if fkbj1 <= fkb2:
                    fkbm = fkbj1 + (fkb2 - fkbj1) * dfl
                    flogm = fkbm * sl2
                    flog2 -= fistep
                    fkb2 += fincr2
                    sl1 = flog1 / fkb1
                    sl2 = flog2 / fkb2
                    break

                fkb1 = 0.0
                fkb2 = 0.0
                j2 = 4
                fincr2 = float(self._m(start, i2 + j2))
                flog2 = float(self._m(start, i2 + 3))
                flog1 = float(self._m(start, i1 + 3))
                flogm = flog1 + (flog2 - flog1) * dfl
                fkbm = 0.0
                fkb2 += fincr2
                flog2 -= fistep
                sl2 = flog2 / fkb2
                if l1 < 4:
                    j1 = 4
                    fincr1 = 0.0
                    sl1 = -900000.0
                else:
                    j1 = 4
                    fincr1 = float(self._m(start, i1 + j1))
                    fkb1 += fincr1
                    flog1 -= fistep
                    sl1 = flog1 / fkb1
                break

            j2 = 4
            fincr2 = float(self._m(start, i2 + j2))
            flog2 = float(self._m(start, i2 + 3))
            flog1 = float(self._m(start, i1 + 3))
            flogm = flog1 + (flog2 - flog1) * dfl
            fkbm = 0.0
            fkb2 += fincr2
            flog2 -= fistep
            sl2 = flog2 / fkb2
            j1 = 4
            fincr1 = 0.0
            sl1 = -900000.0
            break

        while True:
            if sl1 < sl2:
                fkbj1 = ((flog1 / fistep) * fincr1 + fkb1) / (
                    (fincr1 / fistep) * sl2 + 1.0
                )
                fkb = fkbj1 + (fkb2 - fkbj1) * dfl
                flog = fkb * sl2
                if fkb >= fnb:
                    break
                fkbm = fkb
                flogm = flog
                if j2 >= l2:
                    return 0.0
                j2 += 1
                fincr2 = float(self._m(start, i2 + j2))
                flog2 -= fistep
                fkb2 += fincr2
                sl2 = flog2 / fkb2
            else:
                fkbj2 = ((flog2 / fistep) * fincr2 + fkb2) / (
                    (fincr2 / fistep) * sl1 + 1.0
                )
                fkb = fkb1 + (fkbj2 - fkb1) * dfl
                flog = fkb * sl1
                if fkb >= fnb:
                    break
                fkbm = fkb
                flogm = flog
                if j1 >= l1:
                    return 0.0
                j1 += 1
                fincr1 = float(self._m(start, i1 + j1))
                flog1 -= fistep
                fkb1 += fincr1
                sl1 = flog1 / fkb1

        if fkb < fkbm + 1.0e-10:
            return 0.0

        result = flogm + (flog - flogm) * ((fnb - fkbm) / (fkb - fkbm))
        return max(result, 0.0)

    def log_integral_flux_gt_e(
        self,
        *,
        l_shell: float,
        b_over_b0: float,
        energies_mev: Iterable[float],
    ) -> tuple[float, ...]:
        """Evaluate translated RADBELT TRARA1 log10 integral flux Phi(>E)."""

        descriptor = self.descriptor
        escale = float(descriptor[3])
        fscale = float(descriptor[6])
        xnl = min(15.6, abs(float(l_shell)))
        nl = int(xnl * descriptor[4])
        bb0 = max(1.0, float(b_over_b0))
        nb = int((bb0 - 1.0) * descriptor[5])

        i1 = 0
        i2 = self.map_values[0]
        i3 = i2 + self._map_1based(i2 + 1)
        l3 = self._map_1based(i3 + 1)
        e1 = self._map_1based(i1 + 2) / escale
        e2 = self._map_1based(i2 + 2) / escale
        s1 = True
        s2 = True
        s0 = False
        f1 = 1.001
        f2 = 1.002
        f0 = 0.0
        e0 = 0.0
        i0 = 0
        values: list[float] = []

        for energy in energies_mev:
            current_energy = float(energy)
            while (current_energy > e2) and (l3 != 0):
                i0 = i1
                i1 = i2
                i2 = i3
                i3 += l3
                l3 = self._map_1based(i3 + 1)
                e0 = e1
                e1 = e2
                e2 = self._map_1based(i2 + 2) / escale
                s0 = s1
                s1 = s2
                s2 = True
                f0 = f1
                f1 = f2

            if s1:
                f1 = self.trara2_scaled_log_integral(start=i1 + 3, il=nl, ib=nb) / fscale

            if s2:
                f2 = self.trara2_scaled_log_integral(start=i2 + 3, il=nl, ib=nb) / fscale

            s1 = False
            s2 = False
            integral_log = f1 + (f2 - f1) * (current_energy - e1) / (e2 - e1)

            if (not (f2 > 0.0)) and i1 != 0:
                if s0:
                    f0 = (
                        self.trara2_scaled_log_integral(start=i0 + 3, il=nl, ib=nb)
                        / fscale
                    )

                s0 = False
                integral_log = min(
                    integral_log,
                    f0 + (f1 - f0) * (current_energy - e0) / (e1 - e0),
                )

            values.append(max(integral_log, 0.0))

        return tuple(values)

    def integral_flux_gt_e(
        self,
        *,
        l_shell: float,
        b_over_b0: float,
        energy_mev: float,
    ) -> float:
        """Return RADBELT integral flux Phi(>E)."""

        log_flux = self.log_integral_flux_gt_e(
            l_shell=l_shell,
            b_over_b0=b_over_b0,
            energies_mev=(energy_mev,),
        )[0]

        if log_flux <= 0.0:
            return 0.0

        return math.pow(10.0, log_flux)

    def differential_flux_bins(
        self,
        *,
        l_shell: float,
        b_over_b0: float,
        lower_energies_mev: Iterable[float],
        log_bin_ratio: float = RADBELT_LOG_BIN_RATIO,
    ) -> tuple[float, ...]:
        """Return RADBELT differential flux for bins [E_i, E_i * ratio]."""

        lower_energies = tuple(float(value) for value in lower_energies_mev)

        if not lower_energies:
            return ()

        if log_bin_ratio <= 1.0 or not math.isfinite(log_bin_ratio):
            msg = "RADBELT logarithmic bin ratio must be finite and greater than one."
            raise ValueError(msg)

        edges = (*lower_energies, lower_energies[-1] * log_bin_ratio)
        log_integral = self.log_integral_flux_gt_e(
            l_shell=l_shell,
            b_over_b0=b_over_b0,
            energies_mev=edges,
        )
        integral = tuple(10.0**value if value > 0.0 else 0.0 for value in log_integral)
        values: list[float] = []

        for lower_energy, upper_energy, lower_flux, upper_flux in zip(
            edges[:-1],
            edges[1:],
            integral[:-1],
            integral[1:],
            strict=True,
        ):
            if upper_flux > 0.0:
                values.append(abs(upper_flux - lower_flux) / (upper_energy - lower_energy))
            else:
                values.append(0.0)

        return tuple(values)


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
    "RADBELT_LOG_BIN_RATIO",
    "RadbeltAscMap",
    "parse_radbelt_asc_text",
]
