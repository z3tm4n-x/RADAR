#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
electron_shieldose2.py

Python port of the electron + bremsstrahlung dose part of SHIELDOSE-2
(v2.10, S.M. Seltzer / NIST) using ELBRBAS2.DAT.

Purpose
-------
Convert an incident omnidirectional electron spectrum

    J(E) [particles / (MeV cm^2 s)]   -> dose rate [rad/s]
or
    F(E) [particles / (MeV cm^2)]     -> dose [rad]

in a selected detector material behind aluminum shielding.

For the current project the default detector is silicon.

Important
---------
This module calculates dose response, not the transmitted electron energy
spectrum. The database contains ETRAN-based dose response functions for
electrons and bremsstrahlung, not an E_in -> E_out electron transfer matrix.

The incident spectrum convention follows SHIELDOSE-2:
    incident omnidirectional flux in /energy/cm2/unit time

Therefore the default geometry factor is 1/4, matching the original code.

Outputs include:
    - finite aluminum slab transmission-surface electron/brems/total dose
    - semi-infinite aluminum medium electron/brems/total dose
    - SHIELDOSE-2 "1/2 dose at center of aluminum spheres" electron/brems/total

The sphere-center result follows the original SHIELDOSE-2 convention and is
named "sphere_center_half" deliberately.

No hidden pi, 2pi, or 4pi factors are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import argparse
import csv
import math
import re

import numpy as np


RADCON_MEV_PER_G_TO_RAD = 1.6021892e-8
DEFAULT_DETECTOR = "Si"

DETECTOR_NAMES = [
    "Al",
    "Graphite",
    "Si",
    "Air",
    "Bone",
    "CaF2",
    "GaAs",
    "LiF",
    "SiO2",
    "Tissue",
    "H2O",
]

DETECTOR_ALIASES = {
    "al": "Al",
    "aluminum": "Al",
    "aluminium": "Al",
    "graphite": "Graphite",
    "c": "Graphite",
    "si": "Si",
    "silicon": "Si",
    "air": "Air",
    "bone": "Bone",
    "caf2": "CaF2",
    "calcium_fluoride": "CaF2",
    "gaas": "GaAs",
    "lif": "LiF",
    "sio2": "SiO2",
    "silicon_dioxide": "SiO2",
    "tissue": "Tissue",
    "h2o": "H2O",
    "water": "H2O",
}


def _detector_index(detector: str) -> int:
    key = str(detector).strip().lower().replace(" ", "_").replace("-", "_")
    name = DETECTOR_ALIASES.get(key, detector)
    if name not in DETECTOR_NAMES:
        raise ValueError(
            f"Unknown detector {detector!r}. Supported: {', '.join(DETECTOR_NAMES)}"
        )
    return DETECTOR_NAMES.index(name)


def _parse_numbers_from_text(text: str) -> np.ndarray:
    # Accept E/D Fortran exponents.
    pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    vals = [float(tok.replace("D", "E").replace("d", "e"))
            for tok in re.findall(pattern, text)]
    return np.asarray(vals, dtype=float)


def _scof(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of SHIELDOSE-2 SCOF: natural cubic-spline coefficients.

    y(s)=((d[j]*(s-x[j])+c[j])*(s-x[j])+b[j])*(s-x[j])+y[j]
    for s between x[j] and x[j+1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("Need at least two points for interpolation coefficients")

    b = np.zeros(n, dtype=float)
    c = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)

    n1 = n - 1
    s = 0.0
    for j in range(n1):
        d[j] = x[j + 1] - x[j]
        r = (y[j + 1] - y[j]) / d[j]
        c[j] = r - s
        s = r

    s = 0.0
    r = 0.0
    c[0] = 0.0
    c[n - 1] = 0.0

    for j in range(1, n1):
        c[j] = c[j] + r * c[j - 1]
        b[j] = (x[j - 1] - x[j + 1]) * 2.0 - r * s
        s = d[j]
        r = s / b[j]

    for jr in range(n1 - 1, 0, -1):
        c[jr] = (d[jr] * c[jr + 1] - c[jr]) / b[jr]

    for j in range(n1):
        s = d[j]
        r = c[j + 1] - c[j]
        d[j] = r / s
        c[j] = 3.0 * c[j]
        b[j] = (y[j + 1] - y[j]) / s - (c[j] + r) * s

    return b, c, d


def _lcof(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of SHIELDOSE-2 LCOF: linear interpolation coefficients."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("Need at least two points for interpolation coefficients")
    b = np.zeros(n, dtype=float)
    c = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)
    b[:-1] = np.diff(y) / np.diff(x)
    return b, c, d


def _bspol_scalar(s: float, x: np.ndarray, y: np.ndarray,
                  b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Port of SHIELDOSE-2 BSPOL for one point."""
    n = x.size
    if x[0] <= x[-1]:
        if s >= x[-1]:
            j = n - 2
        elif s <= x[0]:
            j = 0
        else:
            j = int(np.searchsorted(x, s, side="right") - 1)
            j = max(0, min(j, n - 2))
    else:
        # Descending grid support, included for completeness.
        xr = x[::-1]
        yr = y[::-1]
        br, cr, dr = _scof(xr, yr)
        return _bspol_scalar(s, xr, yr, br, cr, dr)

    q = s - x[j]
    return ((d[j] * q + c[j]) * q + b[j]) * q + y[j]


def _bspol(s: Union[float, np.ndarray], x: np.ndarray, y: np.ndarray,
           b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Union[float, np.ndarray]:
    arr = np.asarray(s, dtype=float)
    if arr.ndim == 0:
        return float(_bspol_scalar(float(arr), x, y, b, c, d))
    out = np.empty_like(arr, dtype=float)
    it = np.nditer(arr, flags=["multi_index"])
    for val in it:
        out[it.multi_index] = _bspol_scalar(float(val), x, y, b, c, d)
    return out


def _integ_shieldose(delta: float, g: np.ndarray) -> float:
    """Port of SHIELDOSE-2 INTEG.

    The caller passes delta = mesh_spacing / 3.0, exactly as in sd2.for.
    """
    g = np.asarray(g, dtype=float)
    n = g.size
    if n <= 1:
        sigma = 0.0
    elif n % 2 == 1:
        if n == 3:
            sigma = g[0] + 4.0 * g[1] + g[2]
        else:
            sum4 = np.sum(g[1:n-1:2])
            sum2 = np.sum(g[2:n-2:2])
            sigma = g[0] + 4.0 * sum4 + 2.0 * sum2 + g[n - 1]
    else:
        if n == 2:
            sigma = 1.5 * (g[0] + g[1])
        elif n == 4:
            sigma = 1.125 * (g[0] + 3.0 * g[1] + 3.0 * g[2] + g[3])
        elif n == 6:
            sigma = (
                g[0] + 3.875 * g[1] + 2.625 * g[2] +
                2.625 * g[3] + 3.875 * g[4] + g[5]
            )
        elif n == 8:
            sigma = (
                g[0] + 3.875 * g[1] + 2.625 * g[2] +
                2.625 * g[3] + 3.875 * g[4] + 2.0 * g[5] +
                4.0 * g[6] + g[7]
            )
        else:
            sig6 = (
                g[0] + 3.875 * g[1] + 2.625 * g[2] +
                2.625 * g[3] + 3.875 * g[4] + g[5]
            )
            sum4 = np.sum(g[6:n-1:2])
            sum2 = np.sum(g[7:n-2:2])
            sigma = sig6 + g[5] + 4.0 * sum4 + 2.0 * sum2 + g[n - 1]

    return float(delta * sigma)


def _cubic_loglog_interp_values(x_query: np.ndarray, x_tab: np.ndarray,
                                y_tab: np.ndarray) -> np.ndarray:
    """Natural cubic spline in log(x), log(y), SHIELDOSE-style.

    Positive y values are required.
    """
    x_tab = np.asarray(x_tab, dtype=float)
    y_tab = np.asarray(y_tab, dtype=float)
    m = np.isfinite(x_tab) & np.isfinite(y_tab) & (x_tab > 0) & (y_tab > 0)
    if np.count_nonzero(m) < 2:
        raise ValueError("Need at least two positive spectrum points")
    lx = np.log(x_tab[m])
    ly = np.log(y_tab[m])
    order = np.argsort(lx)
    lx = lx[order]
    ly = ly[order]
    # Remove duplicate energies.
    lx, idx = np.unique(lx, return_index=True)
    ly = ly[idx]
    b, c, d = _scof(lx, ly)
    lq = np.log(np.asarray(x_query, dtype=float))
    return np.exp(_bspol(lq, lx, ly, b, c, d))


def _sphere_half_from_slab(depths_gcm2: np.ndarray, slab_dose: np.ndarray) -> np.ndarray:
    """Port of SHIELDOSE-2 SPHERE.

    Returns the SHIELDOSE-2 quantity named "1/2 dose at center of aluminum spheres".
    Requires at least three positive, nonzero slab-dose depths.
    """
    z = np.asarray(depths_gcm2, dtype=float)
    dose = np.asarray(slab_dose, dtype=float)
    out = np.zeros_like(dose, dtype=float)

    m = np.isfinite(z) & np.isfinite(dose) & (z > 0) & (dose > 0)
    if np.count_nonzero(m) < 3:
        return out

    # SHIELDOSE assumes ordered depths and stops at the first nonpositive dose.
    valid_idx = np.where(m)[0]
    # Use only the initial contiguous valid block to match Fortran behavior.
    stop = 0
    while stop < len(z) and m[stop]:
        stop += 1
    if stop < 3:
        # If user supplied non-contiguous order, fall back to all valid sorted points.
        use_idx = valid_idx[np.argsort(z[valid_idx])]
    else:
        use_idx = np.arange(stop)

    zl = np.log(z[use_idx])
    dl = np.log(dose[use_idx])
    if zl.size < 3:
        return out

    b, c, d = _scof(zl, dl)

    n = zl.size
    dx_last = zl[-1] - zl[-2]
    b_last = b[n - 2] + (2.0 * c[n - 2] + 3.0 * d[n - 2] * dx_last) * dx_last

    deriv = np.array(b, copy=True)
    deriv[n - 1] = b_last

    sph = dose[use_idx] * (1.0 - deriv[:n])
    out[use_idx] = np.where(np.isfinite(sph), sph, 0.0)
    out[out < 0] = 0.0
    return out


@dataclass
class Shieldose2ElectronDatabase:
    """Parsed electron/bremsstrahlung subset of ELBRBAS2.DAT."""
    tag: str
    detector: str
    detector_index: int
    ee_log: np.ndarray
    re_log: np.ndarray
    ye_log: np.ndarray
    te_log: np.ndarray
    ar_log: np.ndarray
    rs_log: np.ndarray
    bs: np.ndarray
    zre: np.ndarray
    zs_log: np.ndarray
    zb_log: np.ndarray
    dale_log: np.ndarray
    dalb_log: np.ndarray
    drate: np.ndarray
    dratb: np.ndarray
    coeffs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

    @property
    def electron_energy_min_MeV(self) -> float:
        return float(np.exp(self.ee_log[0]))

    @property
    def electron_energy_max_MeV(self) -> float:
        return float(np.exp(self.ee_log[-1]))

    @property
    def response_energy_min_MeV(self) -> float:
        return float(np.exp(self.te_log[0]))

    @property
    def response_energy_max_MeV(self) -> float:
        return float(np.exp(self.te_log[-1]))


def load_elbrbas2_database(
    path: Union[str, Path] = "elbrbas2.dat",
    detector: str = DEFAULT_DETECTOR,
) -> Shieldose2ElectronDatabase:
    """Load SHIELDOSE-2 electron/bremsstrahlung database.

    Parameters
    ----------
    path:
        Path to ELBRBAS2.DAT.
    detector:
        Detector material. For this project use "Si".
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be ELBRBAS2.DAT")

    tag = lines[0].strip()
    dims = [int(float(x)) for x in lines[1].split()]
    if len(dims) != 7:
        raise ValueError("Could not read ELBRBAS2 dimensions")
    mmaxe, nmaxe, lmaxs_raw, lmaxe, lmaxt, lmaxb, imix = dims

    nums = _parse_numbers_from_text("\n".join(lines[2:]))
    p = 0

    def take(n: int) -> np.ndarray:
        nonlocal p
        if p + n > nums.size:
            raise ValueError("Unexpected end of ELBRBAS2.DAT")
        out = nums[p:p+n].copy()
        p += n
        return out

    ee = take(mmaxe)
    re_arr = take(mmaxe)
    ye = take(mmaxe)
    te = take(nmaxe)
    ar = take(nmaxe)
    rs = take(nmaxe)
    bs_raw = take(lmaxs_raw)
    zre = take(lmaxe)
    zs = take(lmaxt)
    zb = take(lmaxb)

    dale = np.zeros((nmaxe, lmaxs_raw + 1), dtype=float)
    dalb = np.zeros((nmaxe, lmaxt), dtype=float)
    # Keep only selected detector ratios, as in sd2.for.
    idet = _detector_index(detector)
    detector_name = DETECTOR_NAMES[idet]
    drate = np.zeros((nmaxe, lmaxe, 2), dtype=float)
    dratb = np.zeros((nmaxe, lmaxb, 2), dtype=float)

    for n in range(nmaxe):
        dale[n, :lmaxs_raw] = take(lmaxs_raw)
        dale[n, lmaxs_raw] = 1.0e-7
        dalb[n, :] = take(lmaxt)
        for i in range(imix):
            for m in range(2):
                tmp_e = take(lmaxe)
                if i == idet:
                    drate[n, :, m] = tmp_e
                tmp_b = take(lmaxb)
                if i == idet:
                    dratb[n, :, m] = tmp_b

    bs = np.concatenate([bs_raw, np.array([2.0])])

    ee_log = np.log(ee)
    re_log = np.log(re_arr)
    ye_log = np.log(ye)
    te_log = np.log(te)
    ar_log = np.log(ar)
    rs_log = np.log(rs)
    zb_log = np.log(zb)
    zs_log = np.log(zs)

    dale_log = np.log(dale)
    dalb_log = np.log(dalb)

    coeffs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    coeffs["re"] = _scof(ee_log, re_log)
    coeffs["ye"] = _scof(ee_log, ye_log)
    coeffs["ar"] = _scof(te_log, ar_log)
    coeffs["rs"] = _scof(te_log, rs_log)

    # Precompute coefficients for interpolation over incident energy.
    # DALE and DALB are log response values; SHIELDOSE uses LCOF over log TE.
    for l in range(bs.size):
        coeffs[f"dale_te_{l}"] = _lcof(te_log, dale_log[:, l])
    for l in range(lmaxt):
        coeffs[f"dalb_te_{l}"] = _lcof(te_log, dalb_log[:, l])
    # Detector ratios are not logged and are linearly interpolated over log TE.
    for m in range(2):
        for l in range(lmaxe):
            coeffs[f"drate_te_{m}_{l}"] = _lcof(te_log, drate[:, l, m])
        for l in range(lmaxb):
            coeffs[f"dratb_te_{m}_{l}"] = _lcof(te_log, dratb[:, l, m])

    return Shieldose2ElectronDatabase(
        tag=tag,
        detector=detector_name,
        detector_index=idet,
        ee_log=ee_log,
        re_log=re_log,
        ye_log=ye_log,
        te_log=te_log,
        ar_log=ar_log,
        rs_log=rs_log,
        bs=bs,
        zre=zre,
        zs_log=zs_log,
        zb_log=zb_log,
        dale_log=dale_log,
        dalb_log=dalb_log,
        drate=drate,
        dratb=dratb,
        coeffs=coeffs,
    )


def _prepare_energy_mesh(
    energy_MeV: np.ndarray,
    energy_min: Optional[float],
    energy_max: Optional[float],
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    e = np.asarray(energy_MeV, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if e.size < 2:
        raise ValueError("Input electron spectrum must contain at least two positive energies")

    emin = float(np.min(e)) if energy_min is None else float(energy_min)
    emax = float(np.max(e)) if energy_max is None else float(energy_max)
    if not (emin > 0 and emax > emin):
        raise ValueError("Invalid energy range")

    log_e = np.linspace(np.log(emin), np.log(emax), int(n_points))
    grid = np.exp(log_e)
    step = float(log_e[1] - log_e[0])
    return grid, log_e, step


def _spectrum_to_shieldose_grid(
    energy_grid: np.ndarray,
    energy_input_MeV: np.ndarray,
    values_input: np.ndarray,
    eunit: float = 1.0,
) -> np.ndarray:
    """Return SHIELDOSE SP array = E * J(E) on energy_grid."""
    e = np.asarray(energy_input_MeV, dtype=float)
    v = np.asarray(values_input, dtype=float)
    if e.size != v.size:
        raise ValueError("energy and spectrum arrays must have the same length")
    m = np.isfinite(e) & np.isfinite(v) & (e > 0) & (v > 0)
    if np.count_nonzero(m) < 2:
        raise ValueError("Need at least two positive spectrum points")
    j_grid = _cubic_loglog_interp_values(energy_grid, e[m], eunit * v[m])
    return energy_grid * j_grid


def _electron_response_arrays(
    db: Shieldose2ElectronDatabase,
    energy_grid_MeV: np.ndarray,
    depths_gcm2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GE and GB arrays following sd2.for.

    Returns
    -------
    ge, gb:
        Arrays with shape (n_energy, n_depth, 2), where m=0 is finite slab
        transmission surface and m=1 is semi-infinite aluminum medium.
    """
    egrid = np.asarray(energy_grid_MeV, dtype=float)
    tel = np.log(egrid)
    depths = np.asarray(depths_gcm2, dtype=float)
    n_e = egrid.size
    n_z = depths.size

    # Basic electron quantities.
    b, c, d = db.coeffs["re"]
    rine = np.exp(_bspol(tel, db.ee_log, db.re_log, b, c, d))

    b, c, d = db.coeffs["rs"]
    rins = rine * np.exp(_bspol(tel, db.te_log, db.rs_log, b, c, d))

    b, c, d = db.coeffs["ar"]
    ares = np.exp(_bspol(tel, db.te_log, db.ar_log, b, c, d))

    b, c, d = db.coeffs["ye"]
    ylde = np.exp(_bspol(tel, db.ee_log, db.ye_log, b, c, d))

    ge = np.zeros((n_e, n_z, 2), dtype=float)
    gb = np.zeros((n_e, n_z, 2), dtype=float)

    lmaxs = db.bs.size
    lmaxe = db.zre.size
    lmaxt = db.zs_log.size
    lmaxb = db.zb_log.size

    for m in range(2):
        for ne in range(n_e):
            tlog = tel[ne]

            # ---- Electron dose response base, DALE ----
            din = np.zeros(lmaxs, dtype=float)
            for l in range(lmaxs):
                b, c, d = db.coeffs[f"dale_te_{l}"]
                din[l] = _bspol(tlog, db.te_log, db.dale_log[:, l], b, c, d)

            # Detector-material ratio for electron dose, DRATE.
            drin_e = np.ones(lmaxe, dtype=float)
            if not (db.detector_index == 0 and m == 0):
                for l in range(lmaxe):
                    b, c, d = db.coeffs[f"drate_te_{m}_{l}"]
                    val = _bspol(tlog, db.te_log, db.drate[:, l, m], b, c, d)
                    drin_e[l] = max(float(val), 0.0)

            # Spline over z/R_ins for electron response.
            b_bs, c_bs, d_bs = _scof(db.bs, din)
            zrin = depths / rins[ne]
            for i, zr in enumerate(zrin):
                if zr >= db.bs[-1]:
                    ge[ne, i, m] = 0.0
                else:
                    ans = _bspol(zr, db.bs, din, b_bs, c_bs, d_bs)
                    ge[ne, i, m] = egrid[ne] * math.exp(ans) * ares[ne] / rins[ne]

            # Detector ratio over z/R_ine.
            b_ze, c_ze, d_ze = _scof(db.zre, drin_e)
            zrin_e = depths / rine[ne]
            for i, zr in enumerate(zrin_e):
                if zr >= db.zre[-1]:
                    ratio = drin_e[-1]
                else:
                    ratio = _bspol(zr, db.zre, drin_e, b_ze, c_ze, d_ze)
                    ratio = max(float(ratio), 0.0)
                ge[ne, i, m] *= ratio

            # ---- Bremsstrahlung dose response base, DALB ----
            dinb = np.zeros(lmaxt, dtype=float)
            for l in range(lmaxt):
                b, c, d = db.coeffs[f"dalb_te_{l}"]
                dinb[l] = _bspol(tlog, db.te_log, db.dalb_log[:, l], b, c, d)

            # Detector-material ratio for brems dose, DRATB.
            drin_b = np.ones(lmaxb, dtype=float)
            if not (db.detector_index == 0 and m == 0):
                for l in range(lmaxb):
                    b, c, d = db.coeffs[f"dratb_te_{m}_{l}"]
                    val = _bspol(tlog, db.te_log, db.dratb[:, l, m], b, c, d)
                    drin_b[l] = max(float(val), 0.0)

            b_zs, c_zs, d_zs = _lcof(db.zs_log, dinb)
            zrin_log = np.log(depths / rine[ne])
            for i, zl in enumerate(zrin_log):
                ans = _bspol(zl, db.zs_log, dinb, b_zs, c_zs, d_zs)
                gb[ne, i, m] = egrid[ne] * math.exp(ans) * ylde[ne] / rine[ne]

            b_zb, c_zb, d_zb = _lcof(db.zb_log, drin_b)
            zlog = np.log(depths)
            for i, zl in enumerate(zlog):
                if zl >= db.zb_log[-1]:
                    ratio = drin_b[-1]
                else:
                    ratio = _bspol(zl, db.zb_log, drin_b, b_zb, c_zb, d_zb)
                    ratio = max(float(ratio), 0.0)
                gb[ne, i, m] *= ratio

    return ge, gb


@dataclass
class ElectronDoseResult:
    depths_gcm2: np.ndarray
    electron_slab: np.ndarray
    brems_slab: np.ndarray
    total_slab: np.ndarray
    electron_semi_infinite: np.ndarray
    brems_semi_infinite: np.ndarray
    total_semi_infinite: np.ndarray
    electron_sphere_center_half: np.ndarray
    brems_sphere_center_half: np.ndarray
    total_sphere_center_half: np.ndarray
    output_unit: str
    metadata: Dict[str, object]

    def to_rows(self) -> Sequence[Dict[str, float]]:
        rows = []
        for i, z in enumerate(self.depths_gcm2):
            rows.append({
                "depth_gcm2": float(z),
                f"electron_slab_{self.output_unit}": float(self.electron_slab[i]),
                f"brems_slab_{self.output_unit}": float(self.brems_slab[i]),
                f"total_slab_{self.output_unit}": float(self.total_slab[i]),
                f"electron_semi_infinite_{self.output_unit}": float(self.electron_semi_infinite[i]),
                f"brems_semi_infinite_{self.output_unit}": float(self.brems_semi_infinite[i]),
                f"total_semi_infinite_{self.output_unit}": float(self.total_semi_infinite[i]),
                f"electron_sphere_center_half_{self.output_unit}": float(self.electron_sphere_center_half[i]),
                f"brems_sphere_center_half_{self.output_unit}": float(self.brems_sphere_center_half[i]),
                f"total_sphere_center_half_{self.output_unit}": float(self.total_sphere_center_half[i]),
            })
        return rows

    def save_csv(self, path: Union[str, Path]) -> None:
        rows = list(self.to_rows())
        if not rows:
            raise ValueError("No rows to save")
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def calculate_electron_dose(
    energy_MeV: Sequence[float],
    spectrum: Sequence[float],
    depths_gcm2: Sequence[float],
    database: Union[str, Path, Shieldose2ElectronDatabase] = "elbrbas2.dat",
    detector: str = DEFAULT_DETECTOR,
    quantity: Literal["flux", "fluence"] = "flux",
    duration: float = 1.0,
    energy_min_MeV: Optional[float] = None,
    energy_max_MeV: Optional[float] = None,
    n_energy_points: int = 1001,
    omnidirectional_to_planar_factor: float = 0.25,
    eunit: float = 1.0,
) -> ElectronDoseResult:
    """Calculate electron and bremsstrahlung dose behind Al shielding.

    Parameters
    ----------
    energy_MeV:
        Incident electron energies in MeV.
    spectrum:
        If quantity="flux", differential omnidirectional flux
        [particles/(MeV cm^2 s)]. Output is rad/s when duration=1.
        If quantity="fluence", differential omnidirectional fluence
        [particles/(MeV cm^2)]. Output is rad.
    depths_gcm2:
        Aluminum shielding depths in g/cm^2. Use several log-spaced depths if
        the SHIELDOSE-2 sphere-center result is needed, because it is based on
        a derivative with respect to shield depth.
    database:
        Parsed Shieldose2ElectronDatabase or path to ELBRBAS2.DAT.
    detector:
        Detector material. Default is "Si".
    duration:
        Multiplicative duration. For flux input, duration=1 gives rad/s;
        duration=T gives integrated rad over T seconds if the input flux is
        per second. For fluence input normally keep duration=1.
    omnidirectional_to_planar_factor:
        Default 0.25, matching SHIELDOSE-2 for incident omnidirectional flux.
    eunit:
        Conversion factor from spectrum denominator energy unit to MeV.
        Use 1 for /MeV, 1000 for /keV if spectrum values are per keV.
    """
    if isinstance(database, Shieldose2ElectronDatabase):
        db = database
        if detector != db.detector and detector != DEFAULT_DETECTOR:
            # User supplied a parsed DB; trust it, but make the mismatch visible.
            pass
    else:
        db = load_elbrbas2_database(database, detector=detector)

    energy = np.asarray(energy_MeV, dtype=float)
    spec = np.asarray(spectrum, dtype=float)
    depths = np.asarray(depths_gcm2, dtype=float)
    if np.any(~np.isfinite(depths)) or np.any(depths <= 0):
        raise ValueError("All shielding depths must be finite and positive g/cm^2")
    order = np.argsort(depths)
    depths_sorted = depths[order]

    egrid, log_grid, dee = _prepare_energy_mesh(
        energy, energy_min_MeV, energy_max_MeV, n_energy_points
    )
    seg = _spectrum_to_shieldose_grid(egrid, energy, spec, eunit=eunit)

    ge, gb = _electron_response_arrays(db, egrid, depths_sorted)

    delta = (
        float(duration)
        * RADCON_MEV_PER_G_TO_RAD
        * float(omnidirectional_to_planar_factor)
        * dee / 3.0
    )

    electron_slab_sorted = np.zeros_like(depths_sorted, dtype=float)
    brems_slab_sorted = np.zeros_like(depths_sorted, dtype=float)
    electron_semi_sorted = np.zeros_like(depths_sorted, dtype=float)
    brems_semi_sorted = np.zeros_like(depths_sorted, dtype=float)

    for i in range(depths_sorted.size):
        electron_slab_sorted[i] = _integ_shieldose(delta, seg * ge[:, i, 0])
        brems_slab_sorted[i] = _integ_shieldose(delta, seg * gb[:, i, 0])
        electron_semi_sorted[i] = _integ_shieldose(delta, seg * ge[:, i, 1])
        brems_semi_sorted[i] = _integ_shieldose(delta, seg * gb[:, i, 1])

    electron_sphere_sorted = _sphere_half_from_slab(depths_sorted, electron_slab_sorted)
    brems_sphere_sorted = _sphere_half_from_slab(depths_sorted, brems_slab_sorted)

    # Restore original depth order.
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    electron_slab = electron_slab_sorted[inv]
    brems_slab = brems_slab_sorted[inv]
    electron_semi = electron_semi_sorted[inv]
    brems_semi = brems_semi_sorted[inv]
    electron_sphere = electron_sphere_sorted[inv]
    brems_sphere = brems_sphere_sorted[inv]

    if quantity == "flux":
        output_unit = "rad_per_s" if duration == 1.0 else "rad"
    elif quantity == "fluence":
        output_unit = "rad"
    else:
        raise ValueError("quantity must be 'flux' or 'fluence'")

    metadata = {
        "model": "SHIELDOSE-2 electron/brems dose response",
        "database_tag": db.tag,
        "detector": db.detector,
        "incident_spectrum": "omnidirectional",
        "omnidirectional_to_planar_factor": omnidirectional_to_planar_factor,
        "quantity": quantity,
        "duration_multiplier": duration,
        "energy_grid_min_MeV": float(egrid[0]),
        "energy_grid_max_MeV": float(egrid[-1]),
        "n_energy_points": int(n_energy_points),
        "electron_output_spectrum_calculated": False,
        "bremsstrahlung_dose_included": True,
        "sphere_center_quantity": "SHIELDOSE-2 1/2 dose at center of aluminum spheres",
    }

    return ElectronDoseResult(
        depths_gcm2=np.asarray(depths, dtype=float),
        electron_slab=electron_slab,
        brems_slab=brems_slab,
        total_slab=electron_slab + brems_slab,
        electron_semi_infinite=electron_semi,
        brems_semi_infinite=brems_semi,
        total_semi_infinite=electron_semi + brems_semi,
        electron_sphere_center_half=electron_sphere,
        brems_sphere_center_half=brems_sphere,
        total_sphere_center_half=electron_sphere + brems_sphere,
        output_unit=output_unit,
        metadata=metadata,
    )


def load_two_column_spectrum(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column electron spectrum file: E[MeV], J(E)."""
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                e = float(parts[0].replace("D", "E").replace("d", "e"))
                j = float(parts[1].replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if np.isfinite(e) and np.isfinite(j):
                rows.append((e, j))
    if len(rows) < 2:
        raise ValueError(f"Could not read at least two numeric rows from {path}")
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1]


def _parse_depths(s: str) -> np.ndarray:
    vals = []
    for part in re.split(r"[,;\s]+", s.strip()):
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No depths parsed")
    return np.asarray(vals, dtype=float)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SHIELDOSE-2 electron/brems dose in silicon behind Al shielding."
    )
    parser.add_argument("--spectrum", required=True,
                        help="Two-column file: E_MeV, electron flux/fluence")
    parser.add_argument("--database", default="elbrbas2.dat",
                        help="Path to ELBRBAS2.DAT")
    parser.add_argument("--detector", default="Si",
                        help="Detector material, default Si")
    parser.add_argument("--depths", default="0.01,0.0316,0.1,0.316,1,3.16,10,31.6,100",
                        help="Comma/space separated Al depths in g/cm2")
    parser.add_argument("--quantity", choices=["flux", "fluence"], default="flux",
                        help="'flux' gives rad/s for duration=1; 'fluence' gives rad")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Multiplicative duration. For flux input, use seconds if flux is per second.")
    parser.add_argument("--energy-min", type=float, default=None,
                        help="Integration energy min MeV. Default: min input energy.")
    parser.add_argument("--energy-max", type=float, default=None,
                        help="Integration energy max MeV. Default: max input energy.")
    parser.add_argument("--n-energy", type=int, default=1001,
                        help="Number of log energy mesh points")
    parser.add_argument("--eunit", type=float, default=1.0,
                        help="Conversion from input spectrum energy unit to /MeV. Use 1 for /MeV.")
    parser.add_argument("--omni-factor", type=float, default=0.25,
                        help="Omnidirectional-to-planar factor; SHIELDOSE-2 default 0.25")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    args = parser.parse_args(argv)

    energy, spec = load_two_column_spectrum(args.spectrum)
    depths = _parse_depths(args.depths)
    result = calculate_electron_dose(
        energy_MeV=energy,
        spectrum=spec,
        depths_gcm2=depths,
        database=args.database,
        detector=args.detector,
        quantity=args.quantity,
        duration=args.duration,
        energy_min_MeV=args.energy_min,
        energy_max_MeV=args.energy_max,
        n_energy_points=args.n_energy,
        omnidirectional_to_planar_factor=args.omni_factor,
        eunit=args.eunit,
    )
    result.save_csv(args.out)

    print(f"Saved: {args.out}")
    print(f"Output unit: {result.output_unit}")
    print(f"Detector: {result.metadata['detector']}")
    print("Note: sphere_center_half is the SHIELDOSE-2 '1/2 dose at center of aluminum spheres'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
