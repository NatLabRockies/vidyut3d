#!/usr/bin/env python3
"""AMR-aware error norms for the MMS2 manufactured solution.

all_errors.py reads a single level. This one walks the whole hierarchy: each
level contributes only its valid (uncovered) region and the L2 norm is volume
weighted, so levels of different dx combine correctly. Only fluid cells count
(cellmask > 1 - 1e-10); a looser threshold lets in the cut cells the solver
masks out and gives a bogus O(1) max error.

Use it to check that an AMR hierarchy reproduces the uniform grid whose
resolution matches its finest level. It should, provided every coarse-fine
interface clears the immersed boundary by at least two coarse cells - see
Vidyut::check_ib_cf_clearance and vidyut.ib_cf_clearance_warn.

Usage:
  python3 amr_error.py <dir-or-plotfile> [more ...]
"""
import glob
import os
import sys

import numpy as np
import yt

yt.set_log_level(50)

ECH, EPS0 = 1.602176634e-19, 8.8541878128e-12
ALPHA, N0 = ECH / EPS0, 1.0e6
FIELDS = ["Potential", "Efieldx", "Efieldy", "E", "HEp", "cellmask"]
REPORT = ["Potential", "Efieldx", "E", "HEp"]


def exact(X, Y):
    """The manufactured solution: phi = r^4/32, E = -grad phi, species ~ r^2."""
    r2 = X**2 + Y**2
    return {
        "Potential": r2**2 / 32.0,
        "Efieldx": -X * r2 / 8.0,
        "Efieldy": -Y * r2 / 8.0,
        "E": r2 / ALPHA + N0,
        "HEp": 0.5 * r2 / ALPHA + N0,
    }


def level_arrays(ds, lev, fields):
    """Whole-domain arrays for one level, plus a mask of where that level exists."""
    nx, ny = (ds.domain_dimensions[:2] * 2**lev).astype(int)
    present = np.zeros((nx, ny), dtype=bool)
    data = {f: np.zeros((nx, ny)) for f in fields}
    for g in ds.index.grids:
        if g.Level != lev:
            continue
        i0, j0 = g.get_global_startindex()[:2]
        gx, gy = int(g.ActiveDimensions[0]), int(g.ActiveDimensions[1])
        sl = (slice(i0, i0 + gx), slice(j0, j0 + gy))
        present[sl] = True
        for f in fields:
            data[f][sl] = np.asarray(g["boxlib", f])[:, :, 0]
    return present, data


def analyse(pf):
    ds = yt.load(pf)
    maxlev = ds.index.max_level
    lo = ds.domain_left_edge.to_value()[:2]
    dx0 = (ds.domain_right_edge.to_value()[:2] - lo) / ds.domain_dimensions[:2]
    levels = [level_arrays(ds, l, FIELDS) for l in range(maxlev + 1)]

    acc = {k: {"e2": 0.0, "vol": 0.0, "linf": 0.0} for k in REPORT}
    scale = {}
    for lev in range(maxlev + 1):
        present, data = levels[lev]
        dx = dx0 / 2**lev
        nx, ny = present.shape
        X, Y = np.meshgrid(
            lo[0] + (np.arange(nx) + 0.5) * dx[0],
            lo[1] + (np.arange(ny) + 0.5) * dx[1],
            indexing="ij",
        )
        ex = exact(X, Y)
        fluid = present & (data["cellmask"] > 1.0 - 1e-10)
        for k in REPORT:
            if fluid.any():
                scale[k] = max(scale.get(k, 0.0), np.abs(ex[k][fluid]).max())

    for lev in range(maxlev + 1):
        present, data = levels[lev]
        dx = dx0 / 2**lev
        nx, ny = present.shape
        covered = np.zeros_like(present)
        if lev < maxlev:
            covered = levels[lev + 1][0].reshape(nx, 2, ny, 2).any(axis=(1, 3))
        fluid = present & ~covered & (data["cellmask"] > 1.0 - 1e-10)
        if not fluid.any():
            continue
        X, Y = np.meshgrid(
            lo[0] + (np.arange(nx) + 0.5) * dx[0],
            lo[1] + (np.arange(ny) + 0.5) * dx[1],
            indexing="ij",
        )
        ex = exact(X, Y)
        cellvol = dx[0] * dx[1]
        for k in REPORT:
            e = np.abs(data[k] - ex[k])[fluid] / scale[k]
            acc[k]["e2"] += (e**2).sum() * cellvol
            acc[k]["vol"] += fluid.sum() * cellvol
            acc[k]["linf"] = max(acc[k]["linf"], e.max())

    eff = int(ds.domain_dimensions[0] * 2**maxlev)
    return eff, maxlev, {
        k: (np.sqrt(acc[k]["e2"] / acc[k]["vol"]), acc[k]["linf"]) for k in REPORT
    }


def resolve(arg):
    if os.path.isdir(arg):
        cands = sorted(glob.glob(os.path.join(arg, "plt?????")))
        if cands:
            return cands[-1]
    return arg


def main(args):
    hdr = f"{'case':>16} {'eff':>6} {'lev':>4}"
    for k in REPORT:
        hdr += f" {k[:8]+' L2':>13} {'rate':>5}"
    print(hdr)
    prev = None
    for a in args:
        pf = resolve(a)
        if not os.path.exists(pf):
            print(f"{os.path.basename(a):>16}  (no plotfile)")
            continue
        eff, lev, r = analyse(pf)
        row = f"{os.path.basename(os.path.normpath(a)):>16} {eff:>6} {lev:>4}"
        for k in REPORT:
            rate = np.log2(prev[k][0] / r[k][0]) if prev else float("nan")
            row += f" {r[k][0]:13.4e} {rate:5.2f}"
        print(row)
        prev = r


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
