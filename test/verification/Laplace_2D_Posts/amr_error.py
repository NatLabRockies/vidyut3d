#!/usr/bin/env python3
"""Error norms for Laplace_2D_Posts, AMR aware.

Exact solution phi = x^2 - y^2. It is harmonic, so it solves the equation the
case discretises, and its fourth derivatives vanish, so a second-order stencil
reproduces it EXACTLY in the interior on any grid. Everything that shows up
here is therefore immersed-boundary error, produced at the posts and at the
narrow gaps between them.

Each level contributes only its valid (uncovered) cells and the L2 norm is
volume weighted, so levels of different spacing combine correctly. Only fluid
cells count: cellmask > 1 - 1e-10. A looser threshold lets in the cut cells the
solver masks out and gives a bogus O(1) max error.

  python3 amr_error.py <dir-or-plotfile> [more ...] [--near HALF]

--near restricts the norm to |x|,|y| <= HALF, i.e. to the refined patch around
the posts, which is where a locally refined run is supposed to buy accuracy.
"""
import glob
import os
import sys

import numpy as np
import yt

yt.set_log_level(50)


def exact(X, Y):
    return X * X - Y * Y


def analyse(pf, half=None):
    ds = yt.load(pf)
    maxlev = ds.index.max_level
    e2 = vol = linf = 0.0
    ncell = 0
    for g in ds.index.grids:
        cm = np.array(g["boxlib", "cellmask"])[:, :, 0]
        phi = np.array(g["boxlib", "Potential"])[:, :, 0]
        child = np.array(g.child_mask)[:, :, 0].astype(bool)
        le, dd = g.LeftEdge.d, g.dds.d
        nx, ny = cm.shape
        X, Y = np.meshgrid(
            le[0] + (np.arange(nx) + 0.5) * dd[0],
            le[1] + (np.arange(ny) + 0.5) * dd[1],
            indexing="ij",
        )
        sel = (cm > 1.0 - 1e-10) & child
        if half is not None:
            sel &= (np.abs(X) <= half) & (np.abs(Y) <= half)
        if not sel.any():
            continue
        err = np.abs(phi - exact(X, Y))[sel]
        cv = float(dd[0] * dd[1])
        e2 += float((err**2).sum()) * cv
        vol += sel.sum() * cv
        linf = max(linf, float(err.max()))
        ncell += int(sel.sum())
    return maxlev, (np.sqrt(e2 / vol) if vol else float("nan")), linf, ncell


def main():
    args = sys.argv[1:]
    half = None
    if "--near" in args:
        k = args.index("--near")
        half = float(args[k + 1])
        args = args[:k] + args[k + 2 :]
    if half is not None:
        print(f"restricted to |x|,|y| <= {half} (the refined patch)")
    print(f"{'case':10s} {'lev':>3s} {'cells':>9s} {'L2':>12s} {'rate':>6s} "
          f"{'Linf':>12s} {'rate':>6s}")
    print("-" * 64)
    prev = None
    for a in args:
        pf = a
        if os.path.isdir(a) and not os.path.exists(os.path.join(a, "Header")):
            c = sorted(glob.glob(os.path.join(a, "plt*")))
            if not c:
                continue
            pf = c[-1]
        lev, l2, linf, n = analyse(pf, half)
        name = os.path.basename(a.rstrip("/"))
        if prev:
            r2 = np.log2(prev[0] / l2) if l2 > 0 else float("nan")
            ri = np.log2(prev[1] / linf) if linf > 0 else float("nan")
            print(f"{name:10s} {lev:3d} {n:9d} {l2:12.4e} {r2:6.2f} "
                  f"{linf:12.4e} {ri:6.2f}")
        else:
            print(f"{name:10s} {lev:3d} {n:9d} {l2:12.4e} {'':6s} "
                  f"{linf:12.4e} {'':6s}")
        prev = (l2, linf)


if __name__ == "__main__":
    main()
