#!/usr/bin/env python3
"""Relative L2 and Linf error of the last plotfile of each run against the exact solution.

Norms over valid gas cells (cellmask = 1, not covered by a finer level) of all levels, area weighted, relative to
the maximum exact value there. Exact solution (UserSources.H):
phi = r^4/32, n_i = r^2/(2 alpha) + n0, n_e = electron energy = r^2/alpha + n0, alpha = e/eps0, n0 = 1e6.
usage: mms_errors.py <run dir>...      prints one table row per run
"""
import sys, os, glob, time
import numpy as np, yt
yt.set_log_level(50)
A, N0 = 1.602176634e-19 / 8.8541878128e-12, 1.0e6
FIELDS = [("Potential", lambda r2: r2**2 / 32), ("HEp", lambda r2: r2 / (2 * A) + N0),
          ("E", lambda r2: r2 / A + N0), ("Electron_energy", lambda r2: r2 / A + N0)]


def errors(p):
    ds = yt.load(p)
    out, ncell = {}, 0
    for f, fn in FIELDS:
        s = w = mx = lin = 0.0
        for g in ds.index.grids:
            v = (np.array(g[("boxlib", "cellmask")])[:, :, 0] > 1 - 1e-10) & np.array(g.child_mask)[:, :, 0].astype(bool)
            if not v.any():
                continue
            X = np.array(g[("index", "x")])[:, :, 0]; Y = np.array(g[("index", "y")])[:, :, 0]
            e = fn(X**2 + Y**2); d = np.abs(np.array(g[("boxlib", f)])[:, :, 0] - e)[v]
            a = np.prod(g.dds.d[:2]); s += a * (d**2).sum(); w += a * v.sum()
            lin = max(lin, d.max()); mx = max(mx, e[v].max())
            if f == "Potential":
                ncell += g.ActiveDimensions[:2].prod()
        out[f] = (np.sqrt(s / w) / mx, lin / mx)
    return out, ncell, ds.max_level


print(f"{'run':<22}{'levels':>7}{'cells':>9}{'time s':>8}" + "".join(f"{n + ' L2':>12}{n + ' Linf':>12}"
      for n in ("phi", "n_i", "n_e", "energy")))
for run in sys.argv[1:]:
    plts = sorted(p for p in glob.glob(os.path.join(run, "plt*")) if os.path.isdir(p))
    if not plts:
        print(f"{run}: no plotfile"); continue
    e, nc, ml = errors(plts[-1])
    t = "-"
    if os.path.isfile(os.path.join(run, "time")):
        for line in open(os.path.join(run, "time")):
            if line.startswith("real"):
                t = f"{float(line.split()[1]):.0f}"
    name = "/".join(os.path.normpath(run).split(os.sep)[-3:]).replace("/runs/", "/")
    print(f"{name:<22}{ml:>7}{nc:>9}{t:>8}" + "".join(f"{e[f][0]:12.3e}{e[f][1]:12.3e}" for f, _ in FIELDS))
