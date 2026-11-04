#!/usr/bin/env python3
"""Compare a geometry-switch run with the static run of its final geometry.
usage: compare_switch.py runs/<final> runs/<switch>   (last plotfile of each, any AMR depth)"""
import sys, glob, numpy as np, yt
yt.set_log_level(50)
ECH, EPS0 = 1.602176634e-19, 8.8541878128e-12
ALPHA, N0 = ECH / EPS0, 1.0e6
F = ["Potential", "Efieldx", "Efieldy", "E", "HEp", "cellmask"]
def last(d): return sorted(p for p in glob.glob(d + "/plt*"))[-1]
A, B = yt.load(last(sys.argv[1])), yt.load(last(sys.argv[2]))
have = {f for _, f in A.field_list}
F = [f for f in F if f in have]  # a Laplace case has no ion field
print("final :", last(sys.argv[1]), " t =", float(A.current_time))
print("switch:", last(sys.argv[2]), " t =", float(B.current_time))
ga = {(g.Level, tuple(np.round(g.LeftEdge.d[:2], 12))): g for g in A.index.grids}
gb = {(g.Level, tuple(np.round(g.LeftEdge.d[:2], 12))): g for g in B.index.grids}
same = set(ga) == set(gb) and all((ga[k].ActiveDimensions == gb[k].ActiveDimensions).all() for k in ga)
print("identical AMR hierarchy:", same, f"({len(ga)} grids, levels {A.max_level})")
if not same: sys.exit("hierarchies differ")
mx = {}; acc = {f: [0.0, 0.0] for f in F[:-1]}; maskdiff = 0
for k in ga:
    a, b = ga[k], gb[k]
    ma = np.array(a[("boxlib", "cellmask")]); mb = np.array(b[("boxlib", "cellmask")])
    maskdiff += int((ma != mb).sum())
    m = (ma > 1 - 1e-10) & np.array(a.child_mask).astype(bool)
    for f in F[:-1]:
        va = np.array(a[("boxlib", f)]); vb = np.array(b[("boxlib", f)])
        d = np.abs(va - vb)[m]
        if d.size:
            acc[f][0] = max(acc[f][0], d.max()); acc[f][1] = max(acc[f][1], np.abs(va[m]).max())
print("cells whose CMASK differs:", maskdiff)
for f in F[:-1]:
    print(f"  {f:10s} max |switch - final| = {acc[f][0]:.3e}   (relative to max |field| {acc[f][0]/acc[f][1]:.3e})")
