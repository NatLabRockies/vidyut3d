"""Accuracy of the stored wall foot / normal of the pin-to-pin EB case.

For every fluid cell that has a masked face neighbour (the cells bc_ib uses),
compares EB_cp_* / EB_norm_* of the plotfile with the analytical foot on the
paraboloidal pins. Errors are normalised by h^2/R and h/R with R the tip radius
of curvature. Works for 2D (r-z) and 3D plotfiles.

usage: python check_pin_geometry.py [--inputs inputsfile] plt00000 [plt...]
       the pin arrays (prob.dgap, pinrad, nbot, bot_centers, bot_pinrad, ntop,
       top_centers, top_pinrad) are read from the inputs file (default: inputs2d
       for 2D plotfiles, inputs3d for 3D ones)
"""
import os, sys, numpy as np
from pfread import Plotfile

def read_inputs(path):
    v = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, val = line.split("=", 1); v[k.strip()] = val.split()
    return v

def pin_arrays(inp, dim):
    dgap = float(inp.get("prob.dgap", [4e-3])[0]); pinrad = float(inp.get("prob.pinrad", [1e-3])[0])
    out = {}
    for name in ("bot", "top"):
        n = int(inp.get(f"prob.n{name}", [1])[0])
        cv = [float(x) for x in inp.get(f"prob.{name}_centers", [])]
        rv = [float(x) for x in inp.get(f"prob.{name}_pinrad", [])]
        cen = np.zeros((n, 2))
        if cv:
            cv = np.array(cv).reshape(n, dim - 1); cen[:, :dim - 1] = cv
        rad = np.full(n, pinrad) if not rv else (np.full(n, rv[0]) if len(rv) == 1 else np.array(rv))
        out[name] = (cen, rad)
    return dgap, out

def foot_rho(c1, c2, rho0, ax0, rscale):
    p = 1.0 + 2.0 * c1 * (c2 - ax0)
    g = lambda r: 2 * c1 * c1 * r**3 + p * r - rho0
    lo = np.zeros_like(rho0); hi = np.maximum(rho0, rscale)
    for _ in range(60):
        m = g(hi) < 0
        if not m.any(): break
        hi = np.where(m, 2 * hi, hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi); m = g(mid) < 0
        lo = np.where(m, mid, lo); hi = np.where(m, hi, mid)
    return 0.5 * (lo + hi)

args = sys.argv[1:]; inputs = None
if args and args[0] == "--inputs": inputs = args[1]; args = args[2:]
for path in args:
    pf = Plotfile(path); dim = pf.dim
    inp = read_inputs(inputs or ("inputs2d" if dim == 2 else "inputs3d"))
    AX = int(inp.get("prob.pin_axis", [dim - 1])[0]); T = [a for a in range(dim) if a != AX]
    DGAP, pins = pin_arrays(inp, dim)
    names = ["cellmask", "EB_cp_x", "EB_cp_y", "EB_norm_x", "EB_norm_y"]
    if dim == 3: names += ["EB_cp_z", "EB_norm_z"]
    d = pf.read(names)
    mask = d["cellmask"] > 0.5
    nb = np.zeros_like(mask)
    for ax in range(dim):
        for s in (1, -1):
            sh = np.roll(mask, s, axis=ax)
            # roll wraps: neighbours outside the domain count as fluid
            idx = [slice(None)] * dim; idx[ax] = 0 if s == 1 else -1
            sh[tuple(idx)] = True
            nb |= ~sh
    sel = mask & nb
    h = pf.dx[0]
    ijk = np.indices(pf.dims)
    X = np.stack([pf.prob_lo[a] + (ijk[a] + 0.5) * pf.dx[a] for a in range(dim)])[:, sel]
    cp = np.stack([d["EB_cp_x"], d["EB_cp_y"]] + ([d["EB_cp_z"]] if dim == 3 else []))[:, sel]
    nn = np.stack([d["EB_norm_x"], d["EB_norm_y"]] + ([d["EB_norm_z"]] if dim == 3 else []))[:, sel]
    lo, hi = pf.prob_lo[AX], pf.prob_hi[AX]; l = hi - lo
    c2b = lo + 0.5 * (l - DGAP); c2t = c2b + DGAP
    R = min(r**2 / (2.0 * abs(c2b - lo)) for name in pins for r in pins[name][1])
    ax0 = X[AX]
    best = None
    for name, (cen, rad) in pins.items():
        c2 = c2b if name == "bot" else c2t; s = 1.0 if name == "bot" else -1.0
        for c, r in zip(cen, rad):
            c1 = ((lo if name == "bot" else hi) - c2) / r**2
            dxv = X[T[0]] - c[0]; dyv = (X[T[1]] - c[1]) if dim == 3 else np.zeros_like(dxv)
            rho0 = np.hypot(dxv, dyv)
            ex = np.where(rho0 > 0, dxv / np.where(rho0 > 0, rho0, 1), 1.0)
            ey = np.where(rho0 > 0, dyv / np.where(rho0 > 0, rho0, 1), 0.0)
            rs = foot_rho(c1, c2, rho0, ax0, r); as_ = c1 * rs**2 + c2
            d2 = (rs - rho0) ** 2 + (as_ - ax0) ** 2
            nr, na = s * 2 * c1 * rs, -s * np.ones_like(rs); nm = np.hypot(nr, na); nr /= nm; na /= nm
            foot_c = np.zeros_like(X); n_c = np.zeros_like(X)
            foot_c[T[0]] = c[0] + ex * rs; n_c[T[0]] = ex * nr
            if dim == 3: foot_c[T[1]] = c[1] + ey * rs; n_c[T[1]] = ey * nr
            foot_c[AX] = as_; n_c[AX] = na
            if best is None:
                best = (d2, foot_c, n_c)
            else:
                m = d2 < best[0]
                best = (np.where(m, d2, best[0]), np.where(m[None], foot_c, best[1]), np.where(m[None], n_c, best[2]))
    d2, foot, nex = best
    nmag = np.linalg.norm(nn, axis=0)
    nogeom = nmag < 1e-14
    ok = ~nogeom
    nn_u = nn[:, ok] / nmag[ok]
    fe = np.linalg.norm(cp[:, ok] - foot[:, ok], axis=0) / (h * h / R)
    ne = np.linalg.norm(nn_u - nex[:, ok], axis=0) / (h / R)
    print(f"{path}: dim={dim} h={h:.3e} R_tip={R:.3e} h/R={h/R:.3f}  masked cells={int((~mask).sum())}  wall cells={int(sel.sum())}  without geometry={int(nogeom.sum())}")
    print(f"   foot error /(h^2/R): mean {fe.mean():.3f}  p99 {np.percentile(fe,99):.3f}  max {fe.max():.3f}")
    print(f"   normal error /(h/R): mean {ne.mean():.3f}  p99 {np.percentile(ne,99):.3f}  max {ne.max():.3f}")
    dn = np.sqrt(d2[ok])
    print(f"   distance cell centre to wall / h: min {dn.min()/h:.3f}  max {dn.max()/h:.3f}")
