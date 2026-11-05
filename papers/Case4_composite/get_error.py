#!/usr/bin/env python3
"""Error norms for the GEC-like MMS case, overall and by geometric feature.
usage: get_error.py runs/<config> [more configs]   (reads <config>/<n>/plt00001)"""
import sys, os, re, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfread import Plotfile
ECH, EPS0 = 1.602176634e-19, 8.8541878128e-12
ALPHA, N0 = ECH / EPS0, 1.0e6
# geometry (must match inputs2d)
Ro, cox, coy = 2.6, 0.05, -0.03
R1, g, xc, sw, sy = 1.125, 0.5625, 0.04, 0.45, 1.3625
H = 2.0625   # flat top/bottom plates
def corners():
    ye = g + np.sqrt(R1**2 - sw**2)                       # arc meets stem
    xk = lambda yy: np.sqrt(Ro**2 - (yy - coy)**2)        # chamber meets plate
    return {"C rim (flat cap meets arc)": [(xc - R1, g), (xc + R1, g)],
            "E stem meets dome (obtuse)": [(xc - sw, ye), (xc + sw, ye)],
            "F stem meets top plate (90 deg)": [(xc - sw, H), (xc + sw, H)],
            "H aligned convex corner, grid node": [(-R1, -g), (R1, -g)],
            "I aligned concave corner, grid node": [(-R1, -H), (R1, -H)],
            "K chamber meets plates (obtuse)": [(cox - xk(H), H), (cox + xk(H), H), (cox - xk(-H), -H), (cox + xk(-H), -H)]}
def regions(X, Y, dx, rc=0.25, nb=3):
    reg = np.full(X.shape, "J interior", dtype=object); near = nb * dx
    rch = np.hypot(X - cox, Y - coy); rd = np.hypot(X - xc, Y - g)
    reg[np.abs(rch - Ro) < near] = "A chamber wall (curved)"
    reg[(np.abs(rd - R1) < near) & (Y > g)] = "D dome arc (curved)"
    reg[(np.abs(np.abs(X - xc) - sw) < near) & (Y > sy)] = "stem sides (aligned off-grid)"
    reg[(np.abs(np.abs(X) - R1) < near) & (Y < -g)] = "rect sides (on grid lines)"
    reg[np.abs(np.abs(Y) - H) < near] = "plates (on grid lines)"
    reg[(np.abs(Y - g) < near) & (np.abs(X - xc) < R1)] = "B flat cap (on grid line)"
    reg[(np.abs(Y + g) < near) & (np.abs(X) < R1)] = "G rect top (on grid line)"
    for name, pts in corners().items():
        for (px, py) in pts: reg[np.hypot(X - px, Y - py) < rc] = name
    return reg
def load(path):
    ds = Plotfile(path); v = ["Potential", "Efieldx", "Efieldy", "E", "HEp", "Electron_energy", "cellmask"]
    d = ds.read(v); n = ds.dims; dx = ds.dx[0]
    x = ds.prob_lo[0] + (np.arange(n[0]) + 0.5) * dx; y = ds.prob_lo[1] + (np.arange(n[1]) + 0.5) * dx
    X, Y = np.meshgrid(x, y, indexing="ij"); r2 = X**2 + Y**2
    ex = {"Potential": r2**2 / 32, "Efieldx": -X * r2 / 8, "Efieldy": -Y * r2 / 8,
          "E": r2 / ALPHA + N0, "HEp": r2 / (2 * ALPHA) + N0, "Electron_energy": r2 / ALPHA + N0}
    m = d["cellmask"] > 1.0 - 1e-10   # fluid only; cellmask carries the volume fraction
    err = {k: np.abs(d[k] - ex[k]) / np.abs(ex[k][m]).max() for k in ex}   # relative to max exact value
    return n[0], dx, X, Y, m, err
def norms(e, m):
    a = e[m]; return a.mean(), np.sqrt((a**2).mean()), a.max()
def iters(logf):
    its = [int(q) for q in re.findall(r"Final Iter\. (\d+)", open(logf).read())]
    return (np.mean(its), max(its)) if its else (np.nan, np.nan)
def main():
    names = {"Potential": "phi", "Efieldx": "Ex", "Efieldy": "Ey", "E": "ne", "HEp": "ni", "Electron_energy": "Een"}
    for cfg in sys.argv[1:]:
        print("\n" + "=" * 100 + f"\nCONFIG {cfg}\n" + "=" * 100)
        res = sorted(int(os.path.basename(p)) for p in glob.glob(cfg + "/*") if os.path.basename(p).isdigit() and os.path.isdir(f"{p}/plt00001"))
        tab = {}
        for n in res:
            n_, dx, X, Y, m, err = load(f"{cfg}/{n}/plt00001"); tab[n] = (dx, X, Y, m, err)
        for k, s in names.items():
            print(f"\n {s:4s}    n      L1         L2        Linf     order(L2)  order(Linf)   [solver iters mean/max]")
            prev = None
            for n in res:
                dx, X, Y, m, err = tab[n]; l1, l2, li = norms(err[k], m)
                o2 = np.log2(prev[0] / l2) if prev else np.nan; oi = np.log2(prev[1] / li) if prev else np.nan
                it = iters(f"{cfg}/{n}/log.{n}") if k == "Potential" else ("", "")
                print(f"       {n:5d}  {l1:.3e}  {l2:.3e}  {li:.3e}   {o2:6.2f}     {oi:6.2f}       {it[0] if it[0]=='' else f'{it[0]:.1f}/{it[1]}'}")
                prev = (l2, li)
        print("\n Linf error by feature (phi | ne), with observed order between the two finest grids")
        print(f" {'feature':38s}" + "".join(f"  n={n:<4d} phi     ne   " for n in res[-3:]) + "  order phi  order ne")
        feats = None
        for n in res[-3:]:
            dx, X, Y, m, err = tab[n]; tab[n] += (regions(X, Y, dx),)
        feats = sorted(set(np.unique(tab[res[-1]][5][tab[res[-1]][3]])))
        for f in feats:
            row = f" {f:38s}"; vals = []
            for n in res[-3:]:
                dx, X, Y, m, err, reg = tab[n]; sel = m & (reg == f)
                p = err["Potential"][sel].max() if sel.any() else np.nan; e = err["E"][sel].max() if sel.any() else np.nan
                vals.append((p, e)); row += f"  {p:.2e} {e:.2e}  "
            row += f"   {np.log2(vals[-2][0]/vals[-1][0]):6.2f}    {np.log2(vals[-2][1]/vals[-1][1]):6.2f}"
            print(row)
if __name__ == "__main__":
    main()
