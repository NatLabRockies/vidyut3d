"""Errors of the MMS1 spherical-shell case.

usage: python sphere_errors.py <rundir> [<rundir> ...]
Each <rundir> contains sub-folders 32, 64, ... with the final plotfile.
Norms: L2 = sqrt(mean over ALL cells of the masked squared error) (paper
convention), L2f = the same over fluid cells only, Linf over fluid cells.
"""
import sys, glob, os
import numpy as np
import yt

yt.set_log_level(50)
A = float(os.environ.get("MMS_A", 0.5))


def exact(x, y, z):
    r2 = x * x + y * y + z * z
    phi = r2 * r2 + A * z * r2
    ni = 20.0 * r2 + 10.0 * A * z
    gx = 4 * r2 * x + 2 * A * z * x
    gy = 4 * r2 * y + 2 * A * z * y
    gz = 4 * r2 * z + A * (r2 + 2 * z * z)
    return phi, ni, -gx, -gy, -gz


def norms(err, mask):
    e = err * mask
    return (np.sqrt(np.mean(e**2)), np.sqrt(np.sum(e**2) / mask.sum()),
            np.abs(e).max())


for rundir in sys.argv[1:]:
    rows = []
    for sub in sorted((d for d in os.listdir(rundir) if d.isdigit()), key=int):
        plts = sorted(glob.glob(f"{rundir}/{sub}/plt?????"))
        if len(plts) < 2:
            continue
        ds = yt.load(plts[-1])
        ad = ds.all_data()
        x, y, z = (ad[c].to_value() for c in "xyz")
        mask = (ad["cellmask"].to_value() > 1 - 1e-10).astype(float)
        phi, ni, ex, ey, ez = exact(x, y, z)
        ephi = norms(ad["Potential"].to_value() - phi, mask)
        eni = norms(ad["NI"].to_value() - ni, mask)
        ee = np.sqrt((ad["Efieldx"].to_value() - ex)**2
                     + (ad["Efieldy"].to_value() - ey)**2
                     + (ad["Efieldz"].to_value() - ez)**2)
        eef = norms(ee, mask)
        rows.append((int(sub), float(ds.current_time), ephi, eni, eef))
    print(f"\n== {rundir}  (A = {A})")
    for name, idx in (("phi", 2), ("n_i", 3), ("|E|", 4)):
        print(f"  {name}:   N      L2(all)  rate    L2(fluid) rate    Linf     rate")
        prev = None
        for r in rows:
            e = r[idx]
            rt = ["  -  "] * 3 if prev is None else [
                f"{np.log2(prev[i] / e[i]):5.2f}" for i in range(3)]
            print(f"       {r[0]:4d}  {e[0]:.4e} {rt[0]}  {e[1]:.4e} {rt[1]}  {e[2]:.4e} {rt[2]}")
            prev = e
