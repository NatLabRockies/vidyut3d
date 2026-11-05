"""Foot and normal error of the 3D reconstruction on the spherical shell,
for the fluid cells that have a masked face neighbour (the cells used by bc_ib).
usage: python recon_accuracy.py <rundir> ...   (uses <rundir>/<N>/plt00000)"""
import sys, os, numpy as np, yt
yt.set_log_level(50)
RI, RO = 0.5, 1.5
for rundir in sys.argv[1:]:
    print(f"== {rundir}:   N   foot/(h^2/R) mean   p99    max    normal/(h/R) mean  max   cells")
    for sub in sorted((d for d in os.listdir(rundir) if d.isdigit()), key=int):
        ds = yt.load(f"{rundir}/{sub}/plt00000")
        N = ds.domain_dimensions
        cg = ds.covering_grid(0, ds.domain_left_edge, N)
        g = lambda n: cg[n].to_value()
        h = float(ds.domain_width[0] / N[0])
        mask = g("cellmask") > 0.999
        nb = np.zeros_like(mask)
        for ax in range(3):
            for s in (1, -1):
                nb |= ~np.roll(mask, s, axis=ax)
        sel = mask & nb
        X = np.stack([g("x"), g("y"), g("z")])[:, sel]
        n = np.stack([g("EB_norm_x"), g("EB_norm_y"), g("EB_norm_z")])[:, sel]
        cp = np.stack([g("EB_cp_x"), g("EB_cp_y"), g("EB_cp_z")])[:, sel]
        r = np.linalg.norm(X, axis=0); R = np.where(r > 1.0, RO, RI)
        foot = X / r * R
        ns = np.where(r > 1.0, 1.0, -1.0) * X / r
        n = n / np.maximum(np.linalg.norm(n, axis=0), 1e-30)
        fe = np.linalg.norm(cp - foot, axis=0) / (h * h / R)
        ne = np.linalg.norm(n - ns, axis=0) / (h / R)
        print(f"   {int(sub):4d}   {fe.mean():8.3f} {np.percentile(fe,99):8.3f} {fe.max():8.3f}"
              f"      {ne.mean():8.3f} {ne.max():8.3f}   {sel.sum()}")
        if os.environ.get("WORST"):
            i = np.argsort(fe)[-5:]
            for j in i:
                print("      worst: x=", X[:, j].round(3), " r-R(h)=", round((r[j]-R[j])/h, 2), " err=", round(fe[j], 2))
