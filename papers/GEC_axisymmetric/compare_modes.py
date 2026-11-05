#!/usr/bin/env python3
"""Compare GEC axisymmetric runs (e.g. prob.wall_extrap = 0, 1, 2, or three grids).

usage: python3 compare_modes.py label1=rundir1 label2=rundir2 ...   [plt=pltNNNNN]
       Each rundir holds plotfiles; the latest one is used unless plt= is given.
       A rundir may also be the path of a single plotfile.
Writes electronden_modes.png (midplane n_e vs r with the experiment and the
Boeuf-Pitchford fluid result) and prints a summary table.
yt cannot read RZ plotfiles in this environment, so the plain reader pfread.py is used.
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfread import Plotfile

HERE = os.path.dirname(os.path.abspath(__file__))
# digitized reference curves, already in the repo (see README)
DATA = os.path.join(HERE, "..", "..", "test", "verification", "GEC_RF_Cell")
F_RF = 13.56e6
R1, ZCAP = 0.0538, 0.0127          # electrode radius, half gap
cmap = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23", "#662C91", "#A21D21"]
mark = ["o", "s", "d", "p", "h", "*"]

runs, pltname = [], None
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    if k == "plt": pltname = v
    else: runs.append((k, v))
if not runs:
    sys.exit(__doc__)

try:
    plt.rc("text", usetex=True)
except Exception:
    pass
plt.rc("font", size=16); plt.rc("axes", labelsize=20); plt.rc("legend", fontsize=14)
fig, ax = plt.subplots(figsize=(10, 6))
print(f"{'run':14s} {'plotfile':10s} {'cycles':>8s} {'ne peak':>10s} {'r_peak cm':>9s} {'ne axis':>10s} {'Te bulk eV':>10s} {'floor %':>8s} {'phi max V':>9s}")
for n, (label, d) in enumerate(runs):
    if os.path.isfile(os.path.join(d, "Header")):        # d is itself a plotfile
        path = d
    else:
        pfs = sorted(glob.glob(os.path.join(d, "plt?????")))
        path = os.path.join(d, pltname) if pltname else pfs[-1]
    pf = Plotfile(path)
    q = pf.read(["E", "Electron_Temp", "Potential", "cellmask"])
    ne, Te, phi, mask = q["E"], q["Electron_Temp"], q["Potential"], q["cellmask"]
    dr, dz = pf.dx[0], pf.dx[1]
    r = pf.prob_lo[0] + (np.arange(pf.dims[0]) + 0.5) * dr
    z = pf.prob_lo[1] + (np.arange(pf.dims[1]) + 0.5) * dz
    jm = np.argsort(np.abs(z))[:2]                      # two rows straddling the midplane
    ne_mid = ne[:, jm].mean(axis=1)
    ip = np.argmax(ne_mid)
    fluid = mask >= 1
    w = ne * fluid
    Te_bulk = (Te * w).sum() / w.sum() / 11604.5        # density-weighted, eV
    # how often would the linear wall value hit the floor (0.1*cell) under the powered electrode
    j = int(round((ZCAP - pf.prob_lo[1]) / dz)) - 1
    i = slice(0, int(R1 / dr) - 1)
    lin = 1.5 * ne[i, j] - 0.5 * ne[i, j - 1]
    floor_pct = 100.0 * np.mean(lin < 0.1 * ne[i, j])
    print(f"{label:14s} {os.path.basename(path):10s} {pf.time*F_RF:8.1f} {ne_mid[ip]:10.3e} {100*r[ip]:9.2f} {ne_mid[0]:10.3e} {Te_bulk:10.3f} {floor_pct:8.1f} {phi[fluid].max():9.1f}")
    ax.plot(100 * r, ne_mid / 1e15, lw=3, color=cmap[n % 6], label=label)

for fn, lab, kw in [("eden100mtorrOverzetHopkins_APL_63_1993", "Experiment", dict(ls="none", marker="o", ms=8, color="k")),
                    ("eden100mtorrBouefPitchford_PhysRevE_51_2_1995", "Fluid model (Boeuf and Pitchford)", dict(ls="--", lw=2, marker="s", ms=5, mfc="none", color="#B43894"))]:
    a = np.loadtxt(os.path.join(DATA, fn)); a = a[np.argsort(a[:, 0])]
    ax.plot(a[:, 0], a[:, 1], label=lab, **kw)
ax.set_xlabel(r"$r$ (cm)"); ax.set_ylabel(r"$n_e$ ($10^{15}$ m$^{-3}$)")
ax.set_xlim(0, 10.16); ax.set_ylim(0, 4); ax.legend()
fig.tight_layout(); fig.savefig("electronden_modes.png", dpi=300)
print("wrote electronden_modes.png  (floor % = share of powered-electrode faces where the linear wall n_e would be below 0.1 of the cell value)")
