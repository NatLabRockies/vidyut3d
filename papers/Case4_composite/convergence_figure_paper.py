#!/usr/bin/env python3
"""GEC-equivalent MMS convergence figure in the style of potential_error_plot.py
(same colours, markers, line widths and 1st/2nd-order reference lines).
usage: python3 convergence_figure_paper.py      (reads runs/<config>/<n>/plt00001)"""
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_error import load, norms

plt.rc("text", usetex=True)
plt.rc("font", size=16)
plt.rc("axes", titlesize=18, labelsize=20)
plt.rc("legend", fontsize=13)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)

cmap = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23",
        "#662C91", "#A21D21", "#B43894", "#010202"]
markertype = ["o", "s", "d", "p"]

configs = [("fixes_off", "Baseline"),
           ("fixes_on", "Safeguards"),
           ("gec3d_setting", "Isotropic coefficient"),
           ("ray_perface", "Ray value + per-face wall")]
panels = [("Potential", 1, r"$L_2(\phi)$"),
          ("E", 1, r"$L_2(n_e)$"),
          ("HEp", 1, r"$L_2(n_i)$"),
          ("Electron_energy", 1, r"$L_2(E_\epsilon)$")]   # norms() -> (L1, L2, Linf)

data = {}
for cfg, _ in configs:
    res = sorted(int(os.path.basename(p)) for p in glob.glob(f"runs/{cfg}/*")
                 if os.path.basename(p).isdigit() and os.path.isdir(f"{p}/plt00001"))
    rows = []
    for n in res:
        _, dx, X, Y, m, err = load(f"runs/{cfg}/{n}/plt00001")
        rows.append([n] + [norms(err[v], m)[k] for v, k, _ in panels])
    rows = [r for r in rows if np.all(np.isfinite(r)) and max(r[1:]) < 1.0]  # drop diverged runs
    data[cfg] = np.array(rows)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ip, (ax, (_, _, ylabel)) in enumerate(zip(axes.ravel(), panels)):
    for ic, (cfg, label) in enumerate(configs):
        d = data[cfg]
        ax.loglog(d[:, 0], d[:, 1 + ip], lw=3, marker=markertype[ic], ms=10,
                  color=cmap[ic], label=label)
    ref = data["ray_perface"]; idx = 1
    ax.loglog(ref[:, 0], ref[idx, 1 + ip] * (ref[idx, 0] / ref[:, 0]) ** 1,
              lw=3, color=cmap[-1], label="1st order")
    ax.loglog(ref[:, 0], ref[idx, 1 + ip] * (ref[idx, 0] / ref[:, 0]) ** 2,
              lw=3, ls="--", color=cmap[-2], label="2nd order")
    ax.set_xlabel(r"$N_x$")
    ax.set_ylabel(ylabel)
axes[0, 0].legend(loc="lower left")
plt.tight_layout()
plt.savefig("convergence_summary_paper.png", dpi=300, bbox_inches="tight")
print("wrote convergence_summary_paper.png")
