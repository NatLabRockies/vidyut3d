#!/usr/bin/env python3
"""Radial profiles of phi, n_e, n_i and E_eps for MMS2 (paper style, markers only).
usage: python3 mms2_profiles.py   (reads ./<n>/plt00001 for n in FOLDERS)"""
import os
import numpy as np
import yt
import matplotlib.pyplot as plt

yt.set_log_level(50)
plt.rc("text", usetex=True)
plt.rc("font", size=16)
plt.rc("legend", fontsize=16)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
CMAP = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23",
        "#662C91", "#A21D21", "#B43894", "#010202", "#888888"]
MARK = ["s", "d", "o", "p", "h", "s", "*", "p", "v"]
ALPHA = 1.809512801e-08
N0 = 1e6
FOLDERS = ["32", "64", "128", "256", "512"]

FIELDS = [("Potential", lambda r: r**4 / 32.0, r"$\overline{\phi}(r)$", "potential_profile.png"),
          ("E", lambda r: r**2 / ALPHA + N0, r"$\overline{n}_e(r)$", "ne_profile.png"),
          ("HEp", lambda r: 0.5 * r**2 / ALPHA + N0, r"$\overline{n}_i(r)$", "ni_profile.png"),
          ("Electron_energy", lambda r: r**2 / ALPHA + N0, r"$\overline{E}_e(r)$", "een_profile.png")]

data = {}
for fname in FOLDERS:
    path = f"{fname}/plt00001"
    if not os.path.isdir(path):
        continue
    ad = yt.load(path).all_data()
    x = ad["x"].to_value(); y = ad["y"].to_value()
    rad = np.sqrt(x * x + y * y)
    m = ad["cellmask"].to_value(); m[m < 1 - 1e-10] = 0.0
    rb = np.linspace(0.5, 1.5, 41)
    rc = 0.5 * (rb[1:] + rb[:-1])
    inds = np.digitize(rad, rb)
    for name, _, _, _ in FIELDS:
        v = ad[name].to_value()
        prof = np.array([np.average(v[inds == b], weights=m[inds == b])
                         if np.any(inds == b) and np.sum(m[inds == b]) > 0 else np.nan
                         for b in range(1, len(rb))])
        data[(fname, name)] = (rc, prof)

for name, exact, ylabel, out in FIELDS:
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, fname in enumerate(FOLDERS):
        if (fname, name) not in data:
            continue
        rc, prof = data[(fname, name)]
        ax.plot(rc, prof, ls="none", marker=MARK[k + 1], ms=6, color=CMAP[k + 1], label=fname)
    r = np.linspace(0.5, 1.5, 200)
    ax.plot(r, exact(r), color="k", lw=2, label="Exact", zorder=0)
    ax.set_xlim(0.5, 1.5)
    ax.set_xlabel(r"$r$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print("wrote", out)
