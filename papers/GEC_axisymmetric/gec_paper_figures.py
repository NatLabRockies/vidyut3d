#!/usr/bin/env python3
"""Paper figures of the axisymmetric GEC validation.

usage: python3 gec_paper_figures.py <transfer_dir> [outdir]
  <transfer_dir> holds the plotfiles axi2d (first-order wall fluxes), axi2d_2ndorder_mode1
  (linear extrapolation), axi2d_2ndorder_mode2 (logarithmic) and
  axi2d_2ndorder_mode2_longertime.
Writes gec_axi_ne_midplane.png and gec_axi_ne_convergence.png and prints the numbers
quoted in the paper.
Optional extra arguments "label=plotfile_name" (plotfiles in <transfer_dir>) are compared with the
second-order linear run in gec_axi_ne_transport.png, e.g.
  "Ion mobility of Boeuf and Pitchford=axi2d_transport2" "Transport of Boeuf and Pitchford=axi2d_transport1"."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
# digitized reference curves, already in the repo (see README)
DATA = os.path.join(HERE, "..", "..", "test", "verification", "GEC_RF_Cell")
sys.path.insert(0, HERE)
from pfread import Plotfile

args = [a for a in sys.argv[1:] if "=" not in a]
extra = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]   # "label=plotfile_name" (transport study)
T = args[0]
OUT = args[1] if len(args) > 1 else "."
F_RF = 13.56e6
plt.rc("text", usetex=True)
plt.rc("font", size=16); plt.rc("axes", titlesize=18, labelsize=20)
plt.rc("legend", fontsize=15); plt.rc("xtick", labelsize=16); plt.rc("ytick", labelsize=16)
cmap = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23", "#662C91", "#A21D21", "#B43894", "#010202"]


def load(name):
    pf = Plotfile(os.path.join(T, name))
    q = pf.read(["E", "cellmask"])
    ne, mask = q["E"], q["cellmask"] >= 1
    dr, dz = pf.dx[0], pf.dx[1]
    r = pf.prob_lo[0] + (np.arange(pf.dims[0]) + 0.5) * dr
    z = pf.prob_lo[1] + (np.arange(pf.dims[1]) + 0.5) * dz
    jm = np.argsort(np.abs(z))[:2]
    mid = ne[:, jm].mean(axis=1)
    total = (ne * mask * (2 * np.pi * r[:, None]) * dr * dz).sum()
    return dict(r=100 * r, mid=mid / 1e15, total=total, cyc=pf.time * F_RF, dims=pf.dims, dr=dr)


def experiment(ax):
    a = np.loadtxt(os.path.join(DATA, "eden100mtorrOverzetHopkins_APL_63_1993"))
    ax.plot(a[:, 0], a[:, 1], ls="none", marker="o", ms=6, color=cmap[-1], label="Experiment")
    b = np.loadtxt(os.path.join(DATA, "eden100mtorrBouefPitchford_PhysRevE_51_2_1995"))
    b = b[np.argsort(b[:, 0])]
    ax.plot(b[:, 0], b[:, 1], ls=":", lw=2, marker="s", ms=5, mfc="none", color=cmap[6],
            label="Fluid model (Boeuf and Pitchford)")


def finish(ax, fname):
    ax.set_xlabel(r"$r$ (cm)"); ax.set_ylabel(r"$n_e$ ($10^{15}~\mathrm{m^{-3}}$)")
    ax.set_xlim(0, 10.16); ax.set_ylim(0, 4)
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=300); plt.close()


runs = {"First-order wall fluxes": load("axi2d"),
        "Second-order, linear": load("axi2d_2ndorder_mode1"),
        "Second-order, logarithmic": load("axi2d_2ndorder_mode2")}
late = load("axi2d_2ndorder_mode2_longertime")

fig, ax = plt.subplots(figsize=(8, 6))
experiment(ax)
for k, ((lab, d), ls) in enumerate(zip(runs.items(), ["-", "--", "-."])):
    ax.plot(d["r"], d["mid"], lw=2.5, ls=ls, color=cmap[k], label=lab)
finish(ax, "gec_axi_ne_midplane.png")

fig, ax = plt.subplots(figsize=(8, 6))
experiment(ax)
d = runs["Second-order, logarithmic"]
ax.plot(d["r"], d["mid"], lw=2.5, color=cmap[2], label=f"{d['cyc']:.0f} RF cycles")
ax.plot(late["r"], late["mid"], lw=2.5, ls="--", color=cmap[0], label=f"{late['cyc']:.0f} RF cycles")
finish(ax, "gec_axi_ne_convergence.png")

print(f"grid {runs['First-order wall fluxes']['dims']}, dr = {1e3*runs['First-order wall fluxes']['dr']:.4f} mm")
ref = runs["First-order wall fluxes"]
for lab, d in list(runs.items()) + [("logarithmic, later", late)]:
    ip = np.argmax(d["mid"])
    print(f"{lab:28s} {d['cyc']:7.1f} cyc  peak {d['mid'][ip]:.3f} at r={d['r'][ip]:.2f} cm  axis {d['mid'][0]:.4f}  total {d['total']:.4e}"
          f"  | vs first order: peak {100*(d['mid'][ip]/ref['mid'].max()-1):+.1f}% axis {100*(d['mid'][0]/ref['mid'][0]-1):+.1f}% total {100*(d['total']/ref['total']-1):+.1f}%")
m2 = runs["Second-order, logarithmic"]
print(f"time convergence (logarithmic): peak {100*(late['mid'].max()/m2['mid'].max()-1):+.2f}%, axis {100*(late['mid'][0]/m2['mid'][0]-1):+.2f}%, "
      f"total {100*(late['total']/m2['total']-1):+.2f}%, max profile change {100*np.abs(late['mid']-m2['mid']).max()/m2['mid'].max():.2f}% of peak, over {late['cyc']-m2['cyc']:.1f} cycles")

if extra:
    fig, ax = plt.subplots(figsize=(8, 6))
    experiment(ax)
    base = runs["Second-order, linear"]
    ax.plot(base["r"], base["mid"], lw=2.5, color=cmap[1], label="Zero-field ion mobility")
    for k, (lab, name) in enumerate(extra):
        d = load(name); ip = np.argmax(d["mid"])
        ax.plot(d["r"], d["mid"], lw=2.5, ls=["--", "-."][k % 2], color=cmap[[0, 2][k % 2]], label=lab)
        print(f"{lab:40s} {d['cyc']:7.1f} cyc  peak {d['mid'][ip]:.3f} at r={d['r'][ip]:.2f} cm  axis {d['mid'][0]:.4f}  total {d['total']:.4e}"
              f"  | vs zero-field: peak {100*(d['mid'][ip]/base['mid'].max()-1):+.1f}% axis {100*(d['mid'][0]/base['mid'][0]-1):+.1f}%")
    finish(ax, "gec_axi_ne_transport.png")
