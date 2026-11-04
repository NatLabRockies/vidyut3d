#!/usr/bin/env python3
"""Paper figure of Case 4, both panels with an IDENTICAL axes box so that the
domains have the same size when placed side by side:
  gecmms_geometry.{pdf,png}   geometry on the 32 grid with feature labels A-K
  gecmms_potential.{pdf,png}  computed potential, ray value + per-face wall, N_x = 256
usage: python3 case4_figure_paper.py [plotfile]   (default runs/ray_perface/256/plt00001)"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfread import Plotfile

plt.rc("text", usetex=True); plt.rc("font", size=16)
plt.rc("axes", labelsize=20); plt.rc("xtick", labelsize=16); plt.rc("ytick", labelsize=16)
GREY, FLUID = "#8C8C8C", "#DCE9F5"
BOX = [0.13, 0.12, 0.70, 0.70]           # identical axes box in both panels (fig fraction)
FIG = (7.5, 7.0)

def axes_common(ax):
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

# ---------------- (a) geometry
Ro, cox, coy = 2.6, 0.05, -0.03
R1, g, xc, sw, sy = 1.125, 0.5625, 0.04, 0.45, 1.3625
H = 2.0625
n = 1600
xs = np.linspace(-3, 3, n); X, Y = np.meshgrid(xs, xs, indexing="ij")
solid = (np.hypot(X - cox, Y - coy) > Ro) | (np.abs(Y) > H)
solid |= (np.hypot(X - xc, Y - g) < R1) & (Y > g)
solid |= (np.abs(X - xc) < sw) & (Y > sy)
solid |= (np.abs(X) < R1) & (Y < -g)
fig = plt.figure(figsize=FIG); ax = fig.add_axes(BOX)
ax.contourf(X, Y, solid.astype(float), levels=[-0.5, 0.5, 1.5], colors=[FLUID, GREY])
ax.contour(X, Y, solid.astype(float), levels=[0.5], colors="k", linewidths=1.2)
for v in np.arange(-3, 3.0001, 0.1875):
    ax.axhline(v, color="k", lw=0.15, alpha=0.35); ax.axvline(v, color="k", lw=0.15, alpha=0.35)
ye = g + np.sqrt(R1**2 - sw**2); xk = np.sqrt(Ro**2 - (H - coy)**2)
labels = {"A": ((-2.35, 0.55), (-2.52, 0.35)), "B": ((-0.35, 0.25), (-0.35, g)),
          "C": ((-1.55, 0.25), (xc - R1, g)), "D": ((1.45, 1.35), (xc + R1*np.cos(0.6), g + R1*np.sin(0.6))),
          "E": ((0.95, 1.85), (xc + sw, ye)), "F": ((-0.95, 1.80), (xc - sw, H)),
          "G": ((0.35, -0.25), (0.35, -g)), "H": ((1.55, -0.25), (R1, -g)),
          "I": ((1.60, -1.75), (R1, -H)), "K": ((2.30, 2.50), (cox + xk, H))}
for name, (txt, pt) in labels.items():
    ax.annotate(name, xy=pt, xytext=txt, fontsize=16, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.9, color="#B2182B"), color="#B2182B",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#B2182B", lw=0.9))
axes_common(ax)
for ext in ("png", "pdf"): fig.savefig(f"gecmms_geometry.{ext}", dpi=300)
plt.close(fig)

# ---------------- (b) potential
path = sys.argv[1] if len(sys.argv) > 1 else "runs/ray_perface/256/plt00001"
pf = Plotfile(path); q = pf.read(["Potential", "cellmask"])
phi, mask = q["Potential"], q["cellmask"]
x = pf.prob_lo[0] + (np.arange(pf.dims[0]) + 0.5) * pf.dx[0]
y = pf.prob_lo[1] + (np.arange(pf.dims[1]) + 0.5) * pf.dx[1]
Xp, Yp = np.meshgrid(x, y, indexing="ij")
fluid = mask >= 1 - 1e-10
exact = (Xp**2 + Yp**2)**2 / 32.0
fig = plt.figure(figsize=FIG); ax = fig.add_axes(BOX); ax.set_facecolor(GREY)
im = ax.pcolormesh(Xp, Yp, np.ma.masked_where(~fluid, phi), cmap="viridis", shading="auto",
                   vmin=0.0, vmax=np.nanmax(exact[fluid]), rasterized=True)
cax = fig.add_axes([BOX[0] + BOX[2] + 0.02, BOX[1], 0.03, BOX[3]])
cb = fig.colorbar(im, cax=cax); cb.set_label(r"$\phi$")
axes_common(ax)
for ext in ("png", "pdf"): fig.savefig(f"gecmms_potential.{ext}", dpi=300)
err = np.abs(phi - exact)[fluid]
print(f"{path}: max|phi-exact| = {err.max():.3e}, rel = {err.max()/exact[fluid].max():.2e}")
