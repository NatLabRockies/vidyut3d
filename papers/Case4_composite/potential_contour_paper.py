#!/usr/bin/env python3
"""Paper figure for Case 4: contours of the computed potential (ray value + per-face
wall, N_x = 256) with the masked solid in grey (continuous colour map, no contour lines). usage: python3 potential_contour_paper.py [plotfile]"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfread import Plotfile

plt.rc("text", usetex=True); plt.rc("font", size=16)
plt.rc("axes", labelsize=20); plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=16); plt.rc("ytick", labelsize=16)

path = sys.argv[1] if len(sys.argv) > 1 else "runs/ray_perface/256/plt00001"
pf = Plotfile(path)
q = pf.read(["Potential", "cellmask"])
phi, mask = q["Potential"], q["cellmask"]
x = pf.prob_lo[0] + (np.arange(pf.dims[0]) + 0.5) * pf.dx[0]
y = pf.prob_lo[1] + (np.arange(pf.dims[1]) + 0.5) * pf.dx[1]
X, Y = np.meshgrid(x, y, indexing="ij")
fluid = mask >= 1 - 1e-10
exact = (X**2 + Y**2)**2 / 32.0
phim = np.ma.masked_where(~fluid, phi)
levels = np.linspace(0, np.nanmax(exact[fluid]), 13)[1:-1]

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor("#8C8C8C")                      # masked solid
im = ax.pcolormesh(X, Y, phim, cmap="viridis", shading="auto",
                   vmin=0.0, vmax=np.nanmax(exact[fluid]), rasterized=True)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"$\phi$")
ax.set_aspect("equal"); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1])
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"gecmms_potential.{ext}", dpi=300, bbox_inches="tight")
err = np.abs(phi - exact)[fluid]
print(f"{path}: N={pf.dims[0]}, max|phi-exact| = {err.max():.3e}, rel = {err.max()/exact[fluid].max():.2e}")
