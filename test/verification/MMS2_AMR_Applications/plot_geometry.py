#!/usr/bin/env python3
"""Gas cells (light), solid (dark) and refined grids of the last plotfile of each run directory.
usage: plot_geometry.py out.png <run dir>..."""
import sys, os, glob
import numpy as np, yt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
yt.set_log_level(50)
LEV = {1: "#DB9728", 2: "#872C15"}
runs = sys.argv[2:]
fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 2.9), constrained_layout=True, squeeze=False)
for ax, run in zip(axs[0], runs):
    p = sorted(q for q in glob.glob(os.path.join(run, "plt*")) if os.path.isdir(q))[-1]
    ds = yt.load(p)
    for g in sorted(ds.index.grids, key=lambda g: g.Level):
        gl, gh = g.LeftEdge.d[:2], g.RightEdge.d[:2]
        m = np.array(g[("boxlib", "cellmask")])[:, :, 0] > 1 - 1e-10
        ax.imshow(m.T, origin="lower", extent=[gl[0], gh[0], gl[1], gh[1]], cmap="Greys_r", vmin=-0.6, vmax=1,
                  interpolation="nearest")
        if g.Level > 0:
            ax.add_patch(Rectangle(gl, *(gh - gl), fill=False, lw=0.6, ec=LEV[g.Level]))
    lo, hi = ds.domain_left_edge.d[:2], ds.domain_right_edge.d[:2]
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(os.path.normpath(run).split(os.sep)[-3], loc="left")
fig.savefig(sys.argv[1], dpi=140)
print("wrote", sys.argv[1])
