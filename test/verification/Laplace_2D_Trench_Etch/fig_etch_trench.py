"""Trench figure: potential-gradient magnitude around the masked trench at several etch stages, with the
materials drawn from the same implicit function as Prob.H (TrenchIF) and the AMR grids outlined.
usage: python3 fig_etch_trench.py OUT.png PLOTFILE:DEPTH:TIME ...   (yt)
e.g.   runs/staged/plt00003:0:0 runs/staged/plt00007:1.7333333:31 runs/staged/plt00011:3.4666667:63
       runs/staged/plt00016:5.2:94"""
import sys, os
import numpy as np, yt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
yt.set_log_level(50)
out, runs = sys.argv[1], sys.argv[2:]
WT, HM, RB = 2.0, 1.0, 0.3            # as in inputs2d (units of 100 nm)
XL, YL = (-2.2, 2.2), (-6.2, 3.0)
EREF = 1.0 / (4.83 - 0.0)            # uniform field of the flat surface: 1 over the gap to the sheath edge
TIME = {0.0: 0, 1.733: 31, 3.467: 63, 5.2: 94}


def solid(x, y, d, wb):
    """Material map from the case's TrenchIF: 0 gas, 1 substrate (SiO2), 2 mask (Ru)."""
    ax = np.abs(x)
    f_sub = -y
    if d > 0:
        yc = np.clip(y, -d, 0)
        hw = 0.5 * wb + 0.5 * (WT - wb) * (yc + d) / d
        g = np.minimum(hw - ax, y + d)
        r = min(RB, 0.5 * wb); cx, cy = 0.5 * wb - r, -d + r
        corner = (ax > cx) & (y < cy)
        g = np.where(corner, r - np.hypot(ax - cx, y - cy), g)
        f_sub = np.minimum(f_sub, -g)
    f_mask = np.minimum(np.minimum(y, HM - y), ax - 0.5 * WT)
    m = np.zeros_like(x, dtype=int)
    m[f_sub > 0] = 1
    m[f_mask > 0] = 2
    return m


plt.rcParams.update({"font.size": 13, "font.family": "Arial"})
fig, axs = plt.subplots(1, len(runs), figsize=(3.1 * len(runs) + 1.2, 6.6), dpi=250, sharey=True)
xs = np.linspace(*XL, 900); ys = np.linspace(*YL, 1880)
X, Y = np.meshgrid(xs, ys)
mat_cmap = ListedColormap(["#D9C7A7", "#5E6A75"])   # SiO2, Ru mask
for ax, run in zip(axs, runs):
    path, d, tsec = run.split(":")
    d = float(d); wb = 2 - 0.84 * d / 5.2
    ds = yt.load(path); ds.force_periodicity()
    for g in sorted(ds.index.grids, key=lambda g: g.Level):
        gl, gh = g.LeftEdge.d[:2], g.RightEdge.d[:2]
        m = np.array(g[("boxlib", "cellmask")])[:, :, 0] > 1 - 1e-10
        ex = np.array(g[("boxlib", "Efieldx")])[:, :, 0]; ey = np.array(g[("boxlib", "Efieldy")])[:, :, 0]
        emag = np.log10(np.maximum(np.hypot(ex, ey) / EREF, 1e-6))
        im = ax.imshow(np.ma.masked_where(~m, emag).T, origin="lower", extent=[gl[0], gh[0], gl[1], gh[1]],
                       cmap="inferno", vmin=-3, vmax=0.3, interpolation="nearest")
        if g.Level > 0:
            ax.add_patch(Rectangle((gl[0], gl[1]), gh[0] - gl[0], gh[1] - gl[1], fill=False, lw=0.5,
                                   ec="white" if g.Level == 1 else "#ff7f50", zorder=4))
    mat = solid(X, Y, d, wb).astype(float)
    ax.imshow(np.ma.masked_where(mat == 0, mat), origin="lower", extent=[*XL, *YL], cmap=mat_cmap, vmin=1, vmax=2,
              interpolation="nearest", zorder=5)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
    ax.set_title(f"t = {tsec} s", fontsize=14, fontweight="bold", color="#003A69")
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_xticklabels(["-200", "-100", "0", "100", "200"])
    ax.set_xlabel("x (nm)")
axs[0].set_yticks([-6, -4, -2, 0, 2]); axs[0].set_yticklabels(["-600", "-400", "-200", "0", "200"])
axs[0].set_ylabel("y (nm)")
axs[0].text(-1.6, 0.5, "Ru", ha="center", va="center", color="white", fontsize=11, zorder=6)
axs[0].text(-1.6, -2.0, "SiO$_2$", ha="center", va="center", color="#3a3a3a", fontsize=11, zorder=6)
cb = fig.colorbar(im, ax=axs, shrink=0.75, pad=0.015, fraction=0.03); cb.set_label("log$_{10}$ |E| / E$_{flat}$")
fig.savefig(out, dpi=250, bbox_inches="tight", pad_inches=0.05)
print("wrote", out)
