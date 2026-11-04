#!/usr/bin/env python3
"""Movie of the staged opening with errors: gas and AMR boxes (left); relative L2 error of potential, electrons,
ions and electron energy against the exact solution, drawn up to the current step (right).
Norms over valid gas cells of all levels, area weighted, relative to the maximum exact value there.
usage: make_movie_errors.py <run dir> <out.mp4> <plot_int> <switch steps...>"""
import sys, os, glob, subprocess, tempfile, shutil
import numpy as np, yt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
yt.set_log_level(50)
ECH, EPS0 = 1.602176634e-19, 8.8541878128e-12
A, N0 = ECH / EPS0, 1.0e6
S = 29.527559055118
run, out, pint = sys.argv[1], sys.argv[2], int(sys.argv[3])
switch = [int(x) for x in sys.argv[4:]]
FIELDS = [("Potential", lambda r2: r2**2 / 32, "Potential", "#5A5A5A", "-"),
          ("HEp", lambda r2: r2 / (2 * A) + N0, "Ions", "#872C15", "-"),
          ("E", lambda r2: r2 / A + N0, "Electrons", "#1A584D", "-"),
          ("Electron_energy", lambda r2: r2 / A + N0, "Electron energy", "#DB9728", "--")]
PLACES = ["Showerhead closed", "Opening 1: left", "Opening 2: centre", "Opening 3: right", "Opening 4: remaining slots"]
LEV = {1: "#DB9728", 2: "#872C15"}
plts = sorted(p for p in glob.glob(os.path.join(run, "plt*")) if os.path.isdir(p))
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17})
steps = []; hist = {f[0]: [] for f in FIELDS}; linf = {f[0]: [] for f in FIELDS}
for p in plts:
    ds = yt.load(p); steps.append(int(os.path.basename(p)[3:]) * pint)
    for f, fn, *_ in FIELDS:
        s = w = mx = lin = 0.0
        for g in ds.index.grids:
            v = (np.array(g[("boxlib", "cellmask")])[:, :, 0] > 1 - 1e-10) & np.array(g.child_mask)[:, :, 0].astype(bool)
            if not v.any():
                continue
            X = np.array(g[("index", "x")])[:, :, 0]; Y = np.array(g[("index", "y")])[:, :, 0]
            e = fn(X**2 + Y**2); d = np.abs(np.array(g[("boxlib", f)])[:, :, 0] - e)[v]
            a = np.prod(g.dds.d[:2]); s += a * (d**2).sum(); w += a * v.sum()
            lin = max(lin, d.max()); mx = max(mx, e[v].max())
        hist[f].append(np.sqrt(s / w) / mx); linf[f].append(lin / mx)
steps = np.array(steps)
np.savetxt(os.path.splitext(out)[0] + "_history.txt",
           np.column_stack([steps] + [hist[f[0]] for f in FIELDS] + [linf[f[0]] for f in FIELDS]),
           header="step L2: phi ni ne energy; Linf: phi ni ne energy")
tmp = tempfile.mkdtemp()
for n, p in enumerate(plts):
    ds = yt.load(p); step = steps[n]; stage = sum(step > s for s in switch)
    lo, hi = ds.domain_left_edge.d[:2], ds.domain_right_edge.d[:2]
    fig, ax = plt.subplots(1, 2, figsize=(14.4, 4.3), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1]})
    for g in sorted(ds.index.grids, key=lambda g: g.Level):
        gl, gh = g.LeftEdge.d[:2] / S * 1e3, g.RightEdge.d[:2] / S * 1e3
        m = np.array(g[("boxlib", "cellmask")])[:, :, 0] > 1 - 1e-10
        ax[0].imshow(m.T, origin="lower", extent=[gl[0], gh[0], gl[1], gh[1]], cmap="Greys_r", vmin=-0.6, vmax=1,
                     interpolation="nearest")
        if g.Level > 0:
            ax[0].add_patch(Rectangle((gl[0], gl[1]), gh[0] - gl[0], gh[1] - gl[1], fill=False, lw=1.0, ec=LEV[g.Level]))
    ax[0].set_xlim(lo[0] / S * 1e3, hi[0] / S * 1e3); ax[0].set_ylim(lo[1] / S * 1e3, hi[1] / S * 1e3)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)")
    ax[0].set_title(f"{PLACES[stage]}  |  step {step}", color="#003A69", fontweight="bold", loc="left")
    b = ax[1]
    for s in switch:
        b.axvline(s, color="#BBBBBB", lw=1.2, zorder=0)
    upto = (steps <= step) & (steps > 0)
    for f, _, lab, col, ls in FIELDS:
        b.semilogy(steps[upto], np.array(hist[f])[upto], color=col, ls=ls, lw=2.4, label=lab)
    b.set_xlim(0, steps[-1]); b.set_ylim(3e-6, 1e-4); b.set_xlabel("Step")
    b.set_ylabel("Relative L2 error"); b.set_title("Error against exact solution (lines: openings)", loc="left")
    b.grid(axis="y", color="#E6E6E6", lw=0.8)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.legend(loc="upper right", ncol=2, frameon=False, fontsize=14, labelcolor="#222222")
    fig.savefig(os.path.join(tmp, f"f{n:04d}.png"), dpi=130); plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", os.path.join(tmp, "f%04d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", out], check=True)
shutil.copy(os.path.join(tmp, f"f{len(plts) - 1:04d}.png"), os.path.splitext(out)[0] + "_last.png")
print("wrote", out)
for f, _, lab, *_ in FIELDS:
    a = np.array(hist[f])[1:]
    print(f"{lab}: L2 min {a.min():.3e} max {a.max():.3e} last {a[-1]:.3e}")
