# Convergence plot for the Laplace annulus: the inner-cylinder Dirichlet,
# Neumann and Robin sets, plus the AMR hierarchies that reach an effective
# N_x = 512 from four different base grids.
#
# The uniform curves use the same all-cell mean as the paper table. The AMR
# points cannot use ds.all_data(), which would count coarse cells that a finer
# level covers, so they go through amr_error.analyse(norm="domain"), which
# masks each level to its valid region and normalises by the domain volume --
# the same quantity the uniform curves report.
#
# Usage: python3 error_plot_bc.py     (reads dirichlet/, neumann/, robin/)
import os
import sys

import yt
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amr_error import analyse  # noqa: E402

yt.set_log_level(50)
plt.rc("text", usetex=True)
plt.rc("font", size=16)
plt.rc("axes", titlesize=18, labelsize=20)
plt.rc("legend", fontsize=13)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)

cmap = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23",
        "#662C91", "#A21D21", "#B43894", "#010202"]
markertype = ["o", "s", "d"]

rmin, rmax, phi1, phi2 = 0.1, 0.2, 10.0, 20.0
cases = [("dirichlet", "Dirichlet"), ("neumann", "Neumann"), ("robin", "Robin")]
folders = ["32", "64", "128", "256", "512"]

# base grid, number of levels, run directory; all reach an effective 512
amr_runs = [
    (32, 4, "dirichlet/amr32L4_buf8"),
    (64, 3, "dirichlet/amr64L3_buf8"),
    (128, 2, "dirichlet/amr128L2_buf8"),
    (256, 1, "dirichlet/amr256L1_buf8"),
]
amr_markers = ["*", "P", "X", "D"]
amr_sizes = [26, 19, 14, 9]
amr_colors = ["#662C91", "#B43894", "#F47D23", "#008C48"]


def errors(fdir):
    out = []
    for fname in folders:
        path = f"{fdir}/{fname}/plt00001"
        if not os.path.isdir(path):
            continue
        ad = yt.load(path).all_data()
        x = ad["x"].to_value()
        y = ad["y"].to_value()
        mask = (ad["cellmask"].to_value() >= 1.0).astype(float)
        rad = np.sqrt(x * x + y * y)
        lr = np.log(rmax / rmin)
        exact = (phi2 * np.log(rad / rmin) + phi1 * np.log(rmax / rad)) / lr
        V = ad["Potential"].to_value() * mask
        Ex = ad["Efieldx"].to_value() * mask
        Ey = ad["Efieldy"].to_value() * mask
        Er_exact = (phi2 - phi1) / (rad * lr) * mask
        out.append((int(fname),
                    np.sqrt(np.mean((V - exact * mask) ** 2)),
                    np.sqrt(np.mean((np.sqrt(Ex**2 + Ey**2) - Er_exact) ** 2))))
    return np.array(out)


data = {c: errors(c) for c, _ in cases}

for col, ylabel, fname in [(1, r"$L_2(\phi)$", "error_all_bc")]:
    plt.figure(figsize=(10, 6))
    for i, (c, label) in enumerate(cases):
        d = data[c]
        plt.loglog(d[:, 0], d[:, col], lw=3, marker=markertype[i], ms=10,
                   color=cmap[i], label=label)
    ref = data[cases[0][0]]
    idx = 1
    plt.loglog(ref[:, 0], ref[idx, col] * (ref[idx, 0] / ref[:, 0]) ** 1,
               lw=3, color=cmap[-1], label="1st order")
    plt.loglog(ref[:, 0], ref[idx, col] * (ref[idx, 0] / ref[:, 0]) ** 2,
               lw=3, ls="--", color=cmap[-2], label="2nd order")

    # AMR hierarchies, drawn at their finest resolution. They coincide, so the
    # markers are nested from large to small to keep all four visible.
    for (base, nlev, d), mk, ms, cc in zip(amr_runs, amr_markers, amr_sizes,
                                           amr_colors):
        if not os.path.isdir(d):
            print(f"skipping missing {d}")
            continue
        r = analyse(d, norm="domain")
        plt.loglog(r["eff"], r["phi_L2"] if col == 1 else r["E_L2"],
                   marker=mk, ms=ms, mfc="none", mew=2.2, ls="none",
                   color=cc, label=rf"AMR ${base}\!+\!{nlev}$ levels")

    plt.xlabel(r"$N_x$")
    plt.ylabel(ylabel)
    plt.legend(ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{fname}.{ext}", dpi=600, bbox_inches="tight")
    plt.close()
    print("wrote", fname)
