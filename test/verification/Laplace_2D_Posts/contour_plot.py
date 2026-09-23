#!/usr/bin/env python3
"""Figure for Laplace_2D_Posts.

(a) the computed potential, phi = x^2 - y^2, with the posts drawn on top.
(b) |phi - phi_exact| on the uniform fine grid. phi is harmonic with vanishing
    fourth derivatives, so the interior stencil reproduces it exactly and the
    posts are the ONLY source of error in the case. The elliptic solve then
    spreads that error over the domain, so the map is smooth and wide even
    though every bit of it is made at the boundary. The dark curves are sign
    changes, not accurate regions.
(c) the same error, zoomed on the posts.
(d) the refined run at the same finest spacing, same zoom, with the level
    patches drawn. Matching (c) is the result: refinement reproduces the fine
    grid where the geometry is, using far fewer cells.

Usage: python3 contour_plot.py [uni_dir] [amr_dir] [out.png]
"""
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

yt.set_log_level(50)

POST_R, PITCH, NPOST = 0.22, 0.50, 3
SHIFT = (0.031, -0.017)
ZOOM = 1.0


def posts():
    c0 = -0.5 * (NPOST - 1) * PITCH
    return [
        (c0 + i * PITCH + SHIFT[0], c0 + j * PITCH + SHIFT[1])
        for i in range(NPOST)
        for j in range(NPOST)
    ]


def draw_posts(ax):
    for cx, cy in posts():
        ax.add_patch(Circle((cx, cy), POST_R, fc="0.4", ec="k", lw=0.7, zorder=6))


def sample(d, box=None, maxcells=900):
    """Finest-available data on a uniform sampling, as (X, Y, phi, err, ds)."""
    ds = yt.load(sorted(glob.glob(f"{d}/plt*"))[-1])
    ds.force_periodicity()  # smoothed_covering_grid needs ghosts at the box edge
    lev = ds.index.max_level
    dlo, dhi = ds.domain_left_edge.d[:2], ds.domain_right_edge.d[:2]
    dx = (dhi[0] - dlo[0]) / (ds.domain_dimensions[0] * 2**lev)
    lo = np.array([-box, -box]) if box else dlo.copy()
    hi = np.array([box, box]) if box else dhi.copy()
    nx = int(round((hi[0] - lo[0]) / dx))
    if nx > maxcells:  # cap the full-domain panels
        lev = max(0, lev - int(np.ceil(np.log2(nx / maxcells))))
        dx = (dhi[0] - dlo[0]) / (ds.domain_dimensions[0] * 2**lev)
        nx = int(round((hi[0] - lo[0]) / dx))
    le = np.array([lo[0], lo[1], 0.0])
    cg = (ds.smoothed_covering_grid(lev, le, [nx, nx, 1]) if lev > 0
          else ds.covering_grid(0, le, [nx, nx, 1]))
    phi = np.array(cg["boxlib", "Potential"])[:, :, 0]
    cm = np.array(cg["boxlib", "cellmask"])[:, :, 0]
    X, Y = np.meshgrid(
        lo[0] + (np.arange(nx) + 0.5) * dx,
        lo[1] + (np.arange(nx) + 0.5) * dx,
        indexing="ij",
    )
    err = np.abs(phi - (X * X - Y * Y))
    err[cm <= 1 - 1e-10] = np.nan
    phi = np.where(cm > 1 - 1e-10, phi, np.nan)
    return X, Y, phi, err, ds


def levels_on(ax, ds):
    cols = {1: "cyan", 2: "yellow", 3: "lime"}
    for g in ds.index.grids:
        if g.Level == 0:
            continue
        le, re = g.LeftEdge.d, g.RightEdge.d
        ax.add_patch(
            Rectangle(
                (le[0], le[1]), re[0] - le[0], re[1] - le[1],
                fill=False, ec=cols.get(g.Level, "w"), lw=0.8, zorder=5,
            )
        )


def main():
    uni = sys.argv[1] if len(sys.argv) > 1 else "uni512"
    amr = sys.argv[2] if len(sys.argv) > 2 else "amrL3"
    out = sys.argv[3] if len(sys.argv) > 3 else "laplace_2d_posts.png"

    X, Y, phi, err, dsu = sample(uni)
    Xz, Yz, _, errz, _ = sample(uni, ZOOM)
    Xa, Ya, _, erra, dsa = sample(amr, ZOOM)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10.2), constrained_layout=True)

    a = ax[0, 0]
    cf = a.contourf(X, Y, phi, levels=30, cmap="viridis")
    a.contour(X, Y, phi, levels=14, colors="w", linewidths=0.4, alpha=0.6)
    fig.colorbar(cf, ax=a, shrink=0.85, label=r"$\phi$")
    draw_posts(a)
    a.set_title(r"(a) potential, $\phi_{\rm exact}=x^2-y^2$ (harmonic)")

    vmax = np.nanmax(err)
    vmin = vmax / 3.0e3  # show structure instead of saturating
    a = ax[0, 1]
    m = a.pcolormesh(X, Y, err, cmap="magma", norm=LogNorm(vmin, vmax), shading="auto")
    fig.colorbar(m, ax=a, shrink=0.85, label=r"$|\phi-\phi_{\rm exact}|$")
    draw_posts(a)
    a.add_patch(Rectangle((-ZOOM, -ZOOM), 2 * ZOOM, 2 * ZOOM,
                          fill=False, ec="w", lw=1.0, ls="--", zorder=7))
    a.set_title(
        f"(b) error, uniform {uni[3:]}: the posts are the only source,\n"
        "spread by the elliptic solve (dark curves are sign changes)"
    )

    for a, E, Xc, Yc, ttl, ds in (
        (ax[1, 0], errz, Xz, Yz, f"(c) error zoom, uniform {uni[3:]}", None),
        (ax[1, 1], erra, Xa, Ya, f"(d) error zoom, refined {amr}", dsa),
    ):
        m = a.pcolormesh(Xc, Yc, E, cmap="magma", norm=LogNorm(vmin, vmax),
                         shading="auto")
        fig.colorbar(m, ax=a, shrink=0.85, label=r"$|\phi-\phi_{\rm exact}|$")
        if ds is not None:
            levels_on(a, ds)
        draw_posts(a)
        a.set_title(ttl + ("\nlevel patches drawn" if ds is not None else
                           "\ngap = 0.06, unresolved on the base grid"))
        a.set_xlim(-ZOOM, ZOOM)
        a.set_ylim(-ZOOM, ZOOM)

    for a in ax.ravel():
        a.set_aspect("equal")
        a.set_xlabel("x")
    ax[0, 0].set_ylabel("y")
    ax[1, 0].set_ylabel("y")
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
