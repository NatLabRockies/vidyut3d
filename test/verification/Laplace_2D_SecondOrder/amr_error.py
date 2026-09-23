#!/usr/bin/env python3
"""Composite error norms for the annulus Laplace case, AMR-aware.

The single-level script (potential_error_plot.py) uses ds.all_data(), which
gathers cells from every level including coarse cells that a finer level
covers, so an AMR run is counted more than once. Here each level contributes
only its valid region, and the L2 norm is volume weighted so levels of
different dx combine correctly.

Only fluid cells count: cellmask > 1 - 1e-10. A looser threshold lets in the
cut cells that the solver masks out and gives a bogus O(1) max error.

Errors are also split by where they sit, which is what shows whether a
coarse-fine interface degrades the order:
  ib    cells with a cut cell in their 3x3 neighbourhood
  cf    cells within --cf-width of a coarse/fine interface, either side
  int   everything else

Usage:
  python3 amr_error.py -f uni64 uni128 uni256 amr128L2 --plot conv.png
"""
import argparse
import glob
import os

import numpy as np
import yt

yt.set_log_level(50)

RMIN, RMAX, PHI1, PHI2 = 0.1, 0.2, 10.0, 20.0
LR = np.log(RMAX / RMIN)


def exact_phi(r):
    return (PHI2 * np.log(r / RMIN) + PHI1 * np.log(RMAX / r)) / LR


def exact_efield(x, y):
    """E = -grad(phi); the exact field is radial with magnitude (p2-p1)/(r lr)."""
    r2 = x * x + y * y
    return (-(PHI2 - PHI1) * x / (r2 * LR), -(PHI2 - PHI1) * y / (r2 * LR))


def dilate(mask, width):
    """True wherever mask is True within `width` cells (Chebyshev)."""
    out = mask.copy()
    for _ in range(width):
        nxt = out.copy()
        for ax in (0, 1):
            for s in (-1, 1):
                nxt |= np.roll(out, s, axis=ax)
        out = nxt
    return out


def level_arrays(ds, lev, fields):
    """Assemble whole-domain arrays for one level, plus a 'present' mask."""
    nx, ny = (ds.domain_dimensions[:2] * 2 ** lev).astype(int)
    present = np.zeros((nx, ny), dtype=bool)
    data = {f: np.zeros((nx, ny)) for f in fields}
    for g in ds.index.grids:
        if g.Level != lev:
            continue
        i0, j0 = g.get_global_startindex()[:2]
        gx, gy = int(g.ActiveDimensions[0]), int(g.ActiveDimensions[1])
        sl = (slice(i0, i0 + gx), slice(j0, j0 + gy))
        present[sl] = True
        for f in fields:
            data[f][sl] = np.asarray(g["boxlib", f])[:, :, 0]
    return present, data


def analyse(fdir, plotfile=None, cf_width=2):
    if plotfile is None:
        cands = sorted(glob.glob(os.path.join(fdir, "plt?????")))
        if not cands:
            raise FileNotFoundError(f"no plotfiles in {fdir}")
        plotfile = cands[-1]
    ds = yt.load(plotfile)
    maxlev = ds.index.max_level
    fields = ["Potential", "Efieldx", "Efieldy", "cellmask"]

    lo = ds.domain_left_edge.to_value()[:2]
    dx0 = (ds.domain_right_edge.to_value()[:2] - lo) / ds.domain_dimensions[:2]

    levels = [level_arrays(ds, l, fields) for l in range(maxlev + 1)]

    acc = {k: {"e2": 0.0, "vol": 0.0, "linf": 0.0, "n": 0}
           for k in ("all", "ib", "cf", "int")}
    ncell_total = 0

    for lev in range(maxlev + 1):
        present, data = levels[lev]
        dx = dx0 / 2 ** lev
        nx, ny = present.shape

        # covered: a finer level exists over this cell
        covered = np.zeros_like(present)
        if lev < maxlev:
            fine = levels[lev + 1][0]
            covered = fine.reshape(nx, 2, ny, 2).any(axis=(1, 3))

        valid = present & ~covered
        m = data["cellmask"]
        fluid = valid & (m > 1.0 - 1e-10)
        cut = present & (m > 1e-12) & (m < 1.0 - 1e-10)
        ncell_total += int(present.sum())
        if not fluid.any():
            continue

        i = np.arange(nx)[:, None] + 0.5
        j = np.arange(ny)[None, :] + 0.5
        x = lo[0] + i * dx[0] + 0.0 * j
        y = lo[1] + 0.0 * i + j * dx[1]
        r = np.sqrt(x * x + y * y)

        err_phi = np.abs(data["Potential"] - exact_phi(np.where(r > 0, r, 1.0)))
        eex, eey = exact_efield(np.where(x != 0, x, 1e-30), y)
        err_e = np.abs(np.hypot(data["Efieldx"], data["Efieldy"]) - np.hypot(eex, eey))

        # where is each cell relative to the boundary and to a c/f interface?
        near_ib = dilate(cut, 1)
        near_cf = dilate(covered, cf_width)          # coarse side of the interface
        if lev > 0:
            near_cf |= dilate(~present, cf_width)    # fine side, edge of this level

        zones = {
            "all": fluid,
            "ib": fluid & near_ib,
            "cf": fluid & near_cf & ~near_ib,
            "int": fluid & ~near_cf & ~near_ib,
        }
        cellvol = dx[0] * dx[1]
        for key, sel in zones.items():
            if not sel.any():
                continue
            e = err_phi[sel]
            acc[key]["e2"] += float((e ** 2).sum()) * cellvol
            acc[key]["vol"] += float(sel.sum()) * cellvol
            acc[key]["linf"] = max(acc[key]["linf"], float(e.max()))
            acc[key]["n"] += int(sel.sum())
            ee = err_e[sel]
            acc[key]["e2_E"] = acc[key].get("e2_E", 0.0) + float((ee ** 2).sum()) * cellvol
            acc[key]["linf_E"] = max(acc[key].get("linf_E", 0.0), float(ee.max()))

    out = {
        "dir": fdir,
        "plotfile": os.path.basename(plotfile),
        "base": int(ds.domain_dimensions[0]),
        "maxlev": maxlev,
        "eff": int(ds.domain_dimensions[0] * 2 ** maxlev),
        "ncell": ncell_total,
        "phi_L2": np.sqrt(acc["all"]["e2"] / acc["all"]["vol"]),
        "phi_Linf": acc["all"]["linf"],
        "E_L2": np.sqrt(acc["all"]["e2_E"] / acc["all"]["vol"]),
        "E_Linf": acc["all"]["linf_E"],
    }
    for k in ("ib", "cf", "int"):
        ok = acc[k]["vol"] > 0
        out[f"{k}_L2"] = np.sqrt(acc[k]["e2"] / acc[k]["vol"]) if ok else np.nan
        out[f"{k}_Linf"] = acc[k]["linf"] if acc[k]["n"] else np.nan
        out[f"{k}_E_L2"] = np.sqrt(acc[k]["e2_E"] / acc[k]["vol"]) if ok else np.nan
        out[f"{k}_n"] = acc[k]["n"]
    return out


def rate(e_coarse, e_fine):
    if not (np.isfinite(e_coarse) and np.isfinite(e_fine)) or e_fine <= 0:
        return np.nan
    return np.log2(e_coarse / e_fine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fdirs", required=True, nargs="+")
    ap.add_argument("--cf-width", type=int, default=2)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    rows = [analyse(d, cf_width=args.cf_width) for d in args.fdirs]
    rows.sort(key=lambda r: (r["maxlev"], r["eff"]))

    hdr = (f"{'case':<12} {'base':>5} {'lev':>4} {'eff':>5} {'cells':>9} "
           f"{'phi L2':>10} {'rate':>5} {'phi Linf':>10} {'rate':>5} "
           f"{'|E| L2':>10} {'rate':>5}")
    print(hdr)
    print("-" * len(hdr))
    # a family is one refinement path: same number of levels, base doubling
    prev_of = {}
    for r in rows:
        fam = r["maxlev"]
        prev = prev_of.get(fam)
        rl2 = rate(prev["phi_L2"], r["phi_L2"]) if prev else np.nan
        rli = rate(prev["phi_Linf"], r["phi_Linf"]) if prev else np.nan
        rel = rate(prev["E_L2"], r["E_L2"]) if prev else np.nan
        print(f"{os.path.basename(r['dir']):<12} {r['base']:>5} {r['maxlev']:>4} "
              f"{r['eff']:>5} {r['ncell']:>9} "
              f"{r['phi_L2']:>10.3e} {rl2:>5.2f} {r['phi_Linf']:>10.3e} {rli:>5.2f} "
              f"{r['E_L2']:>10.3e} {rel:>5.2f}")
        prev_of[fam] = r

    print()
    hdr2 = (f"{'case':<12} {'IB L2':>10} {'rate':>5} {'CF L2':>10} {'rate':>5} "
            f"{'IB |E|':>10} {'CF |E|':>10} {'IB Linf':>10} {'CF Linf':>10} {'CFcells':>8}")
    print(hdr2)
    print("-" * len(hdr2))
    prev_of = {}
    for r in rows:
        prev = prev_of.get(r["maxlev"])
        rib = rate(prev["ib_L2"], r["ib_L2"]) if prev else np.nan
        rcf = rate(prev["cf_L2"], r["cf_L2"]) if prev else np.nan
        print(f"{os.path.basename(r['dir']):<12} {r['ib_L2']:>10.3e} {rib:>5.2f} "
              f"{r['cf_L2']:>10.3e} {rcf:>5.2f} "
              f"{r['ib_E_L2']:>10.3e} {r['cf_E_L2']:>10.3e} "
              f"{r['ib_Linf']:>10.3e} {r['cf_Linf']:>10.3e} {r['cf_n']:>8}")
        prev_of[r["maxlev"]] = r

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        uni = [r for r in rows if r["maxlev"] == 0]
        amr = [r for r in rows if r["maxlev"] > 0]
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
        for k, (key, name) in enumerate([("phi_L2", r"$\phi$, $L_2$"),
                                         ("phi_Linf", r"$\phi$, $L_\infty$")]):
            h = 0.6 / np.array([r["eff"] for r in uni])
            e = np.array([r[key] for r in uni])
            ax[k].loglog(h, e, "o-", color="#185AA9", label="uniform")
            for r in amr:
                ax[k].loglog(0.6 / r["eff"], r[key], "s", ms=11, mfc="none",
                             mew=2, color="#EE2E2F",
                             label=f"AMR {r['base']}+{r['maxlev']}")
            ax[k].loglog(h, e[0] * (h / h[0]) ** 2, "k--", lw=1,
                         label="2nd order")
            ax[k].set_xlabel(r"$\Delta x$ at the boundary [m]")
            ax[k].set_ylabel(name)
            ax[k].legend(fontsize=9)
            ax[k].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
