#!/usr/bin/env python3
"""
Usage:
    python check_order.py -f /path/to/rundir
where rundir contains subdirectories 32/ and 64/ each with plt00001.
"""

import yt
import numpy as np
import argparse

def exact_phi(x, y, rmin, rmax, phi1, phi2):
    r = np.sqrt(x**2 + y**2)
    return (phi2 * np.log(r/rmin) + phi1 * np.log(rmax/r)) / np.log(rmax/rmin)

def exact_dphidr(r, rmin, rmax, phi1, phi2):
    return (phi2 - phi1) / (r * np.log(rmax/rmin))

def load(fdir, res):
    ds  = yt.load(f"{fdir}/{res}/plt00001")
    ad  = ds.all_data()
    x   = ad["x"].to_value()
    y   = ad["y"].to_value()
    cm  = ad["cellmask"].to_value()
    cm[cm < 1 - 1e-10] = 0.0
    phi = ad["Potential"].to_value() * cm
    ex  = ad["Efieldx"].to_value()   * cm
    ey  = ad["Efieldy"].to_value()   * cm
    return x, y, cm, phi, ex, ey, ds.domain_dimensions[0]

def l2(a, b, mask):
    diff = (a - b) * mask
    return np.sqrt(np.mean(diff**2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fdir", required=True)
    args = parser.parse_args()

    rmin, rmax = 0.1, 0.2
    phi1, phi2 = 10.0, 20.0
    R_mid = 0.5*(rmin + rmax)

    # ----------------------------------------------------------------
    # 1. Solution errors and convergence order
    # ----------------------------------------------------------------
    results = {}
    for res in [32, 64]:
        x, y, cm, phi, ex, ey, nx = load(args.fdir, res)
        r = np.sqrt(x**2 + y**2)

        phi_ex = exact_phi(x, y, rmin, rmax, phi1, phi2)
        dphidr = exact_dphidr(r, rmin, rmax, phi1, phi2)
        ex_ex  = -dphidr * x / r
        ey_ex  = -dphidr * y / r

        err_phi = l2(phi, phi_ex, cm)
        err_ex  = l2(ex,  ex_ex,  cm)
        err_ey  = l2(ey,  ey_ex,  cm)

        results[res] = dict(nx=nx,
                            err_phi=err_phi,
                            err_ex=err_ex,
                            err_ey=err_ey)

        print(f"\n=== Resolution {res} ===")
        print(f"  L2(phi)  = {err_phi:.4e}")
        print(f"  L2(Ex)   = {err_ex:.4e}")
        print(f"  L2(Ey)   = {err_ey:.4e}")

    print("\n=== Observed convergence order (32 -> 64) ===")
    for key, label in [("err_phi","phi"), ("err_ex","Ex"), ("err_ey","Ey")]:
        e32 = results[32][key]
        e64 = results[64][key]
        if e64 > 0 and e32 > 0:
            order = np.log2(e32 / e64)
            tag   = "OK ~2" if order > 1.7 else ("FAIL ~1" if order < 1.3 else "BORDERLINE")
            print(f"  order({label}) = {order:.3f}  ({tag})")

    # ----------------------------------------------------------------
    # 2. Centroid normal-error diagnostic
    # ----------------------------------------------------------------
    print("\n=== Centroid normal-error diagnostic ===")
    for res in [32, 64]:
        ds  = yt.load(f"{args.fdir}/{res}/plt00001")
        ad  = ds.all_data()
        x   = ad["x"].to_value()
        y   = ad["y"].to_value()
        cm  = ad["cellmask"].to_value()

        cpx = ad["EB_cp_x"].to_value()
        cpy = ad["EB_cp_y"].to_value()
        nx_ = ad["EB_norm_x"].to_value()
        ny_ = ad["EB_norm_y"].to_value()

        # Cut cells only: 0 < volfrac < 1
        cut = (cm > 1e-10) & (cm < 1 - 1e-10)
        print(f"\n  res={res}  n_cut_cells={cut.sum()}")
        if cut.sum() == 0:
            print("  WARNING: no cut cells found — check cellmask field")
            continue

        xc  = cpx[cut];  yc  = cpy[cut]
        nxc = nx_[cut];  nyc = ny_[cut]
        nmag = np.sqrt(nxc**2 + nyc**2) + 1e-30
        nhx = nxc / nmag
        nhy = nyc / nmag

        # Which surface does each cut cell belong to?
        r_xb = np.sqrt(xc**2 + yc**2)
        R    = np.where(r_xb < R_mid, rmin, rmax)

        # Exact boundary point via radial projection
        xb_ex = R * xc / r_xb
        yb_ex = R * yc / r_xb

        # Normal-direction error in raw centroid
        err_n = (xc - xb_ex)*nhx + (yc - yb_ex)*nhy

        dx_val = (ds.domain_right_edge[0].v - ds.domain_left_edge[0].v) \
                 / ds.domain_dimensions[0]

        print(f"  dx              = {dx_val:.4e}")
        print(f"  |err_n| max     = {np.max(np.abs(err_n)):.4e}")
        print(f"  |err_n| mean    = {np.mean(np.abs(err_n)):.4e}")
        print(f"  |err_n|/dx mean = {np.mean(np.abs(err_n))/dx_val:.4e}  "
              f"<-- should halve 32->64 if bisection is working")

        # Check bracket sign oracle directly
        # Evaluate surfaceval equivalent at +/- diag_dx along nhat
        diag_dx = 2.0 * np.sqrt(2.0) * dx_val
        p_lo = np.column_stack([xc - diag_dx*nhx, yc - diag_dx*nhy])
        p_hi = np.column_stack([xc + diag_dx*nhx, yc + diag_dx*nhy])

        r_lo = np.sqrt(p_lo[:,0]**2 + p_lo[:,1]**2)
        r_hi = np.sqrt(p_hi[:,0]**2 + p_hi[:,1]**2)

        # surfaceval for inner: r - rmin;  for outer: rmax - r
        is_inner = (r_xb < R_mid).astype(float)
        f_lo = np.where(is_inner, r_lo - rmin, rmax - r_lo)
        f_hi = np.where(is_inner, r_hi - rmin, rmax - r_hi)

        bracket_ok = (f_lo * f_hi < 0.0)
        print(f"  bracket_ok      = {bracket_ok.sum()}/{cut.sum()} cut cells  "
              f"<-- should be 100%")
        if not bracket_ok.all():
            bad = ~bracket_ok
            print(f"  WARNING: {bad.sum()} cut cells have bracket failure")
            print(f"    f_lo range on bad cells: [{f_lo[bad].min():.3e}, {f_lo[bad].max():.3e}]")
            print(f"    f_hi range on bad cells: [{f_hi[bad].min():.3e}, {f_hi[bad].max():.3e}]")
            print(f"    r_xb range on bad cells: [{r_xb[bad].min():.4f}, {r_xb[bad].max():.4f}]")

    # ----------------------------------------------------------------
    # 3. Interpretation guide
    # ----------------------------------------------------------------
    print("""
=== How to interpret ===

  GOOD (bisection working, 2nd order expected):
    order(phi) ~ 2.0
    |err_n|/dx halves from res=32 to res=64

  FAIL MODE A — bracket not firing (bracket_ok < 100%):
    order(phi) ~ 1.0
    |err_n|/dx constant across resolutions
    -> bisection falls back to raw centroid everywhere
    Fix: widen bracket further or check is_inner classification

  FAIL MODE B — bisection working but alpha assembly wrong:
    order(phi) ~ 1.0
    |err_n|/dx halves correctly (bisection IS correcting xb)
    bracket_ok = 100%
    -> the stencil assembly uses 1/dn instead of ccibvec[dir]/d2
    Fix: switch to ccibvec[dir]/d2*outward_normal_dir formula

  FAIL MODE C — is_inner flag wrong for some cut cells:
    bracket_ok < 100% only near midradius
    r_xb on bad cells is close to R_mid (0.15)
    Fix: already handled by using rad_xb not ri in bc_ib
""")

if __name__ == "__main__":
    main()
