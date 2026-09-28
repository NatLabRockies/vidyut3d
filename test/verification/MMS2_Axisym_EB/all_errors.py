import os
import argparse
import numpy as np
import pandas as pd
import yt
import matplotlib.pyplot as plt

# yt is chatty; quiet it for speed
yt.set_log_level(50)

plt.rc("text", usetex=True)

CMAP = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23",
        "#662C91", "#A21D21", "#B43894", "#010202", "#888888"]
MARK = ["s", "d", "o", "p", "h", "s", "*", "p", "v"]

THEORY_COLOR = "black"          # old scripts used white -> invisible lines
ALPHA = 1.809512801e-08         # ECHARGE/EPS0 in species exact solutions
N0 = 1e6
FOLDERS = ["32", "64", "128", "256", "512"]
DPI = 600                       # publication quality

# ------------------------------------------------------------------ #
# Exact solutions
# ------------------------------------------------------------------ #
def phi_exact(r):            return (1.0 / 32.0) * r**4
def gradmag_exact(r):        return (1.0 / 8.0) * r**3
def gradx_exact(x, y):       r2 = x*x + y*y; return (4.0/32.0) * x * r2
def grady_exact(x, y):       r2 = x*x + y*y; return (4.0/32.0) * y * r2
def ne_exact(r):             return r**2 / ALPHA + N0          # electron
def ni_exact(r):             return 0.5 * r**2 / ALPHA + N0    # ion (HEp)
def een_exact(r):            return r**2 / ALPHA + N0          # electron energy


# ------------------------------------------------------------------ #
# Load one plotfile, compute ALL field errors in a single pass
# ------------------------------------------------------------------ #
def process_grid(path):
    ds = yt.load(path)
    dims = ds.domain_dimensions
    ad = ds.all_data()

    x = ad["x"].to_value()
    y = ad["y"].to_value()
    rad = np.sqrt(x * x + y * y)

    cellmask = ad["cellmask"].to_value()
    cellmask[cellmask < 1 - 1e-10] = 0.0
    m = cellmask

    def L2(field, exact):
        return np.sqrt(np.mean((field * m - exact * m) ** 2))

    def get(name):
        try:
            return ad[name].to_value()
        except Exception:
            return None

    out = {"Nx": int(dims[0]), "Ny": int(dims[1])}

    pot = get("Potential")
    if pot is not None:
        out["err_V"] = L2(pot, phi_exact(rad))
    Ex = get("Efieldx"); Ey = get("Efieldy")
    if Ex is not None and Ey is not None:
        out["err_Ex"] = L2(Ex, -gradx_exact(x, y))
        out["err_Ey"] = L2(Ey, -grady_exact(x, y))
        out["err_E"] = L2(np.sqrt(Ex**2 + Ey**2), gradmag_exact(rad))

    ne = get("E")
    if ne is not None:
        out["err_ne"] = L2(ne, ne_exact(rad))
    ni = get("HEp")
    if ni is not None:
        out["err_ni"] = L2(ni, ni_exact(rad))
    een = get("Electron_energy")
    if een is not None:
        out["err_een"] = L2(een, een_exact(rad))

    prof = None
    if pot is not None:
        potm = pot * m
        rb = np.linspace(rad.min(), rad.max(), 100)
        rc = 0.5 * (rb[1:] + rb[:-1])
        inds = np.digitize(rad, rb)
        vavg = np.array([
            np.average(potm[inds == b], weights=m[inds == b])
            if np.any(inds == b) and np.sum(m[inds == b]) > 0 else np.nan
            for b in range(1, len(rb))
        ])
        prof = (rc, vavg)

    return out, prof


def add_orders(df):
    df = df.sort_values("Nx").reset_index(drop=True)
    for col in [c for c in df.columns if c.startswith("err_")]:
        oc = "order_" + col[4:]
        df[oc] = np.nan
        e = df[col].to_numpy()
        for r in range(1, len(df)):
            if e[r] > 0 and e[r - 1] > 0:
                df.loc[r, oc] = np.log2(e[r - 1] / e[r])
    return df


def ref_lines(ax, Nx, err, idx=1):
    Nx = np.asarray(Nx, float); err = np.asarray(err, float)
    if len(Nx) <= idx:
        idx = 0
    ax.loglog(Nx, err[idx] * (Nx[idx] / Nx) ** 1,
              color=THEORY_COLOR, lw=2, ls="-", label="1st order", zorder=0)
    ax.loglog(Nx, err[idx] * (Nx[idx] / Nx) ** 2,
              color=THEORY_COLOR, lw=2, ls="--", label="2nd order", zorder=0)


# (error column, y-axis label, PNG filename key)
PANELS = [
    ("err_V",   r"$L_2(\phi)$",       "V"),
    ("err_E",   r"$L_2(E_r)$",     "E"),
    ("err_Ex",  r"$L_2(E_x)$",     "Ex"),
    ("err_Ey",  r"$L_2(E_y)$",     "Ey"),
    ("err_ne",  r"$L_2(n_e)$",     "ne"),
    ("err_ni",  r"$L_2(n_i)$",     "ni"),
    ("err_een", r"$L_2(E_e)$", "een"),
]


def style_axes(ax):
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description="Unified MMS error tool (PNG per field)")
    ap.add_argument("-f", "--fdirs", required=True, nargs="+",
                    help="Directories containing {16,32,...}/plt00001")
    ap.add_argument("-p", "--prefix", default="",
                    help="Optional filename prefix for output PNGs")
    ap.add_argument("--csv", action="store_true",
                    help="Also dump per-directory error tables to CSV")
    args = ap.parse_args()

    all_dfs = {}
    all_profiles = {}

    for fdir in args.fdirs:
        rows, profs = [], []
        for k, fname in enumerate(sorted(FOLDERS, key=int)):
            path = f"{fdir}/{fname}/plt00001"
            if not os.path.exists(path):
                continue
            try:
                out, prof = process_grid(path)
            except Exception as e:
                print(f"  skip {path}: {e}")
                continue
            rows.append(out)
            if prof is not None:
                profs.append((fname, k, prof))

        if not rows:
            print(f"No data in {fdir}")
            continue

        df = add_orders(pd.DataFrame(rows))
        all_dfs[fdir] = df
        all_profiles[fdir] = profs

        print(f"\n===== {fdir} =====")
        print(df.to_string(index=False))

        if args.csv:
            tag = os.path.basename(fdir.rstrip("/")) or "run"
            csv_name = f"{args.prefix}errors_{tag}.csv"
            df.to_csv(csv_name, index=False)
            print(f"  wrote {csv_name}")

    if not all_dfs:
        print("Nothing to plot.")
        return

    pfx = args.prefix

    # ---- V(r) profile: one PNG per directory ----
    for fdir, profs in all_profiles.items():
        if not profs:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for fname, k, (rc, vavg) in profs:
            ax.plot(rc, vavg, lw=0, marker=MARK[k % len(MARK)], ms=5,
                    color=CMAP[k % len(CMAP)], label=fname)
        xv = np.linspace(0.5, 1.5, 100)
        ax.plot(xv, phi_exact(xv), lw=2, color=THEORY_COLOR,
                label="Exact", zorder=-1)
        ax.set_xlabel(r"$r$", fontsize=22, fontweight="bold")
        ax.set_ylabel(r"$\overline{\phi}(r)$", fontsize=22, fontweight="bold")
        ax.set_xlim(0.5, 1.5)
        style_axes(ax)
        ax.legend(fontsize=10)
        fig.tight_layout()
        tag = os.path.basename(fdir.rstrip("/")) or "run"
        out_png = f"{pfx}potential_{tag}.png"
        fig.savefig(out_png, dpi=DPI)
        plt.close(fig)
        print(f"wrote {out_png}")

    # ---- one PNG per field convergence ----
    for col, ylab, key in PANELS:
        if not any(col in df.columns for df in all_dfs.values()):
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        drew = False
        for i, (fdir, df) in enumerate(all_dfs.items()):
            if col not in df.columns:
                continue
            sub = df.dropna(subset=[col])
            if sub.empty:
                continue
            ax.loglog(sub["Nx"], sub[col], lw=2, marker=MARK[0], ms=5,
                      color=CMAP[i % len(CMAP)], label="Vidyut3D")
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ref_df = None
        for fdir in reversed(args.fdirs):
            d = all_dfs.get(fdir)
            if d is not None and col in d.columns:
                ref_df = d.dropna(subset=[col]); break
        if ref_df is not None and len(ref_df) > 1:
            ref_lines(ax, ref_df["Nx"].to_numpy(), ref_df[col].to_numpy())
        ax.set_xlabel(r"$N_x$", fontsize=22, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=22, fontweight="bold")
        style_axes(ax)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out_png = f"{pfx}error_{key}.png"
        fig.savefig(out_png, dpi=DPI)
        plt.close(fig)
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
