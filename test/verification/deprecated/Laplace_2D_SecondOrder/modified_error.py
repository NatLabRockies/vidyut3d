import os
import yt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

plt.rc("text", usetex=True)
#plt.rc("mathtext", fontset="dejavuserif")
cmap = [
    "#EE2E2F", "#008C48", "#185AA9", "#F47D23",
    "#662C91", "#A21D21", "#B43894", "#010202"
]

markertype = ["s", "d", "o", "p", "h", "s", "*",'s']

# ============================================================
# Exact solutions
# ============================================================
def exact_solution(rad, rmin, rmax, phi1, phi2):
    return (phi2*np.log(rad/rmin) + phi1*np.log(rmax/rad)) / np.log(rmax/rmin)

def exact_solution_grad(rad, rmin, rmax, phi1, phi2):
    return (phi2 - phi1) / (rad * np.log(rmax/rmin))

def exact_solution_gradx(x, y, rmin, rmax, phi1, phi2):
    r = np.sqrt(x*x + y*y)
    return (phi2 - phi1) * x / (r*r * np.log(rmax/rmin))

def exact_solution_grady(x, y, rmin, rmax, phi1, phi2):
    r = np.sqrt(x*x + y*y)
    return (phi2 - phi1) * y / (r*r * np.log(rmax/rmin))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Convergence Plot Tool")
    parser.add_argument("-f", "--fdirs", required=True, type=str, nargs="+")
    args = parser.parse_args()

    rmin, rmax = 0.1, 0.2
    phi1, phi2 = 10.0, 20.0

    for i, fdir in enumerate(args.fdirs):

        lst = []
        folders = ["16", "32", "64", "128", "256", "512", "1024", "2048"]
        folders.sort(key=lambda x: int(x))

        # ============================================================
        # Loop over resolutions
        # ============================================================
        for k, fname in enumerate(folders):
            try:
                ds = yt.load(f"{fdir}/{fname}/plt00001")
            except:
                continue
            dims = ds.domain_dimensions
            ad = ds.all_data()

            x = ad["x"].to_value()
            y = ad["y"].to_value()
            cellmask = ad["cellmask"].to_value()
            cellmask[cellmask < 1 - 1e-10] = 0.0

            potential = ad["Potential"].to_value() * cellmask
            Efieldx = ad["Efieldx"].to_value() * cellmask
            Efieldy = ad["Efieldy"].to_value() * cellmask
            Efield = np.sqrt(Efieldx**2 + Efieldy**2)

            rad = np.sqrt(x*x + y*y)

            # --- Exact solutions ---
            exact = exact_solution(rad, rmin, rmax, phi1, phi2) * cellmask
            exact_efieldx = -exact_solution_gradx(x, y, rmin, rmax, phi1, phi2) * cellmask
            exact_efieldy = -exact_solution_grady(x, y, rmin, rmax, phi1, phi2) * cellmask
            exact_efield = exact_solution_grad(rad, rmin, rmax, phi1, phi2) * cellmask

            error = np.sqrt(np.mean((potential - exact)**2))
            error_efield = np.sqrt(np.mean((Efield - exact_efield)**2))
            error_efieldx = np.sqrt(np.mean((Efieldx - exact_efieldx)**2))
            error_efieldy = np.sqrt(np.mean((Efieldy - exact_efieldy)**2))

            # --- Averaged potential profile ---
            r_bins = np.linspace(rad.min(), rad.max(), 100)
            r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
            inds = np.digitize(rad, r_bins)

            potential_avg = np.array([
                np.average(potential[inds == j], weights=cellmask[inds == j])
                if np.any(inds == j) and np.sum(cellmask[inds == j]) > 0 else np.nan
                for j in range(1, len(r_bins))
            ])

            plt.figure(figsize=(10,6), num=f"potential-{fdir}")
            plt.plot(r_centers, potential_avg, lw=0,
                     marker=markertype[k], ms=5, color=cmap[k],
                     label=f"{fname}")

            lst.append({
                "Nx": dims[0],
                "Ny": dims[1],
                "error": error,
                "error_efield": error_efield,
                "error_efieldx": error_efieldx,
                "error_efieldy": error_efieldy,
            })

        # ============================================================
        # Exact profile
        # ============================================================
        plt.figure(figsize=(10,6), num=f"potential-{fdir}")
        x_vals = np.linspace(rmin, rmax, 200)
        plt.plot(x_vals, exact_solution(x_vals, rmin, rmax, phi1, phi2),
                 lw=2, color=cmap[-1], label="Exact")

        df = pd.DataFrame(lst).sort_values(by="Nx")
        print(df)
        # ============================================================
        # First & second order reference lines
        # ============================================================
        idx = 1
        base_N = df["Nx"].iloc[idx]

        for key in ["error", "error_efield", "error_efieldx", "error_efieldy"]:
            df[f"theory1_{key}"] = df[key].iloc[idx] * (base_N / df["Nx"])**1
            df[f"theory2_{key}"] = df[key].iloc[idx] * (base_N / df["Nx"])**2

        # ============================================================
        # Error plots
        # ============================================================
        def plot_error(name, col, ylabel):
            plt.figure(figsize=(10,6), num=name)
            plt.loglog(df["Nx"], df[col], lw=2,
                       marker=markertype[0], ms=8,
                       color=cmap[i], label="Results")
            plt.loglog(df["Nx"], df[f"theory1_{col}"], lw=2,
                       color=cmap[-1], label="1st order")
            plt.loglog(df["Nx"], df[f"theory2_{col}"], lw=2, ls="--",
                       color=cmap[-2], label="2nd order")
            plt.xlabel(r"$n_x$", fontsize=20, fontweight="bold")
            plt.ylabel(ylabel, fontsize=20, fontweight="bold")
            plt.legend()
            plt.tight_layout()

        plot_error("error", "error", r"$L_2(V)$")
        plot_error("error_efield", "error_efield", r"$L_2(E_r)$")
        plot_error("error_efieldx", "error_efieldx", r"$L_2(E_x)$")
        plot_error("error_efieldy", "error_efieldy", r"$L_2(E_y)$")

    # ============================================================
    # Save plots (clean file names)
    # ============================================================
    for figname, fname in [
        ("error", "error.png"),
        ("error_efield", "error_efield.png"),
        ("error_efieldx", "error_efieldx.png"),
        ("error_efieldy", "error_efieldy.png"),
    ]:
        plt.figure(figname)
        plt.savefig(fname, dpi=1200)
        plt.close()
    plt.figure(f"potential-{fdir}")
    # Potential plot
    for fdir in args.fdirs:
        plt.xlabel(r"$r$", fontsize=20, fontweight="bold")
        plt.ylabel(r"$\\bar{V}(r)$", fontsize=20, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.savefig("potential_results.png", dpi=1200)
        plt.close()


# ============================================================
if __name__ == "__main__":
    main()
