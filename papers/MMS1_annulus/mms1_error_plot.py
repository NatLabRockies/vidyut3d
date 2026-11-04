import os
import yt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

plt.rc("text", usetex=True)
plt.rc("font", size=16)
plt.rc("axes", titlesize=18, labelsize=20)
plt.rc("legend", fontsize=16)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)

cmap = [
    "#EE2E2F", "#008C48", "#185AA9", "#F47D23",
    "#662C91", "#A21D21", "#B43894", "#010202"
]
markertype = ["s", "d", "o", "p", "h", "s", "*", "s"]

def exact_solution(rad, rmin, rmax, phi1, phi2):
    return rad**4

def exact_solution_grad(rad, rmin, rmax, phi1, phi2):
    return 4 * rad**3

def exact_solution_gradx(x, y, rmin, rmax, phi1, phi2):
    r = np.sqrt(x*x + y*y)
    return 4 * x * r**2

def exact_solution_grady(x, y, rmin, rmax, phi1, phi2):
    r = np.sqrt(x*x + y*y)
    return 4 * y * r**2

def exact_NI(rad, rmin, rmax, phi1, phi2):
    return 16 * rad**2


def main():

    parser = argparse.ArgumentParser(description="Extended MMS Potential + NI Plot Tool")
    parser.add_argument("-f", "--fdirs", required=True, type=str, nargs="+")
    args = parser.parse_args()

    rmin = 0.5
    rmax = 1.5
    phi1 = 10.0
    phi2 = 20.0

    for i, fdir in enumerate(args.fdirs):

        lst = []
        lst_NI = []

        folders = ["32", "64", "128", "256", "512"]
        folders.sort(key=lambda x: int(x))

        for k, fname in enumerate(folders):

            ds = yt.load(f"{fdir}/{fname}/plt00001")
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

            NI = ad["NI"].to_value() * cellmask

            rad = np.sqrt(x*x + y*y)

            exact_p = exact_solution(rad, rmin, rmax, phi1, phi2) * cellmask
            error_p = np.sqrt(np.mean((potential - exact_p)**2))

            exact_ex = -exact_solution_gradx(x, y, rmin, rmax, phi1, phi2) * cellmask
            exact_ey = -exact_solution_grady(x, y, rmin, rmax, phi1, phi2) * cellmask
            error_ex = np.sqrt(np.mean((Efieldx - exact_ex)**2))
            error_ey = np.sqrt(np.mean((Efieldy - exact_ey)**2))

            exact_ef = exact_solution_grad(rad, rmin, rmax, phi1, phi2) * cellmask
            error_ef = np.sqrt(np.mean((Efield - exact_ef)**2))

            exact_NI_vals = exact_NI(rad, rmin, rmax, phi1, phi2) * cellmask
            error_NI = np.sqrt(np.mean((NI - exact_NI_vals)**2))

            r_bins = np.linspace(rad.min(), rad.max(), 100)
            r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
            inds = np.digitize(rad, r_bins)

            potential_avg = np.array([
                np.average(potential[inds == idx], weights=cellmask[inds == idx])
                if np.any(inds == idx) and np.sum(cellmask[inds == idx]) > 0 else np.nan
                for idx in range(1, len(r_bins))
            ])

            plt.figure(f"potential-{fdir}")
            plt.plot(
                r_centers, potential_avg,
                lw=0, marker=markertype[k], ms=5,
                color=cmap[k], label=f"{fname}"
            )

            NI_avg = np.array([
                np.average(NI[inds == idx], weights=cellmask[inds == idx])
                if np.any(inds == idx) and np.sum(cellmask[inds == idx]) > 0 else np.nan
                for idx in range(1, len(r_bins))
            ])

            plt.figure(f"NI-{fdir}")
            plt.plot(
                r_centers, NI_avg,
                lw=0, marker=markertype[k], ms=5,
                color=cmap[k], label=f"{fname}"
            )

            lst.append({
                "Nx": dims[0],
                "Ny": dims[1],
                "error": error_p,
                "error_efield": error_ef,
                "error_efieldx": error_ex,
                "error_efieldy": error_ey,
            })

            lst_NI.append({
                "Nx": dims[0],
                "Ny": dims[1],
                "error_NI": error_NI,
            })

        x_vals = np.linspace(rmin, rmax, 200)

        plt.figure(f"potential-{fdir}")
        plt.plot(x_vals, exact_solution(x_vals, rmin, rmax, phi1, phi2),
                 lw=2, color=cmap[-1], label="Exact", zorder=-1)

        plt.figure(f"NI-{fdir}")
        plt.plot(x_vals, exact_NI(x_vals, rmin, rmax, phi1, phi2),
                 lw=2, color=cmap[-1], label="Exact", zorder=-1)

        df = pd.DataFrame(lst)
        df.sort_values(by="Nx", inplace=True)
        print(df)

        df_NI = pd.DataFrame(lst_NI)
        df_NI.sort_values(by="Nx", inplace=True)
        print(df_NI)

        idx = 1

        df["theory1"] = (
            df["error"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"])**1
        )
        df["theory2"] = (
            df["error"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"])**2
        )

        df["theory1_efield"] = (
            df["error_efield"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"])**1
        )
        df["theory1_efieldx"] = (
            df["error_efieldx"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"])**1
        )
        df["theory1_efieldy"] = (
            df["error_efieldy"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"])**1
        )

        df_NI["theory1_NI"] = (
            df_NI["error_NI"].iloc[idx] * (df_NI["Nx"].iloc[idx] / df_NI["Nx"])**1
        )
        df_NI["theory2_NI"] = (
            df_NI["error_NI"].iloc[idx] * (df_NI["Nx"].iloc[idx] / df_NI["Nx"])**2
        )

        plt.figure("error")
        plt.loglog(df["Nx"], df["error"], lw=2, marker=markertype[0], ms=5,
                   color=cmap[i], label="Vidyut3D")

        plt.figure("error_efield")
        plt.loglog(df["Nx"], df["error_efield"], lw=2, marker=markertype[0],
                   ms=5, color=cmap[i], label="Vidyut3D")

        plt.figure("error_efieldx")
        plt.loglog(df["Nx"], df["error_efieldx"], lw=2, marker=markertype[0],
                   ms=5, color=cmap[i], label="Vidyut3D")

        plt.figure("error_efieldy")
        plt.loglog(df["Nx"], df["error_efieldy"], lw=2, marker=markertype[0],
                   ms=5, color=cmap[i], label="Vidyut3D")

        plt.figure("error_NI")
        plt.loglog(df_NI["Nx"], df_NI["error_NI"], lw=2, marker=markertype[0],
                   ms=5, color=cmap[i], label="Vidyut3D")

        if fdir == args.fdirs[-1]:

            plt.figure("error")
            plt.loglog(df["Nx"], df["theory1"], lw=2, color=cmap[-1],
                       label="1st order", zorder=0)
            plt.loglog(df["Nx"], df["theory2"], lw=2, color=cmap[-1],
                       ls="--", label="2nd order", zorder=0)

            plt.figure("error_efield")
            plt.loglog(df["Nx"], df["theory1_efield"], lw=2, color=cmap[-1],
                       label="1st order", zorder=0)

            plt.figure("error_efieldx")
            plt.loglog(df["Nx"], df["theory1_efieldx"], lw=2, color=cmap[-1],
                       label="1st order", zorder=0)

            plt.figure("error_efieldy")
            plt.loglog(df["Nx"], df["theory1_efieldy"], lw=2, color=cmap[-1],
                       label="1st order", zorder=0)

            plt.figure("error_NI")
            plt.loglog(df_NI["Nx"], df_NI["theory1_NI"], lw=2, color=cmap[-1],
                       label="1st order", zorder=0)
            plt.loglog(df_NI["Nx"], df_NI["theory2_NI"], lw=2, color=cmap[-1],
                       ls="--", label="2nd order", zorder=0)

    for fdir in args.fdirs:
        clean = os.path.basename(fdir.rstrip("/"))

        plt.figure(f"potential-{fdir}")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\overline{\phi}(r)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"potential_{clean}.png", dpi=600)
        plt.close()

        plt.figure(f"NI-{fdir}")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\overline{n}_i(r)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"NI_{clean}.png", dpi=600)
        plt.close()

    plt.figure("error")
    plt.xlabel(r"$N_x$")
    plt.ylabel(r"$L_2(\phi)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error.png", dpi=600)
    plt.close()

    plt.figure("error_efield")
    plt.xlabel(r"$N_x$")
    plt.ylabel(r"$L_2(E_r)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_efield.png", dpi=600)
    plt.close()

    plt.figure("error_efieldx")
    plt.xlabel(r"$N_x$")
    plt.ylabel(r"$L_2(E_x)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_efieldx.png", dpi=600)
    plt.close()

    plt.figure("error_efieldy")
    plt.xlabel(r"$N_x$")
    plt.ylabel(r"$L_2(E_y)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_efieldy.png", dpi=600)
    plt.close()

    plt.figure("error_NI")
    plt.xlabel(r"$N_x$")
    plt.ylabel(r"$L_2(n_i)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_NI.png", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
