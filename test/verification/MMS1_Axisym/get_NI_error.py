import os
import yt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import argparse

plt.rc("text", usetex=True)
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
markertype = ["s", "d", "o", "p", "h", "s","*",'p']


def exact_solution(rad, rmin, rmax, phi1, phi2):
    #return (phi2 * np.log(rad / rmin) + phi1 * np.log(rmax / rad)) / np.log(rmax / rmin)
    return 16 * rad ** 2



def main():

    parser = argparse.ArgumentParser(description="A simple plot tool")
    parser.add_argument(
        "-f",
        "--fdirs",
        help="Directories with plot files",
        required=True,
        type=str,
        nargs="+",
    )
    args = parser.parse_args()

    rmin = 0.5
    rmax = 1.5
    phi2 = 20.0
    phi1 = 10.0

    for i, fdir in enumerate(args.fdirs):
        lst = []
        folders = [x for x in os.listdir(fdir)]
        #print(folders)
        folders= ["16","32","64","128","256","512","768","1024"]
        folders.sort(key=lambda x: int(x))
        for k, fname in enumerate(folders):
            try:
                ds = yt.load(f"{fdir}/{fname}/plt00001")
            except:
                continue
            dims = ds.domain_dimensions
            prob_lo = ds.domain_left_edge.d
            prob_hi = ds.domain_right_edge.d

            ad = ds.all_data()
            x = ad["x"].to_value()
            y = ad["y"].to_value()
            cellmask = ad["cellmask"].to_value()
            cellmask[cellmask < 1 - 1e-10] = 0.0
            potential = ad["NI"].to_value() * cellmask
            rad = np.sqrt(x * x + y * y)

            exact = exact_solution(rad, rmin, rmax, phi1, phi2) * cellmask
            error = np.sqrt(np.mean((potential - exact) ** 2))

            r_bins = np.linspace(rad.min(), rad.max(), 100)
            r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
            inds = np.digitize(rad, r_bins)
            potential_avg = np.array(
                [
                    (
                        np.average(potential[inds == i], weights=cellmask[inds == i])
                        if np.any(inds == i) and np.sum(cellmask[inds == i]) > 0
                        else np.nan
                    )
                    for i in range(1, len(r_bins))
                ]
            )

            plt.figure(f"potential-{fdir}")
            plt.plot(
                r_centers,
                potential_avg,
                lw=0,
                marker=markertype[k],
                ms=5,
                color=cmap[k],
                label=f"{fname}",
            )

            lst.append(
                {
                    "Nx": dims[0],
                    "Ny": dims[1],
                    "error": error,
                }
            )

        plt.figure(f"potential-{fdir}")
        x = np.linspace(rmin, rmax, 100)
        exact = exact_solution(x, rmin, rmax, phi1, phi2)
        plt.plot(x, exact, label="Exact", lw=2, color=cmap[-1], zorder=-1)

        df = pd.DataFrame(lst)
        df.sort_values(by="Nx", inplace=True)
        print(df)
        idx = 1
        theory_order = 2
        df["theory2"] = (
            df["error"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"]) ** theory_order
        )
        theory_order = 1
        df["theory1"] = (
            df["error"].iloc[idx] * (df["Nx"].iloc[idx] / df["Nx"]) ** theory_order
        )


        plt.figure("error")
        plt.loglog(
            df["Nx"],
            df["error"],
            lw=2,
            marker=markertype[0],
            ms=5,
            color=cmap[i],
            label=f"vidyut - {fdir}",
        )

        if fdir == args.fdirs[-1]:
            plt.figure("error")
            plt.loglog(
                df["Nx"],
                df["theory1"],
                lw=2,
                color=cmap[-1],
                label="1st order",
                zorder=0,
            )
            plt.loglog(
                df["Nx"],
                df["theory2"],
                lw=2,
                color=cmap[-1],
                ls="--",
                label="2nd order",
                zorder=0,
            )


    fname = "plots.pdf"
    with PdfPages(fname) as pdf:
        for fdir in args.fdirs:
            plt.figure(f"potential-{fdir}")
            ax = plt.gca()
            plt.xlabel(r"$r$", fontsize=22, fontweight="bold")
            plt.ylabel(r"$NI$", fontsize=22, fontweight="bold")
            plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            plt.xlim(rmin, rmax)
            plt.legend()
            plt.tight_layout()
            pdf.savefig(dpi=300)

        plt.figure("error")
        ax = plt.gca()
        plt.xlabel(r"$n_x$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$L_2(NI)$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(dpi=300)


if __name__ == "__main__":
    main()
