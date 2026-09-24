import os
import yt
import numpy as np
import pandas as pd
import argparse


# Laplace annulus geometry (matches modified_error.py)
R_I = 0.1
R_O = 0.2
PHI_I = 10.0
PHI_O = 20.0


def exact_solution_laplace(rad):
    """Analytic Laplace-in-annulus solution with Dirichlet BCs at r_i and r_o."""
    return (PHI_O * np.log(np.maximum(rad, 1e-30) / R_I)
            + PHI_I * np.log(R_O / np.maximum(rad, 1e-30))) / np.log(R_O / R_I)


EXACT = {
    "Potential": exact_solution_laplace,
}


def diagnose(fdir, folder, field_name, n_worst=10, neighbor_radius=1):
    plt_path = f"{fdir}/{folder}/plt00001"
    if not os.path.isdir(plt_path):
        print(f"  [skip] {plt_path} not found")
        return None

    ds = yt.load(plt_path)
    dims = ds.domain_dimensions

    print(f"\n{'='*70}")
    print(f"Diagnosing (Laplace): {fdir}/{folder}  field={field_name}")
    print(f"Grid: {dims}")
    print(f"Fluid annulus: {R_I} <= r <= {R_O}")
    print(f"{'='*70}")

    cg = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )

    field = cg[field_name].to_value()
    cellmask = cg["cellmask"].to_value()
    x = cg["x"].to_value()
    y = cg["y"].to_value()

    rad = np.sqrt(x * x + y * y)
    exact = EXACT[field_name](rad)

    # HARD geometric fluid mask (required because cellmask alone doesn't
    # distinguish covered electrode cells from fluid cells in this run)
    geom_mask = (rad >= R_I) & (rad <= R_O)
    cm_mask   = (cellmask > 0.5)
    fluid_mask = geom_mask & cm_mask
    fluid_float = fluid_mask.astype(float)

    err = np.abs(field - exact) * fluid_float

    n_fluid = int(fluid_mask.sum())
    l2 = np.sqrt(np.sum(((field - exact) * fluid_float) ** 2) / max(n_fluid, 1))
    print(f"# fluid cells                        = {n_fluid}")
    print(f"L2 error (fluid annulus, mean-square)= {l2:.6e}")
    print(f"Max pointwise |error| (fluid annulus)= {err.max():.6e}")

    flat_idx_sorted = np.argsort(err.ravel())[::-1]
    top_flat = flat_idx_sorted[:n_worst]
    top_idx = np.array(np.unravel_index(top_flat, err.shape)).T

    rows = []
    for rank, idx in enumerate(top_idx):
        i, j, k = (idx[0], idx[1], idx[2] if err.ndim == 3 else 0)

        e_val   = err[i, j, k] if err.ndim == 3 else err[i, j]
        f_val   = field[i, j, k] if field.ndim == 3 else field[i, j]
        ex_val  = exact[i, j, k] if exact.ndim == 3 else exact[i, j]
        cm_val  = cellmask[i, j, k] if cellmask.ndim == 3 else cellmask[i, j]
        r_val   = rad[i, j, k] if rad.ndim == 3 else rad[i, j]
        x_val   = x[i, j, k] if x.ndim == 3 else x[i, j]
        y_val   = y[i, j, k] if y.ndim == 3 else y[i, j]

        di_range = range(-neighbor_radius, neighbor_radius + 1)
        neigh_cm = []
        n_cut_neighbors = 0
        n_covered_neighbors = 0
        for di in di_range:
            row = []
            for dj in di_range:
                ii, jj = i + di, j + dj
                if 0 <= ii < cellmask.shape[0] and 0 <= jj < cellmask.shape[1]:
                    v = cellmask[ii, jj, k] if cellmask.ndim == 3 else cellmask[ii, jj]
                    row.append(v)
                    if di == 0 and dj == 0:
                        continue
                    if v < 1 - 1e-10 and v > 1e-10:
                        n_cut_neighbors += 1
                    elif v <= 1e-10:
                        n_covered_neighbors += 1
                else:
                    row.append(np.nan)
            neigh_cm.append(row)

        if cm_val <= 1e-10:
            cell_type = "COVERED"
        elif cm_val < 1 - 1e-10:
            cell_type = "CUT"
        else:
            cell_type = "REGULAR"

        d_inner = r_val - R_I
        d_outer = R_O - r_val
        nearest_wall = "INNER" if d_inner < d_outer else "OUTER"
        d_wall = min(d_inner, d_outer)

        is_near_wall = (n_cut_neighbors > 0
                        or n_covered_neighbors > 0
                        or cell_type != "REGULAR")

        print(f"\n--- Rank {rank+1} worst cell ---")
        print(f"  index (i,j,k)         = ({i},{j},{k})")
        print(f"  location (x,y)        = ({x_val:.5f}, {y_val:.5f})   r = {r_val:.5f}")
        print(f"  nearest wall          = {nearest_wall} (dist = {d_wall:.5f})")
        print(f"  cell type             = {cell_type}   (cellmask = {cm_val:.4f})")
        print(f"  |error|               = {e_val:.4e}")
        print(f"  numerical             = {f_val:.6e}")
        print(f"  exact                 = {ex_val:.6e}")
        print(f"  relative error        = {e_val / max(abs(ex_val), 1e-30):.3e}")
        print(f"  cut neighbors         = {n_cut_neighbors}")
        print(f"  covered neighbors     = {n_covered_neighbors}")
        print(f"  near-wall             = {is_near_wall}")
        print(f"  3x3 neighbor cellmask:")
        for row in neigh_cm:
            print("    " + "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  -- "
                                     for v in row))

        rows.append({
            "rank": rank + 1,
            "i": i, "j": j, "k": k,
            "x": x_val, "y": y_val, "r": r_val,
            "nearest_wall": nearest_wall,
            "d_wall": d_wall,
            "cellmask": cm_val,
            "cell_type": cell_type,
            "abs_error": e_val,
            "numerical": f_val,
            "exact": ex_val,
            "rel_error": e_val / max(abs(ex_val), 1e-30),
            "n_cut_neighbors": n_cut_neighbors,
            "n_covered_neighbors": n_covered_neighbors,
            "near_wall": is_near_wall,
        })

    df = pd.DataFrame(rows)
    csv_out = f"worst_errors_Laplace_{field_name}_{folder}.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved detailed table to {csv_out}")

    n_near_wall = int(df["near_wall"].sum())
    print(f"\nSummary: {n_near_wall}/{len(df)} of the top-{len(df)} "
          f"worst cells are near a wall.")

    n_inner = int((df["nearest_wall"] == "INNER").sum())
    n_outer = int((df["nearest_wall"] == "OUTER").sum())
    print(f"          {n_inner} nearest INNER wall (r={R_I}), "
          f"{n_outer} nearest OUTER wall (r={R_O}).")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Locate worst-error cells for the Laplace annulus verification case."
    )
    parser.add_argument("-f", "--fdirs", required=True, type=str, nargs="+")
    parser.add_argument("--field", default="Potential",
                        help="Field name (default: Potential).")
    parser.add_argument("--folders", nargs="+",
                        default=["16", "32", "64", "128", "256", "512"])
    parser.add_argument("-n", "--n_worst", type=int, default=10)
    parser.add_argument("-r", "--neighbor_radius", type=int, default=1)
    args = parser.parse_args()

    for fdir in args.fdirs:
        for folder in args.folders:
            diagnose(fdir, folder, args.field,
                     n_worst=args.n_worst,
                     neighbor_radius=args.neighbor_radius)


if __name__ == "__main__":
    main()
