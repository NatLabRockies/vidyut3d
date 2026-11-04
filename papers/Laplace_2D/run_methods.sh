#!/usr/bin/env bash
# Laplace annulus, comparison of the wall reconstructions (Table 4 of the paper,
# tab:laplace_nr_methods). Neumann inner wall, Dirichlet outer wall.
#
#   prob.eb_geom_method = 1 PCA plane + quadric, 2 PCA tangent plane,
#                         3 analytical annulus, 4 scaled quadric + Newton
#                         (with prob.eb_quadric_offset=1, the default),
#                         5 nearest cut cell (IB-NG)
#
# Usage: ./run_methods.sh            (NP=<ranks>, default 1)
# Output: methods/m<k>/<n>/plt00001; the error is the mean over all cells of the
# masked squared error (error_plot_bc.py / potential_error_plot.py).
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
EXE=$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")
NP=${NP:-1}

for M in 1 2 3 4 5; do
    for N in 32 64 128 256 512; do
        d="methods/m${M}/${N}"; mkdir -p "$d"
        (cd "$d"; rm -rf plt* chk*
         mpirun -np ${NP} "$EXE" ../../../inputs2d max_step=10 amr.n_cell="${N} ${N}" \
             prob.a_inner_dir=0.0 prob.b_inner_neumann=1.0 \
             prob.eb_geom_method=${M} > log.log 2>&1)
        grep -m1 "IB reconstruction" "$d/log.log"
    done
done
