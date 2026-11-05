#!/usr/bin/env bash
# AMR runs of the Laplace annulus (Dirichlet on both cylinders, inputs2d).
#
# Part 1, convergence: base grids 32/64/128 with two extra levels (effective
# 128/256/512), amr.n_error_buf scaled 6/12/24 so that every coarse-fine
# interface stays at least two coarse cells away from the wall.
#   -> amr32 amr64 amr128
#
# Part 2, Table 3 of the paper (tab:laplace_amr): four hierarchies that all
# reach an effective N_x = 512, n_error_buf = 8, plus the n_error_buf = 2 rows.
#   -> dirichlet/amr32L4_buf8 amr64L3_buf8 amr128L2_buf8 amr256L1_buf8
#      dirichlet/amr128L2_buf2 amr256L1_buf2
# The paper runs used hypre (USE_HYPRE=TRUE, vidyut.use_hypre=1
# vidyut.linsolve_max_coarsening_level=0); errors agree with the native solver
# to about 2e-3 relative. Set HYPRE="vidyut.use_hypre=1 vidyut.linsolve_max_coarsening_level=0"
# to repeat that.
#
#   ./run_amr_convergence.sh [1] [2]      (default: both; NP=<ranks>, default 2)
#   python3 amr_error.py -f amr32 amr64 amr128 --norm domain
#   python3 amr_error.py -f dirichlet/amr*_buf8 --norm domain
#   python3 error_plot_bc.py              (Fig. 2b, needs run.sh output as well)
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
EXE=$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")
NP=${NP:-2}
STEPS=10
HERE=$(pwd)
parts=("$@"); [ ${#parts[@]} -eq 0 ] && parts=(1 2)

run() {  # dir base levels buf
    local d=$1 n=$2 lev=$3 buf=$4
    mkdir -p "$d"
    (cd "$d"; rm -rf plt* chk*
     mpirun -np $NP "$EXE" "$HERE/inputs2d" max_step=$STEPS amr.n_cell="$n $n" \
         amr.max_level=$lev amr.n_error_buf=$buf vidyut.refine_cutcells=1 \
         amr.plot_int=$STEPS amr.chk_int=-1 ${HYPRE} > log.log 2>&1)
    echo "$d: base $n + $lev levels, n_error_buf=$buf done"
    grep -q "WARNING: a coarse-fine" "$d/log.log" && echo "  *** clearance warning, see $d/log.log" || true
}

for p in "${parts[@]}"; do
    case $p in
        1) run amr32 32 2 6; run amr64 64 2 12; run amr128 128 2 24 ;;
        2) run dirichlet/amr32L4_buf8 32 4 8;  run dirichlet/amr64L3_buf8 64 3 8
           run dirichlet/amr128L2_buf8 128 2 8; run dirichlet/amr256L1_buf8 256 1 8
           run dirichlet/amr128L2_buf2 128 2 2; run dirichlet/amr256L1_buf2 256 1 2 ;;
    esac
done
