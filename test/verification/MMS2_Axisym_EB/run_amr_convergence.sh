#!/usr/bin/env bash
# MMS2 on an annulus: does an AMR hierarchy match the uniform grid whose
# resolution equals its finest level?
#
# It does, provided every coarse-fine interface clears the immersed boundary by
# at least two coarse cells. An interface one cell from the wall costs about 20x
# in the max norm and the error then stops converging under refinement, so
# amr.n_error_buf is scaled with the base grid below to hold the physical
# clearance constant. Vidyut prints a warning if the clearance is too small;
# see vidyut.ib_cf_clearance_warn.
#
#   ./run_amr_convergence.sh
#   python3 amr_error.py uni128 uni256 uni512
#   python3 amr_error.py amr32 amr64 amr128
#
# The two tables should agree, row for row.
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
NP=${NP:-4}
STEPS=${STEPS:-720}

# uniform references
for n in 128 256 512; do
    d="uni${n}"; mkdir -p "$d"; (cd "$d"; rm -rf plt* chk*
        mpirun -np $NP "../$EXE" ../inputs2d max_step=$STEPS \
            amr.n_cell=$n $n amr.max_level=0 amr.plot_int=$STEPS amr.chk_int=-1 > log 2>&1)
    echo "uniform $n done"
done

# two extra levels throughout, buffer scaled with the base grid so the refined
# band keeps the same physical width and the interface stays clear of the wall
set -- "32 6" "64 12" "128 24"
for spec in "$@"; do
    n=${spec% *}; buf=${spec#* }
    d="amr${n}"; mkdir -p "$d"; (cd "$d"; rm -rf plt* chk*
        mpirun -np $NP "../$EXE" ../inputs2d max_step=$STEPS \
            amr.n_cell=$n $n amr.max_level=2 amr.n_error_buf=$buf \
            vidyut.refine_cutcells=1 amr.plot_int=$STEPS amr.chk_int=-1 > log 2>&1)
    echo "amr base $n + 2 levels (effective $((n*4)), n_error_buf=$buf) done"
    grep -q "WARNING: a coarse-fine" "$d/log" && echo "  *** clearance warning, see $d/log"
done
