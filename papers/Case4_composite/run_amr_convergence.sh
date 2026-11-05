#!/usr/bin/env bash
# AMR convergence of Case 4 (composite geometry): base grids 32/64/128 with two
# extra levels (effective 128/256/512), refined at the cut cells. Uses the
# ray_perface wall treatment of inputs2d (prob.perface_fallback=1,
# prob.ray_dirichlet=1, prob.corner_fallback=1) and vidyut.freeze_etemp=1.
# Without the three prob switches L2 looks only about 20 % worse, but Linf
# grows about 14x and stops converging.
#
# amr.n_error_buf is scaled with the base grid (6/12/24). Six bodies sit close
# together here, and below n_error_buf=24 at base 128 a few coarse-fine
# interface cells end up one cell from a wall, which costs about 60x in Linf.
# Vidyut prints a warning when that happens (vidyut.ib_cf_clearance_warn).
#
#   ./run_amr_convergence.sh            (NP=<ranks>, default 2; STEPS, default 720)
#   python3 amr_error.py amr32 amr64 amr128
#
# Expected phi L2 (get_error.py norm): 5.5085e-04 | 1.5829e-04 | 3.9736e-05,
# i.e. the uniform 128/256/512 values of runs/ray_perface.
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
EXE=$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")
NP=${NP:-2}
STEPS=${STEPS:-720}
for spec in "32 6" "64 12" "128 24"; do
    n=${spec% *}; buf=${spec#* }
    d="amr${n}"; mkdir -p "$d"
    (cd "$d"; rm -rf plt* chk*
     mpirun -np $NP "$EXE" ../inputs2d max_step=$STEPS amr.n_cell="$n $n" \
         amr.max_level=2 amr.n_error_buf=$buf vidyut.refine_cutcells=1 \
         amr.plot_int=$STEPS amr.chk_int=-1 > log 2>&1)
    echo "amr base $n + 2 levels (effective $((n*4)), n_error_buf=$buf) done"
    grep -q "WARNING: a coarse-fine" "$d/log" && echo "  *** clearance warning, see $d/log" || true
done
