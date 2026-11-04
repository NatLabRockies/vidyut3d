#!/usr/bin/env bash
# AMR convergence of MMS2: base grids 32/64/128 with two extra levels
# (effective 128/256/512), refined at the cut cells.
#
# amr.n_error_buf is scaled with the base grid (6/12/24) so that the refined
# band keeps the same physical width and every coarse-fine interface stays at
# least two coarse cells away from the wall. Vidyut prints a warning when it is
# closer (vidyut.ib_cf_clearance_warn). With more than one level the solver
# selects vidyut.ib_identity_rows=1 by itself (log: "IB solid cells: own row").
#
#   ./run_amr_convergence.sh            (NP=<ranks>, default 2; STEPS, default 720)
#   python3 amr_error.py amr32 amr64 amr128
#
# Expected phi L2 (amr_error.py: relative to max |phi|, fluid cells, volume
# weighted): 1.4346e-03 | 3.5460e-04 | 9.3343e-05.
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
