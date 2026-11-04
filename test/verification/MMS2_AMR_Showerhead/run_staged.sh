#!/usr/bin/env bash
# Staged showerhead opening, MMS, 256x128 + 2 levels.
#   final : every slot open from step 0
#   staged: the plenum is present and the slots open in four groups at unequal steps
# Pass: staged == final at the last step (same discrete steady state).
export FI_PROVIDER=tcp
here=$(cd "$(dirname "$0")" && pwd)
exe=$(ls $here/vidyut2d*.ex | head -1)
AMR="amr.max_level=2 amr.blocking_factor=8 amr.n_error_buf=8 amr.regrid_int=1000000000 vidyut.refine_cutcells=1 prob.cmask_vfrac=1 vidyut.ib_identity_rows=1"
run() { # name plot_int extra...
  local name=$1 pi=$2; shift 2
  local d=$here/runs/$name; mkdir -p $d; cd $d; rm -rf plt* chk*
  /usr/bin/time -p mpirun -np 8 $exe $here/inputs amr.n_cell="256 128" max_step=960 \
      amr.plot_int=$pi amr.chk_int=-1 $AMR "$@" > log 2> time
  echo "$name exit=$? $(grep real time)"
}
run final 960
run staged 6 "eb2.comp_slot_phase=1 1 4 4 2 2 4 4 3 3" "vidyut.geom_switch_steps=120 240 300 450"
