#!/usr/bin/env bash
# Manufactured solution on one application geometry: uniform 256x128, 512x256,
# 1024x512 and AMR 256x128 + 2 levels (finest spacing = 1024x512).
#   usage: ./run_case.sh <streamer|pins|dbd> [run names...]   (default: u256 u512 u1024 amr)
#   output: runs/<case>/<run>/plt*, then python3 mms_errors.py runs/<case>/*
set -e
export FI_PROVIDER=tcp
here=$(cd "$(dirname "$0")" && pwd)
exe=$(ls $here/vidyut2d*.ex | head -1)
case=$1; shift
deck=$here/inputs_$case
[ -f "$deck" ] || { echo "no deck $deck (case must be streamer, pins or dbd)"; exit 1; }
names=${*:-"u256 u512 u1024 amr"}
NP=${NP:-8}; STEPS=${STEPS:-960}
AMR="amr.max_level=2 amr.blocking_factor=8 amr.n_error_buf=8 amr.regrid_int=1000000000 vidyut.refine_cutcells=1 prob.cmask_vfrac=1 vidyut.ib_identity_rows=1"
for n in $names; do
  case $n in
    u256)  args="amr.n_cell=256 128" ;;
    u512)  args="amr.n_cell=512 256" ;;
    u1024) args="amr.n_cell=1024 512" ;;
    amr)   args="amr.n_cell=256 128 $AMR" ;;
    *) echo "unknown run $n"; exit 1 ;;
  esac
  d=$here/runs/$case/$n; mkdir -p $d; cd $d; rm -rf plt* chk*
  /usr/bin/time -p mpirun -np $NP $exe $deck $args max_step=$STEPS amr.plot_int=$STEPS amr.chk_int=-1 > log 2> time
  echo "$case/$n $(grep real time)"
done
