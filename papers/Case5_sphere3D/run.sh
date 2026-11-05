#!/bin/bash
# usage: ./run.sh <rundir> "<N list>" [extra inputs...]
#   ./run.sh runs_m4 "32 64 128"
#   ./run.sh runs_ibng "32 64 128" prob.eb_geom_method=5
# Runs are sequential; never more ranks than performance cores (NP=<ranks> overrides).
set -e
rundir=$1; shift
ns=$1; shift
exe=$(ls $PWD/vidyut3d.*.ex | head -1)
for n in $ns; do
  np=8; [ $n -le 32 ] && np=1; [ $n -eq 64 ] && np=4
  [ -n "$NP" ] && np=$NP
  mkdir -p $rundir/$n
  (cd $rundir/$n && mpirun -np $np $exe ../../inputs3d amr.n_cell="$n $n $n" "$@" > log 2>&1)
  echo "done $rundir/$n: $(grep -c 'STEP' $rundir/$n/log) steps"
done
