#!/usr/bin/env bash
# usage: ./run_config.sh <name> "<extra inputs>" "<resolutions>"
# Runs the resolution sequence for one safeguard configuration into runs/<name>/<n>/
export FI_PROVIDER=tcp
name=$1; extra=$2; res=${3:-"32 64 128 256"}
here=$(cd "$(dirname "$0")" && pwd)
EXE=${EXE:-$(ls -1 "$here"/vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
for n in $res; do
  d=$here/runs/$name/$n; mkdir -p $d; cd $d; rm -rf plt* chk*
  np=2; [ $n -ge 128 ] && np=4; [ $n -ge 256 ] && np=8
  /usr/bin/time -p mpirun -np $np $EXE $here/inputs2d amr.n_cell=$n $n $extra > log.$n 2> time.$n
  echo "$name n=$n exit=$? $(grep real time.$n)"
done
