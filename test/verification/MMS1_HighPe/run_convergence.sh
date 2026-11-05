#!/usr/bin/env bash
# Grid convergence of MMS1_HighPe. Usage:
#   ./run_convergence.sh <deck> [extra inputs...]
# e.g. ./run_convergence.sh inputs_ib vidyut.hyp_wall_ghosts=0 prob.ib_face_flux=0
# Prints N_fluid, L2 and Linf of NI at t = 3 and the observed orders.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
deck=$1; shift
exe=$(ls "$here"/vidyut1d.*.ex | head -1)
chk=$(ls "$here"/l2err/l2err1d.*.ex | head -1)
np=${NP:-2}
len=$(awk '$1=="geometry.prob_hi"{print $3}' "$here/$deck")
tag=$(echo "$deck $*" | tr ' =' '__')
prev=""
for nf in 32 64 128 256; do
    n=$(python3 -c "print(round($nf*$len))")
    dt=$(python3 -c "print(0.2/$nf)")
    d="$here/runs/$tag/$nf"; mkdir -p "$d"; cd "$d"; rm -rf plt*
    mpirun -np "$np" "$exe" "$here/$deck" amr.n_cell=$n vidyut.dt=$dt "$@" > log 2>&1
    p=$(ls -d plt* | tail -1)
    line=$("$chk" "$p" | head -1)
    l2=$(echo "$line" | awk '{print $4}'); li=$(echo "$line" | awk '{print $6}')
    if [ -n "$prev" ]; then
        read -r pl2 pli <<< "$prev"
        r=$(python3 -c "import math;print('%.2f %.2f'%(math.log2($pl2/$l2),math.log2($pli/$li)))")
    else r="- -"; fi
    printf "%4d  L2 %.3e  Linf %.3e  rates %s\n" "$nf" "$l2" "$li" "$r"
    prev="$l2 $li"
done
