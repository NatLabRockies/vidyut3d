#!/usr/bin/env bash
# MMS1 (single-species coupled system) on the annulus, grid sequence of the paper.
# All switches are in inputs2d (method 4 + offset are the Prob.H defaults).
#
# Usage: ./run.sh                 (NP=<ranks>, default 2)
#        EXTRA="prob.eb_geom_method=5" OUT=ibng ./run.sh
# Output: ${OUT:-results}/<n>/plt00001 for n = 32 ... 512, then
#         python3 mms1_error_plot.py -f results
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
EXE=$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")
NP=${NP:-2}
OUT=${OUT:-results}
for N in 32 64 128 256 512; do
    d="${OUT}/${N}"; mkdir -p "$d"
    (cd "$d"; rm -rf plt* chk*
     mpirun -np ${NP} "$EXE" ../../inputs2d amr.n_cell="${N} ${N}" ${EXTRA} > log.${N} 2>&1)
    echo "${OUT} ${N} done: $(grep -m1 'IB reconstruction' "$d/log.${N}")"
done
