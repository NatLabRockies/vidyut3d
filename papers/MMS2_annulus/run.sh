#!/usr/bin/env bash
# MMS2 (electrons, ions, electron energy, potential) on the annulus, grid
# sequence of the paper. All switches are in inputs2d (method 4 + offset are
# the Prob.H defaults, vidyut.freeze_etemp=1).
#
# Usage: ./run.sh              method 4 + offset  -> results/<n>
#        ./run.sh ibng         IB-NG table        -> ibng/<n>  (prob.eb_geom_method=5)
#        NP=<ranks> (default 2)
# Then:  python3 all_errors.py -f results
#        python3 all_errors.py -f ibng -p ibng_
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
EXE=$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")
NP=${NP:-2}
case "${1:-results}" in
    results) OUT=results; EXTRA="" ;;
    ibng)    OUT=ibng;    EXTRA="prob.eb_geom_method=5" ;;
    *) echo "usage: ./run.sh [results|ibng]"; exit 1 ;;
esac
for N in 32 64 128 256 512; do
    d="${OUT}/${N}"; mkdir -p "$d"
    (cd "$d"; rm -rf plt* chk*
     mpirun -np ${NP} "$EXE" ../../inputs2d amr.n_cell="${N} ${N}" ${EXTRA} > log.${N} 2>&1)
    echo "${OUT} ${N} done: $(grep -m1 'IB reconstruction' "$d/log.${N}")"
done
