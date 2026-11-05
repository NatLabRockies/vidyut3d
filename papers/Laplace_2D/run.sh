#!/usr/bin/env bash
# Laplace annulus, grid sequence of the paper (Fig. 2a, Table 2).
#
# Inner-cylinder boundary condition a*phi + b*dphi/dn = f:
#   dirichlet : a=1 b=0      neumann : a=0 b=1      robin : a=1 b=1
# The outer cylinder stays Dirichlet. Each time step is one deferred-correction
# sweep of the second-order Neumann/Robin closure, so max_step=10 is 10 sweeps.
# The first-order closure of the paper (G1) is prob.ib_second_order=0.
#
# Usage: ./run.sh [dirichlet] [neumann] [robin]   (default: all three)
#        NP=4 ./run.sh dirichlet                  (MPI ranks, default 1)
#        EXTRA="prob.ib_second_order=0" OUT=g1 ./run.sh neumann
# Output: [<OUT>/]<bc>/<n>/plt00001 for n = 32 ... 512. Each run first deletes
# the plotfiles in its folder, so give a different OUT for other settings.
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
EXE=$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")
NP=${NP:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${OUT:+${OUT}/}

CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then CASES=(dirichlet neumann robin); fi

for BC in "${CASES[@]}"; do
    case "${BC}" in
        dirichlet) A_IN=1.0; B_IN=0.0 ;;
        neumann)   A_IN=0.0; B_IN=1.0 ;;
        robin)     A_IN=1.0; B_IN=1.0 ;;
        *) echo "unknown case ${BC}"; exit 1 ;;
    esac
    for N in 32 64 128 256 512; do
        d="${PREFIX}${BC}/${N}"; mkdir -p "$d"
        (cd "$d"; rm -rf plt* chk*
         mpirun -np ${NP} "$EXE" "${HERE}/inputs2d" max_step=10 amr.n_cell="${N} ${N}" \
             prob.a_inner_dir=${A_IN} prob.b_inner_neumann=${B_IN} ${EXTRA} > log.log 2>&1)
        echo "${d} done"
    done
done
