#!/usr/bin/env bash
# MMS2 on an annulus with EB geometry.
#
# Reproduces the MMS2 convergence table of the paper. The reconstruction is
# selected at run time and this case defaults to prob.eb_geom_method=4 (scaled
# quadric + Newton) with prob.eb_quadric_offset=1; every log begins with an
# "IB reconstruction:" line saying which one was used. Do not switch methods by
# editing Prob.H.
#
#   ./run.sh                    # grids 32 64 128 256 512
#   python3 all_errors.py -f .  # table and convergence plots
#   python3 mms2_profiles.py    # radial profiles

# stop at the first failed run: without this a crashed solver is followed by a
# successful cd and the sequence reports itself as a clean verification run.
set -e
export FI_PROVIDER=tcp

# 512 is needed as well: the README convergence table and mms2_profiles.py
# (FOLDERS) both include it, so stopping at 256 cannot reproduce the last row.
for DIM in 32 64 128 256 512
do
    NP=1
    if [ "${DIM}" -ge 128 ]; then NP=4; fi

    mkdir -p "${DIM}"
    cd "${DIM}"

    rm -rf plt* chk*
    if ! mpirun -np ${NP} ../*.ex ../inputs2d \
            amr.n_cell="${DIM}" "${DIM}" 1 > log.log 2>&1; then
        echo "grid ${DIM} failed, see ${DIM}/log.log" >&2
        exit 1
    fi

    cd ..
done
