#!/usr/bin/env bash
# MMS2 on an annulus with EB geometry.
#
# Reproduces the MMS2 convergence table of the paper. The reconstruction is
# selected at run time and this case defaults to prob.eb_geom_method=4 (scaled
# quadric + Newton) with prob.eb_quadric_offset=1; every log begins with an
# "IB reconstruction:" line saying which one was used. Do not switch methods by
# editing Prob.H.
#
#   ./run.sh                    # grids 32 64 128 256
#   python3 all_errors.py -f .  # table and convergence plots
#   python3 mms2_profiles.py    # radial profiles

export FI_PROVIDER=tcp mpirun

for DIM in 32 64 128 256
do
    NP=1
    if [ "${DIM}" -ge 128 ]; then NP=4; fi

    mkdir -p "${DIM}"
    cd "${DIM}"

    rm -rf plt* chk*
    mpirun -np ${NP} ../*.ex ../inputs2d amr.n_cell="${DIM}" "${DIM}" 1 > log.log 2>&1

    cd ..
done
