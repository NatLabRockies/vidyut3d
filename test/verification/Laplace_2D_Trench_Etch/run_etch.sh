#!/usr/bin/env bash
# Trench etched in one run (geometry switches before steps 4, 8, 12), a static
# run that has the final trench from step 0, and a restart of the staged run
# from step 6 (between the first and second switch). 128x256 + 2 levels,
# 16 steps, no periodic regrid, so only the switch can move the refined patches.
# Pass (fcompare): staged == static to 1e-12 with identical grids and cellmask,
# restarted == staged exactly. compare_switch.py then prints the gas-cell
# differences (needs yt).
set -o pipefail
export FI_PROVIDER=tcp
here=$(cd "$(dirname "$0")" && pwd)
exe=$(ls $here/vidyut2d*.ex | head -1)
fcompare=${FCOMPARE:-$(ls $here/../../../Submodules/amrex/Tools/Plotfile/fcompare*.ex 2>/dev/null | head -1)}
NP=${NP:-4}
mkdir -p $here/runs/staged $here/runs/final $here/runs/restart
grep -v geom_switch_steps $here/inputs2d > $here/runs/final/inputs_static
cd $here/runs/staged && rm -rf plt* chk*
mpirun -np $NP $exe $here/inputs2d amr.plot_int=1 amr.chk_int=6 > log 2>&1
echo "staged exit=$?"
cd $here/runs/final && rm -rf plt* chk*
mpirun -np $NP $exe inputs_static amr.plot_int=16 amr.chk_int=-1 \
    eb2.etch_stages=0 eb2.trench_depth=5.2 eb2.trench_wbot=1.16 > log 2>&1
echo "final exit=$?"
cd $here/runs/restart && rm -rf plt* chk*
mpirun -np $NP $exe $here/inputs2d amr.restart=../staged/chk00001 amr.plot_int=16 amr.chk_int=-1 > log 2>&1
echo "restart exit=$?"
cd $here
grep "no fluid neighbour" runs/staged/log runs/restart/log && { echo "cells left unfilled"; exit 1; }
if [ -n "$fcompare" ]; then   # built in CI; locally set FCOMPARE=<path>
    $fcompare -r 1e-12 runs/final/plt00001 runs/staged/plt00016 | tail -1 || { echo "staged != static"; exit 1; }
    $fcompare runs/staged/plt00016 runs/restart/plt00001 | tail -1 || { echo "restart != staged"; exit 1; }
fi
python3 compare_switch.py runs/final runs/staged
