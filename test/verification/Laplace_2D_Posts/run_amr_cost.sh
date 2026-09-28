#!/usr/bin/env bash
# Does local refinement reproduce a uniform fine grid when the geometry has
# gaps narrower than a base cell?
#
# phi = x^2 - y^2 is harmonic with vanishing fourth derivatives, so the
# interior stencil is exact and ALL the error comes from the posts and the
# 0.06 gaps between them. The base grid of the refined ladder (64) cannot
# resolve that gap at all - it is 0.64 of a cell.
#
#   uniform : n = 64,128,256,512 at max_level = 0
#   amr     : base 64 plus 1,2,3 levels, refining cut cells plus a buffer band
#
# Read the table in PAIRS at equal finest spacing: amrL1 vs uni128, amrL2 vs
# uni256, amrL3 vs uni512.
set -e
export FI_PROVIDER=tcp
EXE=${EXE:-$(ls -1 ./vidyut2d.*.ex 2>/dev/null | head -1)}
[ -x "$EXE" ] || { echo "build the case first (make)"; exit 1; }
NP=${NP:-4}
BUF=${BUF:-8}

for n in 64 128 256 512; do
    d="uni${n}"; mkdir -p "$d"; (cd "$d"; rm -rf plt* chk*
        mpirun -np $NP "../$EXE" ../inputs2d \
            amr.n_cell=$n $n amr.max_level=0 \
            amr.plot_int=10 amr.chk_int=-1 > log 2>&1)
    echo "uniform $n done"
done

for lev in 1 2 3; do
    d="amrL${lev}"; mkdir -p "$d"; (cd "$d"; rm -rf plt* chk*
        mpirun -np $NP "../$EXE" ../inputs2d \
            amr.n_cell=64 64 amr.max_level=$lev amr.n_error_buf=$BUF \
            vidyut.refine_cutcells=1 \
            amr.plot_int=10 amr.chk_int=-1 > log 2>&1)
    echo "amr base 64 + $lev level(s) -> effective $((64 * (1 << lev))), buf=$BUF"
    grep -q "WARNING: a coarse-fine" "$d/log" && \
        echo "  *** clearance warning, see $d/log" || true
done
