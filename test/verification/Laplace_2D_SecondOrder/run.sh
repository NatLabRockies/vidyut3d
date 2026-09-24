#!/usr/bin/env bash
# Convergence ladder for the Laplace annulus case.
#
# set -e matters here: without it a failed grid is followed by later
# iterations and by a successful `cd ..`, so the script exits 0 and leaves a
# ladder that looks complete but has a missing or stale rung.
set -e

export FI_PROVIDER=tcp

INNER_DIRICHLET=1
OUTER_DIRICHLET=1

for DIM in 16 32 64 128 256 
do
    mkdir -p "${DIM}"

    cd "${DIM}"

    rm -rf plt* chk*
    if ! mpirun -np 1 ../vidyut2d.llvm.MPI.ex ../inputs2d max_step=10 \
            amr.n_cell="${DIM}" "${DIM}" 1; then
        echo "grid ${DIM} failed" >&2
        exit 1
    fi

    cd ..
done

for DIM in 512 1024 
do
    mkdir -p "${DIM}"

    cd "${DIM}"

    rm -rf plt* chk*
    mpirun -np 4 ../vidyut2d.llvm.MPI.ex ../inputs2d max_step=10 amr.n_cell="${DIM}" "${DIM}" 1 
    #ls -1v plt*/Header | tee movie.visit

    cd ..
done

for DIM in 2048
do
    mkdir -p "${DIM}"

    cd "${DIM}"

    rm -rf plt* chk*
    mpirun -np 8 ../vidyut2d.llvm.MPI.ex ../inputs2d max_step=10 amr.n_cell="${DIM}" "${DIM}" 1 
    #ls -1v plt*/Header | tee movie.visit

    cd ..
done



