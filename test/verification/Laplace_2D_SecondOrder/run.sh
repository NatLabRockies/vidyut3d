#!/usr/bin/env bash

export FI_PROVIDER=tcp mpirun

INNER_DIRICHLET=1
OUTER_DIRICHLET=1

for DIM in 16 32 64 128 256 
do
    mkdir -p "${DIM}"

    cd "${DIM}"

    rm -rf plt* chk*
    mpirun -np 1 ../vidyut2d.llvm.MPI.ex ../inputs2d max_step=10 amr.n_cell="${DIM}" "${DIM}" 1 
    #ls -1v plt*/Header | tee movie.visit

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



