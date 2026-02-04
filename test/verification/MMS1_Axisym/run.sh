#!/usr/bin/env bash

export FI_PROVIDER=tcp mpirun

INNER_DIRICHLET=0
OUTER_DIRICHLET=1

for DIM in 16 32 64 
do
    mkdir -p "${DIM}"
    
    cd "${DIM}"
    
    rm -rf plt* chk*
    mpirun -np 1 ../vidyut2d.llvm.MPI.ex ../inputs2d  amr.n_cell="${DIM}" "${DIM}" 1 prob.outer_dirichlet=${OUTER_DIRICHLET} prob.inner_dirichlet=${INNER_DIRICHLET} 
    ls -1v plt*/Header | tee movie.visit
    
    cd ..
done

for DIM in 128 256 512 
do
    mkdir -p "${DIM}"
    
    cd "${DIM}"
    
    rm -rf plt* chk*
    mpirun -np 8 ../vidyut2d.llvm.MPI.ex ../inputs2d  amr.n_cell="${DIM}" "${DIM}" 1 prob.outer_dirichlet=${OUTER_DIRICHLET} prob.inner_dirichlet=${INNER_DIRICHLET} 
    ls -1v plt*/Header | tee movie.visit
    
    cd ..
done
