#!/bin/bash
#SBATCH --job-name=sphere512
#SBATCH --output=sphere512.%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=08:00:00
#SBATCH --account=<allocation>
#SBATCH -p standard

# MMS1 on the spherical shell (paper Case 5) at 512^3: ~134 M cells, ~105 GB in total.
# Two runs, one after the other: linearized wall value (paper table) and ray value.
# Build first in this folder:  make -j 16   (DIM = 3 in the GNUmakefile)
# Load the compiler and MPI modules of your cluster here, e.g.
# module load PrgEnv-gnu

EXE=$(ls $PWD/vidyut3d.*.ex | head -1)
COMMON="amr.n_cell=512 512 512 amr.max_grid_size=64 amr.blocking_factor=16 amr.chk_int=100000"

mkdir -p runs_lin/512 runs_m4/512
( cd runs_lin/512 && srun -N2 -n128 $EXE ../../inputs3d $COMMON prob.ray_dirichlet=5 > log 2>&1 )
( cd runs_m4/512  && srun -N2 -n128 $EXE ../../inputs3d $COMMON prob.ray_dirichlet=1 > log 2>&1 )

# errors (needs python with yt; ~20 GB for the 512^3 plotfile) -- or copy plt00001 back
# python3 sphere_errors.py runs_lin runs_m4
