#!/bin/bash
# Slurm job for the pin decks (NREL Kestrel). Set the account and partition
# for your cluster.
#   sbatch runcpu.sh inputs3d_array_sR12       (3D executable)
#   sbatch runcpu.sh inputs2d_r0p8             (2D r-z executable)
#SBATCH --job-name=PinToPin
#SBATCH --output=PinToPin.%j.out
#SBATCH --nodes=8
#SBATCH --time=192:00:00
#SBATCH --account=<allocation>
#SBATCH -p standard

DECK=${1:-inputs3d_r0p8}
case $DECK in
    inputs2d*) EXE=$(ls ./vidyut2d.*.ex | head -1) ;;
    *)         EXE=$(ls ./vidyut3d.*.ex | head -1) ;;
esac
# Load the compiler and MPI modules of your cluster here, e.g.
# module load PrgEnv-gnu
srun -N8 -n512 ${EXE} ${DECK}
