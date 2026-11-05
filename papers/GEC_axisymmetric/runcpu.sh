#!/bin/bash
# Slurm job of the paper runs (one 64-core node). Set the account and
# partition for your cluster. The wall-flux mode is chosen with
# prob.wall_extrap: 0 first order, 1 linear, 2 logarithmic. Each mode runs in
# its own folder mode<N>/ so the plotfiles of different modes do not mix.
#   sbatch runcpu.sh 1
#SBATCH --job-name=GEC_Axi
#SBATCH --output=GEC_Axi.%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --account=<allocation>
#SBATCH -p standard

MODE=${1:-1}
# Load the compiler and MPI modules of your cluster here, e.g.
# module load PrgEnv-gnu
EXE=$(ls "$PWD"/vidyut2d.*.ex | head -1)
mkdir -p "mode${MODE}"
cd "mode${MODE}" || exit 1
srun -N1 -n64 "${EXE}" ../inputs prob.wall_extrap="${MODE}"
