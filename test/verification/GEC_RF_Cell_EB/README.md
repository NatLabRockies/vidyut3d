# Gaseous Electronics Conference Radio-frequency cell

This case simulates GEC RF cell at 100 mTorr with argon 
plasma chemistry. This run takes about 4 mins to do
a single RF cycle with 64 processors. To get to steady state, 
about 500 cycles is required. This case uses the
cell masking feature in the 2D axisymmetric mode.

### Build instructions

make sure $AMREX_HOME is set to your clone of amrex
`$ export AMREX_HOME=/path/to/amrex`

If you are copying this case folder elsewhere then
make sure $VIDYUT_DIR is set to your clone of vidyut
`$ export VIDYUT_DIR=/path/to/vidyut`

To build a serial executable with gcc do
`$ make -j COMP=gnu`

To build a serial executable with clang++ do
`$ make -j COMP=llvm`

To build a parallel executable with gcc do
`$ make -j COMP=gnu USE_MPI=TRUE`

To build a parallel executable with gcc, mpi and cuda
`$ make -j COMP=gnu USE_CUDA=TRUE USE_MPI=TRUE`

For higher grid resolutions than the default used in this casae, 
AMReX's native MLMG solvers fail to converge. 
For these cases, vidyut has to be built with hypre.
Follow instructions here to get hypre installed - https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#external-solvers
After that do
`$ make -j COMP=gnu USE_MPI=TRUE USE_HYPRE=TRUE HYPRE_DIR=/path/to/hypre`

### Run instructions

Run with inputs2d 
`$ mpirun -n 64 ./*.ex inputs2d`

To use hypre for linear solves, do
`$ mpirun -n 64 ./*.ex inputs2d vidyut.use_hypre=1 vidyut.linsolve_max_coarsening_level=0`

Figure shows electron density contours and distribution along a radial line compared with experiments.

<img src="https://github.com/user-attachments/assets/d676a3f8-8fdd-4301-9390-e65ce8fd7af9" width=400>
<img src="https://github.com/user-attachments/assets/967a3a0d-b90c-4a99-be85-3df1b6da5919" width=400>
