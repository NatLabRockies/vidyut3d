# Near wall non-equilibrium physics in an arc discharge

This case simulates the near wall non-equilibrium region in an 
arc discharge close to the anode. The LTE plasma parameters from an 
arc simulation is used to initialize the arc edge boundary conditions
and the near wall sheath formation is simulated. The plasma densities 
at the arc edge are quite high and therefore the resolutions requirements 
are quite high. Notice, how we use 8192 cells to resolve 500 um.

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

### Run instructions

This is quite a long 1D run. The current at the anode 
can be tracked through the integrated_currents.0 file.
It needs about 3-4 days to get to finish.

### Literature

More information about this case can be obtained in our paper:
H Sitaramanet al. "Elucidating key reducing species beyond ions in hydrogen plasma smelting reduction of iron ore." 
Chemical Engineering Science (2026): 124377. See https://www.sciencedirect.com/science/article/pii/S0009250926010924
