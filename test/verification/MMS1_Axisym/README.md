# Method-of-Manufactured solutions (MMS) test 1

This case solves a coupled system that includes the species-Poisson numerical 
coupling in the plasma fluid model (in axi-symmetric coordinates)

$$\frac{dn}{dt}+\frac{d (\mu n E)}{dx}+\frac{d (\mu n E)}{dy}=\frac{d}{dx}\left(D\frac{dn}{dx}\right) + \frac{d}{dy}\left(D\frac{dn}{dy}\right) + F(r) $$
$$\frac{d^2\phi}{dx^2}+\frac{d^2\phi}{dy^2}=n $$
$$\mu=-1 \quad D=1 \quad E=-\frac{d\phi}{dx}-\frac{d\phi}{dx}$$

The exact solution to the solved system is:

$$n=16r^2 \quad \phi=r^4 \quad r=\sqrt{x^2+y^2} \quad F(r)= - \left(384\mu r^4+64D\right)$$

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

Use the `run.sh` script to run an array of cases with Dirichlet and Neumann boundary conditions.
You will need the `fextract` executable from amrex, which can be built from within 
https://github.com/AMReX-Codes/amrex/tree/development/Tools/Plotfile

Alternatively, you can just do `mpirun -n 1 ./*.ex inputs2d`


