# Laplace solve with axisymmetric coordinate system

This case tests the Laplace equation solution in axisymmetric coordinate system with a robin boundary condition.

$$\nabla^2\phi=0 \quad \frac{d^2\phi}{dr^2}+\frac{1}{r}\frac{d\phi}{dr}=0$$
$$\phi(r=R_{min})=\phi_1 \quad \phi(r=R_{max})=\phi_2$$

The exact solution for this equation is

$$\phi(r)=\frac{1}{\log\left(\frac{R_{max}}{R_{min}}\right)}\left(\phi_2 \log\left(\frac{r}{R_{min}}\right) + \phi_1 \log\left(\frac{R_{max}}{r}\right)\right)$$

The robin BC is implemented as follows: 

At the left IB interface, the flux is given by 

$$ \frac{d\phi}{dn}\right|_{i-1/2} = \frac{\phi_{c} - \phi_{IB}}{dx_i} $$ 

The generic boundary condition iin each direction is given by 

$$ a \phi_{IB} + b \frac{\phi_c - \phi_{IB}}{d}\frac{d_i}{d} = f_i $$ 

This expression for $\phi_{IB}$ is plugged into the fluxes and we solve for the potential. The method is first-order accurate. 

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


