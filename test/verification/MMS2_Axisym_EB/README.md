# Method-of-Manufactured solutions (MMS2) Axisymmetric + EB

This case solves all equations in the plasma model with MMS source terms in axi-symmetric form:

$$\frac{dn_e}{dt}+\frac{d (\mu n_e E)}{dx}+\frac{d (\mu n_e E)}{dy}=\frac{d}{dx}\left(D_e\frac{dn_e}{dx}\right) + \frac{d}{dy}\left(D_e\frac{dn_e}{dy}\right) + k_i n_e +S(n_e) $$
$$\frac{dn_i}{dt}+\frac{d (\mu_i n_i E)}{dx}+\frac{d (\mu_i n_i E)}{dy}=\frac{d}{dx}\left(D_i\frac{dn_i}{dx}\right) + \frac{d}{dy}\left(D_i\frac{dn_i}{dy}\right) + k_i n_e +S(n_i) $$

$$\frac{d^2\phi}{dx^2}+\frac{d^2\phi}{dy^2}=\frac{e(n_e-n_i)}{\epsilon_0} \quad E=-\nabla\phi$$
$$\frac{dE_e}{dt}+\frac{d (\mu_e E_e E)}{dx}+\frac{d (\mu_e E_e E)}{dy}-\frac{d}{dx}\left(D_e\frac{dE_e}{dx}\right) - \frac{d}{dy}\left(D_e\frac{dE_e}{dy}\right)=-e \Gamma_e E - k_i n_e E_i - \frac{3}{2} n_e k_B T_e \nu \frac{2 m_e}{m_h} + S(E_e)$$

These values are assumed for various reaction and transport parameters where e and $$m_p$$ are
electronic charge and proton mass, respectively:

$$k_i=5.0~\exp\left(-\frac{2.0}{k_BTe}\right)$$
$$\mu_e=-1.0 \quad D_e=1.0 \quad \mu_i=0.5 \quad D_i=0.5$$
$$E_i=\frac{4}{e} \quad \nu=10000.0 \quad m_h=4 m_p$$

We assume the exact solution to this system is:

$$n_e=\frac{x^2+y^2}{\alpha} + n_0$$
$$n_i=\frac{x^2+y^2}{2.0 \alpha}+n_0$$
$$\phi=\frac{1}{32}\left(x^2+y^2\right)^2$$
$$E_\epsilon=\frac{x^2+y^2}{\alpha}+n_0$$
$$alpha=\frac{e}{\epsilon_0}$$
$$n_0=10^6$$

We then compute the sources ($$S(n_e),S(n_i), S(E_e)$$)
for each of the equations by substituting the exact solution.

All boundary conditions in this case are Dirichlet type with the
boundary values directly computed from the exact solution.

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


