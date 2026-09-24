# Laplace solve on a Cartesian grid, with EB support

This case tests the Laplace equation on a **Cartesian** grid
(`geometry.coord_sys = 0`) with a **Dirichlet** condition on both immersed
walls. The wall condition is applied through the generic Robin form below, but
`inputs2d` sets `a = 1`, `b = 0` on each wall, which reduces it to pure
Dirichlet; set `prob.b_inner_neumann` or `prob.b_outer_neumann` non-zero to
exercise the Robin path. The point of the case is the use of EB to locate a
point on the surface.

The geometry is an annulus, so the solution depends on $r$ alone and satisfies
the radial Laplace equation below. That radial form is the same in two
Cartesian dimensions and in axisymmetric coordinates, so the exact solution
looks axisymmetric even though the operators being tested are Cartesian.

$$\nabla^2\phi=0 \quad \frac{d^2\phi}{dr^2}+\frac{1}{r}\frac{d\phi}{dr}=0$$
$$\phi(r=R_{min})=\phi_1 \quad \phi(r=R_{max})=\phi_2$$

The exact solution for this equation is

$$\phi(r)=\frac{1}{\log\left(\frac{R_{max}}{R_{min}}\right)}\left(\phi_2 \log\left(\frac{r}{R_{min}}\right) + \phi_1 \log\left(\frac{R_{max}}{r}\right)\right)$$

The robin BC is implemented as follows:

At the left IB interface, the flux is given by

$$ \frac{d\phi}{dn}|_{i-1/2} = \frac{\phi_{c} - \phi_{IB}}{dx_i} $$

The generic boundary condition iin each direction is given by

$$ a \phi_{IB} + b \frac{\phi_c - \phi_{IB}}{d}\frac{d_i}{d} = f_i $$

This expression for $\phi_{IB}$ is plugged into the fluxes and we solve for the
potential. This first-order closure ($G_1$) is what the expression above gives;
the second-order closure used for the convergence results below is described in
the paper.

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

## AMR convergence study

`inputs2d_amr` refines on the EB cut cells (`vidyut.refine_cutcells=1`), so the
refined region is fixed by the geometry and does not move over the run. The study
below checks that refining this way does not cost the scheme its order.

Build with hypre and run as in `test/verification/GEC_RF_Cell/README.md`:

```
make -j COMP=llvm USE_MPI=TRUE USE_HYPRE=TRUE HYPRE_DIR=/path/to/hypre
mpirun -np 4 ./vidyut2d.llvm.MPI.ex inputs2d_amr \
    vidyut.use_hypre=1 vidyut.linsolve_max_coarsening_level=0
```

hypre must be built **without** `--enable-mixedint`: AMReX's `Src/Extern/HYPRE`
assumes `HYPRE_Int == HYPRE_BigInt` and will not compile against a mixedint build
(the Homebrew package is one).

The solution is steady, so `max_step=1` is enough -- plt00001 and plt00002 agree to
every digit. Post-process with `amr_error.py`, not `potential_error_plot.py`: the
latter uses `ds.all_data()`, which gathers coarse cells that a finer level covers and
so double-counts them on an AMR hierarchy.

```
python3 amr_error.py -f uni32 uni64 uni128 uni256 uni512 amr64L2 amr128L2 --plot conv.png
```

### phi error, uniform grid

| N | L2 | rate | Linf | rate |
|---|---|---|---|---|
| 32 | 7.670e-2 | -- | 1.555e-1 | -- |
| 64 | 1.743e-2 | 2.14 | 5.434e-2 | 1.52 |
| 128 | 5.089e-3 | 1.78 | 1.613e-2 | 1.75 |
| 256 | 1.312e-3 | 1.96 | 4.426e-3 | 1.87 |
| 512 | 3.092e-4 | 2.08 | 1.093e-3 | 2.02 |

### The order is held at the immersed boundary

`amr_error.py` splits the error by where it sits. Restricted to the cells the
boundary passes through, every path to an effective 512 grid agrees to within 1 %:

| | uniform 512 | 256 + 1 level | 128 + 2 levels, buf 8 | 128 + 2 levels, buf 2 |
|---|---|---|---|---|
| IB-zone Linf | 1.093e-3 | 1.102e-3 | 1.091e-3 | 1.105e-3 |
| IB-zone L2 | 3.528e-4 | 3.535e-4 | 3.524e-4 | 3.538e-4 |

and the IB-zone L2 rate over the AMR sequence (base 32, 64, 128 with two levels) is
1.87 then 2.10. AMR does not touch the closure.

### The coarse-fine interface is what limits the composite error

With a thin buffer the largest error is not at the immersed boundary but at the
coarse-fine interface. `amr.n_error_buf` controls it (128 base, 2 levels, effective 512):

| n_error_buf | cells | phi L2 | phi Linf | \|E\| L2 | CF-zone Linf |
|---|---|---|---|---|---|
| 2 | 84672 | 4.370e-4 | 1.502e-3 | 3.472e-2 | 1.502e-3 |
| 4 | 114880 | 2.911e-4 | 1.444e-3 | 2.228e-2 | 1.444e-3 |
| 8 | 160512 | 2.759e-4 | 1.091e-3 | 1.617e-2 | 4.599e-4 |
| *uniform 512* | *262144* | *3.092e-4* | *1.093e-3* | *1.107e-2* | -- |

At `n_error_buf=8` the coarse-fine error falls below the boundary error, the maximum
moves back to the boundary, and the composite matches uniform 512 in Linf with 40 %
fewer cells. **Use `amr.n_error_buf >= 4`, preferably 8.**

### Linear solver

The overset mask limits how far MLMG can coarsen, and AMReX takes that limit from
level 0 only, so the native solver degrades badly once AMR levels are added. hypre
fixes it. Same binary, 4 MPI ranks, first potential solve:

| case | hypre iters | hypre s | native iters | native s |
|---|---|---|---|---|
| uniform 512 | 3 | 0.37 | 3 | 1.67 |
| 128 + 2 levels | 24 | 0.37 | 185 | 13.9 |
| 128 + 2 levels, buf 8 | 24 | 0.27 | 176 | 34.9 |
| 256 + 1 level | 20 | 0.30 | 765 | 332.3 |

hypre iteration counts are flat in resolution; the native counts grow roughly with N.
Both reach the same discrete solution (errors agree to <= 2e-3 relative, far below the
discretization error). Without hypre the default `vidyut.linsolve_maxiter=100` aborts
for base >= 128 with levels -- it aborts rather than returning a half-converged answer,
so a convergence table cannot be silently contaminated.
