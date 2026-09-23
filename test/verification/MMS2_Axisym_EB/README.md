# MMS2 on an annulus with embedded-boundary geometry

Fully coupled three-species plasma model (electrons, HEp ions, electron energy
and the potential) on the annulus $0.5 \le r \le 1.5$ inside a
$[-3,3]^2$ domain, with the walls carried as an embedded boundary. The
manufactured solution is

$$\phi = \frac{r^4}{32}, \qquad
  n_e = E_e = \frac{r^2}{\alpha} + n_0, \qquad
  n_i = \frac{r^2}{2\alpha} + n_0,$$

with $\alpha = 1.809512801\times10^{-8}$ and $n_0 = 10^6$.

This is the case that produced the MMS2 results of the paper. The geometric
reconstruction is selected at run time; the defaults here are
`prob.eb_geom_method=4` (scaled quadric fit with the offset and a Newton
closest-point projection) and `prob.eb_quadric_offset=1`. Every log starts with
a line

```
IB reconstruction: prob.eb_geom_method = 4 (scaled quadric + Newton), prob.eb_quadric_offset = 1
```

so a plotfile can always be traced back to the method that produced it. Do not
switch methods by editing `Prob.H`; pass `prob.eb_geom_method=1..5` instead
(1 PCA plane + quadric, 2 PCA tangent plane, 3 analytical annulus,
4 scaled quadric + Newton, 5 nearest cut cell / IB--NG).

### Build

```
export AMREX_HOME=/path/to/amrex        # if not using the submodule
make -j COMP=llvm USE_MPI=TRUE
```

`USE_EB=TRUE` is already set in the `GNUmakefile`.

### Run and check

```
./run.sh
python3 all_errors.py -f .
```

`all_errors.py` masks to the fluid with `cellmask > 1 - 1e-10` and normalizes
the $L_2$ norm by the whole domain. A looser mask threshold lets in the cut
cells that the solver masks out and gives a meaningless error.

### Expected results

$L_2$ errors and convergence rates, method 4 with the offset:

| $N_x$ | $\phi$ | $p$ | $n_e$ | $p$ | $n_i$ | $p$ | $E_e$ | $p$ |
|---|---|---|---|---|---|---|---|---|
| 32  | 8.65e-04 | --   | 1.37e+05 | --   | 8.06e+04 | --   | 1.37e+05 | --   |
| 64  | 3.06e-04 | 1.50 | 2.84e+04 | 2.27 | 2.87e+04 | 1.49 | 2.84e+04 | 2.27 |
| 128 | 8.42e-05 | 1.86 | 5.86e+03 | 2.28 | 9.45e+03 | 1.60 | 5.86e+03 | 2.28 |
| 256 | 2.22e-05 | 1.92 | 1.61e+03 | 1.87 | 2.79e+03 | 1.76 | 1.61e+03 | 1.87 |
| 512 | 5.99e-06 | 1.89 | 4.65e+02 | 1.79 | 7.69e+02 | 1.86 | 4.65e+02 | 1.79 |

### Adaptive mesh refinement

`cellmask` carries the volume fraction, so the cut cells can be tagged directly
and the refined band follows the wall:

```
mpirun -np 4 ./*.ex inputs2d amr.max_level=1 amr.n_error_buf=8 amr.blocking_factor=8 \
    vidyut.refine_cutcells=1 \
    vidyut.use_hypre=1 vidyut.linsolve_max_coarsening_level=0
```

**This does not run to completion yet.** The hierarchy is built and the
potential solve converges on it (21 `hypre` iterations, relative residual
$6\times10^{-13}$), but the implicit species solve stops at a relative residual
of $6\times10^{-5}$ after 1000 iterations and aborts in `ScalarSolve.cpp`.

What the failure needs, established by elimination:

| varied | result |
|---|---|
| `hypre` on or off, `linsolve_max_coarsening_level` 0 or 10 | fails either way |
| `linsolve_reltol` 1e-12, 1e-8, 1e-6 | fails either way, residual floors at 6.9e6 |
| cut-cell or mask-gradient refinement | fails at the same line |
| same case at `amr.max_level=0` | each species converges in **3** iterations |
| planar immersed boundary instead of a curved one (`MMS2`, `inputs_x`) | 200 steps, species in 6--7 iterations |
| potential alone on this geometry (`Laplace_2D_SecondOrder`) | converges, 20--24 iterations |
| refined region covering the whole domain, no coarse--fine interface | 720 steps, species in 9--10 iterations |

So it takes all three of a curved wall, a coarse--fine interface and the species
equation; any two of them are fine. Widening the refined band does not help
until the band covers everything, at which point there is no interface left and
the hierarchy is a uniform fine grid. Forcing the levels to agree about which
cells are active does not help either, in either direction: switching the
covered coarse cut cells on destroys the solution, because the wall closure is
keyed to `cellmask` and not to the solver mask, and switching the fine cells
under them off leaves the residual floor exactly where it was.

The uniform-grid results above are unaffected.
