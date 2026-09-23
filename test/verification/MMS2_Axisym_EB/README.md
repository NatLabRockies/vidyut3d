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

Add `vidyut.ib_identity_rows=1`. Without it the implicit species solve stalls;
with it the case runs to completion, species solves taking 5 to 6 iterations.

### Why the extra option is needed

AMReX applies an overset mask inside the operator but not in the coarse-fine
machinery: `MLCellLinOp::reflux` and `averageDownAndSync` never consult it. A
composite solve therefore refluxes a correction into masked coarse cells at the
coarse-fine interface, and the operator, which returns zero in a masked cell,
cannot remove it. The residual floors at a fixed value and the solve stops.

`vidyut.ib_identity_rows=1` avoids the mask altogether. The solid cells are kept
as ordinary unknowns and instead decoupled: every face coefficient around them
is zeroed, their right-hand side is zeroed, and the potential, whose `a`
coefficient is zero, is given a diagonal so the row is not empty. They solve to
zero and never reach the fluid, and with no mask the AMR machinery behaves
normally.

The option is off by default because it changes the linear system. It does not
change the answer: on a uniform grid the two agree to a relative $10^{-10}$,
which is solver noise. The table above was produced without it, and the paper's
results are unaffected either way.

| | $\phi$ | $n_e$ | $n_i$ | $E_e$ |
|---|---|---|---|---|
| uniform 64, mask | 3.06167e-04 | 2.83838e+04 | 2.87273e+04 | 2.83838e+04 |
| uniform 64, rows | 3.06167e-04 | 2.83838e+04 | 2.87273e+04 | 2.83838e+04 |
| 64 + 1 level, rows | 8.41571e-05 | 5.86154e+03 | 9.44585e+03 | 5.86154e+03 |
| uniform 128 | 8.41571e-05 | 5.86154e+03 | 9.44585e+03 | 5.86154e+03 |

The refined band covers the whole annulus, so the fluid ends up entirely at the
fine spacing and the hierarchy reproduces the uniform grid of that spacing to
every digit.
