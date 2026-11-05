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

The command above already gets identity rows: the default is
`vidyut.ib_identity_rows = -1`, which resolves to 1 whenever `amr.max_level > 0`
(see below), and the run prints `IB solid cells: own row` to say so. Passing
`vidyut.ib_identity_rows=1` explicitly is an override, not a requirement. What
matters is that the overset-mask path must not be used here: with it the
implicit species solve stalls, and with identity rows the case runs to
completion, species solves taking 5 to 6 iterations.

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

The default is `vidyut.ib_identity_rows = -1`, which means *choose*: `Vidyut::ReadParameters`
selects identity rows whenever `amr.max_level > 0` and the overset mask
otherwise. So a single-level run gets the mask and an AMR run gets identity
rows without either being asked for, and the run prints which one it used
("IB solid cells: own row" or "overset mask"). Setting the option to 0 or 1
overrides that choice.

The two do not change the answer: on a uniform grid they agree to a relative
$10^{-10}$, which is solver noise. The table above was produced with the mask,
and the paper's results are unaffected either way.

| | $\phi$ | $n_e$ | $n_i$ | $E_e$ |
|---|---|---|---|---|
| uniform 64, mask | 3.06167e-04 | 2.83838e+04 | 2.87273e+04 | 2.83838e+04 |
| uniform 64, rows | 3.06167e-04 | 2.83838e+04 | 2.87273e+04 | 2.83838e+04 |
| 64 + 1 level, rows | 8.41571e-05 | 5.86154e+03 | 9.44585e+03 | 5.86154e+03 |
| uniform 128 | 8.41571e-05 | 5.86154e+03 | 9.44585e+03 | 5.86154e+03 |

The refined band covers the whole annulus, so the fluid ends up entirely at the
fine spacing and the hierarchy reproduces the uniform grid of that spacing to
every digit.

### Joule heating (`inputs2d_joule`)

The solver advances the electron energy as

$$\partial_t E_e = -\nabla\cdot\Gamma_\varepsilon + q\,\Gamma_e\cdot\mathbf{E} + S_\varepsilon,
\qquad q = -e,$$

where $\Gamma_\varepsilon = \tfrac53(\mu_e E_e \mathbf{E} - D_e\nabla E_e)$ and
$\Gamma_e = \mu_e n_e \mathbf{E} - D_e \nabla n_e$. The Joule term $q\,\Gamma_e\cdot\mathbf{E}$
is added in `Source/ElecEnergySource.cpp`. The manufactured source is therefore

$$S_\varepsilon = \nabla\cdot\Gamma_\varepsilon + e\,\Gamma_e\cdot\mathbf{E}.$$

`UserSources.H` used $-e\,\Gamma_e\cdot\mathbf{E}$ until 2026-11-03. With
`inputs2d` this makes no difference: there $E_e = n_e$, which corresponds to an
electron temperature of order $10^{18}$ eV, and the Joule term is $10^{-22}$ to
$10^{-19}$ of $\nabla\cdot\Gamma_\varepsilon$. The table above is unchanged by
the correction, and `inputs2d` does not test Joule heating at all.

`inputs2d_joule` sets `prob.een_scale` $= \tfrac32 e T_0$ with $T_0 = 0.1$ V, so that

$$E_e = \tfrac32 e T_0\, n_e, \qquad T_e = \tfrac23\frac{E_e}{k_B n_e} = 1160.45\ \mathrm{K}.$$

The ratio $|e\,\Gamma_e\cdot\mathbf{E}| / |\nabla\cdot\Gamma_\varepsilon|$ then
ranges from 0.016 at $r = 0.5$ through 0.29 at $r = 1$ to about 19 at $r = 1.5$,
where $\nabla\cdot\Gamma_\varepsilon$ tends to zero; $S_\varepsilon$ changes sign
near $r = 1.26$. The deck also backs $T_e$ out of
$E_e/n_e$ (`vidyut.freeze_etemp = 0`), solves the energy equation for
$E_e / $`vidyut.elecenergy_scale`, and keeps `vidyut.floor_jh = 0`, because
$\Gamma_e\cdot\mathbf{E} > 0$ here and the floor would clip the term to zero.
Everything else is the same as `inputs2d`.

| Input | Default | Effect |
|---|---|---|
| `prob.een_scale` | 1 | $E_e = $ `een_scale` $\cdot\, n_e$; must be $> 0$ (the run aborts otherwise) |

```
for N in 32 64 128; do mkdir -p j$N; (cd j$N && mpirun -np 1 ../*.ex ../inputs2d_joule amr.n_cell=$N $N 1); done
python3 check_errors.py j32/plt00001 j64/plt00001 j128/plt00001 --deck inputs2d_joule
```

`check_errors.py` reads the plotfiles with numpy only and uses the same mask and
normalization as `all_errors.py` (it reproduces the table above to every printed
digit). `--deck` reads `prob.een_scale` from the input file. `all_errors.py`
assumes $E_e = n_e$ and reports a meaningless energy error for a Joule run.
CI runs the 32 and 64 grids and requires an observed order of at least 1.8 for
$E_e$ and $T_e$. Relative $L_2$ errors:

| $N_x$ | $E_e$ | $p$ | $T_e$ | $p$ | $E_e$, old sign | $p$ |
|---|---|---|---|---|---|---|
| 32  | 4.640e-03 | --   | 5.027e-03 | --   | 1.872e-01 | --   |
| 64  | 1.167e-03 | 1.99 | 1.281e-03 | 1.97 | 1.776e-01 | 0.08 |
| 128 | 2.828e-04 | 2.05 | 2.597e-04 | 2.30 | 1.683e-01 | 0.08 |
| 256 | 6.581e-05 | 2.10 | 5.593e-05 | 2.22 | 1.635e-01 | 0.04 |

With the old sign the energy error stays near 18 % and does not converge.
$\phi$, $n_e$ and $n_i$ are the same as in `inputs2d`, since the energy does not
feed back into them in this case. Because $E_e = $ `een_scale` $\cdot\, n_e$, the
$E_e$ and $n_e$ equations share nearly the same operator and part of their
error cancels in $T_e$, so $E_e$ is the stronger of the two checks (the $n_e$
error is in the main table above).

The Joule source itself is plotted as `Electron_Jheat`. Its $L_\infty$ error
against $-e\,\Gamma_e\cdot\mathbf{E}$, relative to the largest exact value, is
about first order in the fluid cells next to the wall:

| $N_x$ | interior | wall-adjacent cells |
|---|---|---|
| 32  | 4.73e-02 | 3.94e-02 |
| 64  | 1.60e-02 | 3.16e-02 |
| 128 | 6.74e-03 | 1.99e-02 |
| 256 | 3.32e-03 | 1.14e-02 |

The cut-face branch of `Source/ElecEnergySource.cpp` uses the fluid-cell
density as the face value and a one-sided gradient at the face, and the face
field near the wall is first order. With Dirichlet energy walls an $O(1)$
source error confined to one cell layer changes $E_e$ only at $O(h^2)$, which
is why $E_e$ still converges at second order. This test therefore checks the
sign and presence of the Joule term, not its pointwise accuracy, which is
below second order even away from the wall in this table.
