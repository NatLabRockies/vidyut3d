# Laplace equation on an annulus (paper Case 1)

## Purpose

Case 1 of the paper. The potential solves the Laplace equation between
two cylinders, $R_{min}=0.1 \le r \le R_{max}=0.2$, inside the periodic box
$[-0.3,0.3]^2$. Both walls are immersed boundaries carried as an AMReX
embedded boundary (EB). The exact solution is

$$\phi(r)=\frac{\phi_2 \log(r/R_{min}) + \phi_1 \log(R_{max}/r)}{\log(R_{max}/R_{min})},
\qquad \phi_1 = 10,\ \phi_2 = 20 .$$

The inner wall carries $a\phi + b\,\partial\phi/\partial n = f$ with $f$ from the
exact solution: Dirichlet ($b=0$), Neumann ($a=0$) or Robin ($a=b=1$),
selected with `prob.a_inner_dir` and `prob.b_inner_neumann`. The outer wall is
Dirichlet.

The case produced:

- Fig. 2(a) and Table 2: Dirichlet, Neumann and Robin inner walls, grids 32 to 512.
- Table 4 (reconstruction comparison): `prob.eb_geom_method = 1..5`, Neumann inner wall.
  In 2D, method 3 of the working copy that produced the paper looped over
  k = -1..1 and read outside the 2D arrays. This folder loops over k = 0 only.
  The fix changes the method-3 result (at N = 32 the potential differs by up
  to 1.3e-3 relative in 204 cells and the $\phi$ error by about 5 %), so the
  method-3 column of Table 4 should be rerun before it is quoted. Methods 4
  and 5 give bit-identical results at N = 32; methods 1 and 2 do not contain
  the changed loops (not rerun).
- The reconstruction accuracy table (foot and normal), read from `EB_cp_*` and
  `EB_norm_*` in the plotfiles, inner cylinder, cells within 1.6 h of the wall.
- Section 4.1.1, Fig. 2(b) and Table 3: adaptive mesh refinement (AMR).

**Wall closure.** Dirichlet: $G = |n_i|(\phi_{IB}-\phi_P)/d$, implicit.
Neumann and Robin: a first-order implicit term $G_1$ plus a deferred
correction $\omega[G_2(\phi^n) - G_1(\phi^n)]$, where $G_2$ is the face-centre
gradient of a weighted least-squares quadratic fitted to the lagged potential
in a $5^D$ block and to one boundary-condition row at the wall foot of each
cell. The correction changes only the right-hand side. Each time step is one
correction sweep; `max_step=10` is converged (10 and 30 sweeps differ by
0.2 % at N=128). `prob.ib_second_order=0` gives the first-order closure (G1 in
Table 2), `prob.ib_relax` is $\omega$ (default 1).

**Wall foot and normal.** `Prob.H` stores the foot and normal of every regular
cell next to the wall. `prob.eb_geom_method` selects the reconstruction:
1 PCA plane + quadric, 2 PCA tangent plane, 3 analytical annulus,
4 scaled quadric + Newton (default), 5 nearest cut cell (IB-NG).
`prob.eb_quadric_offset=1` (default) fits $w = c_0 + c_1 u + a u^2/2$ in
method 4. Every log starts with a line `IB reconstruction: ...` that records
the method.

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (2D, `USE_EB=TRUE`) |
| `Prob.H`, `ProbParm.H` | geometry, wall foot/normal reconstruction, problem inputs |
| `UserFunctions.H` | `bc_ib` wall closure (Dirichlet, Neumann/Robin with deferred correction) |
| `UserSources.H`, `Chemistry.H`, `Chemistry.cpp` | no sources, no chemistry |
| `inputs2d` | deck (Dirichlet on both walls) |
| `run.sh` | grid sequence 32 to 512 for the Dirichlet, Neumann and Robin inner wall |
| `run_methods.sh` | reconstruction comparison, `prob.eb_geom_method=1..5` |
| `run_amr_convergence.sh` | AMR convergence (part 1) and the hierarchies of Table 3 (part 2) |
| `error_plot_bc.py` | Fig. 2 (uniform curves and AMR points) |
| `potential_error_plot.py` | error plots for one boundary-condition set |
| `amr_error.py` | AMR-aware norms (valid region per level, volume weighted) |

## Data

None. The exact solution is analytical.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE              # AMREX_HOME defaults to Submodules/amrex
./run.sh dirichlet neumann robin              # -> <bc>/<n>/plt00001
python3 error_plot_bc.py                      # Fig. 2
cd dirichlet && python3 ../potential_error_plot.py -f . && cd ..
EXTRA="prob.ib_second_order=0" OUT=g1 ./run.sh neumann   # first-order closure (G1) -> g1/neumann/<n>
./run_methods.sh                              # Table 4 -> methods/m<k>/<n>
./run_amr_convergence.sh 1 2                  # AMR, see below
```

Every run uses `max_step=10`. The paper's norm is the mean over every cell of
the domain, solid included, of the masked squared error
(`np.sqrt(np.mean((phi - exact)**2 * mask))`, mask = `cellmask >= 1`).
`amr_error.py --norm domain` gives the same quantity and also handles AMR;
`--norm fluid` (the default) divides by the fluid area instead and is about
2.0x larger here.

AMR recipe (part 1 of `run_amr_convergence.sh`): base 32/64/128,
`amr.max_level=2`, `amr.n_error_buf` 6/12/24, `vidyut.refine_cutcells=1`,
10 steps, effective grids 128/256/512. Part 2 builds the four hierarchies of
Table 3 (32+4, 64+3, 128+2, 256+1 levels, `amr.n_error_buf=8`) and the
`amr.n_error_buf=2` rows. With more than one level the solver selects
`vidyut.ib_identity_rows=1` by itself (log: `IB solid cells: own row`).

## Expected Results

$L_2(\phi)$, all-cell mean, second-order closure, method 4 with the offset:

| $N_x$ | Dirichlet | $p$ | Neumann | $p$ | Robin | $p$ |
|---|---|---|---|---|---|---|
| 32  | 2.8652e-02 | --   | 8.5230e-02 | --   | 7.9392e-02 | --   |
| 64  | 8.2949e-03 | 1.79 | 8.7334e-03 | 3.29 | 8.0886e-03 | 3.30 |
| 128 | 2.3739e-03 | 1.80 | 2.3852e-03 | 1.87 | 2.2272e-03 | 1.86 |
| 256 | 6.4397e-04 | 1.88 | 4.5398e-04 | 2.39 | 4.2711e-04 | 2.38 |
| 512 | 1.5091e-04 | 2.09 | 9.2674e-05 | 2.29 | 8.7542e-05 | 2.29 |

AMR, part 1 (Dirichlet), measured on this branch:

| base + levels | effective $N_x$ | $L_2(\phi)$ domain | $L_2(\phi)$ fluid | $L_\infty(\phi)$ |
|---|---|---|---|---|
| 32 + 2, buf 6   | 128 | 2.374e-03 | 4.785e-03 | 1.714e-02 |
| 64 + 2, buf 12  | 256 | 6.440e-04 | 1.278e-03 | 4.790e-03 |
| 128 + 2, buf 24 | 512 | 1.509e-04 | 2.971e-04 | 1.168e-03 |

The domain-normalized column equals the uniform Dirichlet column above at the
same resolution. Table 3 of the paper (part 2): the four `buf 8` hierarchies
give $L_2 = 1.341\times10^{-4}$ and $L_\infty = 1.167\times10^{-3}$, against
1.509e-04 and 1.168e-03 for uniform 512. The paper runs used hypre; the native
solver agrees to about 2e-3 relative.

Checked for this folder: the 32 grid of all three boundary conditions and the
AMR rows above reproduce the printed digits.

## CTest

Not in CI.
