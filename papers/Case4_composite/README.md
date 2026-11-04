# Coupled plasma system in a composite geometry (paper Case 4)

## Purpose

Case 4 of the paper. The MMS2 system and manufactured solution
($\phi = r^4/32$, $n_e = E_e = r^2/\alpha + n_0$, $n_i = r^2/(2\alpha) + n_0$)
in a GEC-like geometry (`eb2.geom_type = gecmms`): an off-centre circular
chamber, a dome electrode with a stem, a rectangular electrode and flat top
and bottom plates. The geometry has curved walls, walls on grid lines, walls
off grid lines and convex, concave and obtuse corners. The case produced the
Case 4 convergence figure, the configuration table, Fig. 6 (geometry and
potential at $N_x=256$) and the Case 4 row of the AMR convergence check.

Four wall treatments are compared:

| config | switches |
|---|---|
| `fixes_off` | `prob.corner_fallback=0 prob.d_floor_frac=0 prob.face_fallback=0` |
| `fixes_on` | defaults: `prob.corner_fallback=1 prob.d_floor_frac=0.5 prob.face_fallback=1` |
| `gec3d_setting` | `fixes_on` + `prob.proj_switch=1` (isotropic 1/d coefficient) |
| `ray_perface` | `fixes_on` + `prob.perface_fallback=1 prob.ray_dirichlet=1` (paper's recommendation, `inputs2d` default) |

All four use the offset fit (`prob.eb_quadric_offset=1`). Use the offset fit
with `prob.corner_fallback=1` in production runs: without the corner fallback
some wall feet land up to 27 dx away at $N_x=32$, and `fixes_off` has a 2.8x
larger $\phi$ error there than `fixes_on`. The log line `IB reconstruction: ...`
prints the reconstruction, the offset switch and `prob.corner_fallback`.

Changes relative to the paper's working copy, none of which changes a printed
number:

1. `UserSources.H`: sign of the Joule term of the manufactured
   electron-energy source, `-ECHARGE` to `ECHARGE`, active and commented
   blocks (the solver adds $q\,\Gamma_e\cdot E$ with $q=-e$). The term is about
   $10^{-19}$ of the transport terms.
2. `Prob.H`: `cellmask` carries the volume fraction, and boxes without cut
   cells are skipped when the EB centroids are read; needed for
   `vidyut.refine_cutcells` and refined levels.
3. `get_error.py`: fluid = `cellmask > 1 - 1e-10` (was `> 0.5`), so that the
   norm ignores cut cells now that `cellmask` is the volume fraction. On 0/1
   masks the two are identical.

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (2D, `USE_EB=TRUE`) |
| `Prob.H`, `ProbParm.H` | composite geometry, wall foot/normal reconstruction, safeguards |
| `UserFunctions.H` | `bc_ib` wall closure and wall fluxes |
| `UserSources.H`, `Chemistry.H`, `Chemistry.cpp` | manufactured sources, species and transport |
| `inputs2d` | deck (720 steps, `ray_perface` switches, `vidyut.freeze_etemp=1`) |
| `run_config.sh` | one configuration over a list of grids into `runs/<config>/<n>` |
| `run_paper.sh` | the four configurations of the paper, every switch spelled out |
| `run_amr_convergence.sh` | AMR runs, base 32/64/128 + 2 levels |
| `get_error.py` | norms, overall and by geometric feature |
| `convergence_figure_paper.py` | convergence figure (reads `runs/<config>`) |
| `case4_figure_paper.py`, `potential_contour_paper.py` | Fig. 6 panels (`runs/ray_perface/256/plt00001`) |
| `amr_error.py` | AMR-aware version of the `get_error.py` norm |
| `pfread.py` | minimal plotfile reader used by the scripts |

## Data

None. The exact solution is analytical.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE
./run_paper.sh                      # all four configs, grids 32..512 (RES="32 64" to limit)
python3 get_error.py runs/ray_perface runs/fixes_on
python3 convergence_figure_paper.py
python3 case4_figure_paper.py runs/ray_perface/256/plt00001
./run_amr_convergence.sh            # -> amr32 amr64 amr128
python3 amr_error.py amr32 amr64 amr128
```

`run_config.sh` picks 2/4/8 ranks by grid size; set `EXE` to use another
executable. Required switches (in `inputs2d` and repeated in the scripts):
`prob.perface_fallback=1 prob.ray_dirichlet=1 prob.corner_fallback=1` and
`vidyut.freeze_etemp=1`. Without the three `prob` switches $L_2$ looks only
about 20 % worse, but $L_\infty$ grows about 14x and stops converging.

AMR: `amr.max_level=2`, `amr.n_error_buf` 6/12/24, `vidyut.refine_cutcells=1`,
720 steps. At base 128 the buffer must be at least 24: below that a few
coarse-fine interface cells sit one cell from a wall and $L_\infty$ is up to
60x worse. Vidyut warns when that happens (`vidyut.ib_cf_clearance_warn`).

Norm (`get_error.py`, `amr_error.py`): error relative to the largest exact
value in the fluid, over fluid cells. This differs from Cases 1 to 3.

## Expected Results

$\phi$, `ray_perface`:

| $N_x$ | $L_2$ | $p$ | $L_\infty$ | $p$ |
|---|---|---|---|---|
| 32  | 5.5415e-03 | --   | 1.2832e-02 | --   |
| 64  | 1.9256e-03 | 1.52 | 3.5811e-03 | 1.84 |
| 128 | 5.5085e-04 | 1.81 | 9.6288e-04 | 1.89 |
| 256 | 1.5829e-04 | 1.80 | 2.9013e-04 | 1.73 |
| 512 | 3.9736e-05 | 1.99 | 7.3539e-05 | 1.98 |

$\phi$, the other configurations, $L_2$ / $L_\infty$:

| $N_x$ | `fixes_off` | `fixes_on` | `gec3d_setting` |
|---|---|---|---|
| 32  | 2.5545e-02 / 3.0948e-01 | 9.0744e-03 / 5.8667e-02 | 3.8362e-02 / 8.9559e-02 |
| 64  | 2.4350e-03 / 2.7363e-02 | 2.4824e-03 / 2.8496e-02 | 1.7956e-02 / 4.6207e-02 |
| 128 | 6.6166e-04 / 1.4115e-02 | 6.5418e-04 / 1.3848e-02 | 8.5316e-03 / 2.4752e-02 |
| 256 | 1.7725e-04 / 6.9468e-03 | 1.7881e-04 / 6.8141e-03 | 4.0866e-03 / 1.2475e-02 |
| 512 | 4.6530e-05 / 3.4432e-03 | 4.7617e-05 / 3.3870e-03 | 2.0160e-03 / 6.2004e-03 |

`fixes_off` and `fixes_on` converge at second order in $L_2$ and first order
in $L_\infty$; `gec3d_setting` is first order in both.

AMR (`amr_error.py`, $\phi$ $L_2$): 5.5085e-04 (effective 128) | 1.5829e-04
(256) | 3.9736e-05 (512), equal to `ray_perface` at the same resolution.

Checked for this folder: the 32 grid of `ray_perface` and of `fixes_off`
(2 ranks) and `amr32` reproduce the printed digits of $\phi$, $n_e$ and $n_i$.

## CTest

Not in CI.
