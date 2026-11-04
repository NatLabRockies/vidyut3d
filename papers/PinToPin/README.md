# Pin-to-pin discharge with AMReX-EB geometry (paper Section 5, `sec:pin`)

## Purpose

The pin-to-pin application of Section 5 of the paper: argon
discharge between two arrays of paraboloidal pins, the bottom array powered
with the pulse of `get_applied_potential`, the top array grounded. The paper
compares

1. an axisymmetric single pin (`inputs2d_r0p8`, r-z, 400 x 800, 10 um) with
   the 3D single pin (`inputs3d_r0p8`, 256 x 256 x 384 on the 8 mm box), both
   with pin radius $r_p$ = 0.8 mm at the domain face (tip radius of curvature
   R = 0.16 mm), and
2. a 2x2 array of powered pins at the corners of a square of side s with one
   grounded pin on the axis, all $r_p$ = 0.8 mm, s/R = 6, 12, 25
   (`inputs3d_array_sR06`, `_sR12`, `_sR25`; s = 0.96, 1.92, 4.0 mm, 256^3).
   The single-pin runs are the isolated-pin reference ($s \to \infty$). At
   s/R = 6 the powered pins overlap (s < 2 $r_p$) and join 0.72 mm below the
   tips.

The electrodes are generated once with AMReX EB2 (`PinGeom.H`,
`Prob.H::init_level_with_eb`). The wall foot and normal of every fluid cell
next to an electrode are reconstructed with method 4 and the offset fit (the
reconstruction of Case 5) and stored in `EB_cp_*`/`EB_norm_*`; `bc_ib` reads
them and applies the directional closure. Two details in `Prob.H`:
`EB2::Build(..., extend_domain_face=false)` (the default clamps the implicit
function at the domain faces and biases the fit at the pin base and tip;
AMReX prints a warning for this setting), and cells that the surface touches
only at a node or an edge are re-flagged from cut to regular/covered before
the reconstruction (the log prints the count).

Run-time switches in the decks:

| switch | value in the paper decks (or default) | meaning |
|---|---|---|
| `prob.eb_geom_method` | 4 | 1 PCA + quadric, 2 PCA plane, 3 analytical paraboloid, 4 scaled quadric + Newton, 5 nearest cut cell |
| `prob.eb_quadric_offset` | 1 | offset term of the method-4 fit |
| `prob.sharp_interface_scheme` | 1 | 1 foot/normal closure, 0 wall on the grid face at dx/2 |
| `prob.perface_fallback` | 1 | masked neighbour's wall plane when the stored foot is not across the face |
| `prob.d_floor_frac` | 0.5 | floor on the wall distance in units of dx |
| `prob.wall_extrap` | 1 | wall state of the fluxes: 0 cell value, 1 linear, 2 logarithmic |
| `prob.wall_floor_frac` | 0.1 | lower bound of an extrapolated positive quantity |
| `vidyut.hyp_order` | 1 | first-order advection |
| `vidyut.use_hypre` | 1 | hypre for the potential solve (needs `USE_HYPRE=TRUE`) |

`prob.wall_extrap = 1` in `inputs2d_r0p8` and `inputs3d_r0p8` was set for
this folder from the notes on the paper runs; the working copies of these two
decks had 0. Check against the run logs before quoting.

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (`DIM=2` r-z or `DIM=3`, `USE_EB=TRUE`, `USE_HYPRE=TRUE`) |
| `PinGeom.H` | pin arrays as an EB2 implicit function, analytical foot/normal |
| `Prob.H`, `ProbParm.H` | EB build, foot/normal reconstruction, pin-array inputs |
| `UserFunctions.H` | `bc_ib`, applied potential, wall fluxes |
| `UserSources.H`, `Chemistry.H`, `Chemistry.cpp` | argon chemistry and transport |
| `inputs2d_r0p8`, `inputs3d_r0p8` | single pin, axisymmetric and 3D |
| `inputs3d_array_sR06`, `_sR12`, `_sR25` | 2x2 spacing study |
| `runcpu.sh` | Slurm job; argument = deck |
| `check_pin_geometry.py` | stored foot/normal vs analytical paraboloid |
| `compare_runs.py` | difference of two plotfiles over fluid cells |
| `pfread.py` | minimal plotfile reader used by the scripts |

Pin-array inputs: `prob.pin_axis`, `prob.nbot`/`prob.ntop` (1 to 32),
`prob.bot_centers`/`prob.top_centers`, `prob.bot_pinrad`/`prob.top_pinrad`
(default `prob.pinrad`), `prob.dgap` (tip-to-tip gap).

## Data and acknowledgements

No data file is read by the solver; the argon chemistry and transport are
written as coefficients in `Chemistry.H`/`Chemistry.cpp`. The source of
those rate and transport fits is not recorded in this case. The paper runs were
made on the NREL Kestrel cluster; their plotfiles are not redistributed.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE HYPRE_DIR=/path/to/hypre          # 2D r-z
make -j 4 COMP=llvm USE_MPI=TRUE HYPRE_DIR=/path/to/hypre DIM=3    # 3D
make -j 4 COMP=llvm USE_MPI=TRUE USE_HYPRE=FALSE                   # without hypre
sbatch runcpu.sh inputs2d_r0p8
sbatch runcpu.sh inputs3d_r0p8
sbatch runcpu.sh inputs3d_array_sR12
```

Without hypre set `vidyut.use_hypre=0`; the potential solve then needs the
bottom-solver inputs `vidyut.linsolve_bottom_maxiter=20000
vidyut.linsolve_bottom_reltol=1e-8` (r-z at 400 x 800). Geometry check
(2 steps): `mpirun -np 2 ./vidyut2d.*.ex inputs2d_r0p8 max_step=2
amr.plot_int=1`, then `python3 check_pin_geometry.py --inputs inputs2d_r0p8
plt00000`.

## Expected Results

Geometry check of the reconstruction (`check_pin_geometry.py`, fluid cells
with a masked face neighbour), from the case's development runs with
$r_p$ = 1 mm pins. These are not the paper decks: the check command above,
run on `inputs2d_r0p8` ($r_p$ = 0.8 mm, 400 x 800), gives its own numbers and
is not expected to reproduce this table.



| grid | h/R | wall cells | foot error mean / max ($h^2/R$) | normal error mean / max ($h/R$) |
|---|---|---|---|---|
| 2D 200 x 400 (h = 20 um) | 0.08 | 216 | 0.018 / 0.136 | 0.005 / 0.047 |
| 3D 64^3 (h = 125 um) | 0.5 | 1208 | 0.051 / 0.239 | 0.027 / 0.180 |

The discharge results of Section 5 were not final when this folder was made;
no reference numbers are given here.

Not run for this folder (application case; 2D and 3D builds checked only).

## CTest

Not in CI.
