# Paper cases

The cases that produced the results of the second-order immersed-boundary
paper, so that its tables and figures can be rebuilt with this version of
Vidyut. Each folder holds the sources, decks, run scripts and post-processing
scripts of one case and a `README.md` with the exact commands and the paper's
numbers.

These cases are not part of the regression tests in `test/` and are not run
in CI. They build like any other case: `VIDYUT_DIR` defaults to the
repository root and `AMREX_HOME` to `Submodules/amrex`.

The wall reconstruction everywhere, unless stated, is the scaled quadric fit
with a Newton closest-point projection and the offset term
$w = c_0 + c_1 u + a u^2/2$ (`prob.eb_geom_method=4`,
`prob.eb_quadric_offset=1`). It is chosen at run time, and every log starts
with a line `IB reconstruction: ...` that records it; do not change it by
editing `Prob.H`. The other methods are 1 PCA plane + quadric, 2 PCA tangent
plane, 3 analytical geometry, 5 nearest cut cell (IB-NG).

## Where each result comes from

| Paper item | Folder | Deck and switches | Script |
|---|---|---|---|
| Laplace, Fig. 2(a), Table 2 (Dirichlet / Neumann / Robin, G1 / G2) | `Laplace_2D` | `./run.sh dirichlet neumann robin`; G1 = `prob.ib_second_order=0`; 10 steps = 10 correction sweeps | `error_plot_bc.py`, `potential_error_plot.py` |
| Laplace AMR, Sec. 4.1.1, Fig. 2(b), Table 3 | `Laplace_2D` | `./run_amr_convergence.sh 2`: 32+4, 64+3, 128+2, 256+1 levels, `amr.n_error_buf=8` (and 2), `vidyut.refine_cutcells=1` | `amr_error.py --norm domain`, `error_plot_bc.py` |
| Reconstruction comparison, Table 4 | `Laplace_2D` | `./run_methods.sh`: `prob.eb_geom_method=1..5`, Neumann inner wall | all-cell mean of the masked squared error |
| Reconstruction accuracy table (foot, normal) | `Laplace_2D` | same runs; `EB_cp_*`, `EB_norm_*` in the plotfile, inner cylinder, cells within 1.6 h of the wall | -- |
| Case 2, MMS1 table and figures | `MMS1_annulus` | `./run.sh` | `mms1_error_plot.py -f results` |
| Case 3, MMS2 table and figures | `MMS2_annulus` | `./run.sh`; `vidyut.freeze_etemp=1` (in the deck) | `all_errors.py -f results`, `mms2_profiles.py` |
| Case 3, MMS2 IB-NG table | `MMS2_annulus` | `./run.sh ibng` (`prob.eb_geom_method=5`) | `all_errors.py -f ibng -p ibng_` |
| Case 4, convergence figure and configuration table | `Case4_composite` | `./run_paper.sh`: `fixes_off`, `fixes_on`, `gec3d_setting` (`prob.proj_switch=1`), `ray_perface` (`prob.perface_fallback=1 prob.ray_dirichlet=1`); `vidyut.freeze_etemp=1` | `convergence_figure_paper.py`, `get_error.py` |
| Case 4, Fig. 6 (geometry, potential at 256) | `Case4_composite` | `runs/ray_perface/256` | `case4_figure_paper.py`, `potential_contour_paper.py` |
| AMR convergence check, Laplace / MMS2 / Case 4 | `Laplace_2D`, `MMS2_annulus`, `Case4_composite` | `run_amr_convergence.sh`: base 32/64/128, `amr.max_level=2`, `amr.n_error_buf` 6/12/24, `vidyut.refine_cutcells=1` | `amr_error.py` |
| Case 5, table `tab:mms1_3d_lin` and figure | `Case5_sphere3D` | `./run.sh runs_lin "32 64 128 256" prob.ray_dirichlet=5`; ray value `prob.ray_dirichlet=1`; 512^3 `run512.slurm` | `sphere_errors.py`, `sphere_error_plot.py` |
| Case 5, reconstruction accuracy and "within 2 %" | `Case5_sphere3D` | `runs_m4`, `runs_exact` (`prob.eb_geom_method=3`), `runs_ibng` (`=5`) | `recon_accuracy.py`, `sphere_errors.py` |
| GEC validation, Sec. 5 (`sec:gec`, `sec:gec_order`) | `GEC_axisymmetric` | `inputs`, `prob.wall_extrap` = 0 / 1 / 2, 256 x 256 r-z, about 210 RF cycles | `gec_paper_figures.py`, `compare_modes.py` |
| Pin-to-pin, Sec. 5 (`sec:pin`) | `PinToPin` | `inputs2d_r0p8`, `inputs3d_r0p8`, `inputs3d_array_sR{06,12,25}` | `check_pin_geometry.py`, `compare_runs.py` |

## Notes that change answers

- **Norm.** In Cases 1 to 3 and 5 the paper's $L_2$ is the mean over every
  cell of the domain, solid included, of the masked squared error, with fluid
  = `cellmask > 1 - 1e-10` (`cellmask >= 1` in the Laplace scripts, which
  selects the same cells there). It is smaller than an $L_2$ over fluid cells
  by the square root of the fluid fraction: about 2.0x for Laplace, 2.4x for
  MMS1 and MMS2, 2.2x for Case 5. Case 4 uses errors relative to the largest
  exact value, over fluid cells (`get_error.py`). The MMS2 and Case 4
  `amr_error.py` use the volume-weighted fluid $L_2$ divided by the largest
  exact value; the Laplace `amr_error.py` takes `--norm fluid` (default) or
  `--norm domain` (the paper's norm).
- **Case 4 switches.** `prob.perface_fallback=1 prob.ray_dirichlet=1
  prob.corner_fallback=1`. Without the first two, $L_2$ looks about 20 %
  worse but $L_\infty$ grows about 14x and stops converging. The offset fit is
  meant to be used with `prob.corner_fallback=1`.
- **Frozen electron temperature.** The MMS decks with a plain manufactured
  electron energy (MMS2, Case 4) need `vidyut.freeze_etemp=1`.
- **AMR.** Every coarse-fine interface must stay at least two coarse cells
  from a wall; scale `amr.n_error_buf` with the base grid. With more than one
  level the solver uses `vidyut.ib_identity_rows=1` by itself. The AMR runs
  need the per-level EB rebuild, `vidyut.refine_cutcells` and
  `vidyut.ib_identity_rows` of this branch, and `cellmask` carrying the volume
  fraction (done in each case's `Prob.H`).

## Building

```
cd papers/<case>
make -j 4 COMP=llvm USE_MPI=TRUE        # AMREX_HOME defaults to Submodules/amrex
```

`PinToPin` builds with hypre by default (`HYPRE_DIR=...`, or `USE_HYPRE=FALSE`).
`Case5_sphere3D` builds in 3D, `GEC_axisymmetric` without EB.

## Data and acknowledgements

No case reads an external data file at run time. The GEC figures compare
with two digitized curves already in `test/verification/GEC_RF_Cell`
(Overzet and Hopkins, Appl. Phys. Lett. 63, 1993; Boeuf and Pitchford,
Phys. Rev. E 51, 1376, 1995), and its electron transport is a fit of a
BOLSIG+ run (Hagelaar and Pitchford, Plasma Sources Sci. Technol. 14, 722,
2005); see `GEC_axisymmetric/README.md`. We thank these authors. The GEC,
pin-to-pin and 512^3 Case 5 runs were made on the NREL Kestrel cluster; no
plotfiles are redistributed.
