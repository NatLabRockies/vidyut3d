# Coupled plasma system on an annulus, MMS2 (paper Case 3)

## Purpose

Case 3 of the paper. Electrons, HEp ions, electron energy and the
potential on the annulus $0.5 \le r \le 1.5$ inside $[-3,3]^2$, both walls
carried as an embedded boundary. Manufactured solution:

$$\phi = \frac{r^4}{32}, \qquad n_e = E_e = \frac{r^2}{\alpha} + n_0, \qquad
  n_i = \frac{r^2}{2\alpha} + n_0, \qquad \alpha = e/\epsilon_0,\ n_0 = 10^6 .$$

The case produced the MMS2 table and figures (method 4 with the offset fit, the
`Prob.H` defaults), the MMS2 IB-NG table (`prob.eb_geom_method=5`) and the MMS2
row of the AMR convergence check.

Two changes relative to the paper's working copy, neither of which changes a
printed number:

1. `UserSources.H`: the Joule term of the manufactured electron-energy source
   had the wrong sign. The solver adds $q\,\Gamma_e\cdot E$ with $q=-e$
   (`Source/ElecEnergySource.cpp`), so the source needs $+e\,\Gamma_e\cdot E$.
   The leading `-ECHARGE` is now `ECHARGE`, in the active and in the commented
   reaction-enabled block. At these scales the term is about $10^{-19}$ of the
   transport terms; the 32 grid reproduces the paper to all printed digits.
2. `Prob.H`: `cellmask` carries the volume fraction instead of 0/1, and boxes
   without cut cells are skipped when the EB centroids are read. Both are
   needed for `vidyut.refine_cutcells` and refined levels. The solver reads
   `int(cellmask)`, so the solved cells are unchanged.

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (2D, `USE_EB=TRUE`) |
| `Prob.H`, `ProbParm.H` | geometry, wall foot/normal reconstruction, exact initial state |
| `UserFunctions.H` | `bc_ib` wall closure and wall fluxes |
| `UserSources.H`, `Chemistry.H`, `Chemistry.cpp` | manufactured sources, species and transport |
| `inputs2d` | deck (720 steps, `vidyut.freeze_etemp=1`) |
| `run.sh` | grid sequence 32 to 512: `results/<n>` (method 4) or `ibng/<n>` |
| `run_amr_convergence.sh` | AMR runs, base 32/64/128 + 2 levels |
| `all_errors.py` | table and convergence figures |
| `mms2_profiles.py` | profile figures |
| `amr_error.py` | AMR-aware norms (relative, fluid cells, volume weighted) |

## Data

None. The exact solution is analytical.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE
./run.sh                            # NP=<ranks>, default 2 -> results/<n>
./run.sh ibng                       # IB-NG table -> ibng/<n>
python3 all_errors.py -f results
python3 all_errors.py -f ibng -p ibng_
(cd results && python3 ../mms2_profiles.py)
./run_amr_convergence.sh            # -> amr32 amr64 amr128
python3 amr_error.py amr32 amr64 amr128
```

Switches in `inputs2d`: `vidyut.freeze_etemp=1` (the manufactured electron
energy is a plain scalar, so $T_e$ must not be backed out of it; the paper
runs kept $T_e$ at its initial value). Both walls are Dirichlet, fixed by `bc_ib`. The `prob.*_voltage`, `prob.a_*_dir`, `prob.b_*_neumann` and `prob.*_dirichlet` keys that `Prob.H` reads are not used by this case and are not set in the deck; setting them changes nothing.
AMR: `amr.max_level=2`, `amr.n_error_buf` 6/12/24, `vidyut.refine_cutcells=1`,
720 steps.

Norms. `all_errors.py`: mean over every cell of the domain of the masked
squared error, fluid = `cellmask > 1 - 1e-10`. `amr_error.py`: error relative
to the largest exact value in the fluid, volume-weighted mean over fluid cells.

## Expected Results

$L_2$ errors, method 4 with the offset (`results`):

| $N_x$ | $\phi$ | $p$ | $n_e$ | $p$ | $n_i$ | $p$ | $E_e$ | $p$ |
|---|---|---|---|---|---|---|---|---|
| 32  | 8.646498e-04 | --   | 1.368288e+05 | --   | 8.062207e+04 | --   | 1.368288e+05 | --   |
| 64  | 3.061668e-04 | 1.50 | 2.838394e+04 | 2.27 | 2.872737e+04 | 1.49 | 2.838394e+04 | 2.27 |
| 128 | 8.415710e-05 | 1.86 | 5.861456e+03 | 2.28 | 9.445948e+03 | 1.60 | 5.861456e+03 | 2.28 |
| 256 | 2.217596e-05 | 1.92 | 1.607500e+03 | 1.87 | 2.791141e+03 | 1.76 | 1.607500e+03 | 1.87 |
| 512 | 5.992754e-06 | 1.89 | 4.652216e+02 | 1.79 | 7.695885e+02 | 1.86 | 4.652182e+02 | 1.79 |

IB-NG (`ibng`, `prob.eb_geom_method=5`):

| $N_x$ | $\phi$ | $p$ | $n_e$ | $p$ | $n_i$ | $p$ |
|---|---|---|---|---|---|---|
| 32  | 9.780010e-04 | --   | 5.586961e+05 | --   | 5.293989e+05 | --   |
| 64  | 6.923952e-04 | 0.50 | 2.477572e+05 | 1.17 | 2.253978e+05 | 1.23 |
| 128 | 2.525073e-04 | 1.46 | 8.721837e+04 | 1.51 | 7.306885e+04 | 1.63 |
| 256 | 1.092829e-04 | 1.21 | 3.567441e+04 | 1.29 | 2.819556e+04 | 1.37 |
| 512 | 7.147124e-05 | 0.61 | 2.229592e+04 | 0.68 | 1.756523e+04 | 0.68 |

AMR (`amr_error.py`, $\phi$): 1.4346e-03 (effective 128) | 3.5460e-04 (256) |
9.3343e-05 (512). `amr_error.py` uses a different norm from the tables above:
the volume-weighted $L_2$ over fluid cells only, divided by the largest exact
$\phi$. Converted to that norm, the uniform 512 value of the first table
(5.99e-06) is about 9.35e-05, so the AMR hierarchy matches the uniform grid of
the same finest spacing.

Checked for this folder: the 32 grid of `results` (2 ranks) reproduces all four
values to the printed digits, and `amr32` gives 1.4346e-03.

## CTest

Not in CI. `test/verification/MMS2_Axisym_EB` is a formatted copy of this case
for the repository's own verification set.
