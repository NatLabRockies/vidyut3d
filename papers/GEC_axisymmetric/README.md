# GEC reference cell, axisymmetric (paper Section 5, GEC validation)

## Purpose

The axisymmetric GEC RF reference cell of Section 5 of the paper
(`sec:gec`, `sec:gec_order`): argon, 100 mTorr (13.3 Pa), 300 K, 13.56 MHz,
100 V, six species (E, Ar, Ar+, Ar2+, Ar*, Ar2*) and the electron energy
equation, r-z grid 256 x 256 (dr = 0.397 mm). The electrodes are masked cells
(`vidyut.using_ib=1`); all walls lie on grid faces, so the cases compare wall
states rather than geometric reconstructions.

The wall state used in the electrode and domain-wall fluxes is selected with
`prob.wall_extrap`:

| `prob.wall_extrap` | wall state | plotfile set in the paper |
|---|---|---|
| 0 | cell value, first order | `axi2d` |
| 1 (default) | linear extrapolation $\phi_w = 1.5\phi_i - 0.5\phi_{nb}$, second order | `axi2d_2ndorder_mode1` |
| 2 | logarithmic $\phi_i\sqrt{\phi_i/\phi_{nb}}$ for positive quantities, linear for the field | `axi2d_2ndorder_mode2`, `..._longertime` |

`prob.wall_floor_frac` (default 0.1) bounds an extrapolated positive quantity
from below as a fraction of the cell value; it was never active in the paper
runs. The log prints the setting (`GEC wall fluxes: ...`). The transport model
is a compile-time switch, `-DGEC_TRANSPORT=<n>` in `DEFINES` (0, the default,
is the paper setting: BOLSIG electron transport fit, zero-field ion mobility).

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (2D, `USE_EB=FALSE`) |
| `Prob.H`, `ProbParm.H` | geometry, electrode masks, `prob.wall_extrap` inputs |
| `UserFunctions.H` | wall fluxes (`bc_ib`, `species_bc`) with the extrapolated wall state |
| `UserSources.H` | no user sources |
| `Chemistry.H`, `Chemistry.cpp` | argon chemistry and transport (fits written in the source) |
| `inputs` | deck (`prob.wall_extrap=1`) |
| `runcpu.sh` | Slurm job of the paper runs; argument = `prob.wall_extrap`, output in `mode<N>/` |
| `gec_paper_figures.py` | paper figures and the numbers in the text (final runs) |
| `compare_modes.py` | earlier comparison at 179.4 cycles |
| `pfread.py` | minimal plotfile reader used by the scripts |

## Data and acknowledgements

No data file is read by the solver; the chemistry and transport fits are
written as coefficients in `Chemistry.H`/`Chemistry.cpp` (argon reaction
rates and a fit of a BOLSIG+ run for the electron transport); no
cross-section file is loaded at run time. BOLSIG+ is the Boltzmann solver of
G. J. M. Hagelaar and L. C. Pitchford, "Solving the Boltzmann equation to
obtain electron transport coefficients and rate coefficients for fluid
models", Plasma Sources Sci. Technol. 14, 722 (2005); we thank its authors.
The cross-section set used for that BOLSIG+ run is not recorded in this case.

The figure scripts compare with two digitized curves that are already in the
repository, `test/verification/GEC_RF_Cell/eden100mtorrOverzetHopkins_APL_63_1993`
and `.../eden100mtorrBouefPitchford_PhysRevE_51_2_1995`; the scripts read them
from there and nothing is copied into this folder. We thank the authors of
the two sources:

- L. J. Overzet and M. B. Hopkins, Appl. Phys. Lett. 63 (1993) (measured
  electron density at 100 mTorr). Title and page are still to be added; the
  volume and year are those in the data file name.
- J. P. Boeuf and L. C. Pitchford, "Two-dimensional model of a capacitively
  coupled rf discharge and comparisons with experiments in the Gaseous
  Electronics Conference reference reactor", Phys. Rev. E 51, 1376 (1995),
  doi:10.1103/PhysRevE.51.1376
  (fluid-model electron density; also the source of the transport model of
  `GEC_TRANSPORT=1,2`).

The licence of the two digitized curves is not stated in the repository, so
they are not copied again here.

The paper runs were made on the NREL Kestrel cluster (one node, 64 ranks,
about 4 min per RF cycle). Their plotfiles (`axi2d`, `axi2d_2ndorder_mode1`,
`axi2d_2ndorder_mode2` at 208.8 cycles and `axi2d_2ndorder_mode2_longertime`
at 225.9 cycles) are not redistributed.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE        # cluster: COMP=gnu, see runcpu.sh
for m in 0 1 2; do                      # first order, linear, logarithmic
  mkdir -p mode$m && (cd mode$m && mpirun -np 64 ../vidyut2d.*.ex ../inputs prob.wall_extrap=$m)
done
# collect the plotfiles under the names gec_paper_figures.py expects
mkdir -p figs_in
cp -r mode0/plt00220 figs_in/axi2d                            # 208.8 cycles
cp -r mode1/plt00220 figs_in/axi2d_2ndorder_mode1             # 208.8 cycles
cp -r mode2/plt00220 figs_in/axi2d_2ndorder_mode2             # 208.8 cycles
cp -r mode2/plt00238 figs_in/axi2d_2ndorder_mode2_longertime  # 225.9 cycles
python3 gec_paper_figures.py figs_in <outdir>
```

With `vidyut.dt = 1e-11` s and `amr.plot_int = 7000` a plotfile is written
every 70 ns; at 13.56 MHz `plt00220` is at 208.8 cycles and `plt00238` at
225.9 cycles.

About 210 RF cycles (15.5 us) are needed for the paper numbers; `stop_time`
in the deck allows 1000 cycles. For finer grids build with hypre
(`USE_HYPRE=TRUE HYPRE_DIR=...`) and add `vidyut.use_hypre=1
vidyut.linsolve_max_coarsening_level=0`.

## Expected Results

At 208.8 cycles, first order / linear / logarithmic (`gec_paper_figures.py`):

| quantity | `wall_extrap=0` | `=1` | `=2` |
|---|---|---|---|
| peak $n_e$ [m$^{-3}$], at r = 4.5 cm | 2.164e15 | 2.235e15 | 2.211e15 |
| on-axis $n_e$ [m$^{-3}$] | 0.782e15 | 0.820e15 | 0.809e15 |

Second order vs first order: +3.3 % peak, +4.8 % on axis, +3.1 % total
electrons; logarithmic 1 % below linear. From 209 to 226 cycles the peak
changes by +0.5 % and the profile by less than 1 % of the peak. The
extrapolation floor was never active. Measured peak: 2.2 to 3.2e15
(Overzet and Hopkins); Boeuf and Pitchford fluid model: 1.9e15.

Not run for this folder (application case; build checked only).

## CTest

Not in CI.
