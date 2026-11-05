# Single-species coupled system on an annulus, MMS1 (paper Case 2)

## Purpose

Case 2 of the paper. One ion species and the potential on the annulus
$0.5 \le r \le 1.5$ inside the periodic box $[-3,3]^2$, both walls carried as
an embedded boundary. The ion density is the Poisson source and the drift uses
the computed field, so the coupling is two-way. Manufactured solution:

$$\phi = r^4, \qquad n_i = \nabla^2\phi = 16\,r^2 .$$

The wall reconstruction is method 4 (scaled quadric + Newton) with the offset
fit, the defaults of `Prob.H` (`prob.eb_geom_method=4`,
`prob.eb_quadric_offset=1`); the log line `IB reconstruction: ...` records it.
The case produced the MMS1 table and figures of the paper.

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (2D, `USE_EB=TRUE`) |
| `Prob.H`, `ProbParm.H` | geometry, wall foot/normal reconstruction (`nghost=2`) |
| `UserFunctions.H` | `bc_ib` wall closure |
| `UserSources.H`, `Chemistry.H`, `Chemistry.cpp` | manufactured source, species and transport |
| `inputs2d` | deck (100 steps, `vidyut.dt=0.01`) |
| `run.sh` | grid sequence 32 to 512 into `results/<n>` |
| `mms1_error_plot.py` | table and figures (`-f results`) |

## Data

None. The exact solution is analytical.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE
./run.sh                               # NP=<ranks>, default 2 -> results/<n>/plt00001
python3 mms1_error_plot.py -f results
```

Both walls are Dirichlet, fixed by `bc_ib`. The `prob.*_voltage`, `prob.a_*_dir`, `prob.b_*_neumann` and `prob.*_dirichlet` keys that `Prob.H` reads are not used by this case and are not set in the deck; setting them changes nothing. Other reconstructions: `EXTRA="prob.eb_geom_method=5" OUT=ibng ./run.sh`.

The norm is the mean over every cell of the domain, solid included, of the
masked squared error, with fluid = `cellmask > 1 - 1e-10`.

## Expected Results

| $N_x$ | $L_2(\phi)$ | $p$ | $L_2(n_i)$ | $p$ |
|---|---|---|---|---|
| 32  | 3.995787e-02 | --   | 1.404289e+00 | --   |
| 64  | 2.417303e-03 | 4.05 | 4.134416e-01 | 1.76 |
| 128 | 7.437788e-04 | 1.70 | 1.121568e-01 | 1.88 |
| 256 | 2.020101e-04 | 1.88 | 2.959409e-02 | 1.92 |
| 512 | 5.505090e-05 | 1.88 | 7.826130e-03 | 1.92 |

Checked for this folder: the 32 grid (2 ranks) reproduces both values to the
printed digits.

## CTest

Not in CI.
