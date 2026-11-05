# MMS1 on a spherical shell, 3D (paper Case 5)

## Purpose

Case 5 of the paper. The single-species coupled system of Case 2
between two concentric spheres ($r_i = 0.5$, $r_o = 1.5$) in the periodic box
$[-2,2]^3$. Manufactured solution (`MMSExact.H`):

$$\phi = r^4 + A\,z\,r^2, \qquad n_i = \nabla^2\phi = 20\,r^2 + 10\,A\,z,$$
$$F = -\mu(\nabla n\cdot\nabla\phi + n^2) - D\nabla^2 n, \qquad \mu=-1,\ D=1 .$$

$A$ = `prob.mms_a` (default 0.5) makes the wall data vary along the wall.

Wall value forms (`prob.ray_dirichlet`): 1 value where the grid line meets the
planar wall (ray value), 0 value at the foot of the normal, 5 linearized wall
value (Eq. `eq:3d-linearized` of the paper), 2 diagnostic (exact line/sphere
intersection), 3 and 4 diagnostics that use only the exact distance or only
the exact point of that intersection. Any other value aborts. The paper's table `tab:mms1_3d_lin` is the linearized form
(`prob.ray_dirichlet=5`); the ray value (`=1`) is quoted in the text and
plotted in the figure. Both use method 4 with the 3D offset fit,
$w = c_0 + c_1 u + c_2 v + a u^2/2 + b u v + c v^2/2$
(`prob.eb_geom_method=4 prob.eb_quadric_offset=1`); this 3D fit exists only in
this folder's `Prob.H`. Note: `alpha` is negative in `bc_ib` here
(`acoeff += alpha/dx` with the operator $a\phi + \nabla^2\phi$).

## Files

| File | Content |
|---|---|
| `GNUmakefile`, `Make.package` | build (`DIM=3`, `USE_EB=TRUE`) |
| `MMSExact.H` | exact solution, gradient and source |
| `Prob.H`, `ProbParm.H` | sphere geometry, 3D wall foot/normal reconstruction |
| `UserFunctions.H` | `bc_ib` with the wall value forms of `prob.ray_dirichlet` |
| `UserSources.H`, `Chemistry.H`, `Chemistry.cpp` | manufactured source, species and transport |
| `inputs3d` | deck (100 steps, `prob.mms_a=0.5`, `prob.ray_dirichlet=1`, method 4) |
| `run.sh` | grid sequence into `<rundir>/<n>`, one run at a time |
| `run512.slurm` | the 512^3 cluster job (both wall forms) |
| `sphere_errors.py` | norms: L2 over all cells (paper), L2 over fluid cells, Linf |
| `sphere_error_plot.py` | paper figure (numbers written into the script) |
| `recon_accuracy.py` | foot and normal accuracy of the reconstruction |

## Data

None. The exact solution is analytical. The 512^3 runs were made on the NREL
Kestrel cluster; their plotfiles are not redistributed.

## How to run

```
make -j 4 COMP=llvm USE_MPI=TRUE                               # DIM = 3
./run.sh runs_lin "32 64 128 256" prob.ray_dirichlet=5         # paper table
./run.sh runs_m4  "32 64 128 256" prob.ray_dirichlet=1         # ray value (text, figure)
./run.sh runs_foot "32 64 128" prob.ray_dirichlet=0            # foot value (text)
./run.sh runs_ibng "32 64 128" prob.eb_geom_method=5           # "changed by less than 2 %"
./run.sh runs_exact "32 64 128" prob.eb_geom_method=3
python3 sphere_errors.py runs_lin runs_m4
python3 recon_accuracy.py runs_m4
python3 sphere_error_plot.py
sbatch run512.slurm                                            # 512^3, cluster
```

`run.sh` uses 1/4/8 ranks for 32/64/larger grids (`NP=<ranks>` overrides).
128^3 takes about 6 min on 8 ranks; 256^3 about 80 min and 13 GB; 512^3 about
105 GB.

## Expected Results

All-cell $L_2$ (`sphere_errors.py`, column `L2(all)`):

| $N_x$ | $\phi$ linearized (5) | $p$ | $\phi$ ray (1) | $p$ | $n_i$ linearized | $p$ | $n_i$ ray | $p$ |
|---|---|---|---|---|---|---|---|---|
| 32  | 1.6652e-02 | --   | 4.3620e-02 | --   | 9.7145e-01 | --   | 9.0347e-01 | --   |
| 64  | 1.4172e-03 | 3.55 | 1.3283e-02 | 1.72 | 2.8791e-01 | 1.75 | 2.7247e-01 | 1.73 |
| 128 | 2.9343e-04 | 2.27 | 3.6857e-03 | 1.85 | 8.0938e-02 | 1.83 | 7.6090e-02 | 1.84 |
| 256 | 8.5783e-05 | 1.77 | 9.2305e-04 | 2.00 | 2.1963e-02 | 1.88 | 2.0647e-02 | 1.88 |
| 512 | 2.2361e-05 | 1.94 | 2.2935e-04 | 2.01 | 5.8025e-03 | 1.92 | 5.4297e-03 | 1.93 |

At 512^3 the linearized $\phi$ error is 10.3x ($L_2$) and 12.8x ($L_\infty$)
smaller than the ray value; $L_\infty(\phi)$ = 5.4130e-04 (linearized) and
6.9066e-03 (ray). The analytical geometry (method 3) and IB-NG (method 5) give
$\phi$ errors within 2 % of method 4.

Not run for this folder (3D; build checked only).

Time step: `vidyut.dt = 0.01` is fixed and does not shrink with the grid.
A linear stability estimate (fifth-order upwind drift, backward-Euler
diffusion) gives $v^2\Delta t/2D \approx 1.4$ at the poles on the 256 and
512 grids, so the drift alone would be slightly unstable there; the damping
from the coupling to the potential brings the amplification factor to about
0.88. The paper runs completed; keep this in mind before raising `dt`.

## CTest

Not in CI.
