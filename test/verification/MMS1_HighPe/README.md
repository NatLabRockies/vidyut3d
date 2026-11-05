# MMS1_HighPe: drift-dominated transport at walls

## Purpose

This case checks the order of accuracy of the WENO5 drift flux next to walls
when drift dominates diffusion in the wall cell. That is the regime of ions at
the walls of a low-pressure discharge, where the cell Péclet number
Pe = |μE|h/D is about 100–1000.

It solves one ion species in 1D with a prescribed field:

$$\frac{\partial n}{\partial t}+\frac{\partial (\mu n E)}{\partial x}=\frac{\partial}{\partial x}\left(D\frac{\partial n}{\partial x}\right)+S,\qquad \frac{d^2\phi}{dx^2}=-s$$

$$\mu = 1,\quad D = 10^{-4},\quad E = s\,(1+x),\quad \phi = -s\left(x+\frac{x^2}{2}\right),\quad n = x^2$$

$$S = s\,(2x + 3x^2) - 2D,\qquad n(0) = 0,\quad n(1) = 1$$

Here s = `prob.field_sign`. The drift velocity is 1 at x = 0 and 2 at x = 1,
so the wall-cell Péclet number is 312–625 at N = 32 and 39–78 at N = 256. The field is not
uniform, so the drift velocity at the wall face differs from the one at the
wall cell centre.

There are two decks:

- `inputs`: the walls are the domain faces x = 0 and x = 1;
- `inputs_ib`: the domain is [0, 1.25] and the cells with x > 1 are masked, so
  x = 1 is an immersed-boundary (IB) wall on a cell face.

With s = 1 the flow enters at x = 0 and leaves at x = 1. With s = −1 it is the
reverse.

## Inputs

| Input | Default | Effect |
|---|---|---|
| `vidyut.hyp_wall_ghosts` | 0 | 1: WENO5 stencil cells outside the fluid (beyond a non-periodic, non-axis domain face, or masked IB cells) are filled by linear extrapolation where the flow leaves through the wall. Where it enters, they are linear through the Dirichlet wall value: at domain faces this is the species BC value when the Robin coefficients are b = 0, a ≠ 0, and at IB walls it comes from `user_transport::ib_wall_value`. Without a Dirichlet value (Neumann or Robin inflow faces, IB cases without the hook) they copy the nearest fluid cell. The drift velocity is extrapolated linearly. Periodic, axis and zero-flux (`IHNEUBC`) faces are unchanged. 0: copy of the nearest fluid cell (old behaviour). Must be 0 or 1. 1 requires `vidyut.hyp_order=5` and `amr.blocking_factor` >= 4 on every level; otherwise the run aborts. |
| `prob.field_sign` | 1 | sign s of the prescribed field. Must be 1 or −1; other values abort. |
| `user_transport::ib_wall_value` | not defined | optional function in `UserFunctions.H` (described in `Source/compute_explicit_flux.H`). It gives the Dirichlet value of an IB wall and its distance from the wall cell, so that `hyp_wall_ghosts=1` can use it at IB inflow walls. A case without it, or with a signature that needs a type conversion, falls back to the copy there; the run log prints which one was found. A returned distance below h/2 aborts. This case defines it for x = 1. |
| `prob.ib_face_flux` | 1 | drift flux at the IB wall face in this case's `bc_ib`. 1: value and velocity are extrapolated linearly from the wall cell and its fluid neighbour to the face. 0: wall-cell value and velocity (old closure). Must be 0 or 1; other values abort. |

`vidyut.hyp_wall_ghosts` is a solver input in `Source/`. `prob.ib_face_flux`
belongs to this case: the drift flux at an IB wall is set in each case's
`bc_ib`, so other cases keep their own closure.

The choice between the inflow and the outflow ghost is made from the drift
velocity at the centre of the wall cell. When the field at a wall reverses
in time, as in an RF discharge, the ghost switches with it. Each branch is
stable on its own (eigenvalues of the 1D operator with a stagnation point
within 2.5h of the wall were checked), but the flux is not smooth in time at
the switch.

## Files

| File | Purpose |
|---|---|
| `inputs`, `inputs_ib` | decks for domain walls and for an IB wall, N = 64 fluid cells |
| `UserFunctions.H` | boundary conditions; `bc_ib` holds the IB wall closure |
| `UserSources.H` | manufactured source and the potential source |
| `Chemistry.H` | μ = 1 and D = 1e-4 |
| `l2err/` | L2 and L∞ error of `NI` over the fluid cells; aborts above a tolerance |
| `run_convergence.sh` | grid sweep N = 32–256 with observed orders |

## Data and acknowledgements

No external data are used.

## How to run

    make -j USE_MPI=TRUE
    (cd l2err && make -j)
    ./run_convergence.sh inputs
    ./run_convergence.sh inputs vidyut.hyp_wall_ghosts=0
    ./run_convergence.sh inputs_ib
    ./run_convergence.sh inputs_ib vidyut.hyp_wall_ghosts=0 prob.ib_face_flux=0

Any extra argument is passed to the solver. Runs end at t = 3, which is about
four transit times, with dt = 0.2/N.

## Expected Results

L∞ error of `NI` at t = 3 for s = 1 (outflow wall at x = 1). The orders are
between N and 2N.

| Deck | Switches | N=32 | N=64 | N=128 | N=256 | order at 256 |
|---|---|---|---|---|---|---|
| `inputs` | `hyp_wall_ghosts=0` | 6.37e-2 | 2.64e-2 | 1.07e-2 | 4.89e-3 | 1.13 |
| `inputs` | **`hyp_wall_ghosts=1`** | 3.22e-4 | 7.99e-5 | 1.97e-5 | **4.81e-6** | **2.03** |
| `inputs_ib` | `hyp_wall_ghosts=0`, `ib_face_flux=0` | 6.37e-2 | 2.64e-2 | 1.07e-2 | 4.89e-3 | 1.13 |
| `inputs_ib` | `hyp_wall_ghosts=1`, `ib_face_flux=0` | 3.86e-2 | 1.93e-2 | 9.62e-3 | 4.76e-3 | 1.02 |
| `inputs_ib` | `hyp_wall_ghosts=0`, `ib_face_flux=1` | 5.72e-2 | 1.50e-2 | 2.23e-3 | 3.63e-4 | 2.62 |
| `inputs_ib` | **`hyp_wall_ghosts=1`, `ib_face_flux=1`** | 3.29e-4 | 8.16e-5 | 2.01e-5 | **4.89e-6** | **2.04** |

At the IB wall both changes are needed. With both, the IB deck matches the
domain-wall deck to within 2%.

With s = −1 the flow enters through x = 1:

| Deck | Switches | N=32 | N=64 | N=128 | N=256 | order at 256 |
|---|---|---|---|---|---|---|
| `inputs` | `hyp_wall_ghosts=0` | 2.44e+1 | 1.21e+1 | 3.05e+0 | 7.21e-1 | 2.08 |
| `inputs` | **`hyp_wall_ghosts=1`** | 7.89e-4 | 1.96e-4 | 4.84e-5 | **1.18e-5** | **2.03** |
| `inputs_ib` | `hyp_wall_ghosts=0`, `ib_face_flux=0` | 4.22e-2 | 2.12e-2 | 1.06e-2 | 5.21e-3 | 1.02 |
| `inputs_ib` | `hyp_wall_ghosts=1`, no `ib_wall_value` | 2.92e-2 | 1.47e-2 | 7.34e-3 | 3.62e-3 | 1.02 |
| `inputs_ib` | **`hyp_wall_ghosts=1`** | 5.57e-4 | 1.41e-4 | 3.49e-5 | **8.46e-6** | **2.04** |

At the inflow domain face the old copy never lets the drift flux see the wall
value, and the error is of order one on coarse grids. At the inflow IB wall
the wall value reaches the drift flux through `bc_ib`, but the WENO5 stencil
behind the wall still needs it, which is what `ib_wall_value` supplies.

At D = 1, the existing `MMS1` case (decks `inputs_x` and `inputs_ib_x`,
`vidyut.hyp_order=5`) is second order with the switch on or off. L∞ changes
by less than 2%.

## CTest

The CI workflow runs both decks at N = 64 on 2 ranks with s = 1 and checks
`l2err -linf 1.0e-4`. It also runs both decks with s = −1 and checks
`l2err -linf 3.0e-4`. Each check fails when the change it covers is removed:

| Run | L∞ at N = 64 |
|---|---|
| `inputs` | 7.99e-5 |
| `inputs vidyut.hyp_wall_ghosts=0` | 2.64e-2 |
| `inputs_ib` | 8.16e-5 |
| `inputs_ib vidyut.hyp_wall_ghosts=0` | 1.50e-2 |
| `inputs_ib prob.ib_face_flux=0` | 1.93e-2 |
| `inputs prob.field_sign=-1` (tolerance 3.0e-4) | 1.96e-4 |
| same, `vidyut.hyp_wall_ghosts=0` | 1.21e+1 |
| same, Dirichlet wall value replaced by a copy (one-line mutant) | 1.94e+1 |
| `inputs_ib prob.field_sign=-1` (tolerance 3.0e-4) | 1.41e-4 |
| same, without `ib_wall_value` | 1.47e-2 |

The results on 2 and 3 ranks are identical to all printed digits.
