# Vidyut
<div align="center">
<img src="https://github.com/hsitaram/vidyut3d/blob/main/images/vidyut_image.png" alt="Vidyut Logo">
</div>

## A plasma fluid solver for simulating low-temperature plasmas and plasma-mediated catalysis

Vidyut is a massively-parallel plasma-fluid solver for low-temperature plasmas (LTPs) that supports both local field (LFA) and local mean energy (LMEA) approximations, as well as complex gas and surface-phase chemistry. The solver supports 2D and 3D domains, and uses AMReX's adaptive mesh refinement capabilities to increase the grid resolution around complex structures (e.g. streamer heads and sheaths) while maintaining a tractable problem size. Vidyut specializes in simulating various types of gas-phase discharges, as well as plasma/surface interactions and surface chemistry (e.g. for plasma-mediated catalysis applications). The solver also supports hybrid CPU/GPU parallelization strategies, and has demonstrated excellent scaling on various HPC architectures for problem sizes consisting of O(100 M) control volumes.

## Models and Features

- LFA and LMEA models for solving the plasma-fluid equations with a drift-diffusion approximation
- Support for complex gas and surface-phase chemistry
- Second order semi-implicit scheme that handles drift and reactive source terms explicitly, and diffusive sources implicitly 
- Parallelization via OpenMPI/MPICH and GPU Acceleration with CUDA (NVidia) and HIP (AMD)
- Parallel I/O
- Plotfile format supported by Amrvis, VisIt, ParaView and yt

# Build instructions
* gcc and an MPI library (openMPI/MPICH) for CPU builds. cuda-11.0 is also required for GPU builds
* This tool depends on the AMReX library (https://github.com/AMReX-Codes/amrex) (a submodule of this software)
* Each example/run case must include a Prob.H, ProbParm.H, UserFunctions.H, and UserSources.H - see examples to get started 
* Navigate to the test/run case directory
* Build executable using the GNUMakefile (set USE_MPI and USE_CUDA=TRUE/FALSE depending on architecture and desired parallel execution) and run "make"
* Several test cases can be found in the test directory for getting started using the code

# Visualization instructions

* The outputs for a case are in the form of AMReX plotfiles
* These plot files can be open using AMReX grid reader in ParaView (see https://amrex-codes.github.io/amrex/docs_html/Visualization.html#paraview)
* Alternatively visit can be used. see https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html

# Citation

To cite Vidyut3d, use our computer physics communications paper or the software record:
```
@article{sitaraman2026vidyut3d,
  title={Vidyut3d: a GPU accelerated fluid solver for non-equilibrium plasmas on adaptive grids},
  author={Sitaraman, Hariswaran and Deak, Nicholas and Taneja, Taaresh},
  journal={Computer Physics Communications},
  pages={110236},
  year={2026},
  publisher={Elsevier}
}
```

```
@techreport{
sitaraman2024vidyut3d,
title={Vidyut3d: A Non-Equilibrium Plasma Modeling Tool [SWR-24-101]},
author={Sitaraman, Hariswaran and Deak, Nick},
year={2024},
institution={National Renewable Energy Laboratory (NREL), Golden, CO (United States)}
}
```

# Acknowledgments

This work was authored by the National Renewable Energy Laboratory (NREL) under software record SWR-24-101, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by funding from DOE Laboratory Directed Research and Development (LDRD) and DOE Basic Energy Sciences Materials Sciences and Engineering Division under award DE-SC0024724. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory.

## Update: 256^3 and the linearized wall data (`prob.ray_dirichlet=5`)

phi, all-cell L2 (rate) and Linf (rate):

| variant | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| ray value (1, paper) L2 | 4.36e-2 | 1.33e-2 (1.72) | 3.69e-3 (1.85) | 9.23e-4 (2.00) |
| ray value Linf | 2.05e-1 | 7.76e-2 (1.40) | 3.22e-2 (1.27) | 1.57e-2 (1.03) |
| foot value (0) L2 | 1.50e-2 | 1.43e-3 (3.40) | 4.21e-4 (1.76) | 1.60e-4 (1.40) |
| foot value Linf | 1.15e-1 | 2.95e-2 (1.96) | 1.12e-2 (1.40) | 5.11e-3 (1.13) |
| linearized (5) L2 | 1.67e-2 | 1.42e-3 (3.55) | 2.93e-4 (2.27) | running |
| linearized Linf | 1.06e-1 | 1.91e-2 (2.48) | 5.47e-3 (1.80) | running |

|E| L2 rate at the finest grid: ray 1.44, foot 0.55, linearized 1.34 (128); |E| Linf: ray does
not converge, linearized converges at ~0.8.

- The foot value is only better pre-asymptotically: its phi rate falls 3.4 -> 1.8 -> 1.4 and
  |E| to 0.55, because the tangential gradient of the solution is missing (O(1) flux error).
  The ray value is consistent (rate 2.0 at 256^3).
- The max error of the ray value sits in outer-wall cells with three wall faces where one
  normal component is ~0.03: there s = d_n/|n_dir| ~ 20 h and the wall data are sampled far
  from the cell. Same with the exact geometry -> property of the closure, not of the fit.
- Linearized wall data: phi_wall = phi_b(foot) + grad(phi_b)(foot).(x_ray - x_foot). Equal to
  the ray value for linear data (so Case 4 conclusions hold), never samples far away.
  12x smaller phi error than the ray value at 128^3, Linf rate 1.8, n_i unchanged (+6 %).
  Here the gradient of the wall data is analytical (MMSExact.H); for a general Dirichlet wall
  it is the tangential derivative of the prescribed wall function (zero for an electrode at
  uniform potential, where ray, foot and linearized values coincide).
- NOTE when checking errors by hand: fluid cells are cellmask > 1-1e-10. A threshold of 0.999
  includes cut cells that the solver masks and gives a bogus O(1) max error.
