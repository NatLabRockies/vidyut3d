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
