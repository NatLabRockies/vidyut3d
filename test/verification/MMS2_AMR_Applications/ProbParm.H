#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real inner_radius = 0.1;
    amrex::Real outer_radius = 0.2;
    amrex::Real inner_voltage = 10.0;
    amrex::Real outer_voltage = 20.0;
    int inner_dirichlet = 1;
    int outer_dirichlet = 1;
    amrex::Real a_inner_dir = 1.0;
    amrex::Real b_inner_neumann = 0.0;
    amrex::Real a_outer_dir = 1.0;
    amrex::Real b_outer_neumann = 0.0;
    amrex::Real n0 = 1e6;
    amrex::Real solnfrac = 1.0;
    bool enable_EB = true;
    // --- degenerate-cell safeguards (ported from the 3D GEC case) ---
    int corner_fallback = 1;          // corner-aware foot/normal fallback in Prob.H
    amrex::Real d_floor_frac = 0.5;   // floor on wall distance, in units of dx[dir]; 0 disables
    int proj_switch = 0;              // 1: use isotropic 1/d when |n.dir| < 0.98 (3D GEC setting)
    int perface_fallback = 0;         // 1: per-face wall lookup when the stored foot is not across this face
    int ray_dirichlet = 0;            // 1: take the wall value where the face-direction ray hits the wall
    int face_fallback = 1;            // 1: treat wall as the grid face when no foot/normal is stored
    int cmask_vfrac = 0;              // 0 binary CMASK, 1 vfrac in cut cells (for refine_cutcells)
};
#endif
