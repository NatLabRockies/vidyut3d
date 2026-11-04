#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <UnivConstants.H>

using namespace amrex::literals;

#define MAX_PINS 32

struct ProbParm
{
    amrex::Real pres = 133.0;
    amrex::Real temp = 300.0;
    amrex::Real dgap = 0.0254;
    amrex::Real pulse_freq = 1e6;
    amrex::Real ionden_init = 1e14;
    amrex::Real V1 = 400.0;
    amrex::Real gama = 0.02;
    amrex::Real pinrad = 0.01;
    amrex::Real tpulse = 500.0e-9;
    amrex::Real ramptime = 20.0e-9;
    amrex::Real seedionden = 1e12;

    // Pin arrays (PinGeom.H): bottom = powered, top = grounded. Transverse
    // centres (x in 2D, x y in 3D) and radius at the domain face per pin.
    // Defaults: one pin per array on the axis with radius pinrad.
    int pin_axis = AMREX_SPACEDIM - 1; // 0 x, 1 y, 2 z
    int nbot = 1;
    int ntop = 1;
    amrex::Real bot_c[MAX_PINS][2] = {{0.0, 0.0}};
    amrex::Real top_c[MAX_PINS][2] = {{0.0, 0.0}};
    amrex::Real bot_r[MAX_PINS] = {0.01};
    amrex::Real top_r[MAX_PINS] = {0.01};

    // Geometry is generated once with AMReX EB2 (Prob.H) and stored in the
    // state (EB_cp_*, EB_norm_*); bc_ib reads it back. Requires USE_EB=TRUE.
    bool enable_EB = true;

    // 1: wall closure with the stored foot/normal (sharp interface)
    // 0: wall taken as the grid face at dx/2 (first-order masked closure)
    int sharp_interface_scheme = 1;
    // 1: when the stored foot is not across the current face (cell touching
    //    the wall in two directions near the tip), use the plane through the
    //    EB centroid of the masked neighbour across that face
    int perface_fallback = 1;
    // floor on the wall distance, in units of dx[dir]; 0 disables
    amrex::Real d_floor_frac = 0.5;

    // Wall state used in the wall fluxes:
    //   0 cell-centre value (first order, as the former GeomUtils version)
    //   1 linear extrapolation along the face direction to the wall
    //   2 logarithmic extrapolation for positive quantities
    int wall_extrap = 0;
    // lower bound of an extrapolated positive quantity, as a fraction of the
    // cell-centre value
    amrex::Real wall_floor_frac = 0.1;
};
#endif
