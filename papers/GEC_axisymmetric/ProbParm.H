#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <UnivConstants.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real dgap = 0.0254;
    amrex::Real R1 = 0.0538;
    amrex::Real f = 13.56e6;
    amrex::Real ionden_init = 1e14;
    amrex::Real V1 = 400.0;
    amrex::Real V2 = 0.0;
    amrex::Real bg_gasden = 1.0;
    amrex::Real gama = 0.02;
    // Wall state used in the wall fluxes (bc_ib and species_bc):
    //   0 cell-center value (first order)
    //   1 linear extrapolation to the wall face, 1.5*phi_i - 0.5*phi_nb (second order)
    //   2 logarithmic extrapolation phi_i*sqrt(phi_i/phi_nb) for positive
    //     quantities (second order, positive, exact for exponential profiles);
    //     linear for the electric field
    int wall_extrap = 1;
    // lower bound of an extrapolated positive quantity, as a fraction of the
    // cell-center value (guards against undershoot in a steep sheath)
    amrex::Real wall_floor_frac = 0.1;
};
#endif
