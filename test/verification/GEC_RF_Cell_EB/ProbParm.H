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
    bool enable_EB = true;
};
#endif
