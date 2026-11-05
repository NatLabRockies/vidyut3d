#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real r1 = 0.0; // source term for species 1
    amrex::Real r2 = 1.0; // source term for species 2
    int axdir = 0;
    amrex::Real V1 = 0.0;
    amrex::Real V2 = 0.0;
    // E = field_sign * (1 + x): +1 drifts towards x = 1, -1 towards x = 0
    amrex::Real field_sign = 1.0;
    // drift flux at the IB wall face: 0 = value and velocity of the wall
    // cell, 1 = both extrapolated to the face
    int ib_face_flux = 1;
};
#endif
