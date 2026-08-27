#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <UnivConstants.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real pres=133.0;
    amrex::Real temp=300.0;
    amrex::Real dgap = 0.0254;
    amrex::Real pulse_freq = 1e6;
    amrex::Real ionden_init = 1e14;
    amrex::Real V1 = 400.0;
    amrex::Real gama = 0.02;
    amrex::Real pinrad=0.01;
    amrex::Real tpulse=500.0e-9;  
    amrex::Real ramptime=20.0e-9;  
    amrex::Real seedionden=1e12;
    int sharp_interface_scheme=1;
};
#endif
