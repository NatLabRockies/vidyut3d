#ifndef _USERSOURCES_H_
#define _USERSOURCES_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace user_sources {
AMREX_GPU_DEVICE AMREX_INLINE void add_user_react_sources(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& rxnsrc,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg)
{
    // add any user specific sources
    amrex::Real xloc[2];
    xloc[0] = prob_lo[0] + (i + 0.5) * dx[0];
    xloc[1] = prob_lo[1] + (j + 0.5) * dx[1];
    amrex::Real rad = sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);
    rxnsrc(i, j, k, NI_ID) += (384 * rad * rad * rad * rad - 64);
}

AMREX_GPU_DEVICE AMREX_INLINE void add_user_potential_sources(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& source,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm)
{
    // add any user specific sources
    // add any user specific sources
    source(i, j, k) += phi(i, j, k, NI_ID);
}
} // namespace user_sources
#endif
