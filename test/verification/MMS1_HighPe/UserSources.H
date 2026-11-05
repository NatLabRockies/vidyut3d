#ifndef _REACTIONS_H_
#define _REACTIONS_H_

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
    // n = x^2, E = s (1 + x), mu = 1, D = 1e-4:
    // d(n v)/dx - D d2n/dx2 = s (2 x + 3 x^2) - 2 D
    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real sg = prob_parm.field_sign;
    rxnsrc(i, j, k, NI_ID) += sg * (2.0 * x + 3.0 * x * x) - 2.0e-4;
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
    // prescribed field: d2phi/dx2 = -s, so phi = -s (x + x^2 / 2)
    source(i, j, k) += -prob_parm.field_sign;
}
} // namespace user_sources
#endif
