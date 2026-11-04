#ifndef _MMSEXACT_H_
#define _MMSEXACT_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

// Manufactured solution of the two-way coupled species-Poisson problem on a
// spherical shell (3D) or an annulus (2D). nd = number of space dimensions.
//
//   phi = r^4 + A z r^2          (z is the last coordinate)
//   n_i = lap(phi) = (8+4 nd) r^2 + (4+2 nd) A z
//   dn/dt + div(mu n E) = div(D grad n) + F,   E = -grad(phi)
//   F   = -mu (grad n . grad phi + n^2) - D lap(n)
//
// A = 0 in 2D recovers the annulus problem of MMS1 (phi = r^4, n = 16 r^2,
// F = -(384 mu r^4 + 64 D)). A /= 0 makes the wall data vary along the wall.
namespace mms {

AMREX_GPU_HOST_DEVICE AMREX_INLINE amrex::Real
phi_exact(const amrex::Real* x, amrex::Real A)
{
    amrex::Real r2 = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) r2 += x[d] * x[d];
    const amrex::Real z = x[AMREX_SPACEDIM - 1];
    return r2 * r2 + A * z * r2;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE amrex::Real
ni_exact(const amrex::Real* x, amrex::Real A)
{
    amrex::Real r2 = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) r2 += x[d] * x[d];
    const amrex::Real z = x[AMREX_SPACEDIM - 1];
    const amrex::Real nd = AMREX_SPACEDIM;
    return (8.0 + 4.0 * nd) * r2 + (4.0 + 2.0 * nd) * A * z;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE amrex::Real
ni_source(const amrex::Real* x, amrex::Real A, amrex::Real mu, amrex::Real D)
{
    amrex::Real r2 = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) r2 += x[d] * x[d];
    const amrex::Real z = x[AMREX_SPACEDIM - 1];
    const amrex::Real nd = AMREX_SPACEDIM;
    const amrex::Real cn = 8.0 + 4.0 * nd;
    const amrex::Real cz = (4.0 + 2.0 * nd) * A;

    amrex::Real gdot = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real ez = (d == AMREX_SPACEDIM - 1) ? 1.0 : 0.0;
        const amrex::Real gphi = 4.0 * r2 * x[d] + A * (r2 * ez + 2.0 * z * x[d]);
        const amrex::Real gn = 2.0 * cn * x[d] + cz * ez;
        gdot += gphi * gn;
    }
    const amrex::Real n = cn * r2 + cz * z;
    const amrex::Real lap_n = 2.0 * nd * cn;
    return -mu * (gdot + n * n) - D * lap_n;
}

// gradients of the manufactured solution (used for the wall-data linearization)
AMREX_GPU_HOST_DEVICE AMREX_INLINE void
phi_grad(const amrex::Real* x, amrex::Real A, amrex::Real* g)
{
    amrex::Real r2 = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) r2 += x[d] * x[d];
    const amrex::Real z = x[AMREX_SPACEDIM - 1];
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real ez = (d == AMREX_SPACEDIM - 1) ? 1.0 : 0.0;
        g[d] = 4.0 * r2 * x[d] + A * (r2 * ez + 2.0 * z * x[d]);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
ni_grad(const amrex::Real* x, amrex::Real A, amrex::Real* g)
{
    const amrex::Real nd = AMREX_SPACEDIM;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real ez = (d == AMREX_SPACEDIM - 1) ? 1.0 : 0.0;
        g[d] = 2.0 * (8.0 + 4.0 * nd) * x[d] + (4.0 + 2.0 * nd) * A * ez;
    }
}

} // namespace mms
#endif
