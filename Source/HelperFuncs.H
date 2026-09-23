#ifndef _helperfuncs_H_
#define _helperfuncs_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <UnivConstants.H>
#include <VarDefines.H>

using namespace amrex;
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real get_onesided_grad(
    IntVect face,
    int sgn,
    int idim,
    int comp,
    const GpuArray<Real, AMREX_SPACEDIM> dx,
    Array4<Real> const sb_arr)
{
    // sgn is always outward normal
    IntVect cell = face;
    int cell_adjust = (sgn == 1) ? -1 : 0;
    cell[idim] += cell_adjust;

    IntVect ncell1 = cell;
    IntVect ncell2 = cell;
    ncell1[idim] = cell[idim] - sgn;
    ncell2[idim] = ncell1[idim] - sgn;

    amrex::Real grad1, grad2, grad;
    grad = 0.0;
    // FIXME: can remove this if-else and just
    // use sgn to find grad1 and grad2
    // but wanted to keep it easy to understand
    if (sgn == -1)
    {
        // cells are towards the right
        //  f---x(cell)---|(grad1)---x(ncell1)---|(grad2)---x(ncell2)---|
        grad1 = (sb_arr(ncell1, comp) - sb_arr(cell, comp)) / dx[idim];
        grad2 = (sb_arr(ncell2, comp) - sb_arr(ncell1, comp)) / dx[idim];
    } else
    {
        // cells are towards the right
        //  |(grad2)---x(ncell2)---|(grad1)---x(ncell1)---|---x(cell)---f
        //
        grad1 = (sb_arr(cell, comp) - sb_arr(ncell1, comp)) / dx[idim];
        grad2 = (sb_arr(ncell1, comp) - sb_arr(ncell2, comp)) / dx[idim];
    }
    grad = 2.0 * grad1 - grad2;
    return (grad);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real get_efield_alongdir(
    int i,
    int j,
    int k,
    int dir,
    const GpuArray<int, AMREX_SPACEDIM> domlo,
    const GpuArray<int, AMREX_SPACEDIM> domhi,
    const GpuArray<Real, AMREX_SPACEDIM> dx,
    Array4<Real> const& s_arr)
{
    IntVect cellid{AMREX_D_DECL(i, j, k)};
    Real efield_dir = 0.0;

    if (cellid[dir] > domlo[dir] && cellid[dir] < domhi[dir])
    {
        IntVect ip1{AMREX_D_DECL(i, j, k)};
        IntVect im1{AMREX_D_DECL(i, j, k)};

        ip1[dir] += 1;
        im1[dir] -= 1;

        efield_dir = -0.5 * (s_arr(ip1, POT_ID) - s_arr(im1, POT_ID)) / dx[dir];
    } else if (cellid[dir] == domlo[dir])
    {
        IntVect ip1{AMREX_D_DECL(i, j, k)};
        IntVect ip2{AMREX_D_DECL(i, j, k)};
        ip1[dir] += 1;
        ip2[dir] += 2;

        // fdash = (-f_{i+2}+4f_{i+1}-3f_i)/(2 dx)
        if (domhi[dir] - domlo[dir] > 2)
        {
            efield_dir = -0.5 *
                         (-s_arr(ip2, POT_ID) + 4.0 * s_arr(ip1, POT_ID) -
                          3.0 * s_arr(cellid, POT_ID)) /
                         dx[dir];
        } else
        {
            efield_dir =
                -(s_arr(ip1, POT_ID) - s_arr(cellid, POT_ID)) / dx[dir];
        }
    } else if (cellid[dir] == domhi[dir])
    {
        IntVect im1{AMREX_D_DECL(i, j, k)};
        IntVect im2{AMREX_D_DECL(i, j, k)};
        im1[dir] -= 1;
        im2[dir] -= 2;

        // fdash = (f_{i-2}-4f_{i-1}+3f_i)/(2 dx)
        if (domhi[dir] - domlo[dir] > 2)
        {
            efield_dir = -0.5 *
                         (s_arr(im2, POT_ID) - 4.0 * s_arr(im1, POT_ID) +
                          3.0 * s_arr(cellid, POT_ID)) /
                         dx[dir];
        } else
        {
            efield_dir =
                -(s_arr(cellid, POT_ID) - s_arr(im1, POT_ID)) / dx[dir];
        }
    } else
    {
        // there shouldnt be another option!
    }
    return (efield_dir);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real get_applied_potential(
    Real current_time,
    int domain_end,
    int vprof,
    Real v1,
    Real v2,
    Real vfreq,
    Real vdur,
    Real vcen)
{
    Real voltage;
    Real voltage_amp = (domain_end == -1) ? v1 : v2;
    if (vprof == 1)
    { // Sinusoidal pulse shape
        voltage = sin(2.0 * PI * vfreq * current_time) * voltage_amp;
    } else if (vprof == 2)
    { // Single triangular pulse
        if (current_time <= vcen)
        {
            voltage =
                (vcen - current_time < vdur / 2.0)
                    ? (1.0 - (vcen - current_time) / (vdur / 2.0)) * voltage_amp
                    : 0.0;
        } else
        {
            voltage =
                (current_time - vcen < vdur / 2.0)
                    ? (1.0 - (current_time - vcen) / (vdur / 2.0)) * voltage_amp
                    : 0.0;
        }
    } else
    {
        voltage = voltage_amp;
    }

    return voltage;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
minmod_limiter(amrex::Real left_difference, amrex::Real right_difference)
{
    amrex::Real phi = 1.0;
    if (amrex::Math::abs(right_difference) > 0.0)
    {
        amrex::Real r = left_difference / right_difference;
        phi = std::max(0.0, std::min(1.0, r));
    } else
    {
        phi = 1.0;
    }
    return (phi);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real get_gradlimiter(
    int i, int j, int k, int var, int sweepdir, Array4<Real> const& state_arr)
{
    IntVect iv{AMREX_D_DECL(i, j, k)};
    IntVect ivl{AMREX_D_DECL(i, j, k)};
    IntVect ivr{AMREX_D_DECL(i, j, k)};

    ivl[sweepdir] -= 1;
    ivr[sweepdir] += 1;

    amrex::Real ldiff = state_arr(iv, var) - state_arr(ivl, var);
    amrex::Real rdiff = state_arr(ivr, var) - state_arr(iv, var);

    amrex::Real lim = minmod_limiter(ldiff, rdiff);

    return (lim);
}
#endif
