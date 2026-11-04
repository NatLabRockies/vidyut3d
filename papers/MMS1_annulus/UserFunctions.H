#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <VarDefines.H>

using namespace amrex;
namespace user_transport {

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real current_collector_value(
    IntVect cellid,
    int locid,
    int surfaceloc,
    GpuArray<Real, AMREX_SPACEDIM> problo,
    GpuArray<Real, AMREX_SPACEDIM> probhi,
    GpuArray<int, AMREX_SPACEDIM> domlo,
    GpuArray<int, AMREX_SPACEDIM> domhi,
    GpuArray<Real, AMREX_SPACEDIM> dx)
{
    return (0.0);
}

AMREX_GPU_DEVICE AMREX_INLINE void get_photoion_acoeff(
    int i,
    int j,
    int k,
    int sph_id,
    Array4<Real> const& sb_arr,
    Array4<Real> const& acoeff,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{}

AMREX_GPU_DEVICE AMREX_INLINE void get_photoion_rhs(
    int i,
    int j,
    int k,
    int sph_id,
    Array4<Real> const& sb_arr,
    Array4<Real> const& rhs_arr,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{}

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real surfaceval(
    amrex::Real x,
    amrex::Real y,
    ProbParm const& prob_parm,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    int inner)
{
    amrex::Real rad = std::sqrt(x * x + y * y);
    amrex::Real surfval = 0;
    if (inner)
    {
        surfval = rad - prob_parm.inner_radius;
    } else
    {
        surfval = prob_parm.outer_radius - rad;
    }
    return (surfval);
}



/*AMREX_GPU_DEVICE AMREX_INLINE void bc_ib(
    IntVect face,
    int dir,
    int sgn,
    int solved_comp,
    int rhs_comp,
    Array4<Real> const& sb_arr,
    amrex::Real ascalar,
    amrex::Real bscalar,
    Array4<Real> const& acoeff,
    Array4<Real> const& bcoeff,
    Array4<Real> const& rhs,
    GpuArray<int, AMREX_SPACEDIM> domlo,
    GpuArray<int, AMREX_SPACEDIM> domhi,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    amrex::Real vf_l = sb_arr(lcell, CMASK_ID);
    amrex::Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    IntVect cutcell = (vf_l < vf_r) ? lcell : rcell;

    // interior cell center
    amrex::Real xi[2];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1];

    //amrex::Real ri = std::sqrt(xi[0]*xi[0] + xi[1]*xi[1]);
    amrex::Real ri = std::max(std::sqrt(xi[0]*xi[0] + xi[1]*xi[1]), 1e-14);
    // determine which cylinder surface this cut belongs to
    amrex::Real R =
        (ri > 0.5*(prob_parm.inner_radius + prob_parm.outer_radius))
            ? prob_parm.outer_radius
            : prob_parm.inner_radius;

    // exact boundary intercept along radial normal
    amrex::Real xb[2];
    xb[0] = R * xi[0] / ri;
    xb[1] = R * xi[1] / ri;
    amrex::Real rad = sqrt(xb[0] * xb[0] + xb[1] * xb[1]);
    // Dirichlet value
    amrex::Real bc_value = rad * rad * rad * rad;

    // vector from boundary to cell center
    amrex::Real ccibvec[2];
    ccibvec[0] = xi[0] - xb[0];
    ccibvec[1] = xi[1] - xb[1];

    amrex::Real d2 =
        ccibvec[0]*ccibvec[0]
    + ccibvec[1]*ccibvec[1];

    auto sgn3 = [](amrex::Real v) -> amrex::Real {
        if (v > 1.e-12) return  1.0;
        if (v < -1.e-12) return -1.0;
        return 0.0;
    };

    amrex::Real outward_normal_dir = -sgn3(ccibvec[dir]);
    amrex::Real alpha =
       ccibvec[dir] / d2 * outward_normal_dir;
    if (solved_comp == POT_ID)
    {
        // matrix contribution
        acoeff(intcell) += alpha / dx[dir];
        // rhs contribution
        rhs(intcell,rhs_comp) +=
            bc_value * alpha / dx[dir];
    }
    else if (solved_comp == NI_ID)
    {
        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);
        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));
        amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell) -= Dcoeff / dx[dir] * alpha;
        rhs(intcell, rhs_comp) -= 16 * Dcoeff / dx[dir] * rad * rad * alpha;
        amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;
        if (sb_arr(rcell, CMASK_ID) < 1)
            {
                flux = (vel > 0) ? sb_arr(intcell, solved_comp) : 16 * rad * rad;
                rhs(intcell, rhs_comp) -= flux * vel / dx[dir];
            } else
            {
                flux = (vel > 0) ? 16 * rad * rad : sb_arr(intcell, solved_comp);
                rhs(intcell, rhs_comp) += flux * vel / dx[dir];
            }
        // Determine "away-from-EB" neighbour direction (one extra interior
        // cell)
        /*int step_sign = ((intcell[0] == lcell[0]) && (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                         && (intcell[2] == lcell[2])
#endif
                             )
                            ? -1
                            : +1;
        if (sb_arr(rcell, CMASK_ID) < 1)
        {
            IntVect nb = intcell;
            nb[dir] += step_sign;
            const amrex::Real value =
                0.5 *
                (3.0 * sb_arr(intcell, solved_comp) - sb_arr(nb, solved_comp));
            flux = (vel > 0) ? value : 16 * rad * rad;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir];
        } else
        {
            IntVect nb = intcell;
            nb[dir] += step_sign;
            const amrex::Real value =
                0.5 * (sb_arr(intcell, solved_comp) + sb_arr(nb, solved_comp));
            flux = (vel > 0) ? 16 * rad * rad : value;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir];
        }
    } else
    {
    }
}*/

AMREX_GPU_DEVICE AMREX_INLINE void bc_ib(
    IntVect face,
    int dir,
    int sgn,
    int solved_comp,
    int rhs_comp,
    Array4<Real> const& sb_arr,
    amrex::Real ascalar,
    amrex::Real bscalar,
    Array4<Real> const& acoeff,
    Array4<Real> const& bcoeff,
    Array4<Real> const& rhs,
    GpuArray<int, AMREX_SPACEDIM> domlo,
    GpuArray<int, AMREX_SPACEDIM> domhi,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    Real vf_l = sb_arr(lcell, CMASK_ID);
    Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    //IntVect cutcell = (vf_l < vf_r) ? lcell : rcell;

    auto sgn3 = [] AMREX_GPU_DEVICE (amrex::Real v) -> amrex::Real {
        if (v >  1.0e-12_rt) return  1.0_rt;
        if (v < -1.0e-12_rt) return -1.0_rt;
        return 0.0_rt;
    };

    // Interior cell center
    Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5_rt) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5_rt) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5_rt) * dx[2];
#endif

    // EB centroid seed from cut cell
    Real xb_seed[AMREX_SPACEDIM];
    xb_seed[0] = sb_arr(intcell, CPX_ID);
    xb_seed[1] = sb_arr(intcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_seed[2] = sb_arr(intcell, CPZ_ID);
#endif

    // EB normal from cut cell
    Real n[AMREX_SPACEDIM];
    n[0] = sb_arr(intcell, NX_ID);
    n[1] = sb_arr(intcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
    n[2] = sb_arr(intcell, NZ_ID);
#endif

    Real nmag = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n[d] * n[d];
    nmag = std::sqrt(nmag);

    if (nmag < 1.0e-14_rt) return;

    for (int d = 0; d < AMREX_SPACEDIM; ++d) n[d] /= nmag;

    // Project interior cell center onto tangent plane/line through xb_seed
    Real dxseed[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) dxseed[d] = xi[d] - xb_seed[d];

    Real distn = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) distn += dxseed[d] * n[d];

    Real xb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) xb[d] = xi[d] - distn * n[d];

    // Vector from reconstructed boundary point to interior cell center
    Real ccibvec[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec[d] = xi[d] - xb[d];

    Real d2 = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) d2 += ccibvec[d] * ccibvec[d];

    if (d2 < 1.0e-24_rt) return;

    Real outward_normal_dir = -sgn3(ccibvec[dir]);
    Real alpha = (ccibvec[dir] / d2) * outward_normal_dir;

    if (amrex::Math::abs(alpha) < 1.0e-14_rt) return;

    if (solved_comp == POT_ID)
    {
        // MMS Dirichlet value on immersed boundary: phi = r^4
        Real rb2 = xb[0] * xb[0] + xb[1] * xb[1];
        Real bc_value = rb2 * rb2;

        acoeff(intcell) += alpha / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha / dx[dir];
    }
   else if (solved_comp == NI_ID)
    {
           Real rad = std::sqrt(xb[0] * xb[0] + xb[1] * xb[1]);

        Real ndens = 0.0_rt;
        for (int sp = 0; sp < NUM_SPECIES; sp++) {
            ndens += sb_arr(intcell, sp);
        }

        Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);

        acoeff(intcell) -= Dcoeff / dx[dir] * alpha;
        rhs(intcell, rhs_comp) -= 16.0_rt * Dcoeff / dx[dir] * rad * rad * alpha;
        amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;
        if (sb_arr(rcell, CMASK_ID) < 1)
        {
            flux = (vel > 0) ? sb_arr(intcell, solved_comp) : 16 * rad * rad;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir];
        } else
        {
            flux = (vel > 0) ? 16 * rad * rad : sb_arr(intcell, solved_comp);
            rhs(intcell, rhs_comp) += flux * vel / dx[dir];
        }
    } else
    {
    }
}




AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
int is_dielectric(
    int i,
    int j,
    int k,
    int dir,
    int sign,
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    const GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    const GpuArray<Real, AMREX_SPACEDIM> dx,
    Real time,
    ProbParm const& prob_parm)
{
    return (0);
}

AMREX_GPU_DEVICE AMREX_INLINE void potential_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    Array4<Real> const& phi,
    Array4<Real> const& bc_arr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg,
    amrex::Real app_voltage,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)
{}

AMREX_GPU_DEVICE AMREX_INLINE void photoionization_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    Array4<Real> const& phi,
    Array4<Real> const& bc_arr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg)
{}

AMREX_GPU_DEVICE AMREX_INLINE void species_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    int spec_id,
    int bcspec_id,
    Array4<Real> const& phi,
    Array4<Real> const& bc_arr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)

{}

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real compute_vel(
    IntVect cellid,
    int dir,
    int specid,
    Array4<Real> const& phi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg)
{
    amrex::Real vel = 0.0;
    if (specid == NI_ID)
    {
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(phi(cellid, EFX_ID)) +
            amrex::Math::powi<2>(phi(cellid, EFY_ID)) +
            amrex::Math::powi<2>(phi(cellid, EFZ_ID)));

        amrex::Real ndens = 1.0; // dummy
        amrex::Real mu =
            specMob(specid, phi(cellid, ETEMP_ID), ndens, efieldmag, Tg);
        vel = mu * phi(cellid, EFX_ID + dir);
    }

    return (vel);
}

} // namespace user_transport
#endif
