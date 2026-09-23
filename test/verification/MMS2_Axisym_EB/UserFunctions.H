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

    amrex::Real ri = std::sqrt(xi[0]*xi[0] + xi[1]*xi[1]);

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
    amrex::Real bc_value = rad * rad * rad * rad / 32;

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
Real d_i = std::sqrt(d2);
Real d_eb = std::abs(ccibvec[dir]);

Real theta = (d_eb > 1e-12) ? d_i / d_eb : 1.0;
theta = std::min(1.0, std::max(0.0, theta));
    if (solved_comp == POT_ID)
    {
        // matrix contribution
        acoeff(intcell) += alpha / dx[dir];

        // rhs contribution
        rhs(intcell,rhs_comp) +=
            bc_value * alpha / dx[dir];        
    } 
    else if (
        solved_comp == E_ID || solved_comp == HEp_ID || solved_comp == EEN_ID)
    {
        const amrex::Real n0 = prob_parm.n0;
        const amrex::Real n_alpha = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0 : 1.0;
        const amrex::Real soln = rad * rad / (den_coeff * n_alpha) + n0;
        const amrex::Real fivethird = (solved_comp == EEN_ID) ? 5.0 / 3.0 : 1.0;
        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);
        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));
        amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell) -= Dcoeff / dx[dir] * alpha * fivethird;
        rhs(intcell, rhs_comp) -= soln * Dcoeff / dx[dir] * alpha * fivethird;
        amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;
 // Determine "away-from-EB" neighbour direction (one extra interior
        // cell)
        int step_sign = ((intcell[0] == lcell[0]) && (intcell[1] == lcell[1])
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
                flux = (vel > 0) ? value : soln;
                rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
            } else
            {
                IntVect nb = intcell;
                nb[dir] += step_sign;
                const amrex::Real value =
                    0.5 * (sb_arr(intcell, solved_comp) + sb_arr(nb, solved_comp));
                flux = (vel > 0) ? 16 * rad * rad : value;
                rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
            }        
        if (sb_arr(rcell, CMASK_ID) < 1)
            {
                flux = (vel > 0) ? sb_arr(intcell, solved_comp) : soln;
                rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
            } else
            {
                flux = (vel > 0) ? soln : sb_arr(intcell, solved_comp);
                rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
            }
    } 
    else
    {
    }
}*/

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

    auto sgn3 = [](amrex::Real v) -> amrex::Real {
        if (v >  1.e-12) return  1.0;
        if (v < -1.e-12) return -1.0;
        return 0.0;
    };

    // interior cell center
    amrex::Real xi[2];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1];

    amrex::Real ri = std::sqrt(xi[0]*xi[0] + xi[1]*xi[1]);

    // ---- Analytical boundary intercept (used by species branch) ----
    amrex::Real R =
        (ri > 0.5*(prob_parm.inner_radius + prob_parm.outer_radius))
            ? prob_parm.outer_radius
            : prob_parm.inner_radius;

    amrex::Real xb[2];
    xb[0] = R * xi[0] / ri;
    xb[1] = R * xi[1] / ri;
    amrex::Real rad = std::sqrt(xb[0]*xb[0] + xb[1]*xb[1]);

    amrex::Real ccibvec[2];
    ccibvec[0] = xi[0] - xb[0];
    ccibvec[1] = xi[1] - xb[1];

    amrex::Real d2 = ccibvec[0]*ccibvec[0] + ccibvec[1]*ccibvec[1];

    amrex::Real outward_normal_dir = -sgn3(ccibvec[dir]);
    amrex::Real alpha = ccibvec[dir] / d2 * outward_normal_dir;

    Real d_i  = std::sqrt(d2);
    Real d_eb = std::abs(ccibvec[dir]);
    Real theta = (d_eb > 1e-12) ? d_i / d_eb : 1.0;
    theta = std::min(1.0, std::max(0.0, theta));

    if (solved_comp == POT_ID)
    {
        // ---- EB-centroid / EB-normal reconstruction (from EBUserFunctions.H) ----
        Real xb_seed[AMREX_SPACEDIM];
        xb_seed[0] = sb_arr(cutcell, CPX_ID);
        xb_seed[1] = sb_arr(cutcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
        xb_seed[2] = sb_arr(cutcell, CPZ_ID);
#endif

        Real n_eb[AMREX_SPACEDIM];
        n_eb[0] = sb_arr(cutcell, NX_ID);
        n_eb[1] = sb_arr(cutcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
        n_eb[2] = sb_arr(cutcell, NZ_ID);
#endif

        Real nmag = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n_eb[d] * n_eb[d];
        nmag = std::sqrt(nmag);
        if (nmag < 1.0e-14) return;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) n_eb[d] /= nmag;

        // project interior cell center onto EB tangent plane through xb_seed
        Real dxseed[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) dxseed[d] = xi[d] - xb_seed[d];

        Real distn = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) distn += dxseed[d] * n_eb[d];

        Real xb_eb[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) xb_eb[d] = xi[d] - distn * n_eb[d];

        Real ccibvec_eb[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec_eb[d] = xi[d] - xb_eb[d];

        Real d2_eb = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) d2_eb += ccibvec_eb[d] * ccibvec_eb[d];
        if (d2_eb < 1.0e-24) return;

        Real outward_normal_dir_eb = -sgn3(ccibvec_eb[dir]);
        Real alpha_eb = (ccibvec_eb[dir] / d2_eb) * outward_normal_dir_eb;
        if (amrex::Math::abs(alpha_eb) < 1.0e-14) return;

        // Dirichlet value at reconstructed EB point (original MMS: r^4/32)
        Real rb2 = xb_eb[0]*xb_eb[0] + xb_eb[1]*xb_eb[1];
        Real bc_value = rb2 * rb2 / 32.0;

        acoeff(intcell)          += alpha_eb / dx[dir];
        rhs(intcell, rhs_comp)   += bc_value * alpha_eb / dx[dir];
    }
    else if (
        solved_comp == E_ID || solved_comp == HEp_ID || solved_comp == EEN_ID)
    {
        const amrex::Real n0 = prob_parm.n0;
        const amrex::Real n_alpha = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0 : 1.0;
        const amrex::Real soln = rad * rad / (den_coeff * n_alpha) + n0;
        const amrex::Real fivethird = (solved_comp == EEN_ID) ? 5.0 / 3.0 : 1.0;
        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);
        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));
        amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell) -= Dcoeff / dx[dir] * alpha * fivethird;
        rhs(intcell, rhs_comp) -= soln * Dcoeff / dx[dir] * alpha * fivethird;
        amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;

        int step_sign = ((intcell[0] == lcell[0]) && (intcell[1] == lcell[1])
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
                0.5 * (3.0 * sb_arr(intcell, solved_comp) - sb_arr(nb, solved_comp));
            flux = (vel > 0) ? value : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        } else
        {
            IntVect nb = intcell;
            nb[dir] += step_sign;
            const amrex::Real value =
                0.5 * (sb_arr(intcell, solved_comp) + sb_arr(nb, solved_comp));
            flux = (vel > 0) ? 16 * rad * rad : value;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
    {
    }
}*/

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
    // -----------------------------------------------------------------
    // Identify the interior (fluid) cell and the cut cell across `face`
    // -----------------------------------------------------------------
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    amrex::Real vf_l = sb_arr(lcell, CMASK_ID);
    amrex::Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    IntVect cutcell = (vf_l <  vf_r) ? lcell : rcell;

    auto sgn3 = [](amrex::Real v) -> amrex::Real {
        if (v >  1.e-12) return  1.0;
        if (v < -1.e-12) return -1.0;
        return 0.0;
    };

    // -----------------------------------------------------------------
    // Interior cell center
    // -----------------------------------------------------------------
    amrex::Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5) * dx[2];
#endif

    // -----------------------------------------------------------------
    // EB centroid seed + EB normal from the cut cell (from sb_arr).
    // These are the same geometric quantities the AMReX EB factory used
    // to build the cut-cell apertures, so reconstructing xb, distn from
    // them keeps the wall Dirichlet consistent with the discrete stencil.
    // -----------------------------------------------------------------
    amrex::Real xb_seed[AMREX_SPACEDIM];
    xb_seed[0] = sb_arr(cutcell, CPX_ID);
    xb_seed[1] = sb_arr(cutcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_seed[2] = sb_arr(cutcell, CPZ_ID);
#endif

    amrex::Real n_eb[AMREX_SPACEDIM];
    n_eb[0] = sb_arr(cutcell, NX_ID);
    n_eb[1] = sb_arr(cutcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
    n_eb[2] = sb_arr(cutcell, NZ_ID);
#endif

    amrex::Real nmag = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n_eb[d] * n_eb[d];
    nmag = std::sqrt(nmag);
    if (nmag < 1.0e-14) return;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) n_eb[d] /= nmag;

    // -----------------------------------------------------------------
    // Project xi onto the tangent plane through xb_seed to obtain the
    // true foot xb_eb of the wall normal from xi, plus the signed
    // normal distance distn.
    // -----------------------------------------------------------------
    amrex::Real dxseed[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) dxseed[d] = xi[d] - xb_seed[d];

    amrex::Real distn = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) distn += dxseed[d] * n_eb[d];

    amrex::Real xb_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) xb_eb[d] = xi[d] - distn * n_eb[d];

    amrex::Real ccibvec_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec_eb[d] = xi[d] - xb_eb[d];

    amrex::Real d2_eb = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) d2_eb += ccibvec_eb[d] * ccibvec_eb[d];
    if (d2_eb < 1.0e-24) return;

    amrex::Real outward_normal_dir_eb = -sgn3(ccibvec_eb[dir]);
    amrex::Real alpha_eb =
        (ccibvec_eb[dir] / d2_eb) * outward_normal_dir_eb;
    if (amrex::Math::abs(alpha_eb) < 1.0e-14) return;

    // Radius at the reconstructed EB foot; used by species Dirichlet.
    amrex::Real rad_eb =
        std::sqrt(xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1]);

    // =================================================================
    // POT_ID: MMS Dirichlet phi = r^4 / 32 at reconstructed EB point
    // =================================================================
    if (solved_comp == POT_ID)
    {
        amrex::Real rb2 = xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1];
        amrex::Real bc_value = rb2 * rb2 / 32.0;

        acoeff(intcell)        += alpha_eb / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha_eb / dx[dir];
    }
    // =================================================================
    // Species (E, HEp, EEN): diffusion Dirichlet + drift-flux upwind
    // All wall values are evaluated at xb_eb (radius rad_eb).
    // =================================================================
    else if (solved_comp == E_ID   ||
             solved_comp == HEp_ID ||
             solved_comp == EEN_ID)
    {
        const amrex::Real n0        = prob_parm.n0;
        const amrex::Real n_alpha   = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0 : 1.0;
        const amrex::Real fivethird =
            (solved_comp == EEN_ID) ? 5.0 / 3.0 : 1.0;

        // Manufactured Dirichlet value at the reconstructed EB foot
        const amrex::Real soln =
            rad_eb * rad_eb / (den_coeff * n_alpha) + n0;

        // Local plasma state at intcell for transport coefficients
        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; ++sp)
            ndens += sb_arr(intcell, sp);

        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        // ---- Diffusion contribution (EB-consistent stencil) ----
        amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID),
                     ndens, emag, Tg);
        acoeff(intcell) -=
            Dcoeff / dx[dir] * alpha_eb * fivethird;
        rhs(intcell, rhs_comp) -=
            soln * Dcoeff / dx[dir] * alpha_eb * fivethird;

        // ---- Advection (drift) contribution ----
        // vel = mu * E_dir; wall value for upwind = soln (density at xb_eb)
        amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID),
                    ndens, emag, Tg);
        amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;

        // "Away from EB" neighbor for one-sided extrapolation
        int step_sign = ((intcell[0] == lcell[0]) &&
                         (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                      && (intcell[2] == lcell[2])
#endif
                        ) ? -1 : +1;

        IntVect nb = intcell;
        nb[dir] += step_sign;

        if (sb_arr(rcell, CMASK_ID) < 1)
        {
            // interior cell on the "lo" side of the face
            const amrex::Real value =
                0.5 * (3.0 * sb_arr(intcell, solved_comp)
                           - sb_arr(nb,      solved_comp));
            flux = (vel > 0.0) ? value : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }
        else
        {
            // interior cell on the "hi" side of the face
            const amrex::Real value =
                0.5 * (sb_arr(intcell, solved_comp)
                     + sb_arr(nb,      solved_comp));
            flux = (vel > 0.0) ? soln : value;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
    {
        // no-op for other solved components
    }
}*/


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
    // -----------------------------------------------------------------
    // Small-cut-cell / degenerate-geometry guard thresholds.
    //  - DISTN_MIN_FRAC  : minimum |distn| / dx[dir] before we accept the cell
    //  - ALPHA_MAX_FACTOR: cap on |alpha_eb| * dx[dir]^2 (dimensionless)
    // Cells failing either guard are skipped; the interior stencil there
    // is left to the operator alone. Empirically 0.10 / 100.0 removes the
    // pathological 1/d^2 blow-up without noticeably reducing wall coverage.
    // -----------------------------------------------------------------
    constexpr amrex::Real DISTN_MIN_FRAC   = 0.10;
    constexpr amrex::Real ALPHA_MAX_FACTOR = 100.0;

    // -----------------------------------------------------------------
    // Identify the interior (fluid) cell and the cut cell across `face`
    // -----------------------------------------------------------------
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    amrex::Real vf_l = sb_arr(lcell, CMASK_ID);
    amrex::Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    IntVect cutcell = (vf_l <  vf_r) ? lcell : rcell;

    auto sgn3 = [](amrex::Real v) -> amrex::Real {
        if (v >  1.e-12) return  1.0;
        if (v < -1.e-12) return -1.0;
        return 0.0;
    };

    // -----------------------------------------------------------------
    // Interior cell center
    // -----------------------------------------------------------------
    amrex::Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5) * dx[2];
#endif

    // -----------------------------------------------------------------
    // EB centroid seed + EB normal from the cut cell (from sb_arr).
    // -----------------------------------------------------------------
    amrex::Real xb_seed[AMREX_SPACEDIM];
    xb_seed[0] = sb_arr(cutcell, CPX_ID);
    xb_seed[1] = sb_arr(cutcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_seed[2] = sb_arr(cutcell, CPZ_ID);
#endif

    amrex::Real n_eb[AMREX_SPACEDIM];
    n_eb[0] = sb_arr(cutcell, NX_ID);
    n_eb[1] = sb_arr(cutcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
    n_eb[2] = sb_arr(cutcell, NZ_ID);
#endif

    amrex::Real nmag = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n_eb[d] * n_eb[d];
    nmag = std::sqrt(nmag);
    if (nmag < 1.0e-14) return;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) n_eb[d] /= nmag;

    // -----------------------------------------------------------------
    // Project xi onto the tangent plane through xb_seed to get true foot
    // xb_eb of the wall normal from xi, plus signed normal distance distn.
    // -----------------------------------------------------------------
    amrex::Real dxseed[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) dxseed[d] = xi[d] - xb_seed[d];

    amrex::Real distn = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) distn += dxseed[d] * n_eb[d];

    amrex::Real xb_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) xb_eb[d] = xi[d] - distn * n_eb[d];

    amrex::Real ccibvec_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec_eb[d] = xi[d] - xb_eb[d];

    amrex::Real d2_eb = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) d2_eb += ccibvec_eb[d] * ccibvec_eb[d];
    if (d2_eb < 1.0e-24) return;

    // -----------------------------------------------------------------
    // Guard 1: reject degenerate cuts where the interior center is
    // essentially on the wall (|distn| tiny compared to dx).
    // -----------------------------------------------------------------
    const amrex::Real d_norm = std::sqrt(d2_eb);
    if (d_norm < DISTN_MIN_FRAC * dx[dir]) return;

    amrex::Real outward_normal_dir_eb = -sgn3(ccibvec_eb[dir]);
    amrex::Real alpha_eb =
        (ccibvec_eb[dir] / d2_eb) * outward_normal_dir_eb;
    if (amrex::Math::abs(alpha_eb) < 1.0e-14) return;

    // -----------------------------------------------------------------
    // Guard 2: cap |alpha_eb| so no single row of the matrix acquires
    // a runaway coefficient that swamps the neighbors.
    // -----------------------------------------------------------------
    const amrex::Real alpha_max = ALPHA_MAX_FACTOR / (dx[dir] * dx[dir]);
    if (amrex::Math::abs(alpha_eb) > alpha_max) return;

    // Radius at the reconstructed EB foot; used by species Dirichlet.
    const amrex::Real rad_eb =
        std::sqrt(xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1]);

    // =================================================================
    // POT_ID: MMS Dirichlet phi = r^4 / 32 at reconstructed EB point
    // =================================================================
    if (solved_comp == POT_ID)
    {
        const amrex::Real rb2 = xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1];
        const amrex::Real bc_value = rb2 * rb2 / 32.0;

        acoeff(intcell)        += alpha_eb / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha_eb / dx[dir];
    }
    // =================================================================
    // Species (E, HEp, EEN): diffusion Dirichlet + drift-flux upwind
    // All wall values are evaluated at xb_eb (radius rad_eb).
    // =================================================================
    else if (solved_comp == E_ID   ||
             solved_comp == HEp_ID ||
             solved_comp == EEN_ID)
    {
        const amrex::Real n0        = prob_parm.n0;
        const amrex::Real n_alpha   = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0 : 1.0;
        const amrex::Real fivethird =
            (solved_comp == EEN_ID) ? 5.0 / 3.0 : 1.0;

        // Manufactured Dirichlet value at the reconstructed EB foot
        const amrex::Real soln =
            rad_eb * rad_eb / (den_coeff * n_alpha) + n0;

        // Local plasma state at intcell for transport coefficients
        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; ++sp)
            ndens += sb_arr(intcell, sp);

        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        // ---- Diffusion contribution (EB-consistent stencil) ----
        const amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID),
                     ndens, emag, Tg);
        acoeff(intcell) -=
            Dcoeff / dx[dir] * alpha_eb * fivethird;
        rhs(intcell, rhs_comp) -=
            soln * Dcoeff / dx[dir] * alpha_eb * fivethird;

        // ---- Advection (drift) contribution ----
        const amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID),
                    ndens, emag, Tg);
        const amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;

        int step_sign = ((intcell[0] == lcell[0]) &&
                         (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                      && (intcell[2] == lcell[2])
#endif
                        ) ? -1 : +1;

        IntVect nb = intcell;
        nb[dir] += step_sign;

        if (sb_arr(rcell, CMASK_ID) < 1)
        {
            const amrex::Real value =
                0.5 * (3.0 * sb_arr(intcell, solved_comp)
                           - sb_arr(nb,      solved_comp));
            flux = (vel > 0.0) ? value : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }
        else
        {
            const amrex::Real value =
                0.5 * (sb_arr(intcell, solved_comp)
                     + sb_arr(nb,      solved_comp));
            flux = (vel > 0.0) ? soln : value;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
    {
        // no-op for other solved components
    }
}*/

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
    // Cap runaway alpha at oblique cut cells. Empirically the best
    // known configuration for this MMS test.
    //constexpr amrex::Real ALPHA_MAX_FACTOR = 100000.0;

    // -----------------------------------------------------------------
    // Identify the interior (fluid) cell and the cut cell across `face`
    // -----------------------------------------------------------------
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    amrex::Real vf_l = sb_arr(lcell, CMASK_ID);
    amrex::Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    IntVect cutcell = (vf_l <  vf_r) ? lcell : rcell;

    auto sgn3 = [](amrex::Real v) -> amrex::Real {
        if (v >  1.e-12) return  1.0;
        if (v < -1.e-12) return -1.0;
        return 0.0;
    };

    // -----------------------------------------------------------------
    // Interior cell center
    // -----------------------------------------------------------------
    amrex::Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5) * dx[2];
#endif

    // -----------------------------------------------------------------
    // EB centroid seed + EB normal from the cut cell (or, with the
    // Prob.H fix, the analytical wall foot/normal at the fluid cell).
    // -----------------------------------------------------------------
    amrex::Real xb_seed[AMREX_SPACEDIM];
    xb_seed[0] = sb_arr(intcell, CPX_ID);
    xb_seed[1] = sb_arr(intcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_seed[2] = sb_arr(intcell, CPZ_ID);
#endif

    amrex::Real n_eb[AMREX_SPACEDIM];
    n_eb[0] = sb_arr(intcell, NX_ID);
    n_eb[1] = sb_arr(intcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
    n_eb[2] = sb_arr(intcell, NZ_ID);
#endif

    amrex::Real nmag = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n_eb[d] * n_eb[d];
    nmag = std::sqrt(nmag);
    if (nmag < 1.0e-14) return;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) n_eb[d] /= nmag;

    // -----------------------------------------------------------------
    // Project xi onto the tangent plane through xb_seed to get the true
    // wall foot xb_eb of the wall normal from xi.
    // -----------------------------------------------------------------
    amrex::Real dxseed[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) dxseed[d] = xi[d] - xb_seed[d];

    amrex::Real distn = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) distn += dxseed[d] * n_eb[d];

    amrex::Real xb_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) xb_eb[d] = xi[d] - distn * n_eb[d];

    amrex::Real ccibvec_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec_eb[d] = xi[d] - xb_eb[d];

    amrex::Real d2_eb = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) d2_eb += ccibvec_eb[d] * ccibvec_eb[d];
    if (d2_eb < 1.0e-24) return;

    amrex::Real outward_normal_dir_eb = -sgn3(ccibvec_eb[dir]);
    amrex::Real alpha_eb =
        (ccibvec_eb[dir] / d2_eb) * outward_normal_dir_eb;
    if (amrex::Math::abs(alpha_eb) < 1.0e-14) return;

    // Cap |alpha_eb| rather than dropping rows.
    /*const amrex::Real alpha_max = ALPHA_MAX_FACTOR / (dx[dir] * dx[dir]);
    if (amrex::Math::abs(alpha_eb) > alpha_max)
    {
        alpha_eb = std::copysign(alpha_max, alpha_eb);
    }

    // Radius at the reconstructed EB foot; used by species Dirichlet.
    const amrex::Real rad_eb =
        std::sqrt(xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1]);

    // =================================================================
    // POT_ID: MMS Dirichlet phi = r^4 / 32 at reconstructed EB point
    // =================================================================
    if (solved_comp == POT_ID)
    {
        const amrex::Real rb2 = xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1];
        const amrex::Real bc_value = rb2 * rb2 / 32.0;

        acoeff(intcell)        += alpha_eb / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha_eb / dx[dir];
    }
    // =================================================================
    // Species (E, HEp, EEN): diffusion Dirichlet + drift-flux upwind
    // =================================================================
    else if (solved_comp == E_ID   ||
             solved_comp == HEp_ID ||
             solved_comp == EEN_ID)
    {
        const amrex::Real n0        = prob_parm.n0;
        const amrex::Real n_alpha   = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0 : 1.0;
        const amrex::Real fivethird =
            (solved_comp == EEN_ID) ? 5.0 / 3.0 : 1.0;

        const amrex::Real soln =
            rad_eb * rad_eb / (den_coeff * n_alpha) + n0;

        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; ++sp) ndens += sb_arr(intcell, sp);

        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        // ---- Diffusion Dirichlet contribution ----
        const amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell)        -= Dcoeff / dx[dir] * alpha_eb * fivethird;
        rhs(intcell, rhs_comp) -= soln * Dcoeff / dx[dir] * alpha_eb * fivethird;
// ---- Advection (drift) contribution ----
        /*const amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID),
                    ndens, emag, Tg);
        const amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;

        int step_sign = ((intcell[0] == lcell[0]) &&
                         (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                      && (intcell[2] == lcell[2])
#endif
                        ) ? -1 : +1;

        IntVect nb = intcell;
        nb[dir] += step_sign;

        if (sb_arr(rcell, CMASK_ID) < 1)
        {
            const amrex::Real value =
                0.5 * (3.0 * sb_arr(intcell, solved_comp)
                           - sb_arr(nb,      solved_comp));
            flux = (vel > 0.0) ? value : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }
        else
        {
            const amrex::Real value =
                0.5 * (sb_arr(intcell, solved_comp)
                     + sb_arr(nb,      solved_comp));
            flux = (vel > 0.0) ? soln : value;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
     // ---- Advection (drift): 2nd-order wall-consistent reconstruction ----
       const amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        const amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);

        const amrex::Real d_norm = std::sqrt(d2_eb);   // wall-face distance

        int step_sign = ((intcell[0] == lcell[0]) &&
                         (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                      && (intcell[2] == lcell[2])
#endif
                        ) ? -1 : +1;
        IntVect nb = intcell;
        nb[dir] += step_sign;

        const amrex::Real phi_i  = sb_arr(intcell, solved_comp);
        const amrex::Real phi_nb = sb_arr(nb,      solved_comp);

        // 2nd-order one-sided interior gradient (away from wall), then
        // linearly extrapolate the interior profile to the EB face at
        // distance d_norm. No limiter (smooth MMS) -> preserves O(h^2).
        const amrex::Real slope     = (phi_i - phi_nb) / dx[dir];
        const amrex::Real phi_wall_extrap = phi_i - slope * d_norm;

        amrex::Real flux;
        if (sb_arr(rcell, CMASK_ID) < 1.0_rt)
        {
            // EB on hi side: vel>0 -> toward wall (extrapolate), vel<0 -> from wall (soln exact)
            flux = (vel > 0.0_rt) ? phi_wall_extrap : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }
        else
        {
            // EB on lo side: vel>0 -> from wall (soln exact), vel<0 -> toward wall (extrapolate)
            flux = (vel > 0.0_rt) ? soln : phi_wall_extrap;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
    {
        // no-op for other solved components
    }
        
}*/

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

    // Match the driver's interior-cell choice EXACTLY (integer CMASK, ==1).
    // The driver flags this face because one side has mask 1 and the other 0.
    IntVect intcell = (int(sb_arr(lcell, CMASK_ID)) == 1) ? lcell : rcell;
    IntVect cutcell = (int(sb_arr(lcell, CMASK_ID)) == 1) ? rcell : lcell;

    auto sgn3 = [] AMREX_GPU_DEVICE(amrex::Real v) -> amrex::Real {
        if (v >  1.0e-12_rt) return  1.0_rt;
        if (v < -1.0e-12_rt) return -1.0_rt;
        return 0.0_rt;
    };

    // -----------------------------------------------------------------
    // Boundary point = EB centroid stored on the CUT cell, used DIRECTLY
    // (this is what MMS1 does and what gives 2nd order). Do NOT project
    // along the EB normal -- the projected foot uses a *borrowed* seed
    // and freezes an O(dx) tangential error -> 1st order.
    // -----------------------------------------------------------------
    amrex::Real xib[AMREX_SPACEDIM];
    xib[0] = sb_arr(cutcell, CPX_ID);
    xib[1] = sb_arr(cutcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xib[2] = sb_arr(cutcell, CPZ_ID);
#endif

    const amrex::Real rad = std::sqrt(xib[0] * xib[0] + xib[1] * xib[1]);

    // Vector from EB centroid to interior cell center; axis-aligned closure.
    amrex::Real ccibvec[AMREX_SPACEDIM];
    ccibvec[0] = prob_lo[0] + (intcell[0] + 0.5_rt) * dx[0] - xib[0];
    ccibvec[1] = prob_lo[1] + (intcell[1] + 0.5_rt) * dx[1] - xib[1];
#if (AMREX_SPACEDIM == 3)
    ccibvec[2] = prob_lo[2] + (intcell[2] + 0.5_rt) * dx[2] - xib[2];
#endif

    amrex::Real d2 = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) d2 += ccibvec[d] * ccibvec[d];
    if (d2 < 1.0e-24_rt) return;

    const amrex::Real outward_normal_dir = -sgn3(ccibvec[dir]);
    const amrex::Real alpha = ccibvec[dir] / d2 * outward_normal_dir;
    if (amrex::Math::abs(alpha) < 1.0e-14_rt) return;

    // =================================================================
    // POT_ID: Dirichlet phi = r^4 / 32 at the EB centroid
    // =================================================================
    if (solved_comp == POT_ID)
    {
        const amrex::Real bc_value = rad * rad * rad * rad / 32.0_rt;

        acoeff(intcell)        += alpha / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha / dx[dir];
    }
    // =================================================================
    // Species (E, HEp, EEN): diffusion Dirichlet + drift-flux upwind
    // (structure identical to MMS1's NI_ID branch)
    // =================================================================
    else if (solved_comp == E_ID   ||
             solved_comp == HEp_ID ||
             solved_comp == EEN_ID)
    {
        const amrex::Real n0        = prob_parm.n0;
        const amrex::Real n_alpha   = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0_rt : 1.0_rt;
        const amrex::Real fivethird = (solved_comp == EEN_ID) ? 5.0_rt / 3.0_rt : 1.0_rt;

        // Manufactured Dirichlet value at the EB centroid.
        const amrex::Real soln = rad * rad / (den_coeff * n_alpha) + n0;

        amrex::Real ndens = 0.0_rt;
        for (int sp = 0; sp < NUM_SPECIES; ++sp) ndens += sb_arr(intcell, sp);

        const amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        // ---- Diffusion Dirichlet contribution ----
        const amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell)        -= Dcoeff / dx[dir] * alpha * fivethird;
        rhs(intcell, rhs_comp) -= soln * Dcoeff / dx[dir] * alpha * fivethird;

        // ---- Advection (drift) contribution: plain upwind, no MUSCL ----
        const amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        const amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0_rt;

        if (int(sb_arr(rcell, CMASK_ID)) < 1)
        {
            flux = (vel > 0.0_rt) ? sb_arr(intcell, solved_comp) : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }
        else
        {
            flux = (vel > 0.0_rt) ? soln : sb_arr(intcell, solved_comp);
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
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
    // -----------------------------------------------------------------
    // Identify the interior (fluid) cell and the cut cell across `face`
    // -----------------------------------------------------------------
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    amrex::Real vf_l = sb_arr(lcell, CMASK_ID);
    amrex::Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    IntVect cutcell = (vf_l <  vf_r) ? lcell : rcell;

    auto sgn3 = [] AMREX_GPU_DEVICE (amrex::Real v) -> amrex::Real {
        if (v >  1.0e-12_rt) return  1.0_rt;
        if (v < -1.0e-12_rt) return -1.0_rt;
        return 0.0_rt;
    };

    // -----------------------------------------------------------------
    // Interior cell center
    // -----------------------------------------------------------------
    amrex::Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5_rt) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5_rt) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5_rt) * dx[2];
#endif

    // -----------------------------------------------------------------
    // Curvature-corrected wall foot + normal, stored on the interior
    // (regular) cell by Prob.H PASS 2 (quadric fit). USE DIRECTLY.
    // Do NOT re-project onto a tangent plane: the quadric foot is already
    // the foot-of-normal from xi on the true (curved) surface, so any
    // re-projection only re-introduces an O(h^2) tangential error.
    // -----------------------------------------------------------------
    amrex::Real xb_eb[AMREX_SPACEDIM];
    xb_eb[0] = sb_arr(intcell, CPX_ID);
    xb_eb[1] = sb_arr(intcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_eb[2] = sb_arr(intcell, CPZ_ID);
#endif

    amrex::Real n_eb[AMREX_SPACEDIM];
    n_eb[0] = sb_arr(intcell, NX_ID);
    n_eb[1] = sb_arr(intcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
    n_eb[2] = sb_arr(intcell, NZ_ID);
#endif

    amrex::Real nmag = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n_eb[d] * n_eb[d];
    nmag = std::sqrt(nmag);
    if (nmag < 1.0e-14_rt) return;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) n_eb[d] /= nmag;

    // Vector from wall foot to interior cell center (no re-projection).
    amrex::Real ccibvec_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec_eb[d] = xi[d] - xb_eb[d];

    amrex::Real d2_eb = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        d2_eb += ccibvec_eb[d] * ccibvec_eb[d];
    if (d2_eb < 1.0e-24_rt) return;

    amrex::Real outward_normal_dir_eb = -sgn3(ccibvec_eb[dir]);
    amrex::Real alpha_eb =
        (ccibvec_eb[dir] / d2_eb) * outward_normal_dir_eb;
    if (amrex::Math::abs(alpha_eb) < 1.0e-14_rt) return;

    // Radius at the reconstructed EB foot; used by species Dirichlet.
    const amrex::Real rad_eb =
        std::sqrt(xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1]);

    // =================================================================
    // POT_ID: MMS Dirichlet phi = r^4 / 32 at the (quadric) EB foot.
    // =================================================================
    if (solved_comp == POT_ID)
    {
        //const amrex::Real rb2 = xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1];
        //const amrex::Real bc_value = rb2 * rb2 / 32.0_rt;
        const amrex::Real rb2 = xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1];
        const amrex::Real bc_value = rb2 * rb2 / 32.0_rt;

        acoeff(intcell)        += alpha_eb / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha_eb / dx[dir];
    }
    // =================================================================
    // Species (E, HEp, EEN): diffusion Dirichlet + 2nd-order drift flux
    // All wall values are evaluated at xb_eb (radius rad_eb).
    // =================================================================
    else if (solved_comp == E_ID   ||
             solved_comp == HEp_ID ||
             solved_comp == EEN_ID)
    {
        const amrex::Real n0        = prob_parm.n0;
        const amrex::Real n_alpha   = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0_rt : 1.0_rt;
        const amrex::Real fivethird =
            (solved_comp == EEN_ID) ? 1.0 : 1.0_rt;

        const amrex::Real soln =
            rad_eb * rad_eb / (den_coeff * n_alpha) + n0;

        amrex::Real ndens = 0.0_rt;
        for (int sp = 0; sp < NUM_SPECIES; ++sp) ndens += sb_arr(intcell, sp);

        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        // ---- Diffusion Dirichlet contribution ----
        const amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell)        -= Dcoeff / dx[dir] * alpha_eb ;
        rhs(intcell, rhs_comp) -= soln * Dcoeff / dx[dir] * alpha_eb ;



        // ---- Advection (drift): 2nd-order wall-consistent reconstruction ----
        const amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        const amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);

        const amrex::Real d_norm = std::sqrt(d2_eb);   // wall-face distance

        int step_sign = ((intcell[0] == lcell[0]) &&
                         (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                      && (intcell[2] == lcell[2])
#endif
                        ) ? -1 : +1;
        IntVect nb = intcell;
        nb[dir] += step_sign;

        const amrex::Real phi_i  = sb_arr(intcell, solved_comp);
        const amrex::Real phi_nb = sb_arr(nb,      solved_comp);

        // Linear extrapolation of the interior profile to the EB face at
        // distance d_norm (2nd order; no limiter for smooth MMS).
        const amrex::Real slope           = (phi_i - phi_nb) / dx[dir];
        // slope points towards the wall (nb is the neighbour away from it), so the
        // extrapolation to the wall adds slope*d_norm. (Was '-', which returned the
        // value a distance d_norm INSIDE the fluid; fixed 2026-09-20.)
        const amrex::Real phi_wall_extrap = phi_i + slope * d_norm;

        amrex::Real flux;
        if (sb_arr(rcell, CMASK_ID) < 1.0_rt)
        {
            // EB on hi side: vel>0 -> toward wall (extrapolate),
            //                vel<0 -> from wall (soln exact)
            flux = (vel > 0.0_rt) ? phi_wall_extrap : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }

        else
        {
            // EB on lo side: vel>0 -> from wall (soln exact),
            //                vel<0 -> toward wall (extrapolate)
            flux = (vel > 0.0_rt) ? soln : phi_wall_extrap;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
    {
        // no-op for other solved components
    }
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
    // -----------------------------------------------------------------
    // Identify the interior (fluid) cell and the cut cell across `face`
    // -----------------------------------------------------------------
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    amrex::Real vf_l = sb_arr(lcell, CMASK_ID);
    amrex::Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;
    IntVect cutcell = (vf_l <  vf_r) ? lcell : rcell;

    auto sgn3 = [] AMREX_GPU_DEVICE (amrex::Real v) -> amrex::Real {
        if (v >  1.0e-12_rt) return  1.0_rt;
        if (v < -1.0e-12_rt) return -1.0_rt;
        return 0.0_rt;
    };

    // -----------------------------------------------------------------
    // Interior cell center
    // -----------------------------------------------------------------
    amrex::Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5_rt) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5_rt) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5_rt) * dx[2];
#endif

    // -----------------------------------------------------------------
    // O(h^2) quadric foot + normal from Prob.H PASS 2 (interior cell).
    // GEOMCHK confirmed these are unit, finite, and land on the wall.
    // -----------------------------------------------------------------
    amrex::Real xb_eb[AMREX_SPACEDIM];
    xb_eb[0] = sb_arr(intcell, CPX_ID);
    xb_eb[1] = sb_arr(intcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_eb[2] = sb_arr(intcell, CPZ_ID);
#endif

    amrex::Real n_eb[AMREX_SPACEDIM];
    n_eb[0] = sb_arr(intcell, NX_ID);
    n_eb[1] = sb_arr(intcell, NY_ID);
#if (AMREX_SPACEDIM == 3)
    n_eb[2] = sb_arr(intcell, NZ_ID);
#endif

    amrex::Real nmag = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += n_eb[d] * n_eb[d];
    nmag = std::sqrt(nmag);
    if (!(nmag > 1.0e-8_rt)) return;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) n_eb[d] /= nmag;

    // Vector from wall foot to interior cell center (no re-projection).
    amrex::Real ccibvec_eb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec_eb[d] = xi[d] - xb_eb[d];

    amrex::Real d2_eb = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        d2_eb += ccibvec_eb[d] * ccibvec_eb[d];
    if (!(d2_eb > 1.0e-24_rt)) return;

    // -----------------------------------------------------------------
    // NORMAL-CONSISTENT CLOSURE WEIGHT (MLMG-safe).
    //
    //   dn      = signed normal distance from cell center to the foot
    //   alpha_eb = -(n_eb[dir]^2) / dn_abs      (ALWAYS <= 0)
    //
    // Downstream the code applies alpha_eb / dx[dir], so the net diagonal
    // contribution is  -(n_eb[dir]^2) / (dn_abs * dx[dir]).  Summed over
    // the face directions (sum_dir n_eb[dir]^2 = 1) this reconstructs the
    // true normal derivative d/dn -> 2nd order.
    //
    // The sign is FIXED negative, exactly like the original stable weight
    //   -|ccibvec[dir]| / d2_eb,
    // so the diagonal keeps its sense and MLMG stays diagonally dominant.
    // (The previous crash came from a cell-varying sign; this removes it.)
    // -----------------------------------------------------------------
    amrex::Real dn = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) dn += ccibvec_eb[d] * n_eb[d];

    // Skip directions where the wall normal is ~perpendicular to `dir`;
    // the transverse direction (also visited) imposes the BC there.
    if (n_eb[dir] * n_eb[dir] < 1.0e-6_rt) return;

    // Floor the normal distance (5% of a cell) against small-cut blowup.
    const amrex::Real dn_abs = std::max(std::abs(dn), 0.05_rt * dx[dir]);

    const amrex::Real alpha_eb =
        -(n_eb[dir] * n_eb[dir]) / dn_abs;      // divided by dx[dir] below
    if (amrex::Math::abs(alpha_eb) < 1.0e-14_rt) return;

    const amrex::Real rad_eb =
        std::sqrt(xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1]);

    // =================================================================
    // POT_ID: MMS Dirichlet phi = r^4 / 32 at the quadric EB foot.
    // =================================================================
    if (solved_comp == POT_ID)
    {
        const amrex::Real rb2 = xb_eb[0] * xb_eb[0] + xb_eb[1] * xb_eb[1];
        const amrex::Real bc_value = rb2 * rb2 / 32.0_rt;

        acoeff(intcell)        += alpha_eb / dx[dir];
        rhs(intcell, rhs_comp) += bc_value * alpha_eb / dx[dir];
    }
    // =================================================================
    // Species (E, HEp, EEN): diffusion Dirichlet + drift flux.
    // =================================================================
    else if (solved_comp == E_ID   ||
             solved_comp == HEp_ID ||
             solved_comp == EEN_ID)
    {
        const amrex::Real n0        = prob_parm.n0;
        const amrex::Real n_alpha   = ECHARGE / EPS0;
        const amrex::Real den_coeff = (solved_comp == HEp_ID) ? 2.0_rt : 1.0_rt;
        const amrex::Real fivethird =
            (solved_comp == EEN_ID) ? 5.0_rt / 3.0_rt : 1.0_rt;

        const amrex::Real soln =
            rad_eb * rad_eb / (den_coeff * n_alpha) + n0;

        amrex::Real ndens = 0.0_rt;
        for (int sp = 0; sp < NUM_SPECIES; ++sp) ndens += sb_arr(intcell, sp);

        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        // ---- Diffusion Dirichlet (normal-consistent closure) ----
        const amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        acoeff(intcell)        -= Dcoeff / dx[dir] * alpha_eb * fivethird;
        rhs(intcell, rhs_comp) -= soln * Dcoeff / dx[dir] * alpha_eb * fivethird;

        // ---- Advection (drift): linear wall-consistent reconstruction ----
        const amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        const amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);

        // Use the true normal wall-face distance for the extrapolation.
        const amrex::Real d_norm = dn_abs;

        int step_sign = ((intcell[0] == lcell[0]) &&
                         (intcell[1] == lcell[1])
#if (AMREX_SPACEDIM == 3)
                      && (intcell[2] == lcell[2])
#endif
                        ) ? -1 : +1;
        IntVect nb = intcell;
        nb[dir] += step_sign;

        const amrex::Real phi_i  = sb_arr(intcell, solved_comp);
        const amrex::Real phi_nb = sb_arr(nb,      solved_comp);

        const amrex::Real slope           = (phi_i - phi_nb) / dx[dir];
        const amrex::Real phi_wall_extrap = phi_i - slope * d_norm;

        amrex::Real flux;
        if (sb_arr(rcell, CMASK_ID) < 1.0_rt)
        {
            flux = (vel > 0.0_rt) ? phi_wall_extrap : soln;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir] * fivethird;
        }
        else
        {
            flux = (vel > 0.0_rt) ? soln : phi_wall_extrap;
            rhs(intcell, rhs_comp) += flux * vel / dx[dir] * fivethird;
        }
    }
    else
    {
        // no-op for other solved components
    }
}*/


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
    Array4<Real> const& bcarr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg,
    Real app_voltage,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)
{
    // note: bdryLo/bdryHi grabs the face indices from bx that are at the
    // boundary since they are face indices, the bdry normal index is 0/n+1, n
    // is number of cells so the ghost cell index at left side is i-1 while it
    // is i on the right
    IntVect cell_int{AMREX_D_DECL(i, j, k)};
    IntVect ghost_cell{AMREX_D_DECL(i, j, k)};
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;

    int gcell_adjust, cell_adjust;

    // ghost_cell is one behind
    gcell_adjust = (sgn == -1) ? -1 : 0;
    // cell is one behind
    cell_adjust = (sgn == 1) ? -1 : 0;

    ghost_cell[dir] += gcell_adjust;
    cell_int[dir] += cell_adjust;

    if (sgn == -1)
    { // lo sides
        robin_a(ghost_cell) = 1.0;
        robin_b(ghost_cell) = 0.0;
        robin_f(ghost_cell) = 0.0;
    } else
    {
        robin_a(ghost_cell) = 1.0;
        robin_b(ghost_cell) = 0.0;
        robin_f(ghost_cell) = 0.0;
    }
}


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
{
    IntVect cell_int{AMREX_D_DECL(i, j, k)};
    IntVect ghost_cell{AMREX_D_DECL(i, j, k)};
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;

    int gcell_adjust, cell_adjust;

    // ghost_cell is one behind
    gcell_adjust = (sgn == -1) ? -1 : 0;
    // cell is one behind
    cell_adjust = (sgn == 1) ? -1 : 0;

    ghost_cell[dir] += gcell_adjust;
    cell_int[dir] += cell_adjust;

    if (sgn == -1 && dir == 1)
    { // lo sides
        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    } else if (sgn == 1 && dir == 1)
    {
        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    } else
    {
        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    }
}

AMREX_GPU_DEVICE AMREX_INLINE void species_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    int spec_id,
    int bcspec_id,
    Array4<Real> const& phi,
    Array4<Real> const& bcarr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)

{
    // note: bdryLo/bdryHi grabs the face indices from bx that are at the
    // boundary since they are face indices, the bdry normal index is 0/n+1, n
    // is number of cells so the ghost cell index at left side is i-1 while it
    // is i on the right
    IntVect cell_int{AMREX_D_DECL(i, j, k)};
    IntVect ghost_cell{AMREX_D_DECL(i, j, k)};
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;

    int gcell_adjust, cell_adjust;

    gcell_adjust = (sgn == -1) ? -1 : 0;
    cell_adjust = (sgn == 1) ? -1 : 0;

    ghost_cell[dir] += gcell_adjust;
    cell_int[dir] += cell_adjust;

    if (sgn == -1)
    {
        // lo sides
        // default homogenous Neumann
        robin_a(ghost_cell, bcspec_id) = 0.0;
        robin_b(ghost_cell, bcspec_id) = 1.0;
        robin_f(ghost_cell, bcspec_id) = 0.0;

        if (spec_id == E_ID || spec_id == HEp_ID || spec_id == EEN_ID)
        {
            bcarr(ghost_cell, bcspec_id) = 0.0 + prob_parm.n0;
        }

    } else
    {
        robin_a(ghost_cell, bcspec_id) = 0.0;
        robin_b(ghost_cell, bcspec_id) = 1.0;
        robin_f(ghost_cell, bcspec_id) = 0.0;

        amrex::Real alpha = ECHARGE / EPS0;

        if (spec_id == E_ID)
        {
            bcarr(ghost_cell, bcspec_id) = 1.0 / alpha + prob_parm.n0;
        }

        if (spec_id == HEp_ID)
        {
            bcarr(ghost_cell, bcspec_id) = 1.0 / (2.0 * alpha) + prob_parm.n0;
        }

        if (spec_id == EEN_ID)
        {
            bcarr(ghost_cell, bcspec_id) = 1.0 / alpha + prob_parm.n0;
        }
    }
}


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
    amrex::Real efieldmag = std::sqrt(
        amrex::Math::powi<2>(phi(cellid, EFX_ID)) +
        amrex::Math::powi<2>(phi(cellid, EFY_ID)) +
        amrex::Math::powi<2>(phi(cellid, EFZ_ID)));

    amrex::Real ndens = 1.0; // dummy
    amrex::Real mu =
        specMob(specid, phi(cellid, ETEMP_ID), ndens, efieldmag, Tg);
    vel = mu * phi(cellid, EFX_ID + dir);
    return (vel);
}

} // namespace user_transport

#endif
