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

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real surfacefunc(
    amrex::Real x,
    amrex::Real y,
    ProbParm const& prob_parm,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi)
{
    amrex::Real rad = std::sqrt(x * x + y * y);
    Real sgn = 0.0;
    if ((prob_parm.inner_radius <= rad) && (rad <= prob_parm.outer_radius))
    {
        sgn = 1.0;
    } else
    {
        sgn = -1.0;
    }
    return (sgn);
}

AMREX_GPU_DEVICE AMREX_INLINE void get_surface_point(
    IntVect cutcell,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<int, AMREX_SPACEDIM> domlo,
    GpuArray<int, AMREX_SPACEDIM> domhi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    ProbParm const& prob_parm,
    Real xib[AMREX_SPACEDIM])
{
    int niter = 50;
    Real tol = 1e-10;

    // guess
    xib[0] = prob_lo[0] + (cutcell[0] + 0.5) * dx[0];
    xib[1] = prob_lo[1] + (cutcell[1] + 0.5) * dx[1];
    amrex::Real radib = std::sqrt(xib[0] * xib[0] + xib[1] * xib[1]);
    amrex::Real midrad =
        0.5 * (prob_parm.inner_radius + prob_parm.outer_radius);
    int inner = (radib < midrad) ? 1 : 0;

    amrex::Real sgn_ib =
        surfacefunc(xib[0], xib[1], prob_parm, prob_lo, prob_hi);

    // complementary point to ib
    // if xib is inside cib_c is outside
    // if xib is outside xib_c is inside
    Real xib_c[AMREX_SPACEDIM];
    Real xmid[AMREX_SPACEDIM];

    for (int jj = 0; jj < AMREX_SPACEDIM; jj++)
    {
        int found = 0;
        for (int ii = 0; ii < AMREX_SPACEDIM; ii++)
        {
            xib_c[0] = prob_lo[0] + (cutcell[0] + ii) * dx[0];
            xib_c[1] = prob_lo[1] + (cutcell[1] + jj) * dx[1];

            if (surfacefunc(xib_c[0], xib_c[1], prob_parm, prob_lo, prob_hi) !=
                sgn_ib)
            {
                found = 1;
                break;
            }
        }
        if (found)
        {
            break;
        }
    }

    int iter = 0;
    for (iter = 0; iter < niter; iter++)
    {
        xmid[0] = 0.5 * (xib_c[0] + xib[0]);
        xmid[1] = 0.5 * (xib_c[1] + xib[1]);

        if (amrex::Math::abs(surfaceval(
                xmid[0], xmid[1], prob_parm, prob_lo, prob_hi, inner)) < tol)
        {
            break;
        }

        Real sgnmid =
            surfacefunc(xmid[0], xmid[1], prob_parm, prob_lo, prob_hi);
        Real sgn1 = surfacefunc(xib[0], xib[1], prob_parm, prob_lo, prob_hi);
        Real sgn2 =
            surfacefunc(xib_c[0], xib_c[1], prob_parm, prob_lo, prob_hi);

        if (sgnmid == sgn1)
        {
            xib[0] = xmid[0];
            xib[1] = xmid[1];
        } else
        {
            xib_c[0] = xmid[0];
            xib_c[1] = xmid[1];
        }
    }

    // final value
    xib[0] = xmid[0];
    xib[1] = xmid[1];
}

AMREX_GPU_DEVICE AMREX_INLINE void bc_ib(
    IntVect face,
    int dir,
    int sgn,
    int solved_comp,
    int rhs_comp,
    Array4<Real> const& sb_arr,
    Array4<Real> const& acoeff,
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
    amrex::Real xloc[2], xib[2];
    xloc[0] = prob_lo[0] + (lcell[0] + 0.5) * dx[0];
    xloc[1] = prob_lo[1] + (lcell[1] + 0.5) * dx[1];
    xloc[dir] += 0.5 * dx[dir];
    // amrex::Real rad = sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);

    // interior cell
    IntVect intcell = (sgn == 1) ? lcell : rcell;
    IntVect cutcell = (sgn == 1) ? rcell : lcell;
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;
    amrex::Real dx2 = dx[dir] * dx[dir];

    get_surface_point(
        cutcell, prob_lo, prob_hi, domlo, domhi, dx, prob_parm, xib);
    amrex::Real rad = sqrt(xib[0] * xib[0] + xib[1] * xib[1]);
    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);
    // amrex::Print()<<intcell<<std::endl;
    if (solved_comp == POT_ID)
    {
        if (0.5 * (prob_parm.inner_radius + prob_parm.outer_radius) < rad)
        {
            amrex::Real phi_dirc = rad * rad * rad * rad;
            amrex::Real dphidn =
                4 * rad * rad * rad * std::fabs(xib[dir]) / rad;
            amrex::Real bc_value = prob_parm.a_outer_dir * phi_dirc +
                                   prob_parm.b_outer_neumann * dphidn;
            amrex::Real ccibvec[AMREX_SPACEDIM];
            ccibvec[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0] - xib[0];
            ccibvec[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1] - xib[1];
            amrex::Real d_ccib2 =
                ccibvec[0] * ccibvec[0] + ccibvec[1] * ccibvec[1];
            amrex::Real d_ccib_sqrt = std::sqrt(d_ccib2);
            const amrex::Real alpha =
                ccibvec[dir] / d_ccib2 * outward_normal[dir];
            acoeff(intcell) += alpha / dx[dir] *
                               (1 + prob_parm.b_outer_neumann * alpha /
                                        (prob_parm.a_outer_dir -
                                         prob_parm.b_outer_neumann * alpha));
            rhs(intcell, rhs_comp) +=
                bc_value * alpha / dx[dir] * 1.0 /
                (prob_parm.a_outer_dir - prob_parm.b_outer_neumann * alpha);
            /*acoeff(intcell) += alpha / dx[dir] ;
            rhs(intcell, rhs_comp) += bc_value * alpha / dx[dir] ;*/
            /*amrex::Print() << " Applying outer Dirichlet BC at cell " << alpha
                           << " with bc_value = " << bc_value
                           << " and a_outer_dir = " << prob_parm.a_outer_dir
                           << " and b_outer_neumann = "
                           << prob_parm.b_outer_neumann << "\n";*/
        } else
        {
            amrex::Real phi_dirc = rad * rad * rad * rad;
            amrex::Real dphidn =
                -4 * rad * rad * rad * std::fabs(xib[dir]) / rad;
            amrex::Real bc_value = prob_parm.a_outer_dir * phi_dirc +
                                   prob_parm.b_outer_neumann * dphidn;
            amrex::Real ccibvec[AMREX_SPACEDIM];
            ccibvec[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0] - xib[0];
            ccibvec[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1] - xib[1];
            amrex::Real d_ccib2 =
                ccibvec[0] * ccibvec[0] + ccibvec[1] * ccibvec[1];
            amrex::Real d_ccib_sqrt = std::sqrt(d_ccib2);
            const amrex::Real alpha =
                ccibvec[dir] / d_ccib2 * outward_normal[dir];
            acoeff(intcell) += alpha / dx[dir] *
                               (1 + prob_parm.b_outer_neumann * alpha /
                                        (prob_parm.a_outer_dir -
                                         prob_parm.b_outer_neumann * alpha));
            rhs(intcell, rhs_comp) +=
                bc_value * alpha / dx[dir] * 1.0 /
                (prob_parm.a_outer_dir - prob_parm.b_outer_neumann * alpha);
            /*acoeff(intcell) += alpha / dx[dir] ;
            rhs(intcell, rhs_comp) += bc_value * alpha / dx[dir]*/
            ;
        }
    } else if (solved_comp == NI_ID)
    {
        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);
        amrex::Real emag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));
        amrex::Real Dcoeff =
            specDiff(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        amrex::Real ccibvec[AMREX_SPACEDIM];
        ccibvec[0] = prob_lo[0] + (intcell[0] + 0.5) * dx[0] - xib[0];
        ccibvec[1] = prob_lo[1] + (intcell[1] + 0.5) * dx[1] - xib[1];
        amrex::Real d_ccib2 = ccibvec[0] * ccibvec[0] + ccibvec[1] * ccibvec[1];
        amrex::Real d_ccib_sqrt = std::sqrt(d_ccib2);
        const amrex::Real alpha = ccibvec[dir] / d_ccib2 * outward_normal[dir];
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
