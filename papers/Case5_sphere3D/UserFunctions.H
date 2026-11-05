#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <VarDefines.H>
#include <MMSExact.H>

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

    // Point at which the wall data are evaluated: the intersection of the
    // grid line through the cell centre with the planar wall (ray value), or
    // the foot of the normal.
    Real xw[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) xw[d] = xb[d];
    if (prob_parm.ray_dirichlet)
    {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) xw[d] = xi[d];
        xw[dir] = xi[dir] - d2 / ccibvec[dir];
    }
    if (prob_parm.ray_dirichlet >= 2)
    {
        // Diagnostic only (uses the analytical sphere): exact intersection of
        // the grid line with the curved wall, distance tau along the line.
        Real ri2 = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) ri2 += xi[d] * xi[d];
        const Real rmid = 0.5_rt * (prob_parm.inner_radius + prob_parm.outer_radius);
        const bool outer = (ri2 > rmid * rmid);
        const Real Rw = outer ? prob_parm.outer_radius : prob_parm.inner_radius;
        const Real edir = (vf_l >= vf_r) ? 1.0_rt : -1.0_rt; // towards the masked cell
        const Real b = xi[dir] * edir;
        const Real c = ri2 - Rw * Rw;
        const Real disc = b * b - c;
        Real tau = -1.0_rt;
        if (disc >= 0.0_rt) tau = outer ? (-b + std::sqrt(disc)) : (-b - std::sqrt(disc));
        if (tau > 0.45_rt * dx[dir] && tau < 3.0_rt * dx[dir])
        {
            if (prob_parm.ray_dirichlet != 3)
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) xw[d] = xi[d];
                xw[dir] = xi[dir] + tau * edir;
            }
            if (prob_parm.ray_dirichlet != 4) alpha = -1.0_rt / tau; // alpha < 0 in this hook
        }
    }
    Real phi_wall = mms::phi_exact(xw, prob_parm.mms_a);
    Real ni_wall = mms::ni_exact(xw, prob_parm.mms_a);
    if (prob_parm.ray_dirichlet == 5)
    {
        // Wall data linearized about the foot: value at the foot plus the
        // tangential derivative of the wall data times the in-plane offset of
        // the ray point. Identical to the ray value for linear data, but the
        // data are never evaluated far from the cell at oblique faces.
        // xw is the planar ray point here; xb is the foot.
        Real gp[AMREX_SPACEDIM], gn[AMREX_SPACEDIM];
        mms::phi_grad(xb, prob_parm.mms_a, gp);
        mms::ni_grad(xb, prob_parm.mms_a, gn);
        phi_wall = mms::phi_exact(xb, prob_parm.mms_a);
        ni_wall = mms::ni_exact(xb, prob_parm.mms_a);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            // (xw - xb) lies in the wall plane, so only the tangential part
            // of the gradient contributes
            phi_wall += gp[d] * (xw[d] - xb[d]);
            ni_wall += gn[d] * (xw[d] - xb[d]);
        }
    }

    if (solved_comp == POT_ID)
    {
        acoeff(intcell) += alpha / dx[dir];
        rhs(intcell, rhs_comp) += phi_wall * alpha / dx[dir];
    }
   else if (solved_comp == NI_ID)
    {
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
        rhs(intcell, rhs_comp) -= ni_wall * Dcoeff / dx[dir] * alpha;
        amrex::Real mu =
            specMob(solved_comp, sb_arr(intcell, ETEMP_ID), ndens, emag, Tg);
        amrex::Real vel = mu * sb_arr(intcell, EFX_ID + dir);
        amrex::Real flux = 0.0;
        if (sb_arr(rcell, CMASK_ID) < 1)
        {
            flux = (vel > 0) ? sb_arr(intcell, solved_comp) : ni_wall;
            rhs(intcell, rhs_comp) -= flux * vel / dx[dir];
        } else
        {
            flux = (vel > 0) ? ni_wall : sb_arr(intcell, solved_comp);
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
