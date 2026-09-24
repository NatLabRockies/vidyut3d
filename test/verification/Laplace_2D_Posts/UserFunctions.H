#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

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

    // Interior (solve) cell: the fluid side (higher volume fraction).
    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;

    // Interior cell center (from the fluid interior cell)
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

    // Project cell center onto tangent plane/line through xb_seed
    Real dxseed[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) dxseed[d] = xi[d] - xb_seed[d];

    Real distn = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) distn += dxseed[d] * n[d];

    Real xb[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) xb[d] = xi[d] - distn * n[d];

    // Boundary value: the exact solution evaluated at the reconstructed wall
    // point. phi = x^2 - y^2 is harmonic, so it solves the Laplace equation
    // the case actually discretises, and its fourth derivatives vanish, so the
    // interior stencil reproduces it EXACTLY on any grid. Every bit of error
    // in this case therefore comes from the immersed boundary - which is the
    // whole point: it makes the gap, and only the gap, set the accuracy.
    Real bc_value = xb[0] * xb[0] - xb[1] * xb[1];
    // Vector from reconstructed boundary point to interior cell center
    Real ccibvec[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) ccibvec[d] = xi[d] - xb[d];

    Real d2 = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) d2 += ccibvec[d] * ccibvec[d];

    if (d2 < 1.0e-24_rt) return;

    auto sgn3 = [] AMREX_GPU_DEVICE(Real v) -> Real {
        if (v > 1.0e-12_rt) return 1.0_rt;
        if (v < -1.0e-12_rt) return -1.0_rt;
        return 0.0_rt;
    };

    Real outward_normal_dir = sgn3(ccibvec[dir]);
    Real alpha = (ccibvec[dir] / d2) * outward_normal_dir;

    if (amrex::Math::abs(alpha) < 1.0e-14_rt) return;
    // ---------------------------------------------------------------------
    // Select boundary coefficients
    // ---------------------------------------------------------------------

    // Every post is a Dirichlet wall holding the exact solution, so there is
    // no inner/outer electrode to tell apart as in the annulus this came from.
    const Real a = 1.0_rt;
    const Real b = 0.0_rt;

    const Real coeff = alpha / dx[dir];
    constexpr Real eps = 1.0e-14_rt;

    // acoeff(intcell) +=
    //                (bscalar / ascalar) * bcoeff(intcell, 0.0) / (0.5 * dx2);
    //            rhs(intcell, rhs_comp) +=
    //                bscalar * bcoeff(intcell, 0.0) * phi_dirc / (0.5 * dx2);

    // ---------------------------------------------------------------------
    // Pure Dirichlet
    //      a * phi = f
    // ---------------------------------------------------------------------

    // The accumulations below are atomic on purpose. set_explicit_fluxes_at_ib
    // launches one thread per FACE, and in a narrow gap a single fluid cell can
    // have wall faces on both sides in the same direction - the
    // covered|valid|covered pattern. Two threads then update the same acoeff
    // and rhs entries, and a plain += loses one of the two wall contributions
    // on a GPU. It is correct on a CPU, where the faces are visited in turn,
    // which is why the convergence numbers in the README are unaffected. The
    // hierarchy this case runs has 14 such cells on the base grid and 22 on
    // each of the two intermediate levels, so the race is reachable here and
    // not a theoretical concern.
    if (std::abs(a) > eps && std::abs(b) < eps)
    {
        amrex::HostDevice::Atomic::Add(
            acoeff.ptr(intcell),
            (bscalar / ascalar) * bcoeff(intcell, 0.0) * coeff);
        amrex::HostDevice::Atomic::Add(
            rhs.ptr(intcell, rhs_comp),
            (bscalar * bcoeff(intcell, 0.0)) * coeff * (bc_value / a));
    }

    // ---------------------------------------------------------------------
    // Pure Neumann
    //      b * dphi/dn = f
    // ---------------------------------------------------------------------

    else if (std::abs(a) < eps && std::abs(b) > eps)
    {
        amrex::HostDevice::Atomic::Add(
            rhs.ptr(intcell, rhs_comp),
            (bscalar * bcoeff(intcell, 0.0)) * coeff * bc_value / (b * alpha));
    }
    // Robin
    //      a * phi + b * dphi/dn = f
    // ---------------------------------------------------------------------

    // Robin:  a * phi + b * dphi/dn = f
    else
    {
        const Real denom = a + b * alpha;
        if (std::abs(denom) < eps) return;

        amrex::HostDevice::Atomic::Add(
            acoeff.ptr(intcell),
            (bscalar / ascalar) * bcoeff(intcell, 0.0) * coeff * (a / denom));
        amrex::HostDevice::Atomic::Add(
            rhs.ptr(intcell, rhs_comp),
            (bscalar * bcoeff(intcell, 0.0)) * coeff * bc_value / denom);
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
{
    // The posts leave the domain box in the fluid, unlike the annulus this
    // case came from, so the box has to carry the exact solution. It is an
    // ordinary grid-aligned Dirichlet boundary and contributes no immersed
    // boundary error of its own.
    IntVect ghost_cell{AMREX_D_DECL(i, j, k)};
    IntVect cell_int{AMREX_D_DECL(i, j, k)};
    ghost_cell[dir] += (sgn == -1) ? -1 : 0;
    cell_int[dir] += (sgn == 1) ? -1 : 0;

    amrex::Real xf[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        xf[d] = prob_lo[d] + (cell_int[d] + 0.5) * dx[d];
    }
    xf[dir] = (sgn == -1) ? prob_lo[dir] : prob_hi[dir];

    const amrex::Real phi_exact = xf[0] * xf[0] - xf[1] * xf[1];
    robin_a(ghost_cell) = 1.0;
    robin_b(ghost_cell) = 0.0;
    robin_f(ghost_cell) = phi_exact;
    bc_arr(ghost_cell) = phi_exact;
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
    IntVect iv,
    int dir,
    int specid,
    Array4<Real> const& phi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg)
{
    amrex::Real efieldmag = std::sqrt(
        amrex::Math::powi<2>(phi(iv, EFX_ID)) +
        amrex::Math::powi<2>(phi(iv, EFY_ID)) +
        amrex::Math::powi<2>(phi(iv, EFZ_ID)));

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(iv, sp);

    amrex::Real mu = specMob(specid, phi(iv, ETEMP_ID), ndens, efieldmag, Tg);

    return (mu * phi(iv, EFX_ID + dir));
}
} // namespace user_transport
#endif
