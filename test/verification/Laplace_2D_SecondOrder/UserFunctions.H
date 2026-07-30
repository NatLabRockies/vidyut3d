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
    if (solved_comp != POT_ID) return;

    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    Real vf_l = sb_arr(lcell, CMASK_ID);
    Real vf_r = sb_arr(rcell, CMASK_ID);

    IntVect intcell = (vf_l >= vf_r) ? lcell : rcell;

    // Interior cell center
    Real xi[AMREX_SPACEDIM];
    xi[0] = prob_lo[0] + (intcell[0] + 0.5_rt) * dx[0];
    xi[1] = prob_lo[1] + (intcell[1] + 0.5_rt) * dx[1];
#if (AMREX_SPACEDIM == 3)
    xi[2] = prob_lo[2] + (intcell[2] + 0.5_rt) * dx[2];
#endif

    // Seed boundary point from EB data
    Real xb_seed[AMREX_SPACEDIM];
    xb_seed[0] = sb_arr(intcell, CPX_ID);
    xb_seed[1] = sb_arr(intcell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
    xb_seed[2] = sb_arr(intcell, CPZ_ID);
#endif

    // Local EB normal from EB data
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

    // Boundary value:
    // for current annulus test, still classify by radius
    Real rb = std::sqrt(xb[0]*xb[0] + xb[1]*xb[1]);
    Real Rmid = 0.5_rt * (prob_parm.inner_radius + prob_parm.outer_radius);

    amrex::Real dphidr =
                (prob_parm.outer_voltage - prob_parm.inner_voltage) /
                std::log(prob_parm.outer_radius / prob_parm.inner_radius) / rb;
    amrex::Real dphidn = dphidr * std::fabs(xb[dir]) / rb;
    Real bc_value = (rb > Rmid)
        ? prob_parm.a_outer_dir * prob_parm.outer_voltage + prob_parm.b_outer_neumann * dphidn
        : prob_parm.a_inner_dir * prob_parm.inner_voltage - prob_parm.b_inner_neumann * dphidn;
    // Keep original Dirichlet geometry path intact.
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

    const Real outward_normal_dir = sgn3(ccibvec[dir]);
    const Real alpha_dir = (ccibvec[dir] / d2) * outward_normal_dir;
    if (amrex::Math::abs(alpha_dir) < 1.0e-14_rt) return;
    const Real coeff_dir = alpha_dir / dx[dir];

    const Real hn = std::abs(distn);
    if (hn < 1.0e-14_rt) return;

    // This closure is only valid when this face is a covered/uncovered
    // interface and the cut-face bcoeff has been nulled by null_bcoeff_at_ib.
    AMREX_ASSERT((vf_l >= 1.0_rt && vf_r < 1.0_rt) ||
                 (vf_l < 1.0_rt && vf_r >= 1.0_rt));

    Real n_in[AMREX_SPACEDIM];
    const Real nsign = (distn >= 0.0_rt) ? 1.0_rt : -1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) n_in[d] = nsign * n[d];

    Real hstep = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        hstep = amrex::max(hstep, std::abs(n_in[d]) * dx[d]);
    }
    if (hstep < 1.0e-14_rt) hstep = amrex::min(dx[0], dx[1]);

    const Real h2 = hn + hstep;
    const Real denom_h = h2 - hn;
    if (denom_h < 1.0e-14_rt) return;

    Real x_img[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) x_img[d] = xb[d] + h2 * n_in[d];

    const Real phi_i = sb_arr(intcell, POT_ID);
    Real phi_ii = phi_i;

#if (AMREX_SPACEDIM == 2)
    const Real gx = (x_img[0] - prob_lo[0]) / dx[0] - 0.5_rt;
    const Real gy = (x_img[1] - prob_lo[1]) / dx[1] - 0.5_rt;
    const int i0 = static_cast<int>(std::floor(gx));
    const int j0 = static_cast<int>(std::floor(gy));
    const Real tx = gx - static_cast<Real>(i0);
    const Real ty = gy - static_cast<Real>(j0);

    Real wsum = 0.0_rt;
    Real vsum = 0.0_rt;
    for (int ii = 0; ii < 2; ++ii)
    {
        for (int jj = 0; jj < 2; ++jj)
        {
            const IntVect iv(AMREX_D_DECL(i0 + ii, j0 + jj, 0));
            const bool in_domain = (iv[0] >= domlo[0] && iv[0] <= domhi[0] &&
                                    iv[1] >= domlo[1] && iv[1] <= domhi[1]);
            if (!in_domain) continue;
            if (sb_arr(iv, CMASK_ID) < 1.0_rt) continue;

            const Real wx = (ii == 0) ? (1.0_rt - tx) : tx;
            const Real wy = (jj == 0) ? (1.0_rt - ty) : ty;
            const Real w = wx * wy;
            wsum += w;
            vsum += w * sb_arr(iv, POT_ID);
        }
    }
    if (wsum > 1.0e-14_rt)
    {
        phi_ii = vsum / wsum;
    }
    else
    {
        int major_dir = 0;
        if (std::abs(n_in[1]) > std::abs(n_in[0])) major_dir = 1;
        IntVect nb = intcell;
        nb[major_dir] += (n_in[major_dir] >= 0.0_rt) ? 1 : -1;
        const bool nb_in_domain =
            (nb[0] >= domlo[0] && nb[0] <= domhi[0] && nb[1] >= domlo[1] &&
             nb[1] <= domhi[1]);
        if (nb_in_domain && sb_arr(nb, CMASK_ID) >= 1.0_rt)
        {
            phi_ii = sb_arr(nb, POT_ID);
        }
    }
#endif

    // Nonuniform 3-point one-sided derivative at the IB point:
    // dphi/dn = c0*phi_b + c1*phi_i + c2*phi_ii
    const Real c0 = -(hn + h2) / (hn * h2);
    const Real c1 = h2 / (hn * (h2 - hn));
    const Real c2 = -hn / (h2 * (h2 - hn));

    // Keep the Robin denominator structure using alpha and absorb the
    // higher-order term via beta.
    const Real alpha = -c0;
    const Real beta = c2 * (phi_ii - phi_i);

    if (amrex::Math::abs(alpha) < 1.0e-14_rt) return;
    // ---------------------------------------------------------------------
    // Select boundary coefficients
    // ---------------------------------------------------------------------

    Real a, b;

    if (rb > Rmid)
    {
        a = prob_parm.a_outer_dir;
        b = prob_parm.b_outer_neumann;
    }
    else
    {
        a = prob_parm.a_inner_dir;
        b = prob_parm.b_inner_neumann;
    }

    const Real coeff = alpha / hn;
    constexpr Real eps = 1.0e-14_rt;

//acoeff(intcell) +=
 //               (bscalar / ascalar) * bcoeff(intcell, 0.0) / (0.5 * dx2);
//            rhs(intcell, rhs_comp) +=
//                bscalar * bcoeff(intcell, 0.0) * phi_dirc / (0.5 * dx2);

    // ---------------------------------------------------------------------
    // Pure Dirichlet
    //      a * phi = f
    // ---------------------------------------------------------------------

    if (std::abs(a) > eps && std::abs(b) < eps)
    {
        /*acoeff(intcell) += coeff;

        rhs(intcell, rhs_comp) +=
            coeff * (bc_value / a);*/
      acoeff(intcell)+= (bscalar / ascalar) * bcoeff(intcell, 0.0) * coeff_dir;
      rhs(intcell, rhs_comp) +=
          (bscalar * bcoeff(intcell, 0.0)) * coeff_dir * (bc_value / a);
    }

    // ---------------------------------------------------------------------
    // Pure Neumann
    //      b * dphi/dn = f
    // ---------------------------------------------------------------------

   else if (std::abs(a) < eps && std::abs(b) > eps)
    {
      const Real grad_term = (bc_value / b) - beta;
      rhs(intcell, rhs_comp) +=
          (bscalar * bcoeff(intcell, 0.0)) * coeff * grad_term / alpha;
   } 
    // Robin
    //      a * phi + b * dphi/dn = f
    // ---------------------------------------------------------------------

// Robin:  a * phi + b * dphi/dn = f
   else
    {
       const Real bc_eff = bc_value - b * beta;
       const Real denom = a + b * alpha;              
       if (std::abs(denom) < eps) return;

       acoeff(intcell) += (bscalar / ascalar) * bcoeff(intcell, 0.0) * coeff * (a / denom);  
       rhs(intcell, rhs_comp) +=
           (bscalar * bcoeff(intcell, 0.0)) * coeff * bc_eff / denom;
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
