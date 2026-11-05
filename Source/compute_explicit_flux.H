#ifndef _compute_explicit_flux_H_
#define _compute_explicit_flux_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <UnivConstants.H>
#include <VarDefines.H>
#include <PlasmaChem.H>
#include <type_traits>
#include <utility>

using namespace amrex;

#define heps 1.0e-30

//===========================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
gk(amrex::Real wk, amrex::Real gama_k)
{
    amrex::Real gkval =
        wk * (gama_k + gama_k * gama_k - 3.0 * gama_k * wk + wk * wk) /
        (gama_k * gama_k + wk * (1 - 2.0 * gama_k));
    return (gkval);
}
//===========================================================================
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real weno(
    amrex::Real phi_im2,
    amrex::Real phi_im1,
    amrex::Real phi_i,
    amrex::Real phi_ip1,
    amrex::Real phi_ip2,
    int scheme = 1)
{

    // see description of all flavors of WENO here
    // Rathan, Samala, and G. Naga Raju. "A modified fifth-order WENO scheme
    // for hyperbolic conservation laws." Computers & Mathematics with
    // Applications 75.5 (2018): 1531-1549.

    amrex::Real eps = 1e-20;

    amrex::Real phi_iphalf_1 =
        (1.0 / 3.0) * phi_im2 - (7.0 / 6.0) * phi_im1 + (11.0 / 6.0) * phi_i;
    amrex::Real phi_iphalf_2 =
        -(1.0 / 6.0) * phi_im1 + (5.0 / 6.0) * phi_i + (1.0 / 3.0) * phi_ip1;
    amrex::Real phi_iphalf_3 =
        (1.0 / 3.0) * phi_i + (5.0 / 6.0) * phi_ip1 - (1.0 / 6.0) * phi_ip2;

    amrex::Real beta_1 =
        (13.0 / 12.0) * amrex::Math::powi<2>(phi_im2 - 2.0 * phi_im1 + phi_i) +
        (1.0 / 4.0) *
            amrex::Math::powi<2>(phi_im2 - 4.0 * phi_im1 + 3.0 * phi_i);

    amrex::Real beta_2 =
        (13.0 / 12.0) * amrex::Math::powi<2>(phi_im1 - 2.0 * phi_i + phi_ip1) +
        (1.0 / 4.0) * amrex::Math::powi<2>(phi_im1 - phi_ip1);

    amrex::Real beta_3 =
        (13.0 / 12.0) * amrex::Math::powi<2>(phi_i - 2.0 * phi_ip1 + phi_ip2) +
        (1.0 / 4.0) *
            amrex::Math::powi<2>(3.0 * phi_i - 4.0 * phi_ip1 + phi_ip2);

    amrex::Real gama_1 = (1.0 / 10.0);
    amrex::Real gama_2 = (6.0 / 10.0);
    amrex::Real gama_3 = (3.0 / 10.0);

    amrex::Real w1, w2, w3;
    amrex::Real alpha_1, alpha_2, alpha_3;

    // this would return to fifth order scheme
    /*alpha_1 = gama_1;
      alpha_2 = gama_2;
      alpha_3 = gama_3;*/

    // WENO JS
    if (scheme == 1)
    {
        alpha_1 = gama_1 / amrex::Math::powi<2>(beta_1 + eps);
        alpha_2 = gama_2 / amrex::Math::powi<2>(beta_2 + eps);
        alpha_3 = gama_3 / amrex::Math::powi<2>(beta_3 + eps);
    }
    // WENO M
    else if (scheme == 2)
    {
        // do JS stuff
        alpha_1 = gama_1 / amrex::Math::powi<2>(beta_1 + eps);
        alpha_2 = gama_2 / amrex::Math::powi<2>(beta_2 + eps);
        alpha_3 = gama_3 / amrex::Math::powi<2>(beta_3 + eps);

        amrex::Real sum_alpha = (alpha_1 + alpha_2 + alpha_3);

        w1 = alpha_1 / sum_alpha;
        w2 = alpha_2 / sum_alpha;
        w3 = alpha_3 / sum_alpha;

        alpha_1 = gk(w1, gama_1);
        alpha_2 = gk(w2, gama_2);
        alpha_3 = gk(w3, gama_3);

    }
    // WENO Z
    else if (scheme == 3)
    {
        amrex::Real tau = std::abs(beta_1 - beta_3);
        alpha_1 = gama_1 * (1.0 + amrex::Math::powi<2>(tau / (beta_1 + eps)));
        alpha_2 = gama_2 * (1.0 + amrex::Math::powi<2>(tau / (beta_2 + eps)));
        alpha_3 = gama_3 * (1.0 + amrex::Math::powi<2>(tau / (beta_3 + eps)));

    } else
    {
        amrex::Abort("Unknown WENO scheme\n");
    }

    amrex::Real sum_alpha = (alpha_1 + alpha_2 + alpha_3);

    w1 = alpha_1 / sum_alpha;
    w2 = alpha_2 / sum_alpha;
    w3 = alpha_3 / sum_alpha;

    amrex::Real phi_iphalf =
        w1 * phi_iphalf_1 + w2 * phi_iphalf_2 + w3 * phi_iphalf_3;

    return (phi_iphalf);
}
//====================================================================
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void weno_reconstruct(
    amrex::Real uim2,
    amrex::Real uim1,
    amrex::Real ui,
    amrex::Real uip1,
    amrex::Real uip2,
    amrex::Real uip3,
    amrex::Real& umhalf,
    amrex::Real& uphalf,
    int weno_scheme)
{
    uphalf = weno(uim2, uim1, ui, uip1, uip2, weno_scheme);
    umhalf = weno(uip3, uip2, uip1, ui, uim1, weno_scheme);
}
//====================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real get_slope_lim(
    amrex::Real c,
    amrex::Real diff_left,
    amrex::Real diff_mid,
    amrex::Real diff_right)
{
    // MINBEE vesion of limiter for
    // WAF scheme, see page 502,
    // Toro's Riemann solver textbook
    amrex::Real r, lim;
    r = 0.0;
    r = (c >= 0.0) ? diff_left / (diff_mid + heps)
                   : diff_right / (diff_mid + heps);

    lim = 0.0;
    if (r <= 0.0)
    {
        lim = 1.0;
    } else if (r > 0.0 && r <= 1.0)
    {
        // NOTE: something funky can happen
        // if c or advective CFL is greater than 1
        // But we should be safe with
        // explicit advection :)
        lim = 1.0 - (1.0 - amrex::Math::abs(c)) * r;
    } else
    {
        lim = amrex::Math::abs(c);
    }
    return (lim);
}
//====================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real get_secondorder_WAF_flux(
    amrex::Real vel_L,
    amrex::Real vel_R,
    amrex::Real phi_Lm1,
    amrex::Real phi_L,
    amrex::Real phi_R,
    amrex::Real phi_Rp1,
    amrex::Real dx,
    amrex::Real dt)
{
    // See TVD Version of WAF schemes
    // Page 501, Toro's Riemann solver texbook

    amrex::Real diff_left = phi_L - phi_Lm1;
    amrex::Real diff_mid = phi_R - phi_L;
    amrex::Real diff_right = phi_Rp1 - phi_R;
    amrex::Real vel_mid = 0.5 * (vel_L + vel_R);
    amrex::Real vel_sgn = std::copysign(1.0, vel_mid);
    amrex::Real c = vel_mid * dt / dx;
    amrex::Real fL = phi_L * vel_L;
    amrex::Real fR = phi_R * vel_R;

    amrex::Real lim = get_slope_lim(c, diff_left, diff_mid, diff_right);

    amrex::Real flux = 0.5 * (fL + fR) - 0.5 * vel_sgn * lim * (fR - fL);

    return (flux);
}
//====================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real get_firstorder_upwind_flux(
    amrex::Real vel_L, amrex::Real vel_R, amrex::Real phi_L, amrex::Real phi_R)
{
    amrex::Real fL = phi_L * vel_L;
    amrex::Real fR = phi_R * vel_R;
    amrex::Real vel_mid = 0.5 * (vel_L + vel_R);
    amrex::Real vel_sgn = std::copysign(1.0, vel_mid);
    amrex::Real flux = fL * 0.5 * (1.0 + vel_sgn) + fR * 0.5 * (1.0 - vel_sgn);
    return (flux);
}
//====================================================================
// Optional per-case hook for the WENO5 stencil cells behind an IB inflow
// wall (vidyut.hyp_wall_ghosts=1). A case defines, in UserFunctions.H,
//
//   AMREX_GPU_DEVICE AMREX_INLINE bool ib_wall_value(
//       IntVect cell, int dir, int sgn, int specid,
//       Array4<Real> const& phi, GpuArray<Real, AMREX_SPACEDIM> prob_lo,
//       GpuArray<Real, AMREX_SPACEDIM> dx, Real time,
//       ProbParm const& prob_parm, Real& dist, Real& value)
//
// in namespace user_transport. cell is the fluid cell next to the wall and
// sgn = +1 if the wall is on its high side along dir. It returns true when
// the wall has a Dirichlet value for specid, with value set to it and dist to
// the distance from the cell centre to the wall along dir (at least h/2).
// The template below is found only when a case does not define the hook
// (or defines it with parameter types that need a conversion), and the ghost
// cells then copy the fluid cell. The run log says which one was found.
namespace user_transport {
struct no_ib_wall_value
{};
template <typename... Args>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE no_ib_wall_value ib_wall_value(Args&&...)
{
    return {};
}
} // namespace user_transport

// whether the case defines ib_wall_value with a matching signature
inline constexpr bool ib_wall_value_defined = !std::is_same_v<
    decltype(user_transport::ib_wall_value(
        std::declval<const IntVect&>(),
        0,
        0,
        0,
        std::declval<Array4<Real> const&>(),
        std::declval<const GpuArray<Real, AMREX_SPACEDIM>&>(),
        std::declval<const GpuArray<Real, AMREX_SPACEDIM>&>(),
        Real(0),
        std::declval<ProbParm const&>(),
        std::declval<Real&>(),
        std::declval<Real&>())),
    user_transport::no_ib_wall_value>;

// true/false from a case hook, false from the fallback
AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool ib_hook_result(bool known)
{
    return known;
}
AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
ib_hook_result(user_transport::no_ib_wall_value)
{
    return false;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool ib_inflow_wall(
    const IntVect& cell,
    int dir,
    int sgn,
    int specid,
    Array4<Real> const& phi,
    const GpuArray<Real, AMREX_SPACEDIM>& prob_lo,
    const GpuArray<Real, AMREX_SPACEDIM>& dx,
    amrex::Real time,
    ProbParm const& prob_parm,
    amrex::Real& dist,
    amrex::Real& value)
{
    const bool known = ib_hook_result(
        user_transport::ib_wall_value(
            cell, dir, sgn, specid, phi, prob_lo, dx, time, prob_parm, dist,
            value));
    // a fluid cell is never cut, so the wall is at least h/2 away
    if (known && dist < 0.499 * dx[dir])
    {
        amrex::Abort("user_transport::ib_wall_value returned dist < h/2");
    }
    return known;
}
//====================================================================
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux(
    int i,
    int j,
    int k,
    int sweepdir,
    int specid,
    int fluxid,
    Array4<Real> const& phi, // state variable
    const GpuArray<int, AMREX_SPACEDIM> bclo,
    const GpuArray<int, AMREX_SPACEDIM> bchi,
    const GpuArray<int, AMREX_SPACEDIM> domlo,
    const GpuArray<int, AMREX_SPACEDIM> domhi,
    Array4<Real> const& fx,
    amrex::Real Tg,
    amrex::Real Pg,
    amrex::Real time,
    const GpuArray<Real, AMREX_SPACEDIM>& dx,
    amrex::Real dt,
    ProbParm const& prob_parm,
    int hyp_order,
    int user_defined_vel,
    int weno_scheme,
    int using_ib,
    Array4<Real const> const& bc_a,
    Array4<Real const> const& bc_b,
    Array4<Real const> const& bc_f,
    int wall_ghosts,
    const GpuArray<Real, AMREX_SPACEDIM>& prob_lo)
{

    //|--left1--|--left--|--right--|--right1--|
    //|--(i-2)--|--(i-1)--|--i--|--(i+1)--|
    // remember: face i separates (i-1) and i cells

    IntVect face{AMREX_D_DECL(i, j, k)};
    IntVect left{AMREX_D_DECL(i, j, k)};
    IntVect right{AMREX_D_DECL(i, j, k)};
    left[sweepdir] -= 1;

    Real phi_L = phi(left, specid);
    Real phi_R = phi(right, specid);
    Real vel_L;
    Real vel_R;

    int mask_L = (using_ib == 1) ? int(phi(left, CMASK_ID)) : 1;
    int mask_R = (using_ib == 1) ? int(phi(right, CMASK_ID)) : 1;

    int valid_left_and_right_cells = mask_L * mask_R;

    // default value if face involves covered cells
    fx(face, fluxid) = 0.0;

    if (valid_left_and_right_cells)
    {
        if (user_defined_vel == 1)
        {
            vel_L = user_transport::compute_vel(
                left, sweepdir, specid, phi, dx, time, prob_parm, Tg, Pg);
            vel_R = user_transport::compute_vel(
                right, sweepdir, specid, phi, dx, time, prob_parm, Tg, Pg);
        } else
        {
            vel_L = plasmachem_transport::compute_vel(
                left, sweepdir, specid, phi, dx, time, prob_parm, Tg, Pg);
            vel_R = plasmachem_transport::compute_vel(
                right, sweepdir, specid, phi, dx, time, prob_parm, Tg, Pg);
        }
        if (hyp_order == 1)
        {
            fx(face, fluxid) =
                get_firstorder_upwind_flux(vel_L, vel_R, phi_L, phi_R);
        } else if (hyp_order == 2)
        {
            IntVect left1{AMREX_D_DECL(i, j, k)};
            IntVect right1{AMREX_D_DECL(i, j, k)};
            left1[sweepdir] -= 2;
            right1[sweepdir] += 1;

            int mask_L1 = (using_ib == 1) ? int(phi(left1, CMASK_ID)) : 1;
            int mask_R1 = (using_ib == 1) ? int(phi(right1, CMASK_ID)) : 1;

            // reduce to 1st order near IB
            amrex::Real phi_Lm1 =
                (mask_L1 == 1) ? phi(left1, specid) : phi(left, specid);
            amrex::Real phi_Rp1 =
                (mask_R1 == 1) ? phi(right1, specid) : phi(right, specid);

#if AMREX_SPACEDIM > 1
            amrex::Real dxmin = amrex::min(AMREX_D_DECL(dx[0], dx[1], dx[2]));
#else
            amrex::Real dxmin = dx[0];
#endif
            fx(face, fluxid) = get_secondorder_WAF_flux(
                vel_L, vel_R, phi_Lm1, phi_L, phi_R, phi_Rp1, dxmin, dt);
        } else if (hyp_order == 5)
        {
            IntVect left1{AMREX_D_DECL(i, j, k)};
            IntVect right1{AMREX_D_DECL(i, j, k)};
            IntVect left2{AMREX_D_DECL(i, j, k)};
            IntVect right2{AMREX_D_DECL(i, j, k)};
            left1[sweepdir] -= 2;
            right1[sweepdir] += 1;
            left2[sweepdir] -= 3;
            right2[sweepdir] += 2;

            int mask_L1 = (using_ib == 1) ? int(phi(left1, CMASK_ID)) : 1;
            int mask_L2 = (using_ib == 1) ? int(phi(left2, CMASK_ID)) : 1;

            int mask_R1 = (using_ib == 1) ? int(phi(right1, CMASK_ID)) : 1;
            int mask_R2 = (using_ib == 1) ? int(phi(right2, CMASK_ID)) : 1;

            amrex::Real vel_Lm1, vel_Lm2, vel_Rp1, vel_Rp2;

            if (user_defined_vel == 1)
            {
                vel_Lm1 = (mask_L1 == 1) ? user_transport::compute_vel(
                                               left1, sweepdir, specid, phi, dx,
                                               time, prob_parm, Tg, Pg)
                                         : vel_L;
                vel_Lm2 = (mask_L2 == 1) ? user_transport::compute_vel(
                                               left2, sweepdir, specid, phi, dx,
                                               time, prob_parm, Tg, Pg)
                                         : vel_L;
                vel_Rp1 = (mask_R1 == 1) ? user_transport::compute_vel(
                                               right1, sweepdir, specid, phi,
                                               dx, time, prob_parm, Tg, Pg)
                                         : vel_R;
                vel_Rp2 = (mask_R2 == 1) ? user_transport::compute_vel(
                                               right2, sweepdir, specid, phi,
                                               dx, time, prob_parm, Tg, Pg)
                                         : vel_R;
            } else
            {
                vel_Lm1 = (mask_L1 == 1) ? plasmachem_transport::compute_vel(
                                               left1, sweepdir, specid, phi, dx,
                                               time, prob_parm, Tg, Pg)
                                         : vel_L;
                vel_Lm2 = (mask_L2 == 1) ? plasmachem_transport::compute_vel(
                                               left2, sweepdir, specid, phi, dx,
                                               time, prob_parm, Tg, Pg)
                                         : vel_L;
                vel_Rp1 = (mask_R1 == 1) ? plasmachem_transport::compute_vel(
                                               right1, sweepdir, specid, phi,
                                               dx, time, prob_parm, Tg, Pg)
                                         : vel_R;
                vel_Rp2 = (mask_R2 == 1) ? plasmachem_transport::compute_vel(
                                               right2, sweepdir, specid, phi,
                                               dx, time, prob_parm, Tg, Pg)
                                         : vel_R;
            }

            amrex::Real ulm2 =
                (mask_L2 == 1) ? phi(left2, specid) : phi(left, specid);
            amrex::Real ulm1 =
                (mask_L1 == 1) ? phi(left1, specid) : phi(left, specid);
            amrex::Real ul = phi(left, specid);
            amrex::Real ur = phi(right, specid);
            amrex::Real urp1 =
                (mask_R1 == 1) ? phi(right1, specid) : phi(right, specid);
            amrex::Real urp2 =
                (mask_R2 == 1) ? phi(right2, specid) : phi(right, specid);

            // WENO5 stencil cells outside the fluid with
            // vidyut.hyp_wall_ghosts=1. A copy of the nearest fluid cell makes
            // WENO5 pick a constant, first-order sub-stencil next to an
            // outflow wall. The drift velocity is always extrapolated
            // linearly. The value is extrapolated linearly where the flow
            // leaves through the wall. Where it enters, the value is linear
            // through the Dirichlet wall value if one is known (species BC
            // data at domain faces, user_transport::ib_wall_value at IB
            // walls) and a copy of the nearest fluid cell otherwise. Linear
            // extrapolation against the flow is unstable at a domain inflow
            // face, and was unstable at a curved IB inflow wall in 2D
            // (N = 256); at a grid-aligned IB wall it was stable.
            //
            // IB: masked stencil cells (int(CMASK) != 1, so cut cells too).
            // The face itself has two fluid cells, so ul and ur are fluid.
            if (wall_ghosts == 1 && using_ib == 1)
            {
                amrex::Real tw = 0.0; // distance to the inflow wall
                amrex::Real uw = 0.0; // Dirichlet value at that wall
                if (mask_L1 != 1)
                {
                    vel_Lm1 = 2.0 * vel_L - vel_R;
                    vel_Lm2 = 2.0 * vel_Lm1 - vel_L;
                    if (vel_L < 0.0)
                    {
                        ulm1 = 2.0 * ul - ur;
                        ulm2 = 2.0 * ulm1 - ul;
                    } else if (ib_inflow_wall(
                                   left, sweepdir, -1, specid, phi, prob_lo, dx,
                                   time, prob_parm, tw, uw))
                    {
                        ulm1 = ul + (uw - ul) * dx[sweepdir] / tw;
                        ulm2 = ul + (uw - ul) * 2.0 * dx[sweepdir] / tw;
                    } else
                    {
                        ulm1 = ul;
                        ulm2 = ul;
                    }
                } else if (mask_L2 != 1)
                {
                    vel_Lm2 = 2.0 * vel_Lm1 - vel_L;
                    if (vel_Lm1 < 0.0)
                    {
                        ulm2 = 2.0 * ulm1 - ul;
                    } else if (ib_inflow_wall(
                                   left1, sweepdir, -1, specid, phi, prob_lo,
                                   dx, time, prob_parm, tw, uw))
                    {
                        ulm2 = ulm1 + (uw - ulm1) * dx[sweepdir] / tw;
                    } else
                    {
                        ulm2 = ulm1;
                    }
                }
                if (mask_R1 != 1)
                {
                    vel_Rp1 = 2.0 * vel_R - vel_L;
                    vel_Rp2 = 2.0 * vel_Rp1 - vel_R;
                    if (vel_R > 0.0)
                    {
                        urp1 = 2.0 * ur - ul;
                        urp2 = 2.0 * urp1 - ur;
                    } else if (ib_inflow_wall(
                                   right, sweepdir, 1, specid, phi, prob_lo, dx,
                                   time, prob_parm, tw, uw))
                    {
                        urp1 = ur + (uw - ur) * dx[sweepdir] / tw;
                        urp2 = ur + (uw - ur) * 2.0 * dx[sweepdir] / tw;
                    } else
                    {
                        urp1 = ur;
                        urp2 = ur;
                    }
                } else if (mask_R2 != 1)
                {
                    vel_Rp2 = 2.0 * vel_Rp1 - vel_R;
                    if (vel_Rp1 > 0.0)
                    {
                        urp2 = 2.0 * urp1 - ur;
                    } else if (ib_inflow_wall(
                                   right1, sweepdir, 1, specid, phi, prob_lo,
                                   dx, time, prob_parm, tw, uw))
                    {
                        urp2 = urp1 + (uw - urp1) * dx[sweepdir] / tw;
                    } else
                    {
                        urp2 = urp1;
                    }
                }
            }

            // Domain faces: stencil cells beyond a non-periodic face. The
            // Dirichlet value (robin b = 0) is taken from the species BC data
            // in the first ghost cell, and the value at the face is matched.
            if (wall_ghosts == 1)
            {
                const int d = sweepdir;
                amrex::Real* uu[6] = {&ulm2, &ulm1, &ul, &ur, &urp1, &urp2};
                amrex::Real* vv[6] = {&vel_Lm2, &vel_Lm1, &vel_L,
                                      &vel_R,   &vel_Rp1, &vel_Rp2};
                IntVect base = face;
                base[d] -= 3;
                for (int q = 0; q < 6; ++q)
                {
                    IntVect c = base;
                    c[d] += q;
                    // periodic, axis and symmetry (zero-flux) faces keep the
                    // FillPatch ghosts
                    const bool lo =
                        (c[d] < domlo[d] && bclo[d] != PERBC &&
                         bclo[d] != AXISBC && bclo[d] != IHNEUBC);
                    const bool hi =
                        (c[d] > domhi[d] && bchi[d] != PERBC &&
                         bchi[d] != AXISBC && bchi[d] != IHNEUBC);
                    if (!lo && !hi)
                    {
                        continue;
                    }
                    IntVect c0 = c;
                    IntVect c1 = c;
                    IntVect gb = c;
                    c0[d] = lo ? domlo[d] : domhi[d];
                    c1[d] = lo ? domlo[d] + 1 : domhi[d] - 1;
                    gb[d] = lo ? domlo[d] - 1 : domhi[d] + 1;
                    if (using_ib == 1 &&
                        (phi(c0, CMASK_ID) < 1.0 || phi(c1, CMASK_ID) < 1.0))
                    {
                        continue;
                    }
                    const amrex::Real m = lo ? amrex::Real(domlo[d] - c[d])
                                             : amrex::Real(c[d] - domhi[d]);
                    const amrex::Real w0 =
                        (user_defined_vel == 1)
                            ? user_transport::compute_vel(
                                  c0, d, specid, phi, dx, time, prob_parm, Tg,
                                  Pg)
                            : plasmachem_transport::compute_vel(
                                  c0, d, specid, phi, dx, time, prob_parm, Tg,
                                  Pg);
                    const amrex::Real w1 =
                        (user_defined_vel == 1)
                            ? user_transport::compute_vel(
                                  c1, d, specid, phi, dx, time, prob_parm, Tg,
                                  Pg)
                            : plasmachem_transport::compute_vel(
                                  c1, d, specid, phi, dx, time, prob_parm, Tg,
                                  Pg);
                    *vv[q] = w0 + m * (w0 - w1);
                    const amrex::Real p0 = phi(c0, specid);
                    const amrex::Real p1 = phi(c1, specid);
                    // the BC data exist only in boxes that touch the face;
                    // blocking_factor >= 4 (checked at start-up) keeps
                    // every stencil cell beyond a face inside such a box
                    const auto gd = gb.dim3();
                    const bool has_bc = bc_a.contains(gd.x, gd.y, gd.z);
                    const amrex::Real ra = has_bc ? bc_a(gb, fluxid) : 0.0;
                    const amrex::Real rb = has_bc ? bc_b(gb, fluxid) : 1.0;
                    const bool inflow = lo ? (w0 > 0.0) : (w0 < 0.0);
                    if (inflow && rb == 0.0 && ra != 0.0)
                    {
                        const amrex::Real pb = bc_f(gb, fluxid) / ra;
                        *uu[q] = pb - (2.0 * m - 1.0) * (p0 - pb);
                    } else if (inflow)
                    {
                        *uu[q] = p0;
                    } else
                    {
                        *uu[q] = p0 + m * (p0 - p1);
                    }
                }
            }

            amrex::Real flm2 = ulm2 * vel_Lm2;
            amrex::Real flm1 = ulm1 * vel_Lm1;
            amrex::Real fl = ul * vel_L;
            amrex::Real fr = ur * vel_R;
            amrex::Real frp1 = urp1 * vel_Rp1;
            amrex::Real frp2 = urp2 * vel_Rp2;

            amrex::Real fmhalf = 0.0;
            amrex::Real fphalf = 0.0;
            amrex::Real umhalf = 0.0;
            amrex::Real uphalf = 0.0;
            amrex::Real lambda_max = amrex::Math::abs(0.5 * (vel_L + vel_R));

            weno_reconstruct(
                flm2, flm1, fl, fr, frp1, frp2, fmhalf, fphalf, weno_scheme);
            weno_reconstruct(
                ulm2, ulm1, ul, ur, urp1, urp2, umhalf, uphalf, weno_scheme);

            fx(face, fluxid) = 0.5 * (fphalf + lambda_max * uphalf) +
                               0.5 * (fmhalf - lambda_max * umhalf);
        } else
        {
            amrex::Abort("hyp_order not implemented yet\n");
        }
    }

    if (face[sweepdir] == domlo[sweepdir])
    {
        if ((bclo[sweepdir] == IHNEUBC) || (bclo[sweepdir] == AXISBC))
        {
            fx(face, fluxid) = 0.0;
        }
    }
    if (face[sweepdir] == domhi[sweepdir] + 1)
    {
        if ((bchi[sweepdir] == IHNEUBC) || (bchi[sweepdir] == AXISBC))
        {
            fx(face, fluxid) = 0.0;
        }
    }
}
//====================================================================
#endif
