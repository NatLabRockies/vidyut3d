#ifndef _USERSOURCES_H_
#define _USERSOURCES_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace user_sources {

// =====================================================================
// PURE-TRANSPORT MMS source.
//
// productionRate() in this case's Chemistry.H returns zero wdot and zero
// enerExch, so CKWC contributes nothing and no reaction or energy-exchange
// term is added here. Note that ScalarSolve.cpp calls CKWC unconditionally,
// and that add_user_react_sources uses -= , so it ADDS to the reaction
// source rather than replacing it: anything productionRate left non-zero
// would enter the solved equations with no matching term in the source
// below, and this would stop being a consistent MMS. The chemistry is
// therefore disabled in Chemistry.H explicitly, not left to cancel by
// accident - see the note there.
//
// Verifying against TRANSPORT ONLY removes the reaction-source
// inconsistency (different Te / ne / nHE between mechanism and manufactured
// source) that floored n_e / n_i at 1st order and shifted the net-charge
// mean.
//
// CARTESIAN operators (coord_sys = 0). Manufactured solution:
//   n_e = r^2/alpha + n0,   n_i = r^2/(2 alpha) + n0
//   phi = r^4/32,           E   = -grad(phi),   alpha = e/eps0
// Cartesian identities:
//   grad^2 phi = r^2/2,  grad^2 n_e = 4/alpha,  grad^2 n_i = 2/alpha
//   grad n_e . grad phi = r^4/(4 alpha)
//   grad n_i . grad phi = r^4/(8 alpha)
//   |grad phi|^2 = r^6/64
// =====================================================================
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
    amrex::Real xloc[2];
    xloc[0] = prob_lo[0] + (i + 0.5) * dx[0];
    xloc[1] = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real rad = std::sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);

    const amrex::Real n0 = prob_parm.n0;
    const amrex::Real alpha = ECHARGE / EPS0;

    const amrex::Real r2 = rad * rad;
    const amrex::Real r4 = r2 * r2;
    const amrex::Real r6 = r4 * r2;

    // Manufactured field values.
    const amrex::Real ne = r2 / alpha + n0;
    const amrex::Real ni = r2 / (2.0 * alpha) + n0;

    // Transport coefficients (match Chemistry.H specMob/specDiff).
    const amrex::Real mue = -1.0;
    const amrex::Real De = 1.0;
    const amrex::Real mui = 0.5;
    const amrex::Real Di = 0.5;

    // ---- Cartesian differential quantities ----
    const amrex::Real lapphi = 0.5 * r2;                  // grad^2 phi
    const amrex::Real lapne = 4.0 / alpha;                // grad^2 n_e
    const amrex::Real lapni = 2.0 / alpha;                // grad^2 n_i
    const amrex::Real gradne_dot_gp = r4 / (4.0 * alpha); // grad n_e . grad phi
    const amrex::Real gradni_dot_gp = r4 / (8.0 * alpha); // grad n_i . grad phi

    // ---- Electron continuity (transport only) ----
    //   S_e = -mu_e(grad n_e . grad phi + n_e grad^2 phi) - De grad^2 n_e
    rxnsrc(i, j, k, E_ID) -=
        mue * (gradne_dot_gp + ne * lapphi) // div(mu_e n_e E)
        + De * lapne;                       // -De grad^2 n_e

    // ---- Ion continuity (transport only) ----
    //   S_i = -mu_i(grad n_i . grad phi + n_i grad^2 phi) - Di grad^2 n_i
    rxnsrc(i, j, k, HEp_ID) -=
        mui * (gradni_dot_gp + ni * lapphi) // div(mu_i n_i E)
        + Di * lapni;                       // -Di grad^2 n_i

    // ---- Electron energy: advection + diffusion + Joule (NO reaction,
    //      NO elastic loss -- chemistry disabled) ----
    const amrex::Real term1 =
        -5.0 / 3.0 * (mue * (gradne_dot_gp + ne * lapphi) + De * lapne);

    const amrex::Real E_mag_sq = r6 / 64.0;          // |grad phi|^2
    const amrex::Real gradne_dot_E = -gradne_dot_gp; // grad n_e . E
    const amrex::Real term2 =
        -ECHARGE * (mue * ne * E_mag_sq - De * gradne_dot_E);

    rxnsrc(i, j, k, EEN_ID) += term1 + term2;
}

/* ---- REACTION-ENABLED version (Te_solved + nHE consistent). Restore
        together with the reaction block in Chemistry.H productionRate. ----
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
    amrex::Real xloc[2];
    xloc[0] = prob_lo[0] + (i + 0.5) * dx[0];
    xloc[1] = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real rad = std::sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);

    const amrex::Real n0    = prob_parm.n0;
    const amrex::Real alpha = ECHARGE / EPS0;

    const amrex::Real r2 = rad * rad;
    const amrex::Real r4 = r2 * r2;
    const amrex::Real r6 = r4 * r2;

    const amrex::Real ne = r2 / alpha + n0;
    const amrex::Real ni = r2 / (2.0 * alpha) + n0;

    const amrex::Real mue = -1.0;
    const amrex::Real De  =  1.0;
    const amrex::Real mui =  0.5;
    const amrex::Real Di  =  0.5;

    const amrex::Real lapphi        = 0.5 * r2;
    const amrex::Real lapne         = 4.0 / alpha;
    const amrex::Real lapni         = 2.0 / alpha;
    const amrex::Real gradne_dot_gp = r4 / (4.0 * alpha);
    const amrex::Real gradni_dot_gp = r4 / (8.0 * alpha);

    const amrex::Real Te_solved = phi(i, j, k, ETEMP_ID);
    const amrex::Real nHE       = phi(i, j, k, HE_ID);
    const amrex::Real Ta        = 2.0 / K_B;
    const amrex::Real Rrate     = 5.0 * std::exp(-Ta / Te_solved) * ne * nHE;

    rxnsrc(i, j, k, E_ID) -=
        mue * (gradne_dot_gp + ne * lapphi) + De * lapne + Rrate;

    rxnsrc(i, j, k, HEp_ID) -=
        mui * (gradni_dot_gp + ni * lapphi) + Di * lapni + Rrate;

    const amrex::Real term1 =
        -5.0 / 3.0 *
        (mue * (gradne_dot_gp + ne * lapphi) + De * lapne);

    const amrex::Real E_mag_sq     = r6 / 64.0;
    const amrex::Real gradne_dot_E = -gradne_dot_gp;
    const amrex::Real term2 =
        -ECHARGE * (mue * ne * E_mag_sq - De * gradne_dot_E);

    const amrex::Real nu = 10000.0;
    const amrex::Real term3 =
        4.0 * Rrate +
        3.0 / 2.0 * ne * K_B * (Te_solved - Tg) * nu * (2.0 * ME / (4.0 *
M_AMU));

    rxnsrc(i, j, k, EEN_ID) += term1 + term2 + term3;
}
---- end reaction-enabled version ---- */

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
    // add any user specific sources
}

} // namespace user_sources
#endif