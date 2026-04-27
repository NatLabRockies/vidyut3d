#ifndef _USERSOURCES_H_
#define _USERSOURCES_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace user_sources {
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
    // add any user specific sources
    amrex::Real xloc[2];
    xloc[0] = prob_lo[0] + (i + 0.5) * dx[0];
    xloc[1] = prob_lo[1] + (j + 0.5) * dx[1];
    amrex::Real rad = sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);
    // rxnsrc(i, j, k, NI_ID) += (384 * rad * rad * rad * rad - 64);
    amrex::Real n0 = prob_parm.n0;
    amrex::Real alpha = ECHARGE / EPS0;
    amrex::Real Ee = (rad * rad) / alpha + n0;
    amrex::Real ne = (rad * rad) / alpha + n0;
    amrex::Real ni = (rad * rad) / (2.0 * alpha) + n0;

    const amrex::Real Te = (2.0 / 3.0) * (Ee / ne) / K_B;
    const amrex::Real Ta = 2.0 / K_B;
    const amrex::Real mue = -1.0;
    const amrex::Real De = 1.0;
    const amrex::Real mui = 0.5;
    const amrex::Real Di = 0.5;
    const amrex::Real ki = 5.0 * std::exp(-2.0 / (K_B * Te));
    const amrex::Real Ei = 4 / ECHARGE;
    const amrex::Real nu = 10000;

    const amrex::Real dphidr = rad * rad * rad / 8;
    const amrex::Real pdhi2dr2 = 3.0 / 8.0 * rad * rad;
    const amrex::Real dnedr = 2.0 * rad / alpha;
    const amrex::Real d2nedr2 = 2.0 / alpha;
    //! ne  mue * dnedr * dphidr + mue ne dphi2dr2 + mue ne dpidr + De d2ndr2 +
    //! ki * ne + De / r * dne/dr + Fe = 0
    rxnsrc(i, j, k, E_ID) -= mue * dnedr * dphidr + mue * ne * pdhi2dr2 +
                             mue * ne * dphidr + De * d2nedr2 + ki * ne +
                             De / rad * dnedr;
    //! ni  mui * dnidr * dphidr + mui ni dphi2dr2 + mui ni dpidr + Di d2ndr2 +
    //! ki * ni + Di / r * dne/dr + Fi = 0
    const amrex::Real dnidr = rad / alpha;
    const amrex::Real d2nidr2 = 1.0 / alpha;
    rxnsrc(i, j, k, HEp_ID) -= mui * dnidr * dphidr + mui * ni * pdhi2dr2 +
                               mui * ni * dphidr + Di * d2nidr2 + ki * ne +
                               Di / rad * dnidr;
    //! FEe= -  5/3 * (mue * dnedr * dphidr + mue ne dphi2dr2 + mue ne dpidr +
    //! De d2ndr2 + De / r * dne/dr) + Other Terms
    const amrex::Real term1 =
        -5.0 / 3.0 *
        (mue * dnedr * dphidr + mue * ne * pdhi2dr2 + mue * ne * dphidr +
         De * d2nedr2 + De / rad * dnedr);
    // Other Terms = e vec(gamma_E).vec(E)+ ki * ne * Ei + 3//2 * ne * kb * Tr *
    // nu * 2 me/mh Term 2 e Gammae E = mue ne - De * (\vec{E}.grad(ne))
    const amrex::Real dotterm = -rad * rad * rad * rad / 4.0;
    const amrex::Real term2 =
        5.0 / 3.0 * mue * Ee * ECHARGE - 5.0 / 3.0 * ECHARGE * De * dotterm;
    // Term 3  + ki * ne Ei + 3/2 ne kb Te mu 2 * me/mh
    // 3.0 / 2.0 * K_B * ne * (Te - Tg) * nu * (2.0 * ME / (4.0 * M_AMU))
    const amrex::Real izrate =
        5.0 * std::exp(-Ta / Te) * ne * phi(i, j, k, HE_ID);
    const amrex::Real term3 = 4.0 * izrate + 3.0 / 2.0 * ne * K_B * (Te - Tg) *
                                                 nu *
                                                 (2.0 * ME / (4.0 * M_AMU));
    rxnsrc(i, j, k, EEN_ID) += term1 + term2 + term3;
}

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
    // add any user specific sources
}
} // namespace user_sources
#endif
