#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace user_transport {
AMREX_GPU_HOST_DEVICE AMREX_INLINE amrex::Real current_collector_value(
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
{
    const amrex::Real cm_to_m = 0.01;
    const amrex::Real Torr_to_Pa = 133.322;

    amrex::Real pO2 =
        150.0 * Torr_to_Pa; // assuming air - update this as per need
    amrex::Real lambda_j[3];
    lambda_j[0] = 0.0553 / ((cm_to_m) * (Torr_to_Pa));
    lambda_j[1] = 0.1460 / ((cm_to_m) * (Torr_to_Pa));
    lambda_j[2] = 0.89 / ((cm_to_m) * (Torr_to_Pa));

    acoeff(i, j, k) = amrex::Math::powi<2>(lambda_j[sph_id] * pO2);
}

AMREX_GPU_DEVICE AMREX_INLINE void get_photoion_rhs(
    int i,
    int j,
    int k,
    int sph_id,
    Array4<Real> const& phi_arr,
    Array4<Real> const& rhs_arr,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{
    const amrex::Real cm_to_m = 0.01;
    const amrex::Real Torr_to_Pa = 133.322;

    amrex::Real pO2 =
        150.0 * Torr_to_Pa; // assuming air - update this as per need
    amrex::Real pq = 30.0 * Torr_to_Pa; // From Bourdon et al., 2007 Plasma
                                        // Sources Sci. Technol. 16 656
    amrex::Real A_j[3];
    amrex::Real quenching_fact = pq / (pq + Pg);
    amrex::Real photoion_eff = 0.075;

    A_j[0] = 1.986e-4 / ((cm_to_m * cm_to_m) * (Torr_to_Pa * Torr_to_Pa));
    A_j[1] = 0.0051 / ((cm_to_m * cm_to_m) * (Torr_to_Pa * Torr_to_Pa));
    ;
    A_j[2] = 0.4886 / ((cm_to_m * cm_to_m) * (Torr_to_Pa * Torr_to_Pa));
    ;

    amrex::Real e_num_density = phi_arr(i, j, k, E_ID);
    // amrex::Real Te = phi_arr(i,j,k,ETEMP_ID);
    // amrex::Real O2_num_density = phi_arr(i,j,k,O2_ID);
    amrex::Real N2_num_density = phi_arr(i, j, k, N2_ID);
    amrex::Real EN = phi_arr(i, j, k, REF_ID);
    amrex::Real efield_mag = EN * N2_num_density * 1e-21;
    amrex::Real alpha =
        (1.1944e6 + 4.3666e26 / amrex::Math::powi<3>(efield_mag)) *
        std::exp(-2.73e7 / efield_mag);
    amrex::Real mu_e = 2.3987 * std::pow(efield_mag, -0.26);
    amrex::Real rate_N2_ion = alpha * mu_e * efield_mag * e_num_density;

    rhs_arr(i, j, k) = (A_j[sph_id] * pO2 * pO2) *
                       (quenching_fact * photoion_eff * 1.0 * rate_N2_ion);
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
{}

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
        robin_a(ghost_cell) = 1.0;
        robin_b(ghost_cell) = 0.0;
        robin_f(ghost_cell) = app_voltage;
        bc_arr(ghost_cell) = app_voltage;
    } else if (sgn == 1 && dir == 1)
    {
        robin_a(ghost_cell) = 1.0;
        robin_b(ghost_cell) = 0.0;
        robin_f(ghost_cell) = app_voltage;
        bc_arr(ghost_cell) = app_voltage;
    } else
    {
        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
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

    // for all species
    if (sgn == -1)
    { // lo sides
        robin_a(ghost_cell, bcspec_id) = 0.0;
        robin_b(ghost_cell, bcspec_id) = 1.0;
        robin_f(ghost_cell, bcspec_id) = 0.0;
    } else
    {
        robin_a(ghost_cell, bcspec_id) = 0.0;
        robin_b(ghost_cell, bcspec_id) = 1.0;
        robin_f(ghost_cell, bcspec_id) = 0.0;
    }
}

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
