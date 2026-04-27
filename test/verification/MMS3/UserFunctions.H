#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <VarDefines.H>

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
    if (dir == prob_parm.axdir)
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
