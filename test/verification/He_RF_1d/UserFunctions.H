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
    IntVect intcell = face;
    int cell_adjust = (sgn == 1) ? -1 : 0;
    intcell[dir] += cell_adjust;

    amrex::Real x[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        x[d] = prob_lo[d] + (intcell[d] + 0.5) * dx[d];
    }
    x[dir] += sgn * 0.5 * dx[dir];

    int axdir = 0;
    amrex::Real xloc = x[axdir];
    amrex::Real midx = 0.5 * (prob_lo[axdir] + prob_hi[axdir]);

    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;
    amrex::Real dx2 = dx[dir] * dx[dir];

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);

    if (solved_comp == POT_ID)
    {
        // We are solving del^2 (phi)=source
        // in vidyut
        // so del(phi)|i+1/2 - del(phi)|i-1/2 = source

        // for both left or right being the interior cell
        // Dirichlet condition works out to the same sign
        int left_side = (xloc < midx) ? 1 : 0;
        amrex::Real phi_dirc =
        (left_side) ? prob_parm.V1 * std::sin(2.0 * PI * prob_parm.f * time)
        : prob_parm.V2;
        acoeff(intcell) -= 1.0 / (0.5 * dx2);
        rhs(intcell, rhs_comp) -= phi_dirc / (0.5 * dx2);
    } else if (solved_comp == E_ID || solved_comp == EEN_ID)
    {
        // |--x--|--x--|-----|
        //(f_{i+1/2}-f_{ib})/dx=rhs
        //
        //|-----|--x--|--x--|
        //(f_{ib}-f_{i-1/2})/dx=rhs
        amrex::Real ne = sb_arr(intcell, E_ID);
        amrex::Real Te = sb_arr(intcell, ETEMP_ID);
        amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);
        amrex::Real flx = ne * cbar / 4.0;
        if (solved_comp == EEN_ID)
        {
            flx *= (2.0 * K_B * Te);
        }
        rhs(intcell, rhs_comp) -= flx / dx[dir];
    } else if (solved_comp == HEp_ID)
    {
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(sb_arr(intcell, EFX_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFY_ID)) +
            amrex::Math::powi<2>(sb_arr(intcell, EFZ_ID)));

        amrex::Real mu = specMob(
            solved_comp, sb_arr(intcell, ETEMP_ID), ndens, efieldmag, Tg);
        amrex::Real efield = sb_arr(intcell, EFX_ID + dir);
        amrex::Real ion_drift_flux = mu * sb_arr(intcell, solved_comp) * efield;
        amrex::Real ion_drift_flux_dotn = ion_drift_flux * outward_normal[dir];
        amrex::Real flx =
        (ion_drift_flux_dotn >= 0.0) ? ion_drift_flux_dotn : 0.0;

        rhs(intcell, rhs_comp) -= flx / dx[dir];
    } else if (solved_comp == HEm_ID || solved_comp == HE2m_ID)
    {
        amrex::Real cbar = sqrt(8.0 * K_B * Tg / PI * imw(solved_comp));
        amrex::Real flx = sb_arr(intcell, solved_comp) * cbar / 4.0;
        rhs(intcell, rhs_comp) -= flx / dx[dir];
    } else
    {
        // zero flux
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
    Real Tg,
    Real Pg,
    Real app_voltage,
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

    if (sgn == -1)
    { // lo sides
        robin_a(ghost_cell) = 1.0;
        robin_b(ghost_cell) = 0.0;
        robin_f(ghost_cell) = app_voltage;
        bc_arr(ghost_cell) = app_voltage;
    } else if (sgn == 1)
    {
        robin_a(ghost_cell) = 1.0;
        robin_b(ghost_cell) = 0.0;
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
    Real Tg,
    Real Pg,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)

{
    const int im1 = (dir == 0) ? i - 1 : i;
    const int jm1 = (dir == 1) ? j - 1 : j;
    const int km1 = (dir == 2) ? k - 1 : k;

    IntVect cell_int{AMREX_D_DECL(i, j, k)};
    IntVect ghost_cell{AMREX_D_DECL(i, j, k)};
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = Real(sgn);

    int gcell_adjust, cell_adjust;

    // ghost_cell is one behind
    gcell_adjust = (sgn == -1) ? -1 : 0;
    // cell is one behind
    cell_adjust = (sgn == 1) ? -1 : 0;

    ghost_cell[dir] += gcell_adjust;
    cell_int[dir] += cell_adjust;

    // default
    robin_a(ghost_cell, bcspec_id) = 0.0;
    robin_b(ghost_cell, bcspec_id) = 1.0;
    robin_f(ghost_cell, bcspec_id) = 0.0;

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(cell_int, sp);

    if (spec_id == E_ID)
    {
        amrex::Real ne = phi(cell_int, E_ID);
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

        amrex::Real D =
            specDiff(spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

        amrex::Real Te = phi(cell_int, ETEMP_ID);
        amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);
        amrex::Real flx = ne * cbar / 4.0 * outward_normal[dir];

        // why -flx/D?
        // because diffusive flux -D dc/dx = flx
        // so dc/dx = -flx/D
        bc_arr(ghost_cell, bcspec_id) = (-flx / D);
    }

    if (spec_id == HEp_ID)
    {
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

        amrex::Real mu =
            specMob(spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

        amrex::Real D =
            specDiff(spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

        amrex::Real efield = phi(cell_int, EFX_ID + dir);
        amrex::Real ion_drift_flux = mu * phi(cell_int, spec_id) * efield;
        amrex::Real ion_drift_flux_dotn = ion_drift_flux * outward_normal[dir];

        // if driftvel is pointing towards the wall
        // then flux=v n - D dndx = v n
        // else  flux = v n - D dndx = 0
        bc_arr(ghost_cell, bcspec_id) =
            (ion_drift_flux_dotn >= 0.0) ? (-ion_drift_flux / D) : 0.0;
    }

    if (spec_id == HEm_ID || spec_id == HE2m_ID)
    {
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

        amrex::Real D =
            specDiff(spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

        amrex::Real cbar = sqrt(8.0 * K_B * Tg / PI * imw(spec_id));
        amrex::Real flx =
            phi(i, j, k, spec_id) * cbar / 4.0 * outward_normal[dir];
        bc_arr(ghost_cell, bcspec_id) = (-flx / D);
    }

    if (spec_id == EEN_ID)
    {
        amrex::Real ne = phi(cell_int, E_ID);
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
            amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

        amrex::Real Te = phi(cell_int, ETEMP_ID);

        amrex::Real D =
            specDiff(spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

        amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);

        amrex::Real flx =
            (2.0 * K_B * Te) * ne * cbar / 4.0 * outward_normal[dir];
        bc_arr(ghost_cell, bcspec_id) = (-flx / D);
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
    amrex::Real efieldmag = std::sqrt(
        amrex::Math::powi<2>(phi(cellid, EFX_ID)) +
        amrex::Math::powi<2>(phi(cellid, EFY_ID)) +
        amrex::Math::powi<2>(phi(cellid, EFZ_ID)));

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(cellid, sp);

    amrex::Real mu =
        specMob(specid, phi(cellid, ETEMP_ID), ndens, efieldmag, Tg);

    return (mu * phi(cellid, EFX_ID + dir));
}
} // namespace user_transport
#endif
