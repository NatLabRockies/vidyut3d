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
{}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
int is_dielectric(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
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

    amrex::Real x = prob_lo[0] + (cell_int[0] + 0.5) * dx[0];
    amrex::Real y = prob_lo[1] + (cell_int[1] + 0.5) * dx[1];
    bc_arr(ghost_cell) = 0.0;

    // default condition
    robin_a(ghost_cell) = 0.0;
    robin_b(ghost_cell) = 1.0;
    robin_f(ghost_cell) = 0.0;

    // lo z side
    if (sgn == -1 && dir == 2)
    {
        int inside_trode1 = 0;
        int inside_trode2 = 0;
        if (!prob_parm.circle_electrodes)
        {
            inside_trode1 =
                (x >= prob_parm.trode1_minx && y >= prob_parm.trode1_miny &&
                 x <= prob_parm.trode1_maxx && y <= prob_parm.trode1_maxy);

            inside_trode2 =
                (x >= prob_parm.trode2_minx && y >= prob_parm.trode2_miny &&
                 x <= prob_parm.trode2_maxx && y <= prob_parm.trode2_maxy);
        } else
        {
            inside_trode1 =
                ((x * x +
                  (y - prob_parm.trode1_cent) * (y - prob_parm.trode1_cent)) <
                 prob_parm.trode1_rad * prob_parm.trode1_rad);

            inside_trode2 =
                ((x * x +
                  (y - prob_parm.trode2_cent) * (y - prob_parm.trode2_cent)) <
                 prob_parm.trode2_rad * prob_parm.trode2_rad);
        }

        if (inside_trode1)
        {
            robin_a(ghost_cell) = 1.0;
            robin_b(ghost_cell) = 0.0;
            robin_f(ghost_cell) =
                prob_parm.v1 * std::cos(2.0 * PI * prob_parm.freq1 * time);
        } else if (inside_trode2)
        {
            robin_a(ghost_cell) = 1.0;
            robin_b(ghost_cell) = 0.0;
            robin_f(ghost_cell) =
                prob_parm.v2 * std::cos(2.0 * PI * prob_parm.freq2 * time);
        } else
        {
            robin_a(ghost_cell) = 0.0;
            robin_b(ghost_cell) = 1.0;
            robin_f(ghost_cell) = 0.0;
        }
    }
    // hi z side
    else if (sgn == 1 && dir == 2)
    {
        int inside_subs = 0;
        if (!prob_parm.circle_electrodes)
        {
            inside_subs =
                (x >= prob_parm.subs_minx && y >= prob_parm.subs_miny &&
                 x <= prob_parm.subs_maxx && y <= prob_parm.subs_maxy);
        } else
        {
            inside_subs =
                ((x * x + y * y) < prob_parm.subs_rad * prob_parm.subs_rad);
        }
        if (inside_subs)
        {
            robin_a(ghost_cell) = 1.0;
            robin_b(ghost_cell) = 0.0;
            robin_f(ghost_cell) = 0.0;
        }
    } else
    {
        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
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

    robin_a(ghost_cell) = 0.0;
    robin_b(ghost_cell) = 1.0;
    robin_f(ghost_cell) = 0.0;
    bc_arr(ghost_cell) = 0.0;
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

    amrex::Real x = prob_lo[0] + (cell_int[0] + 0.5) * dx[0];
    amrex::Real y = prob_lo[1] + (cell_int[1] + 0.5) * dx[1];

    // default
    robin_a(ghost_cell, bcspec_id) = 0.0;
    robin_b(ghost_cell, bcspec_id) = 1.0;
    robin_f(ghost_cell, bcspec_id) = 0.0;
    bc_arr(ghost_cell, bcspec_id) = 0.0;

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(cell_int, sp);

    if (dir == 2)
    {
        int inside_trode1 = 0;
        int inside_trode2 = 0;
        int inside_subs = 0;

        if (!prob_parm.circle_electrodes)
        {
            inside_trode1 =
                (x >= prob_parm.trode1_minx && y >= prob_parm.trode1_miny &&
                 x <= prob_parm.trode1_maxx && y <= prob_parm.trode1_maxy);

            inside_trode2 =
                (x >= prob_parm.trode2_minx && y >= prob_parm.trode2_miny &&
                 x <= prob_parm.trode2_maxx && y <= prob_parm.trode2_maxy);

            inside_subs =
                (x >= prob_parm.subs_minx && y >= prob_parm.subs_miny &&
                 x <= prob_parm.subs_maxx && y <= prob_parm.subs_maxy);
        } else
        {
            inside_trode1 =
                ((x * x +
                  (y - prob_parm.trode1_cent) * (y - prob_parm.trode1_cent)) <
                 prob_parm.trode1_rad * prob_parm.trode1_rad);

            inside_trode2 =
                ((x * x +
                  (y - prob_parm.trode2_cent) * (y - prob_parm.trode2_cent)) <
                 prob_parm.trode2_rad * prob_parm.trode2_rad);

            inside_subs =
                ((x * x + y * y) < prob_parm.subs_rad * prob_parm.subs_rad);
        }
        if (((sgn == -1) && (inside_trode1 || inside_trode2)) ||
            ((sgn == +1) && inside_subs))
        {
            if (spec_id == E_ID)
            {
                amrex::Real ne = phi(cell_int, E_ID);
                amrex::Real efieldmag = std::sqrt(
                    std::pow(phi(cell_int, EFX_ID), 2.0) +
                    std::pow(phi(cell_int, EFY_ID), 2.0) +
                    std::pow(phi(cell_int, EFZ_ID), 2.0));

                amrex::Real D = specDiff(
                    spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

                // electron loss flux
                amrex::Real Te = phi(cell_int, ETEMP_ID);
                amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);
                amrex::Real flx = ne * cbar / 4.0 * outward_normal[dir];

                // secondary emission
                amrex::Real gama = 0.0;
                if (sgn == -1 && inside_trode1)
                {
                    gama = prob_parm.gama_trode1;
                } else if (sgn == -1 && inside_trode2)
                {
                    gama = prob_parm.gama_trode2;
                } else
                {
                    gama = prob_parm.gama_subs;
                }
                amrex::Real flx_se = 0.0;
                for (int ion_id = FIRST_ION; ion_id < (FIRST_ION + NUM_IONS);
                     ion_id++)
                {
                    amrex::Real mu = specMob(
                        ion_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

                    amrex::Real ion_drift_flux_dotn =
                        mu * phi(cell_int, ion_id) *
                        phi(cell_int, EFX_ID + dir) * outward_normal[dir];
                    flx_se += (ion_drift_flux_dotn > 0.0)
                                  ? gama * ion_drift_flux_dotn
                                  : 0.0;
                }

                //-1 for inward flux
                flx_se *= (-1.0) * outward_normal[dir];
                flx += flx_se;

                // why -flx/D?
                // because diffusive flux -D dc/dx = flx
                // so dc/dx = -flx/D
                bc_arr(ghost_cell, bcspec_id) = (-flx / D);
            }

            if (spec_id == ARp_ID || spec_id == AR2p_ID)
            {
                amrex::Real efieldmag = std::sqrt(
                    std::pow(phi(cell_int, EFX_ID), 2.0) +
                    std::pow(phi(cell_int, EFY_ID), 2.0) +
                    std::pow(phi(cell_int, EFZ_ID), 2.0));

                amrex::Real mu = specMob(
                    spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

                amrex::Real D = specDiff(
                    spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

                amrex::Real efield = phi(cell_int, EFX_ID + dir);
                amrex::Real ion_drift_flux =
                    mu * phi(cell_int, spec_id) * efield;
                amrex::Real ion_drift_flux_dotn =
                    ion_drift_flux * outward_normal[dir];

                // if driftvel is pointing towards the wall
                // then flux=v n - D dndx = v n
                // else  flux = v n - D dndx = 0
                bc_arr(ghost_cell, bcspec_id) =
                    (ion_drift_flux_dotn >= 0.0) ? (-ion_drift_flux / D) : 0.0;
            }

            if (spec_id == ARm_ID || spec_id == AR2m_ID)
            {
                amrex::Real efieldmag = std::sqrt(
                    std::pow(phi(cell_int, EFX_ID), 2.0) +
                    std::pow(phi(cell_int, EFY_ID), 2.0) +
                    std::pow(phi(cell_int, EFZ_ID), 2.0));

                amrex::Real D = specDiff(
                    spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

                amrex::Real cbar = sqrt(8.0 * K_B * Tg / PI * imw(spec_id));
                amrex::Real flx =
                    phi(i, j, k, spec_id) * cbar / 4.0 * outward_normal[dir];
                bc_arr(ghost_cell, bcspec_id) = (-flx / D);
            }

            if (spec_id == EEN_ID)
            {
                amrex::Real ne = phi(cell_int, E_ID);
                amrex::Real efieldmag = std::sqrt(
                    std::pow(phi(cell_int, EFX_ID), 2.0) +
                    std::pow(phi(cell_int, EFY_ID), 2.0) +
                    std::pow(phi(cell_int, EFZ_ID), 2.0));

                amrex::Real Te = phi(cell_int, ETEMP_ID);

                amrex::Real D = specDiff(
                    spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, Tg);

                amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);

                amrex::Real flx =
                    (2.0 * K_B * Te) * ne * cbar / 4.0 * outward_normal[dir];
                bc_arr(ghost_cell, bcspec_id) = (-flx / D);
            }
        }
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
        std::pow(phi(iv, EFX_ID), 2.0) + std::pow(phi(iv, EFY_ID), 2.0) +
        std::pow(phi(iv, EFZ_ID), 2.0));
    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(iv, sp);

    amrex::Real mu = specMob(specid, phi(iv, ETEMP_ID), ndens, efieldmag, Tg);

    return (mu * phi(iv, EFX_ID + dir));
}
} // namespace user_transport
#endif
