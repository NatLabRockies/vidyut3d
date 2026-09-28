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
    // I know the only surface we are looking at is
    // yhi in this case
    amrex::Real value = 0.0;
    if (cellid[0] == domhi[0])
    {
        value = 1.0;
    }
    return (value);
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

    robin_a(ghost_cell) = 1.0;
    robin_b(ghost_cell) = 0.0;
    robin_f(ghost_cell) = 0.0;

    if (sgn == -1) // left side
    {
        // set dphidx=-electric_field
        // bc_arr(ghost_cell)=0.0;
        if (prob_parm.arc_edge_dirichlet)
        {
            bc_arr(ghost_cell) = 0.0;
        } else
        {
            bc_arr(ghost_cell) = -prob_parm.arc_efield;
        }
    } else
    {
        bc_arr(ghost_cell) = prob_parm.wallpot;
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

    int chrgspec[NUM_SPECIES + 1] = {0}; // include electron energy
    CKCHRG(chrgspec);

    // special case
    amrex::Real gastemp = P_NTP / K_B / ndens;

    if (spec_id == E_ID)
    {
        if (sgn == -1)
        {
            bc_arr(ghost_cell, bcspec_id) = prob_parm.E_arc;
        } else
        {

            amrex::Real ne = phi(cell_int, E_ID);
            amrex::Real efieldmag = std::sqrt(
                amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

            amrex::Real D = specDiff(
                spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, gastemp);

            amrex::Real Te = phi(cell_int, ETEMP_ID);
            amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);
            amrex::Real flx_th =
                ne * cbar / 4.0 * outward_normal[dir]; // thermal

            amrex::Real phi_w = prob_parm.workfunc;

            // Field emission (Fowler-Nordheim equation)
            amrex::Real beta_fe = prob_parm.beta_fe;
            amrex::Real flx_fe =
                -1.54e-6 * std::pow(10.0, 4.52 / std::sqrt(phi_w)) *
                amrex::Math::powi<2>(beta_fe * efieldmag) / phi_w *
                std::exp(
                    -6.53e9 * std::pow(phi_w, 1.5) / (beta_fe * efieldmag)) /
                ECHARGE * outward_normal[dir]; // field emission

            // Thermionic emission
            amrex::Real lamdar = prob_parm.lamdar_te;
            amrex::Real Ag = 1.20173e6; // A/m2/K2
            amrex::Real Tw = prob_parm.walltemp;
            amrex::Real Tw_eV = Tw / eV;
            // inward flux from the wall
            amrex::Real flx_te = -lamdar * Ag * Tw * Tw *
                                 std::exp(-phi_w / Tw_eV) / ECHARGE *
                                 outward_normal[dir];

            // secondary emission
            amrex::Real Hpflux = 0.0;
            amrex::Real flx_se = 0.0;
            for (int ion_id = FIRST_ION; ion_id < (FIRST_ION + NUM_IONS);
                 ion_id++)
            {
                amrex::Real mu = specMob(
                    ion_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, gastemp);

                amrex::Real ion_drift_flux_dotn = mu * phi(cell_int, ion_id) *
                                                  phi(cell_int, EFX_ID + dir) *
                                                  outward_normal[dir];
                flx_se += (ion_drift_flux_dotn > 0.0)
                              ? prob_parm.gama_se * ion_drift_flux_dotn
                              : 0.0;
                if (ion_id == Hp_ID)
                {
                    Hpflux = ion_drift_flux_dotn;
                }
            }
            //-1 for inward flux
            flx_se *= (-1.0) * outward_normal[dir];

            // total flux
            amrex::Real flx = flx_th + flx_fe + flx_te + flx_se;
            amrex::AllPrint()
                << "dx in nm, flx th,fe,te,se:" << dx[0] * 1e9 << "\t" << flx_th
                << "\t" << flx_fe << "\t" << flx_te << "\t" << flx_se << "\n";
            amrex::AllPrint()
                << "dx (nm), electron and ion current density (mega Amps/m2):"
                << dx[0] * 1e9 << "\t" << flx * ECHARGE / 1e6 << "\t"
                << -flx_se / prob_parm.gama_se * ECHARGE / 1e6 << "\n";
            amrex::AllPrint() << "H+ flux, E field:" << Hpflux << "\t"
                              << phi(cell_int, EFX_ID + dir) << "\n";
            bc_arr(ghost_cell, bcspec_id) = (-flx / D);
        }
    }

    if (amrex::Math::abs(chrgspec[spec_id]) > 0 && spec_id != E_ID &&
        spec_id != EEN_ID) // all solved ions
    {
        if (sgn == -1)
        {
            if (spec_id == ARp_ID)
            {
                bc_arr(ghost_cell, bcspec_id) = prob_parm.Arp_arc;
            } else if (spec_id == Hp_ID)
            {
                bc_arr(ghost_cell, bcspec_id) = prob_parm.Hp_arc;
            } else
            {
                bc_arr(ghost_cell, bcspec_id) = prob_parm.spec_arc;
            }
        } else
        {
            amrex::Real efieldmag = std::sqrt(
                amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

            amrex::Real mu = specMob(
                spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, gastemp);

            amrex::Real D = specDiff(
                spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, gastemp);

            amrex::Real efield = phi(cell_int, EFX_ID + dir);
            amrex::Real ion_drift_flux = mu * phi(cell_int, spec_id) * efield;
            amrex::Real ion_drift_flux_dotn =
                ion_drift_flux * outward_normal[dir];

            // if driftvel is pointing towards the wall
            // then flux=v n - D dndx = v n
            // else  flux = v n - D dndx = 0
            bc_arr(ghost_cell, bcspec_id) =
                (ion_drift_flux_dotn >= 0.0) ? (-ion_drift_flux / D) : 0.0;
        }
    }

    if (amrex::Math::abs(chrgspec[spec_id]) == 0 &&
        spec_id != EEN_ID) // all solved neutrals
    {
        if (sgn == -1)
        {
            if (spec_id == H_ID)
            {
                bc_arr(ghost_cell, bcspec_id) = prob_parm.H_arc;
            } else
            {
                bc_arr(ghost_cell, bcspec_id) = prob_parm.spec_arc;
            }
        } else
        {
            amrex::Real efieldmag = std::sqrt(
                amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

            amrex::Real D = specDiff(
                spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, gastemp);

            amrex::Real cbar = sqrt(8.0 * K_B * gastemp / PI * imw(spec_id));
            amrex::Real flx =
                phi(i, j, k, spec_id) * cbar / 4.0 * outward_normal[dir];
            if (spec_id == H_ID || spec_id == H2v1_ID || spec_id == H2v2_ID ||
                spec_id == H2v3_ID)
            {
                flx *= prob_parm.stickfac;
            }
            bc_arr(ghost_cell, bcspec_id) = (-flx / D);
        }
    }

    if (spec_id == EEN_ID)
    {
        if (sgn == -1)
        {
            bc_arr(ghost_cell, bcspec_id) =
                1.5 * prob_parm.E_arc * K_B * prob_parm.bulkelectemp;
        } else
        {
            amrex::Real ne = phi(cell_int, E_ID);
            amrex::Real efieldmag = std::sqrt(
                amrex::Math::powi<2>(phi(cell_int, EFX_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFY_ID)) +
                amrex::Math::powi<2>(phi(cell_int, EFZ_ID)));

            amrex::Real Te = phi(cell_int, ETEMP_ID);

            amrex::Real D = specDiff(
                spec_id, phi(cell_int, ETEMP_ID), ndens, efieldmag, gastemp);

            amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);

            amrex::Real flx =
                (2.0 * K_B * Te) * ne * cbar / 4.0 * outward_normal[dir];
            bc_arr(ghost_cell, bcspec_id) = (-flx / D);
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

    amrex::Real efieldmag = std::sqrt(
        amrex::Math::powi<2>(phi(cellid, EFX_ID)) +
        amrex::Math::powi<2>(phi(cellid, EFY_ID)) +
        amrex::Math::powi<2>(phi(cellid, EFZ_ID)));

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(cellid, sp);

    // special case
    amrex::Real gastemp = P_NTP / K_B / ndens;

    amrex::Real mu =
        specMob(specid, phi(cellid, ETEMP_ID), ndens, efieldmag, gastemp);

    return (mu * phi(cellid, EFX_ID + dir));
}
} // namespace user_transport
#endif
