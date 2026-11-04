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

// ---------------------------------------------------------------------------
// Second-order wall state for the wall fluxes.
//
// All walls of this case lie on grid faces, so the wall is exactly dx/2 from
// the center of the adjacent fluid cell `cell`. `nb` is the neighbor of `cell`
// AWAY from the wall. The slope (phi_i - phi_nb)/dx points towards the wall, so
// the extrapolation to the wall face is
//     phi_wall = phi_i + 0.5*(phi_i - phi_nb) = 1.5*phi_i - 0.5*phi_nb .
// For positive quantities (densities, Te) the result is bounded from below by
// floor_frac*phi_i. mode 0 returns the cell value (first order), and the cell
// value is also used when `nb` is not a fluid cell.
// ---------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_INLINE amrex::Real gec_wall_value(
    Array4<Real> const& arr,
    IntVect cell,
    IntVect nb,
    int comp,
    bool positive,
    int mode,
    amrex::Real floor_frac)
{
    const amrex::Real phi_i = arr(cell, comp);
    if (mode == 0 || arr(nb, CMASK_ID) < 1.0) return phi_i;

    const amrex::Real phi_nb = arr(nb, comp);
    amrex::Real phi_w = 1.5 * phi_i - 0.5 * phi_nb;
    if (positive)
    {
        if (mode == 2 && phi_i > 0.0 && phi_nb > 0.0)
        {
            phi_w = phi_i * std::sqrt(phi_i / phi_nb);
        }
        phi_w = amrex::max(phi_w, floor_frac * phi_i);
    }
    return phi_w;
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
{
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;
    amrex::Real xloc[2];
    xloc[0] = prob_lo[0] + (lcell[0] + 0.5) * dx[0];
    xloc[1] = prob_lo[1] + (lcell[1] + 0.5) * dx[1];
    xloc[dir] += 0.5 * dx[dir];

    amrex::Real midy = 0.5 * (prob_lo[1] + prob_hi[1]);

    // interior cell
    IntVect intcell = (sgn == 1) ? lcell : rcell;
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;
    amrex::Real dx2 = dx[dir] * dx[dir];

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);

    // neighbor of the interior cell away from the wall, and the second-order
    // wall state (see gec_wall_value)
    IntVect nbcell = intcell;
    nbcell[dir] -= sgn;
    const int wmode = prob_parm.wall_extrap;
    const amrex::Real wfloor = prob_parm.wall_floor_frac;
    auto wallval = [&] AMREX_GPU_DEVICE(int comp, bool positive) -> amrex::Real {
        return gec_wall_value(
            sb_arr, intcell, nbcell, comp, positive, wmode, wfloor);
    };

    if (solved_comp == POT_ID)
    {
        // We are solving -B del.(beta * grad(phi))=source
        // in vidyut where B=1 and beta=-1
        // This is unlike the species equations where
        // B=1 and beta > 0
        // FVM for cell i at the boundary
        //(-B beta grad(phi).\hat{n}|internal
        //-B beta grad(phi).\hat{n}|boundary)/h = source
        // now grad(phi).\hat{n} is always (phi_d-phi_i)/(h/2)
        // therefore, diagonal contribution is always +B beta/(0.5*h^2)
        // rhs contribution is always +B beta phi_d/(0.5*h^2)
        // remember B*beta=-1 for Poisson
        // and B*beta=beta>0 for other equations
        //
        // remember flux.n is (-beta grad(phi)).n

        // for both left or right being the interior cell
        // Dirichlet condition works out to the same sign
        if (dir == 1 && xloc[0] <= prob_parm.R1)
        {
            int top_side = (xloc[dir] > midy) ? 1 : 0;
            amrex::Real phi_dirc =
                (top_side)
                    ? prob_parm.V1 * std::cos(2.0 * PI * prob_parm.f * time)
                    : prob_parm.V2;
            acoeff(intcell) +=
                (bscalar / ascalar) * bcoeff(intcell, 0.0) / (0.5 * dx2);
            rhs(intcell, rhs_comp) +=
                bscalar * bcoeff(intcell, 0.0) * phi_dirc / (0.5 * dx2);
        }
        /*else if(dir==0 && xloc[0]<prob_hi[0])
        {
           //do nothing
           //dphidn=0
        }*/
        else
        {
            amrex::Real phi_dirc = 0.0;
            acoeff(intcell) +=
                (bscalar / ascalar) * bcoeff(intcell, 0.0) / (0.5 * dx2);
            rhs(intcell, rhs_comp) +=
                bscalar * bcoeff(intcell, 0.0) * phi_dirc / (0.5 * dx2);
        }
    } else if (solved_comp == E_ID || solved_comp == EEN_ID)
    {
        // |--x--|--x--|-----|
        //(f_{i+1/2}-f_{ib})/dx=rhs
        //
        //|-----|--x--|--x--|
        //(f_{ib}-f_{i-1/2})/dx=rhs
        amrex::Real ne = wallval(E_ID, true);
        amrex::Real Te = wallval(ETEMP_ID, true);
        amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);
        amrex::Real flx = ne * cbar / 4.0;
        if (solved_comp == EEN_ID)
        {
            flx *= (2.0 * K_B * Te);
        }
        if (solved_comp == E_ID)
        {
            amrex::Real efieldmag = std::sqrt(
                amrex::Math::powi<2>(wallval(EFX_ID, false)) +
                amrex::Math::powi<2>(wallval(EFY_ID, false)) +
                amrex::Math::powi<2>(wallval(EFZ_ID, false)));
            // ion flux
            amrex::Real mu_ARp = specMob(ARp_ID, Te, ndens, efieldmag, Tg);

            amrex::Real mu_AR2p = specMob(AR2p_ID, Te, ndens, efieldmag, Tg);

            amrex::Real efield = wallval(EFX_ID + dir, false);

            amrex::Real ARp_drift_flux =
                mu_ARp * wallval(ARp_ID, true) * efield;
            amrex::Real AR2p_drift_flux =
                mu_AR2p * wallval(AR2p_ID, true) * efield;

            amrex::Real ARp_drift_flux_dotn =
                ARp_drift_flux * outward_normal[dir];
            amrex::Real ARp_flux =
                (ARp_drift_flux_dotn > 0.0) ? ARp_drift_flux_dotn : 0.0;

            amrex::Real AR2p_drift_flux_dotn =
                AR2p_drift_flux * outward_normal[dir];
            amrex::Real AR2p_flux =
                (AR2p_drift_flux_dotn > 0.0) ? AR2p_drift_flux_dotn : 0.0;

            amrex::Real gama = prob_parm.gama;

            flx -= gama * (ARp_flux + AR2p_flux); // net outward flux
        }
        rhs(intcell, rhs_comp) -= bscalar * flx / dx[dir];
    } else if (solved_comp == ARp_ID || solved_comp == AR2p_ID)
    {
        amrex::Real efieldmag = std::sqrt(
            amrex::Math::powi<2>(wallval(EFX_ID, false)) +
            amrex::Math::powi<2>(wallval(EFY_ID, false)) +
            amrex::Math::powi<2>(wallval(EFZ_ID, false)));

        amrex::Real mu = specMob(
            solved_comp, wallval(ETEMP_ID, true), ndens, efieldmag, Tg);
        amrex::Real efield = wallval(EFX_ID + dir, false);
        amrex::Real ion_drift_flux = mu * wallval(solved_comp, true) * efield;
        amrex::Real ion_drift_flux_dotn = ion_drift_flux * outward_normal[dir];
        amrex::Real flx =
            (ion_drift_flux_dotn >= 0.0) ? ion_drift_flux_dotn : 0.0;

        rhs(intcell, rhs_comp) -= bscalar * flx / dx[dir];
    } else if (solved_comp == ARm_ID || solved_comp == AR2m_ID)
    {
        amrex::Real cbar = sqrt(8.0 * K_B * Tg / PI * imw(solved_comp));
        amrex::Real flx = wallval(solved_comp, true) * cbar / 4.0;
        rhs(intcell, rhs_comp) -= bscalar * flx / dx[dir];
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

    robin_a(ghost_cell) = 1.0;
    robin_b(ghost_cell) = 0.0;
    robin_f(ghost_cell) = 0.0;
    bc_arr(ghost_cell) = 0.0;
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

    // neighbor of the interior cell away from the wall, and the second-order
    // wall state (see gec_wall_value)
    IntVect nbcell = cell_int;
    nbcell[dir] -= sgn;
    const int wmode = prob_parm.wall_extrap;
    const amrex::Real wfloor = prob_parm.wall_floor_frac;
    auto wallval = [&] AMREX_GPU_DEVICE(int comp, bool positive) -> amrex::Real {
        return gec_wall_value(
            phi, cell_int, nbcell, comp, positive, wmode, wfloor);
    };
    const amrex::Real Te_w = wallval(ETEMP_ID, true);
    const amrex::Real efieldmag_w = std::sqrt(
        amrex::Math::powi<2>(wallval(EFX_ID, false)) +
        amrex::Math::powi<2>(wallval(EFY_ID, false)) +
        amrex::Math::powi<2>(wallval(EFZ_ID, false)));
    const amrex::Real efield_w = wallval(EFX_ID + dir, false);

    if (spec_id == E_ID)
    {
        amrex::Real ne = wallval(E_ID, true);
        amrex::Real D = specDiff(spec_id, Te_w, ndens, efieldmag_w, Tg);

        amrex::Real cbar = sqrt(8.0 * K_B * Te_w / PI / ME);
        amrex::Real flx = ne * cbar / 4.0 * outward_normal[dir];

        // ion flux
        amrex::Real mu_ARp = specMob(ARp_ID, Te_w, ndens, efieldmag_w, Tg);
        amrex::Real mu_AR2p = specMob(AR2p_ID, Te_w, ndens, efieldmag_w, Tg);

        amrex::Real ARp_drift_flux = mu_ARp * wallval(ARp_ID, true) * efield_w;
        amrex::Real AR2p_drift_flux =
            mu_AR2p * wallval(AR2p_ID, true) * efield_w;

        amrex::Real ARp_drift_flux_dotn = ARp_drift_flux * outward_normal[dir];
        amrex::Real ARp_flux =
            (ARp_drift_flux_dotn > 0.0) ? ARp_drift_flux_dotn : 0.0;

        amrex::Real AR2p_drift_flux_dotn =
            AR2p_drift_flux * outward_normal[dir];
        amrex::Real AR2p_flux =
            (AR2p_drift_flux_dotn > 0.0) ? AR2p_drift_flux_dotn : 0.0;

        amrex::Real gama = prob_parm.gama;

        // flux is inwards
        flx -= gama * (ARp_flux + AR2p_flux) * outward_normal[dir];

        // why -flx/D?
        // because diffusive flux -D dc/dx = flx
        // so dc/dx = -flx/D
        bc_arr(ghost_cell, bcspec_id) = (-flx / D);
    }

    if (spec_id == ARp_ID || spec_id == AR2p_ID)
    {
        amrex::Real mu = specMob(spec_id, Te_w, ndens, efieldmag_w, Tg);
        amrex::Real D = specDiff(spec_id, Te_w, ndens, efieldmag_w, Tg);

        amrex::Real ion_drift_flux = mu * wallval(spec_id, true) * efield_w;
        amrex::Real ion_drift_flux_dotn = ion_drift_flux * outward_normal[dir];

        // if driftvel is pointing towards the wall
        // then flux=v n - D dndx = v n
        // else  flux = v n - D dndx = 0
        bc_arr(ghost_cell, bcspec_id) =
            (ion_drift_flux_dotn >= 0.0) ? (-ion_drift_flux / D) : 0.0;
    }

    if (spec_id == ARm_ID || spec_id == AR2m_ID)
    {
        amrex::Real D = specDiff(spec_id, Te_w, ndens, efieldmag_w, Tg);

        amrex::Real cbar = sqrt(8.0 * K_B * Tg / PI * imw(spec_id));
        // NOTE: the first-order version used phi(i, j, k, spec_id), which is the
        // ghost cell on high-side boundaries. The wall state of the interior
        // cell is used here on both sides.
        amrex::Real flx =
            wallval(spec_id, true) * cbar / 4.0 * outward_normal[dir];
        bc_arr(ghost_cell, bcspec_id) = (-flx / D);
    }

    if (spec_id == EEN_ID)
    {
        amrex::Real ne = wallval(E_ID, true);
        amrex::Real D = specDiff(spec_id, Te_w, ndens, efieldmag_w, Tg);

        amrex::Real cbar = sqrt(8.0 * K_B * Te_w / PI / ME);

        amrex::Real flx =
            (2.0 * K_B * Te_w) * ne * cbar / 4.0 * outward_normal[dir];
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
