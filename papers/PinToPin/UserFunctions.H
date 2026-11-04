#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <PinGeom.H>

using namespace amrex;
namespace user_transport
{
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

    AMREX_GPU_DEVICE AMREX_INLINE
    void get_photoion_acoeff(int i,int j,int k,int sph_id,
                             Array4<Real> const& sb_arr,
                             Array4<Real> const& acoeff,
                             GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                             GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                             GpuArray<Real, AMREX_SPACEDIM> dx,
                             const Real time,
                             ProbParm const& prob_parm,
                             Real Tg, Real Pg)
    {

    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void get_photoion_rhs(int i,int j,int k,int sph_id,
                          Array4<Real> const& sb_arr,
                          Array4<Real> const& rhs_arr,
                          GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                          GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                          GpuArray<Real, AMREX_SPACEDIM> dx,
                          const Real time,
                          ProbParm const& prob_parm,
                          Real Tg, Real Pg)
    {

    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real get_applied_potential(amrex::Real current_time,
                                      amrex::Real voltage_amp,
                                      amrex::Real pulse_width,
                                      amrex::Real pulse_ramp,
                                      amrex::Real pulse_freq)
    {
        amrex::Real voltage=0.0;
        amrex::Real pulse_reptime=1.0/pulse_freq;
        int pulsenum=int(current_time*pulse_freq);
        amrex::Real inpulse_time=current_time-pulsenum*pulse_reptime;
        int inside_pulse=(inpulse_time<pulse_width);

        if(inside_pulse)
        {
            if(inpulse_time < pulse_ramp)
            {
                voltage = voltage_amp*(inpulse_time/pulse_ramp);
            }
            else if(inpulse_time >= pulse_ramp && inpulse_time < (pulse_width-pulse_ramp))
            {
                voltage = voltage_amp;
            }
            else if(inpulse_time>=(pulse_width-pulse_ramp) && inpulse_time<pulse_width)
            {
                voltage = voltage_amp*(1-(inpulse_time-(pulse_width-pulse_ramp))/pulse_ramp);
            }
        }
        else
        {
            voltage=0.0;
        }
        //voltage=voltage_amp;
        return(voltage);
    }


    // -----------------------------------------------------------------------
    // Wall state for the wall fluxes. `nb` is the neighbour of the fluid cell
    // AWAY from the wall along the face direction, `s_over_dx` the distance
    // from the cell centre to the wall along that direction in units of dx
    // (1/2 for a wall on the grid face). mode 0 returns the cell value (first
    // order), 1 the linear extrapolation phi_i + s/dx (phi_i - phi_nb), 2 the
    // logarithmic one for positive quantities. Positive quantities are bounded
    // from below by floor_frac*phi_i. The cell value is used when `nb` is not
    // a fluid cell.
    // -----------------------------------------------------------------------
    AMREX_GPU_DEVICE AMREX_INLINE amrex::Real pin_wall_value(
        Array4<Real> const& arr,
        IntVect cell,
        IntVect nb,
        int comp,
        bool positive,
        int mode,
        amrex::Real s_over_dx,
        amrex::Real floor_frac)
    {
        const amrex::Real phi_i = arr(cell, comp);
        if (mode == 0 || arr(nb, CMASK_ID) < 1.0) return phi_i;

        const amrex::Real phi_nb = arr(nb, comp);
        amrex::Real phi_w = phi_i + s_over_dx * (phi_i - phi_nb);
        if (positive)
        {
            if (mode == 2 && phi_i > 0.0 && phi_nb > 0.0)
            {
                phi_w = phi_i * std::pow(phi_i / phi_nb, s_over_dx);
            }
            phi_w = amrex::max(phi_w, floor_frac * phi_i);
        }
        return phi_w;
    }

    // -----------------------------------------------------------------------
    // Immersed-boundary closure for the face `face` (direction `dir`) between
    // the fluid cell and a masked (electrode) cell. sgn = +1 when the fluid
    // cell is on the low side of the face.
    //
    // Geometry: the wall foot xb and the unit normal n of the fluid cell were
    // reconstructed once at initialisation (Prob.H) and are read from the
    // state; no surface search is done here. The cell centre is projected on
    // the plane through the stored foot, c = xi - xb, d_n = |c|, and the
    // directional closure coefficient for this face is
    //     |n_dir| / d_n = |c_dir| / |c|^2
    // which reduces to 1/(dx/2) when the wall lies on the grid face. With
    // prob.sharp_interface_scheme = 0 the wall is taken as the grid face.
    //
    // Potential: Dirichlet, V(t) on the powered (bottom) pin, 0 on the top
    // pin; the operator is  ascalar*acoeff*phi - bscalar*div(bcoeff grad phi)
    // with bcoeff = -1, so the diagonal and rhs get
    //     (bscalar/ascalar) bcoeff (|n_dir|/d_n)/dx   and
    //      bscalar bcoeff phi_d (|n_dir|/d_n)/dx .
    // Species: explicit wall fluxes as in the former GeomUtils version;
    // thermal fluxes are weighted by |n_dir| (share of the wall seen by this
    // face), the ion drift flux uses the field component normal to the face.
    // -----------------------------------------------------------------------
    AMREX_GPU_DEVICE AMREX_INLINE
    void bc_ib(IntVect face,
               int dir, int sgn,
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
               Real Tg, Real Pg)
    {
        IntVect lcell = face;
        IntVect rcell = face;
        lcell[dir] -= 1;

        // interior (fluid) cell and the masked cell across the face
        IntVect intcell = (sgn == 1) ? lcell : rcell;
        IntVect cutcell = (sgn == 1) ? rcell : lcell;
        // side of the wall relative to the fluid cell along dir
        const amrex::Real side = (sgn == 1) ? 1.0 : -1.0;
        amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
        outward_normal[dir] = sgn;

        amrex::Real xi[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            xi[d] = prob_lo[d] + (intcell[d] + 0.5) * dx[d];

        // ---- wall foot xb and unit normal n for this face ----
        amrex::Real xb[AMREX_SPACEDIM], n[AMREX_SPACEDIM];
        auto face_wall = [&]() {
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                xb[d] = xi[d];
                n[d] = 0.0;
            }
            xb[dir] = xi[dir] + 0.5 * dx[dir] * side;
            n[dir] = side;
        };
        // plane through the stored point p with stored normal m of `cell`
        auto plane_wall = [&](IntVect cell) -> bool {
            amrex::Real m[AMREX_SPACEDIM], p[AMREX_SPACEDIM];
            m[0] = sb_arr(cell, NX_ID);
            m[1] = sb_arr(cell, NY_ID);
            p[0] = sb_arr(cell, CPX_ID);
            p[1] = sb_arr(cell, CPY_ID);
#if (AMREX_SPACEDIM == 3)
            m[2] = sb_arr(cell, NZ_ID);
            p[2] = sb_arr(cell, CPZ_ID);
#endif
            amrex::Real mm = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) mm += m[d] * m[d];
            mm = std::sqrt(mm);
            if (mm < 1.0e-14) return false;
            amrex::Real dist = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                m[d] /= mm;
                dist += (xi[d] - p[d]) * m[d];
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                xb[d] = xi[d] - dist * m[d];
                n[d] = m[d];
            }
            return true;
        };

        bool have = false;
        if (prob_parm.sharp_interface_scheme)
        {
            have = plane_wall(intcell);
            // The stored foot is the nearest wall point of the cell. Near the
            // tip a cell can touch the electrode across two faces; if the foot
            // is not across THIS face (offset along dir too small or on the
            // wrong side) use the wall stored on the masked neighbour instead.
            if (have && prob_parm.perface_fallback != 0 &&
                (xi[dir] - xb[dir]) * side > -0.25 * dx[dir])
            {
                have = plane_wall(cutcell);
            }
            if (have && (xi[dir] - xb[dir]) * side > -1.0e-3 * dx[dir])
            {
                have = false;
            }
        }
        if (!have) face_wall();

        amrex::Real ccib[AMREX_SPACEDIM];
        amrex::Real d2 = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            ccib[d] = xi[d] - xb[d];
            d2 += ccib[d] * ccib[d];
        }
        // floor on the wall distance: a fluid cell centre cannot be closer
        // than dx/2 to the wall, so this only acts on a bad reconstruction
        const amrex::Real d_floor = prob_parm.d_floor_frac * dx[dir];
        d2 = amrex::max(d2, d_floor * d_floor);
        const amrex::Real c_dir = amrex::Math::abs(ccib[dir]);
        // |n_dir|/d_n, the directional closure coefficient of this face
        const amrex::Real geom_coef = c_dir / d2;
        // share of the wall seen by this face
        const amrex::Real n_abs_dir = amrex::Math::abs(n[dir]);
        // distance to the wall along dir (ray), in units of dx, bounded to the
        // range a fluid cell allows; used for the extrapolated wall state
        amrex::Real s_over_dx =
            (c_dir > 1.0e-3 * dx[dir]) ? (d2 / c_dir) / dx[dir] : 0.5;
        s_over_dx = amrex::Clamp(s_over_dx, 0.5, 1.5);

        // neighbour of the fluid cell away from the wall
        IntVect nbcell = intcell;
        nbcell[dir] -= sgn;
        const int wmode = prob_parm.wall_extrap;
        const amrex::Real wfloor = prob_parm.wall_floor_frac;
        auto wallval = [&](int comp, bool positive) -> amrex::Real {
            return pin_wall_value(
                sb_arr, intcell, nbcell, comp, positive, wmode, s_over_dx, wfloor);
        };

        amrex::Real ndens = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sb_arr(intcell, sp);

        if (solved_comp == POT_ID)
        {
            // which electrode: by the axial position of the wall foot
            const int AX = prob_parm.pin_axis;
            const amrex::Real midht = 0.5 * (prob_lo[AX] + prob_hi[AX]);
            amrex::Real phi_dirc = (xb[AX] > midht) ? 0.0 :
                get_applied_potential(time, prob_parm.V1, prob_parm.tpulse,
                                      prob_parm.ramptime, prob_parm.pulse_freq);

            acoeff(intcell) +=
                (bscalar / ascalar) * bcoeff(intcell, 0) * geom_coef / dx[dir];
            rhs(intcell, rhs_comp) +=
                bscalar * bcoeff(intcell, 0) * phi_dirc * geom_coef / dx[dir];
        }
        else if (solved_comp == E_ID || solved_comp == EEN_ID)
        {
            // thermal flux of electrons (and electron energy) to the wall
            amrex::Real ne = wallval(E_ID, true);
            amrex::Real Te = wallval(ETEMP_ID, true);
            amrex::Real cbar = sqrt(8.0 * K_B * Te / PI / ME);
            amrex::Real flx = ne * cbar / 4.0;
            if (solved_comp == EEN_ID)
            {
                flx *= (2.0 * K_B * Te);
            }
            rhs(intcell, rhs_comp) -= bscalar * flx / dx[dir] * n_abs_dir;
        }
        else if (solved_comp == ARp_ID)
        {
            // ion drift flux through the face when directed into the wall
            amrex::Real efieldmag = std::sqrt(
                amrex::Math::powi<2>(wallval(EFX_ID, false)) +
                amrex::Math::powi<2>(wallval(EFY_ID, false)) +
                amrex::Math::powi<2>(wallval(EFZ_ID, false)));

            amrex::Real mu = specMob(solved_comp, wallval(ETEMP_ID, true),
                                     ndens, efieldmag, Tg);
            amrex::Real efield = wallval(EFX_ID + dir, false);
            amrex::Real ion_drift_flux = mu * wallval(solved_comp, true) * efield;
            amrex::Real ion_drift_flux_dotn = ion_drift_flux * outward_normal[dir];
            amrex::Real flx = (ion_drift_flux_dotn >= 0.0) ? ion_drift_flux_dotn : 0.0;

            rhs(intcell, rhs_comp) -= bscalar * flx / dx[dir];
        }
        else // any of the neutrals
        {
            amrex::Real cbar = sqrt(8.0 * K_B * Tg / PI * imw(solved_comp));
            amrex::Real flx = wallval(solved_comp, true) * cbar / 4.0;
            rhs(intcell, rhs_comp) -= bscalar * flx / dx[dir] * n_abs_dir;
        }
    }
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    int is_dielectric(int i, int j, int k, int dir, int sgn,
                      const GpuArray<Real,AMREX_SPACEDIM> prob_lo,
                      const GpuArray<Real,AMREX_SPACEDIM> prob_hi,
                      const GpuArray<Real,AMREX_SPACEDIM> dx,
                      Real time,
                      ProbParm const& prob_parm)
    {
        int val=0;
        return(val);
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void potential_bc(int i, int j, int k,
                      int dir, int sgn,
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
                      amrex::Real Tg,amrex::Real Pg,
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

        amrex::Real y = prob_lo[1] + (cell_int[1] + 0.5) * dx[1];

        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void photoionization_bc(int i, int j, int k,
                            int dir, int sgn,
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
                            amrex::Real Tg,amrex::Real Pg)
    {
        IntVect cell_int{AMREX_D_DECL(i,j,k)};
        IntVect ghost_cell{AMREX_D_DECL(i,j,k)};
        amrex::Real outward_normal[AMREX_SPACEDIM]={0.0};
        outward_normal[dir]=sgn;

        int gcell_adjust,cell_adjust;

        //ghost_cell is one behind
        gcell_adjust= (sgn==-1)?-1:0;
        //cell is one behind
        cell_adjust = (sgn==1)?-1:0;

        ghost_cell[dir]+=gcell_adjust;
        cell_int[dir]+=cell_adjust;

        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void species_bc(int i,int j,int k, int dir, int sgn,
                    int spec_id, int bcspec_id,
                    Array4<Real> const &phi,
                    Array4<Real> const& bc_arr,
                    Array4<Real> const& robin_a,
                    Array4<Real> const& robin_b,
                    Array4<Real> const& robin_f,
                    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                    GpuArray<Real, AMREX_SPACEDIM> dx,
                    const Real time,
                    ProbParm const& prob_parm,
                    amrex::Real Tg,amrex::Real Pg,
                    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
                    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
                    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)

    {
        IntVect cell_int{AMREX_D_DECL(i,j,k)};
        IntVect ghost_cell{AMREX_D_DECL(i,j,k)};
        amrex::Real outward_normal[AMREX_SPACEDIM]={0.0};
        outward_normal[dir]=sgn;

        int gcell_adjust,cell_adjust;

        //ghost_cell is one behind
        gcell_adjust= (sgn==-1)?-1:0;
        //cell is one behind
        cell_adjust = (sgn==1)?-1:0;

        ghost_cell[dir]+=gcell_adjust;
        cell_int[dir]+=cell_adjust;

        amrex::Real y=prob_lo[1]+(cell_int[1]+0.5)*dx[1];
        int chrgspec[NUM_SPECIES+1]={0}; //include electron energy
        CKCHRG(chrgspec);

        //default
        robin_a(ghost_cell,bcspec_id) = 0.0;
        robin_b(ghost_cell,bcspec_id) = 1.0;
        robin_f(ghost_cell,bcspec_id) = 0.0;
        bc_arr(ghost_cell,bcspec_id) = 0.0;

    }

    AMREX_GPU_DEVICE AMREX_INLINE
    amrex::Real compute_vel(IntVect iv, int dir,
                            int specid,
                            Array4<Real> const& phi,
                            GpuArray<Real, AMREX_SPACEDIM> dx,
                            const Real time,
                            ProbParm const& prob_parm,
                            amrex::Real Tg,amrex::Real Pg)
    {
        amrex::Real efieldmag=std::sqrt( amrex::Math::powi<2>(phi(iv,EFX_ID))+
                                        amrex::Math::powi<2>(phi(iv,EFY_ID))+
                                        amrex::Math::powi<2>(phi(iv,EFZ_ID)) );
        amrex::Real ndens = 0.0;
        for(int sp=0; sp<NUM_SPECIES; sp++) ndens += phi(iv,sp);

        amrex::Real mu = specMob(specid,phi(iv,ETEMP_ID),
                                 ndens,efieldmag,Tg);

        return(mu*phi(iv,EFX_ID+dir));
    }
}
#endif
