#include <Vidyut.H>
#include <UserSources.H>
#include <Chemistry.H>
#include <UserFunctions.H>
#include <PlasmaChem.H>
#include <ProbParm.H>
#include <AMReX_MLABecLaplacian.H>
#include <HelperFuncs.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real charge_flux(
    int i,
    int j,
    int k,
    int sgn,
    int dir,
    int eidx,
    const GpuArray<Real, AMREX_SPACEDIM> dx,
    amrex::Real gastemp,
    Array4<Real> const& sb_arr)
{
    IntVect face{AMREX_D_DECL(i, j, k)};
    IntVect icell{AMREX_D_DECL(i, j, k)};
    IntVect icell_prvs{AMREX_D_DECL(i, j, k)};
    amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
    outward_normal[dir] = sgn;
    int cell_adjust;

    cell_adjust = (sgn == 1) ? -1 : 0;
    icell[dir] += cell_adjust;
    // i+1 at domain lo and i-1 at
    // domain hi
    icell_prvs[dir] = icell[dir] - sgn;

    amrex::Real etemp = sb_arr(icell, ETEMP_ID);
    amrex::Real ebyn = sb_arr(icell, REF_ID);
    amrex::Real efield_mag = sb_arr(icell, EFX_ID) * sb_arr(icell, EFX_ID);
#if AMREX_SPACEDIM > 1
    efield_mag += sb_arr(icell, EFY_ID) * sb_arr(icell, EFY_ID);
#if AMREX_SPACEDIM == 3
    efield_mag += sb_arr(icell, EFZ_ID) * sb_arr(icell, EFZ_ID);
#endif
#endif
    efield_mag = std::sqrt(efield_mag);
    amrex::Real q_times_flux = 0.0;

    // linear interpolation among two cells
    // amrex::Real efield_n1 = (1.5 * sb_arr(icell, EFX_ID + dir) -
    //                        0.5 * sb_arr(icell_prvs, EFX_ID + dir)) *
    //                       outward_normal[dir];

    amrex::Real phi_boundary =
        (1.5 * sb_arr(icell, POT_ID) - 0.5 * sb_arr(icell_prvs, POT_ID));

    // doesnt matter the sgn, always directed outward
    amrex::Real efield_n = -(phi_boundary - sb_arr(icell, POT_ID)) / dx[dir];

    amrex::Real numdens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) numdens += sb_arr(icell, sp);

    for (int sp = 0; sp < NUM_SPECIES; sp++)
    {
        amrex::Real chrg = plasmachem::get_charge(sp);

        // if(amrex::Math::abs(chrg) > 0 && sp!=eidx)
        if (amrex::Math::abs(chrg) > 0)
        {
            amrex::Real specdens = sb_arr(icell, sp);
            amrex::Real gradn_n =
                get_onesided_grad(face, sgn, dir, sp, dx, sb_arr) *
                outward_normal[dir];
            // amrex::Real
            // gradn_n=(sb_arr(icell,sp)-sb_arr(icell_prvs,sp))/dx[dir];
            amrex::Real mu = specMob(sp, etemp, numdens, efield_mag, gastemp);
            amrex::Real dcoeff =
                specDiff(sp, etemp, numdens, efield_mag, gastemp);

            amrex::Real flx = mu * specdens * efield_n - dcoeff * gradn_n;

            // add only when flux is directed towards the surface
            if (flx > 0.0)
            {
                q_times_flux += chrg * flx;
            }
        }
    }

    q_times_flux *= ECHARGE;
    return (q_times_flux);
}

void Vidyut::update_surf_charge(
    Vector<MultiFab>& Sborder, Real current_time, Real dt)
{
    BL_PROFILE("Vidyut::update_surf_charge()");
    amrex::Real gastemp = gas_temperature;
    amrex::Real tstep = dt;
    ProbParm const* localprobparm = d_prob_parm;
    int eidx = E_IDX;
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        // advance from the old time level: this function is called once per
        // timestep corrector and phi_new holds the averaged state after the
        // first iteration. The face loops below then accumulate, so a cell
        // with more than one dielectric face gets all contributions
        amrex::MultiFab::Copy(
            phi_new[ilev], phi_old[ilev], SRFCH_ID, SRFCH_ID, 1, 0);

        // set boundary conditions
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Array4<Real> sb_arr = Sborder[ilev].array(mfi);
            Array4<Real> phi_arr = phi_new[ilev].array(mfi);
            Real time = current_time; // for GPU capture

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                // note: bdryLo/bdryHi grabs the face indices from bx that are
                // the boundary since they are face indices, the bdry normal
                // index is 0/n+1, n is number of cells so the ghost cell index
                // at left side is i-1 while it is i on the right
                if (bx.smallEnd(idim) == domain.smallEnd(idim))
                {
                    int sign = -1;
                    amrex::ParallelFor(
                        amrex::bdryLo(bx, idim),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            int dielectricflag = user_transport::is_dielectric(
                                i, j, k, idim, sign, prob_lo, prob_hi, dx, time,
                                *localprobparm);
                            if (dielectricflag)
                            {
                                IntVect icell{AMREX_D_DECL(i, j, k)};
                                int cell_adjust = (sign == 1) ? -1 : 0;
                                icell[idim] += cell_adjust;
                                amrex::Real q_times_flux = charge_flux(
                                    i, j, k, sign, idim, eidx, dx, gastemp,
                                    sb_arr);
                                phi_arr(icell, SRFCH_ID) +=
                                    q_times_flux * tstep;
                            }
                        });
                }
                if (bx.bigEnd(idim) == domain.bigEnd(idim))
                {
                    int sign = 1;
                    amrex::ParallelFor(
                        amrex::bdryHi(bx, idim),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            int dielectricflag = user_transport::is_dielectric(
                                i, j, k, idim, sign, prob_lo, prob_hi, dx, time,
                                *localprobparm);

                            if (dielectricflag)
                            {
                                IntVect icell{AMREX_D_DECL(i, j, k)};
                                int cell_adjust = (sign == 1) ? -1 : 0;
                                icell[idim] += cell_adjust;
                                amrex::Real q_times_flux = charge_flux(
                                    i, j, k, sign, idim, eidx, dx, gastemp,
                                    sb_arr);
                                phi_arr(icell, SRFCH_ID) +=
                                    q_times_flux * tstep;
                            }
                        });
                }
            }
        }
    }
}
