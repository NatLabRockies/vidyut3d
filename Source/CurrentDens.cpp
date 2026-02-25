#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLTensorOp.H>
#include <ProbParm.H>
#include <Vidyut.H>
#include <Chemistry.H>
#include <UserFunctions.H>
#include <HelperFuncs.H>
#include <compute_explicit_flux.H>
#include <AMReX_MLABecLaplacian.H>

void Vidyut::compute_disp_current_den(amrex::Real time, amrex::Real tstep)
{
    amrex::Real dt = tstep;
    amrex::Real curtime = time;
    for (int lev = 0; lev <= finest_level; lev++)
    {
        const auto dx = geom[lev].CellSizeArray();
        auto prob_lo = geom[lev].ProbLoArray();
        auto prob_hi = geom[lev].ProbHiArray();
        const Box& domain = geom[lev].Domain();
        const int* domlo_arr = geom[lev].Domain().loVect();
        const int* domhi_arr = geom[lev].Domain().hiVect();

        GpuArray<int, AMREX_SPACEDIM> domlo = {
            AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
        GpuArray<int, AMREX_SPACEDIM> domhi = {
            AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

        for (MFIter mfi(phi_new[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array4<Real> phinew_arr = phi_new[lev].array(mfi);
            Array4<Real> phiold_arr = phi_old[lev].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (phinew_arr(i, j, k, CMASK_ID) == 1.0 && curtime > 0.0)
                {
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        phinew_arr(i, j, k, DCURX_ID + d) =
                            EPS0 *
                            (phinew_arr(i, j, k, EFX_ID + d) -
                             phiold_arr(i, j, k, EFX_ID + d)) /
                            dt;
                    }
                } else
                {
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        phinew_arr(i, j, k, DCURX_ID + d) = 0.0;
                    }
                }
            });
        }
    }
}

void Vidyut::compute_current_den(Vector<MultiFab>& Sborder)
{
    BL_PROFILE("Vidyut::compute_cur_den()");
    Vector<Array<MultiFab, AMREX_SPACEDIM>> ecurrentden(finest_level + 1);
    Vector<Array<MultiFab, AMREX_SPACEDIM>> icurrentden(finest_level + 1);

    // allocate flux, expl_src, Sborder
    for (int lev = 0; lev <= finest_level; lev++)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray ba = grids[lev];
            ba.surroundingNodes(idim);

            ecurrentden[lev][idim].define(ba, dmap[lev], 1, 0);
            ecurrentden[lev][idim].setVal(0.0);

            icurrentden[lev][idim].define(ba, dmap[lev], 1, 0);
            icurrentden[lev][idim].setVal(0.0);
        }

        compute_current_density_at_level(
            lev, Sborder[lev], ecurrentden[lev], icurrentden[lev]);
    }

    // =======================================================
    // Average down the fluxes before using them to update phi
    // =======================================================
    for (int lev = finest_level; lev > 0; lev--)
    {
        average_down_faces(
            amrex::GetArrOfConstPtrs(ecurrentden[lev]),
            amrex::GetArrOfPtrs(ecurrentden[lev - 1]), refRatio(lev - 1),
            Geom(lev - 1));

        average_down_faces(
            amrex::GetArrOfConstPtrs(icurrentden[lev]),
            amrex::GetArrOfPtrs(icurrentden[lev - 1]), refRatio(lev - 1),
            Geom(lev - 1));
    }

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        const Array<const MultiFab*, AMREX_SPACEDIM> allecurrent = {
            AMREX_D_DECL(
                &ecurrentden[ilev][0], &ecurrentden[ilev][1],
                &ecurrentden[ilev][2])};

        const Array<const MultiFab*, AMREX_SPACEDIM> allicurrent = {
            AMREX_D_DECL(
                &icurrentden[ilev][0], &icurrentden[ilev][1],
                &icurrentden[ilev][2])};
        average_face_to_cellcenter(phi_new[ilev], ECURX_ID, allecurrent);
        average_face_to_cellcenter(phi_new[ilev], ICURX_ID, allicurrent);
    }
}

void Vidyut::compute_current_density_at_level(
    int lev,
    MultiFab& Sborder,
    Array<MultiFab, AMREX_SPACEDIM>& ecurrden,
    Array<MultiFab, AMREX_SPACEDIM>& icurrden)
{

    BL_PROFILE("vidyut::compute_current_density()");
    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    int ncomp = Sborder.nComp();
    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;
    int ib_enabled = using_ib;
    int eidx = E_IDX;

    const int* domlo_arr = geom[lev].Domain().loVect();
    const int* domhi_arr = geom[lev].Domain().hiVect();

    GpuArray<int, AMREX_SPACEDIM> domlo = {
        AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
    GpuArray<int, AMREX_SPACEDIM> domhi = {
        AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array<Box, AMREX_SPACEDIM> face_boxes;
            face_boxes[0] = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
            face_boxes[1] = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
            face_boxes[2] = mfi.nodaltilebox(2);
#endif
#endif
            Array4<Real> sborder_arr = Sborder.array(mfi);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> e_j_arr{AMREX_D_DECL(
                ecurrden[0].array(mfi), ecurrden[1].array(mfi),
                ecurrden[2].array(mfi))};

            GpuArray<Array4<Real>, AMREX_SPACEDIM> i_j_arr{AMREX_D_DECL(
                icurrden[0].array(mfi), icurrden[1].array(mfi),
                icurrden[2].array(mfi))};

            for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
            {
                amrex::ParallelFor(
                    face_boxes[idim],
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        IntVect face{AMREX_D_DECL(i, j, k)};
                        IntVect lcell{AMREX_D_DECL(i, j, k)};
                        IntVect rcell{AMREX_D_DECL(i, j, k)};
                        lcell[idim] -= 1;

                        i_j_arr[idim](face) = 0.0;
                        e_j_arr[idim](face) = 0.0;

                        int mask_L = int(sborder_arr(lcell, CMASK_ID));
                        int mask_R = int(sborder_arr(rcell, CMASK_ID));
                        int mask_tot = mask_L + mask_R;

                        // 1 when both mask_L and mask_R are 0
                        int covered_interface = (!mask_L) * (!mask_R);
                        // 1 when both mask_L and mask_R are 1
                        int regular_interface = (mask_L) * (mask_R);
                        // 1-0 or 0-1 interface
                        int cov_uncov_interface =
                            (mask_L) * (!mask_R) + (!mask_L) * (mask_R);

                        int left_phys_boundary =
                            (face[idim] == domlo[idim]);
                        int right_phys_boundary =
                            (face[idim] == (domhi[idim] + 1));

                        if (!covered_interface)
                        {
                            // by 2 for regular and 1 for cov/uncov
                            amrex::Real etemp =
                                (sborder_arr(lcell, ETEMP_ID) * mask_L +
                                 sborder_arr(rcell, ETEMP_ID) * mask_R) /
                                mask_tot;

                            amrex::Real nspec[NUM_SPECIES];
                            amrex::Real ndens = 0.0;
                            for (int sp = 0; sp < NUM_SPECIES; sp++)
                            {
                                nspec[sp] = (sborder_arr(lcell, sp) * mask_L +
                                             sborder_arr(rcell, sp) * mask_R) /
                                            mask_tot;
                                ndens += nspec[sp];
                            }

                            Real Esum = 0.0;
                            amrex::Real efx =
                                (sborder_arr(lcell, EFX_ID) * mask_L +
                                 sborder_arr(rcell, EFX_ID) * mask_R) /
                                mask_tot;
                            Esum += amrex::Math::powi<2>(efx);
#if AMREX_SPACEDIM > 1
                            amrex::Real efy =
                                (sborder_arr(lcell, EFY_ID) * mask_L +
                                 sborder_arr(rcell, EFY_ID) * mask_R) /
                                mask_tot;

                            Esum += amrex::Math::powi<2>(efy);
#if AMREX_SPACEDIM == 3
                            amrex::Real efz =
                                (sborder_arr(lcell, EFZ_ID) * mask_L +
                                 sborder_arr(rcell, EFZ_ID) * mask_R) /
                                mask_tot;

                            Esum += amrex::Math::powi<2>(efz);
#endif
#endif
                            amrex::Real efield_mag = std::sqrt(Esum);

                            // could just average EFX/Y/Z field to get this
                            // but feel this is more accurate
                            amrex::Real face_efield = 0.0;
                            amrex::Real face_specden_grad[NUM_SPECIES] = {0.0};
                            if (regular_interface)
                            {
                                if (!left_phys_boundary && !right_phys_boundary)
                                {
                                    face_efield = -1.0 *
                                                  (sborder_arr(rcell, POT_ID) -
                                                   sborder_arr(lcell, POT_ID)) /
                                                  dx[idim];

                                    for (int sp = 0; sp < NUM_SPECIES; sp++)
                                    {
                                        face_specden_grad[sp] =
                                            (sborder_arr(rcell, sp) -
                                             sborder_arr(lcell, sp)) /
                                            dx[idim];
                                    }
                                } else if (left_phys_boundary)
                                {
                                    face_efield =
                                        -1.0 * get_onesided_grad(
                                                   face, -1, idim, POT_ID, dx,
                                                   sborder_arr);
                                    for (int sp = 0; sp < NUM_SPECIES; sp++)
                                    {
                                        face_specden_grad[sp] =
                                            get_onesided_grad(
                                                face, -1, idim, sp, dx,
                                                sborder_arr);
                                    }
                                } else // right boundary
                                {
                                    face_efield =
                                        -1.0 * get_onesided_grad(
                                                   face, +1, idim, POT_ID, dx,
                                                   sborder_arr);
                                    for (int sp = 0; sp < NUM_SPECIES; sp++)
                                    {
                                        face_specden_grad[sp] =
                                            get_onesided_grad(
                                                face, +1, idim, sp, dx,
                                                sborder_arr);
                                    }
                                }
                            } else // cov-uncov interface
                            {
                                int sgn = (mask_L == 1) ? 1 : -1;
                                face_efield =
                                    -1.0 * get_onesided_grad(
                                               face, sgn, idim, POT_ID, dx,
                                               sborder_arr);
                                for (int sp = 0; sp < NUM_SPECIES; sp++)
                                {
                                    face_specden_grad[sp] = get_onesided_grad(
                                        face, sgn, idim, sp, dx, sborder_arr);
                                }
                            }

                            Real ion_current = 0.0;
                            for (int sp = 0; sp < NUM_SPECIES; sp++)
                            {
                                amrex::Real chrg = plasmachem::get_charge(sp);
                                if (amrex::Math::abs(chrg) > 0 && sp != eidx)
                                {
                                    amrex::Real mu = specMob(
                                        sp, etemp, ndens, efield_mag,
                                        captured_gastemp);
                                    amrex::Real dcoeff = specDiff(
                                        sp, etemp, ndens, efield_mag,
                                        captured_gastemp);
                                    amrex::Real ionflux =
                                        mu * nspec[sp] * face_efield -
                                        dcoeff * face_specden_grad[sp];
                                    ion_current += chrg * ECHARGE * ionflux;
                                }
                            }
                            i_j_arr[idim](face) = ion_current;

                            // electrons
                            amrex::Real mu_e = specMob(
                                eidx, etemp, ndens, efield_mag,
                                captured_gastemp);
                            amrex::Real dcoeff_e = specDiff(
                                eidx, etemp, ndens, efield_mag,
                                captured_gastemp);
                            e_j_arr[idim](face) =
                                -ECHARGE * (mu_e * nspec[eidx] * face_efield -
                                            dcoeff_e * face_specden_grad[eidx]);
                        }
                    });
            }
        }
    }
}

void Vidyut::compute_integrated_currents()
{
    for (int locs = 0; locs < ncurrent_locs; locs++)
    {
        // note that the surfloc can be the same for different locs
        // this means the current collector value can adjusted to mask
        // out some parts of the same surface for different locs
        int surfloc = current_loc_surfaces[locs];
        amrex::Real outward_normal = (surfloc % 2 == 0) ? -1.0 : 1.0;
        int dir = int(surfloc / 2);
        amrex::Real condcurrent = 0.0;
        amrex::Real dispcurrent = 0.0;
        amrex::Real surfarea = 0.0;
        for (int lev = 0; lev <= finest_level; lev++)
        {
            const auto dx = geom[lev].CellSizeArray();
            amrex::Real cellvolume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
            amrex::Real cellarea = cellvolume / dx[dir];

            // Get the boundary ids
            const int* domlo_p = geom[lev].Domain().loVect();
            const int* domhi_p = geom[lev].Domain().hiVect();
            auto problo = geom[lev].ProbLoArray();
            auto probhi = geom[lev].ProbHiArray();
            const int* domlo_arr = geom[lev].Domain().loVect();
            const int* domhi_arr = geom[lev].Domain().hiVect();

            GpuArray<int, AMREX_SPACEDIM> domlo = {
                AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
            GpuArray<int, AMREX_SPACEDIM> domhi = {
                AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

            amrex::iMultiFab level_mask;
            if (lev < finest_level)
            {
                level_mask = makeFineMask(
                    phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                    phi_new[lev + 1].boxArray(), amrex::IntVect(2), 1, 0);
            } else
            {
                level_mask.define(
                    phi_new[lev].boxArray(), phi_new[lev].DistributionMap(), 1,
                    0, amrex::MFInfo());
                level_mask.setVal(1);
            }

            condcurrent += amrex::ReduceSum(
                phi_new[lev], level_mask, 0,
                [=] AMREX_GPU_HOST_DEVICE(
                    Box const& bx, Array4<Real const> const& fab,
                    Array4<int const> const& mask_arr) -> Real {
                    // surface integral part
                    Real si_part = 0.0;
                    amrex::Loop(
                        bx, [=, &si_part](int i, int j, int k) noexcept {
                            IntVect cellid(AMREX_D_DECL(i, j, k));
                            si_part += cellarea * outward_normal *
                                       (fab(cellid, ECURX_ID + dir) +
                                        fab(cellid, ICURX_ID + dir)) *
                                       mask_arr(cellid) *
                                       user_transport::current_collector_value(
                                           cellid, locs, surfloc, problo,
                                           probhi, domlo, domhi, dx);
                        });
                    return si_part;
                });

            dispcurrent += amrex::ReduceSum(
                phi_new[lev], level_mask, 0,
                [=] AMREX_GPU_HOST_DEVICE(
                    Box const& bx, Array4<Real const> const& fab,
                    Array4<int const> const& mask_arr) -> Real {
                    // surface integral part
                    Real si_part = 0.0;
                    amrex::Loop(
                        bx, [=, &si_part](int i, int j, int k) noexcept {
                            IntVect cellid(AMREX_D_DECL(i, j, k));
                            si_part += cellarea * outward_normal *
                                       fab(cellid, DCURX_ID + dir) *
                                       mask_arr(cellid) *
                                       user_transport::current_collector_value(
                                           cellid, locs, surfloc, problo,
                                           probhi, domlo, domhi, dx);
                        });
                    return si_part;
                });

            surfarea += amrex::ReduceSum(
                phi_new[lev], level_mask, 0,
                [=] AMREX_GPU_HOST_DEVICE(
                    Box const& bx, Array4<Real const> const& fab,
                    Array4<int const> const& mask_arr) -> Real {
                    // surface integral part
                    Real si_part = 0.0;
                    amrex::Loop(
                        bx, [=, &si_part](int i, int j, int k) noexcept {
                            IntVect cellid(AMREX_D_DECL(i, j, k));
                            si_part += cellarea * mask_arr(cellid) *
                                       user_transport::current_collector_value(
                                           cellid, locs, surfloc, problo,
                                           probhi, domlo, domhi, dx);
                        });
                    return si_part;
                });
        }

        ParallelAllReduce::Sum(condcurrent, ParallelContext::CommunicatorSub());
        ParallelAllReduce::Sum(dispcurrent, ParallelContext::CommunicatorSub());
        ParallelAllReduce::Sum(surfarea, ParallelContext::CommunicatorSub());
        integrated_conduction_currents[locs] = condcurrent;
        integrated_displacement_currents[locs] = dispcurrent;
        integrated_current_areas[locs] = surfarea;
    }
}
