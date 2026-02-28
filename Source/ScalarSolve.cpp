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
#include <UserSources.H>
#include <compute_explicit_flux.H>
#include <AMReX_MLABecLaplacian.H>

void Vidyut::compute_dsdt(
    int startspec,
    int numspec,
    int lev,
    Array<MultiFab, AMREX_SPACEDIM>& flux,
    MultiFab& rxn_src,
    MultiFab& dsdt,
    Real time,
    Real dt)
{
    BL_PROFILE("Vidyut::compute_dsdt()");
    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    int captured_startspec = startspec;
    int captured_numspec = numspec;
    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;

    auto rxn_arrays = rxn_src.const_arrays();
    auto dsdt_arrays = dsdt.arrays();

    auto fluxx_arrays = flux[0].arrays();
#if AMREX_SPACEDIM > 1
    auto fluxy_arrays = flux[1].arrays();
#if AMREX_SPACEDIM == 3
    auto fluxz_arrays = flux[2].arrays();
#endif
#endif

    // update residual
    amrex::ParallelFor(
        dsdt, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
            auto dsdt_arr = dsdt_arrays[nbx];
            auto rxn_arr = rxn_arrays[nbx];

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{AMREX_D_DECL(
                fluxx_arrays[nbx], fluxy_arrays[nbx], fluxz_arrays[nbx])};
            for (int c = captured_startspec;
                 c < (captured_startspec + captured_numspec); c++)
            {
                dsdt_arr(i, j, k, c) =
                    (flux_arr[0](i, j, k, c - captured_startspec) -
                     flux_arr[0](i + 1, j, k, c - captured_startspec)) /
                        dx[0] +
                    rxn_arr(i, j, k, c);
#if AMREX_SPACEDIM > 1
                dsdt_arr(i, j, k, c) +=
                    (flux_arr[1](i, j, k, c - captured_startspec) -
                     flux_arr[1](i, j + 1, k, c - captured_startspec)) /
                    dx[1];
#if AMREX_SPACEDIM == 3
                dsdt_arr(i, j, k, c) +=
                    (flux_arr[2](i, j, k, c - captured_startspec) -
                     flux_arr[2](i, j, k + 1, c - captured_startspec)) /
                    dx[2];
#endif
#endif
            }
        });
}

void Vidyut::update_explsrc_at_all_levels(
    int startspec,
    int numspec,
    Vector<MultiFab>& Sborder,
    Vector<MultiFab>& rxn_src,
    Vector<MultiFab>& expl_src,
    Vector<int>& bc_lo,
    Vector<int>& bc_hi,
    amrex::Real cur_time)
{
    BL_PROFILE("Vidyut::update_explsrc_at_all_levels()");
    Vector<Array<MultiFab, AMREX_SPACEDIM>> flux(finest_level + 1);

    // allocate flux, expl_src, Sborder
    for (int lev = 0; lev <= finest_level; lev++)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray ba = grids[lev];
            ba.surroundingNodes(idim);

            flux[lev][idim].define(ba, dmap[lev], numspec, 0);
            flux[lev][idim].setVal(0.0);
        }
    }

    for (int lev = 0; lev <= finest_level; lev++)
    {
        expl_src[lev].setVal(0.0);
    }

    if (do_transport)
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            compute_scalar_transport_flux(
                startspec, numspec, lev, Sborder[lev], flux[lev], bc_lo, bc_hi,
                cur_time);
        }
    }

    // =======================================================
    // Average down the fluxes before using them to update phi
    // =======================================================
    for (int lev = finest_level; lev > 0; lev--)
    {
        average_down_faces(
            amrex::GetArrOfConstPtrs(flux[lev]),
            amrex::GetArrOfPtrs(flux[lev - 1]), refRatio(lev - 1),
            Geom(lev - 1));
    }

    for (int lev = 0; lev <= finest_level; lev++)
    {
        compute_dsdt(
            startspec, numspec, lev, flux[lev], rxn_src[lev], expl_src[lev],
            cur_time, dt[lev]);
    }

    // Additional source terms for axisymmetric geometry
    if (geom[0].IsRZ())
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            compute_axisym_correction(
                startspec, numspec, lev, Sborder[lev], expl_src[lev], cur_time);
        }
    }

    if (using_ib)
    {
        null_field_in_covered_cells(expl_src, Sborder, startspec, numspec);
    }
}

void Vidyut::update_rxnsrc_at_all_levels(
    Vector<MultiFab>& Sborder, Vector<MultiFab>& rxn_src, amrex::Real cur_time)
{
    BL_PROFILE("Vidyut::update_rxnsrc_at_all_levels()");
    amrex::Real time = cur_time;
    ProbParm const* localprobparm = d_prob_parm;

    // Zero out reactive source MFs
    for (int lev = 0; lev <= finest_level; lev++)
    {
        rxn_src[lev].setVal(0.0);
    }

    for (int lev = 0; lev <= finest_level; lev++)
    {
        amrex::Real captured_gastemp = gas_temperature;
        amrex::Real captured_gaspres = gas_pressure;
        const auto dx = geom[lev].CellSizeArray();
        auto prob_lo = geom[lev].ProbLoArray();
        auto prob_hi = geom[lev].ProbHiArray();

        auto sborder_arrays = Sborder[lev].arrays();
        auto rxn_arrays = rxn_src[lev].arrays();

        // update residual
        amrex::ParallelFor(
            rxn_src[lev],
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                auto sborder_arr = sborder_arrays[nbx];
                auto rxn_arr = rxn_arrays[nbx];

                // Create array with species concentrations
                amrex::Real spec_C[NUM_SPECIES];
                amrex::Real spec_wdot[NUM_SPECIES];
                amrex::Real Te = sborder_arr(i, j, k, ETEMP_ID);
                amrex::Real EN = sborder_arr(i, j, k, REF_ID);
                amrex::Real ener_exch = 0.0;
                for (int sp = 0; sp < NUM_SPECIES; sp++)
                    spec_C[sp] = sborder_arr(i, j, k, sp) / N_A;

                // Get molar production rates
                CKWC(captured_gastemp, spec_C, spec_wdot, Te, EN, &ener_exch);

                // Convert from mol/m3-s to 1/m3-s and add to scalar react
                // source MF
                for (int sp = 0; sp < NUM_SPECIES; sp++)
                    rxn_arr(i, j, k, sp) = spec_wdot[sp] * N_A;
                rxn_arr(i, j, k, NUM_SPECIES) = ener_exch;

                // Add on user-defined reactive sources
                user_sources::add_user_react_sources(
                    i, j, k, sborder_arr, rxn_arr, prob_lo, prob_hi, dx, time,
                    *localprobparm, captured_gastemp, captured_gaspres);
            });
    }

    if (using_ib)
    {
        null_field_in_covered_cells(rxn_src, Sborder, 0, NUM_SPECIES + 1);
    }
}

void Vidyut::compute_scalar_transport_flux(
    int startspec,
    int numspec,
    int lev,
    MultiFab& Sborder,
    Array<MultiFab, AMREX_SPACEDIM>& flux,
    Vector<int>& bc_lo,
    Vector<int>& bc_hi,
    Real current_time)
{
    BL_PROFILE("Vidyut::compute_scalar_transport_flux()");
    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    // class member variable
    int captured_hyporder = hyp_order;
    int captured_wenoscheme = weno_scheme;
    int userdefvel = user_defined_vel;

    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;
    amrex::Real lev_dt = dt[lev];

    int captured_startspec = startspec;
    int captured_numspec = numspec;
    int captured_using_ib = using_ib;

    // Get the boundary ids
    const int* domlo_arr = geom[lev].Domain().loVect();
    const int* domhi_arr = geom[lev].Domain().hiVect();

    GpuArray<int, AMREX_SPACEDIM> domlo = {
        AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
    GpuArray<int, AMREX_SPACEDIM> domhi = {
        AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

    GpuArray<int, AMREX_SPACEDIM> bclo = {
        AMREX_D_DECL(bc_lo[0], bc_lo[1], bc_lo[2])};
    GpuArray<int, AMREX_SPACEDIM> bchi = {
        AMREX_D_DECL(bc_hi[0], bc_hi[1], bc_hi[2])};

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Box bx_x = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
            Box bx_y = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
            Box bx_z = mfi.nodaltilebox(2);
#endif
#endif

            Real time = current_time; // for GPU capture

            Array4<Real> sborder_arr = Sborder.array(mfi);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{AMREX_D_DECL(
                flux[0].array(mfi), flux[1].array(mfi), flux[2].array(mfi))};

            // amrex::Print()<<"bx:"<<bx<<"\n";
            // amrex::Print()<<"bx_x:"<<bx_x<<"\n";
            amrex::ParallelFor(bx_x, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int c = captured_startspec;
                     c < (captured_startspec + captured_numspec); c++)
                {
                    compute_flux(
                        i, j, k, 0, c, (c - captured_startspec), sborder_arr,
                        bclo, bchi, domlo, domhi, flux_arr[0], captured_gastemp,
                        captured_gaspres, time, dx, lev_dt, *localprobparm,
                        captured_hyporder, userdefvel, captured_wenoscheme,
                        captured_using_ib);
                }
            });

#if AMREX_SPACEDIM > 1
            amrex::ParallelFor(bx_y, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int c = captured_startspec;
                     c < (captured_startspec + captured_numspec); c++)
                {
                    compute_flux(
                        i, j, k, 1, c, (c - captured_startspec), sborder_arr,
                        bclo, bchi, domlo, domhi, flux_arr[1], captured_gastemp,
                        captured_gaspres, time, dx, lev_dt, *localprobparm,
                        captured_hyporder, userdefvel, captured_wenoscheme,
                        captured_using_ib);
                }
            });

#if AMREX_SPACEDIM == 3
            amrex::ParallelFor(bx_z, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int c = captured_startspec;
                     c < (captured_startspec + captured_numspec); c++)
                {
                    compute_flux(
                        i, j, k, 2, c, (c - captured_startspec), sborder_arr,
                        bclo, bchi, domlo, domhi, flux_arr[2], captured_gastemp,
                        captured_gaspres, time, dx, lev_dt, *localprobparm,
                        captured_hyporder, userdefvel, captured_wenoscheme,
                        captured_using_ib);
                }
            });
#endif
#endif
        }
    }
}

void Vidyut::compute_axisym_correction(
    int startspec,
    int numspec,
    int lev,
    MultiFab& Sborder,
    MultiFab& dsdt,
    Real time)
{
    BL_PROFILE("Vidyut::compute_axisym_correction()");
    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;
    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    int captured_startspec = startspec;
    int captured_numspec = numspec;

    auto sborder_arrays = Sborder.const_arrays();
    auto dsdt_arrays = dsdt.arrays();

    // Evaluate cell-centered axisymmetric source terms (Gamma_k / r)
    amrex::ParallelFor(
        dsdt, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
            auto s_arr = sborder_arrays[nbx];
            auto dsdt_arr = dsdt_arrays[nbx];

            // calculate r which is always x
            // ideally x will be positive for axisymmetric cases
            amrex::Real rval = amrex::Math::abs(prob_lo[0] + (i + 0.5) * dx[0]);

            // Calculate the advective source term component
            amrex::Real etemp = s_arr(i, j, k, ETEMP_ID);
            amrex::Real ndens = 0.0;
            amrex::Real Esum = 0.0;
            for (int sp = 0; sp < NUM_SPECIES; sp++)
                ndens += s_arr(i, j, k, sp);
            for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
                Esum += amrex::Math::powi<2>(s_arr(i, j, k, EFX_ID + dim));
            amrex::Real efield_mag = std::sqrt(Esum);
            for (int c = captured_startspec;
                 c < (captured_startspec + captured_numspec); c++)
            {
                amrex::Real mu =
                    specMob(c, etemp, ndens, efield_mag, captured_gastemp);
                dsdt_arr(i, j, k, c) -=
                    mu * s_arr(i, j, k, c) * s_arr(i, j, k, EFX_ID) / rval;
            }
        });
}

void Vidyut::implicit_solve_scalar(
    Real current_time,
    Real dt,
    int startspec,
    int numspec,
    Vector<MultiFab>& Sborder,
    Vector<MultiFab>& Sborder_old,
    Vector<MultiFab>& dsdt_expl,
    Vector<int>& bc_lo,
    Vector<int>& bc_hi,
    Vector<Array<MultiFab, AMREX_SPACEDIM>>& grad_fc)
{
    // BL_PROFILE("Vidyut::implicit_solve_species(" + std::to_string( spec_id )
    // + ")");
    BL_PROFILE("Vidyut::implicit_solve_scalar()");

    // FIXME: add these as inputs
    int max_coarsening_level = linsolve_max_coarsening_level;
    int linsolve_verbose = solver_verbose;
    int captured_startspec = startspec;
    int captured_numspec = numspec;

    amrex::GpuArray<amrex::Real, MAX_CURRENT_LOCS> int_currents = {{0.0}};
    amrex::GpuArray<amrex::Real, MAX_CURRENT_LOCS> int_current_areas = {{0.0}};
    amrex::GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces = {{0}};
    if (track_integrated_currents)
    {
        for (int i = 0; i < ncurrent_locs; i++)
        {
            int_currents[i] = integrated_conduction_currents[i];
            int_current_areas[i] = integrated_current_areas[i];
            int_current_surfaces[i] = current_loc_surfaces[i];
        }
    }

    int electron_flag = 0;
    int electron_energy_flag = 0;

    // FIXME: may be a better way
    if (numspec == 1)
    {
        electron_flag = (startspec == E_IDX) ? 1 : 0;
        electron_energy_flag = (startspec == EEN_ID) ? 1 : 0;
    }

    //==================================================
    // amrex solves
    // read small a as alpha, b as beta

    //(A a - B del.(b del)) phi = f
    //
    // A and B are scalar constants
    // a and b are scalar fields
    // f is rhs
    // in this case: A=0,a=0,B=1,b=conductivity
    // note also the negative sign
    //====================================================
    ProbParm const* localprobparm = d_prob_parm;

    const Real tol_rel = linsolve_reltol;
    const Real tol_abs = linsolve_abstol;

    // set A and B, A=1/dt, B=1
    Real ascalar = 1.0;
    Real bscalar = vidyut_timescale;
    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;
    int userdefspec = user_defined_species;
    int eidx = E_IDX;

    Real dt_scaled = dt / vidyut_timescale;

    LPInfo info;
    info.setAgglomeration(true);
    info.setConsolidation(true);
    info.setMaxCoarseningLevel(max_coarsening_level);

#ifdef AMREX_USE_HYPRE
    if (use_hypre && linsolve_verbose)
    {
        amrex::Print() << "using hypre\n";
    }
#endif

    // default to inhomogNeumann since it is defaulted to flux = 0.0 anyways
    std::array<LinOpBCType, AMREX_SPACEDIM> bc_linsolve_lo = {AMREX_D_DECL(
        LinOpBCType::Robin, LinOpBCType::Robin, LinOpBCType::Robin)};

    std::array<LinOpBCType, AMREX_SPACEDIM> bc_linsolve_hi = {AMREX_D_DECL(
        LinOpBCType::Robin, LinOpBCType::Robin, LinOpBCType::Robin)};

    int mixedbc = 0;
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        // lower side bcs
        if (bc_lo[idim] == PERBC)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Periodic;
        }
        if (bc_lo[idim] == DIRCBC)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_lo[idim] == HNEUBC)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Neumann;
        }
        if (bc_lo[idim] == IHNEUBC)
        {
            bc_linsolve_lo[idim] = LinOpBCType::inhomogNeumann;
        }
        if (bc_lo[idim] == ROBINBC)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Robin;
            mixedbc = 1;
        }
        if (bc_lo[idim] == AXISBC)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Neumann;
        }

        // higher side bcs
        if (bc_hi[idim] == PERBC)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Periodic;
        }
        if (bc_hi[idim] == DIRCBC)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_hi[idim] == HNEUBC)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Neumann;
        }
        if (bc_hi[idim] == IHNEUBC)
        {
            bc_linsolve_hi[idim] = LinOpBCType::inhomogNeumann;
        }
        if (bc_hi[idim] == ROBINBC)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Robin;
            mixedbc = 1;
        }
        if (bc_hi[idim] == AXISBC)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Neumann;
        }
    }

    Vector<MultiFab> specdata(finest_level + 1);
    Vector<MultiFab> acoeff(finest_level + 1);
    Vector<MultiFab> bcoeff(finest_level + 1);
    Vector<MultiFab> solution(finest_level + 1);
    Vector<MultiFab> rhs(finest_level + 1);

    Vector<MultiFab> robin_a(finest_level + 1);
    Vector<MultiFab> robin_b(finest_level + 1);
    Vector<MultiFab> robin_f(finest_level + 1);
    Vector<iMultiFab> solvemask(finest_level + 1);

    const int num_grow = 1;

    // define all relevant arrays
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        specdata[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);

        // FIXME: for now acoeff is a single component
        // as soon as AMREX changes this we should shift
        acoeff[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);

        bcoeff[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);
        solution[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);
        rhs[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);

        // FIXME: Robin BCs are suspect with multi component mlabec
        // fix this after amrex fixes things
        robin_a[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);
        robin_b[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);
        robin_f[ilev].define(grids[ilev], dmap[ilev], numspec, num_grow);

        if (using_ib)
        {
            solvemask[ilev].define(grids[ilev], dmap[ilev], 1, 0);
            solvemask[ilev].setVal(1);
        }
    }

    if (using_ib)
    {
        set_solver_mask(solvemask, Sborder);
        linsolve_ptr.reset(new MLABecLaplacian(
            Geom(0, finest_level), boxArray(0, finest_level),
            DistributionMap(0, finest_level), GetVecOfConstPtrs(solvemask),
            info, {}, numspec));
    } else
    {
        linsolve_ptr.reset(new MLABecLaplacian(
            Geom(0, finest_level), boxArray(0, finest_level),
            DistributionMap(0, finest_level), info, {}, numspec));
    }

    linsolve_ptr->setDomainBC(bc_linsolve_lo, bc_linsolve_hi);
    linsolve_ptr->setScalars(ascalar, bscalar);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        // Copy args (FabArray<FAB>& dst, FabArray<FAB> const& src,
        // int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
        specdata[ilev].setVal(0.0);
        amrex::Copy(
            specdata[ilev], Sborder_old[ilev], startspec, 0, numspec, num_grow);

        // will use time scaling later
        acoeff[ilev].setVal(1.0 / dt);
        bcoeff[ilev].setVal(1.0);

        // default to homogenous Neumann
        robin_a[ilev].setVal(0.0);
        robin_b[ilev].setVal(1.0);
        robin_f[ilev].setVal(0.0);

        rhs[ilev].setVal(0.0);

        /*LINCOMB cheat sheet==============
         * \brief dst = a*x + b*y
         *
         * \param dst     destination FabArray
         * \param a       scalar a
         * \param x       FabArray x
         * \param xcomp   starting component of x
         * \param b       scalar b
         * \param y       FabArray y
         * \param ycomp   starting component of y
         * \param dstcomp starting component of destination
         * \param numcomp number of components
         * \param nghost  number of ghost cells
         static void LinComb (FabArray<FAB>& dst,
         value_type a, const FabArray<FAB>& x, int xcomp,
         value_type b, const FabArray<FAB>& y, int ycomp,
         int dstcomp, int numcomp, const IntVect& nghost);
         ====================================*/

        // adding U^n/dt and explicit sources
        MultiFab::LinComb(
            rhs[ilev], 1.0 / dt, Sborder_old[ilev], startspec, 1.0,
            dsdt_expl[ilev], startspec, 0, numspec, 0);

        /*===============
          static void Copy (MultiFab&       dst,
          const MultiFab& src,
          int             srccomp,
          int             dstcomp,
          int             numcomp,
          int             nghost);
          ================*/

        amrex::Copy(
            specdata[ilev], Sborder[ilev], startspec, 0, numspec, num_grow);

        // fill cell centered diffusion coefficients and rhs
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Real time = current_time; // for GPU capture

            Array4<Real> sb_arr = Sborder[ilev].array(mfi);
            Array4<Real> acoeff_arr = acoeff[ilev].array(mfi);
            Array4<Real> bcoeff_arr = bcoeff[ilev].array(mfi);

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // FIXME:may be use updated efields here
                amrex::Real Esum = 0.0;
                for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
                    Esum += amrex::Math::powi<2>(sb_arr(i, j, k, EFX_ID + dim));
                amrex::Real efield_mag = std::sqrt(Esum);

                amrex::Real ndens = 0.0;
                for (int sp = 0; sp < NUM_SPECIES; sp++)
                    ndens += sb_arr(i, j, k, sp);

                for (int specid = captured_startspec;
                     specid < (captured_startspec + captured_numspec); specid++)
                {
                    bcoeff_arr(i, j, k, specid - captured_startspec) = specDiff(
                        specid, sb_arr(i, j, k, ETEMP_ID), ndens, efield_mag,
                        captured_gastemp);
                }
            });
        }

        // average cell coefficients to faces, this includes boundary faces
        Array<MultiFab, AMREX_SPACEDIM> face_bcoeff;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(
                bcoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
            face_bcoeff[idim].define(
                ba, bcoeff[ilev].DistributionMap(), numspec, 0);
        }
        // true argument for harmonic averaging
        amrex::average_cellcenter_to_face(
            GetArrOfPtrs(face_bcoeff), bcoeff[ilev], geom[ilev], numspec, true);

        // set boundary conditions
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Array4<Real> bc_arr = specdata[ilev].array(mfi);
            Array4<Real> sb_arr = Sborder[ilev].array(mfi);
            Real time = current_time; // for GPU capture

            Array4<Real> robin_a_arr = robin_a[ilev].array(mfi);
            Array4<Real> robin_b_arr = robin_b[ilev].array(mfi);
            Array4<Real> robin_f_arr = robin_f[ilev].array(mfi);

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                if (!geom[ilev].isPeriodic(idim))
                {
                    // note: bdryLo/bdryHi grabs the face indices from bx that
                    // are the boundary since they are face indices, the bdry
                    // normal index is 0/n+1, n is number of cells so the ghost
                    // cell index at left side is i-1 while it is i on the right
                    if (bx.smallEnd(idim) == domain.smallEnd(idim))
                    {
                        amrex::ParallelFor(
                            amrex::bdryLo(bx, idim),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                if (userdefspec == 1)
                                {
                                    for (int specid = captured_startspec;
                                         specid < (captured_startspec +
                                                   captured_numspec);
                                         specid++)
                                    {
                                        user_transport::species_bc(
                                            i, j, k, idim, -1, specid,
                                            specid - captured_startspec, sb_arr,
                                            bc_arr, robin_a_arr, robin_b_arr,
                                            robin_f_arr, prob_lo, prob_hi, dx,
                                            time, *localprobparm,
                                            captured_gastemp, captured_gaspres,
                                            int_currents, int_current_areas,
                                            int_current_surfaces);
                                    }
                                } else
                                {
                                    for (int specid = captured_startspec;
                                         specid < (captured_startspec +
                                                   captured_numspec);
                                         specid++)
                                    {
                                        plasmachem_transport::species_bc(
                                            i, j, k, idim, -1, specid,
                                            specid - captured_startspec, sb_arr,
                                            bc_arr, robin_a_arr, robin_b_arr,
                                            robin_f_arr, prob_lo, prob_hi, dx,
                                            time, *localprobparm,
                                            captured_gastemp, captured_gaspres,
                                            int_currents, int_current_areas,
                                            int_current_surfaces);
                                    }
                                }
                            });
                    }
                    if (bx.bigEnd(idim) == domain.bigEnd(idim))
                    {
                        amrex::ParallelFor(
                            amrex::bdryHi(bx, idim),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                if (userdefspec == 1)
                                {
                                    for (int specid = captured_startspec;
                                         specid < (captured_startspec +
                                                   captured_numspec);
                                         specid++)
                                    {
                                        user_transport::species_bc(
                                            i, j, k, idim, +1, specid,
                                            specid - captured_startspec, sb_arr,
                                            bc_arr, robin_a_arr, robin_b_arr,
                                            robin_f_arr, prob_lo, prob_hi, dx,
                                            time, *localprobparm,
                                            captured_gastemp, captured_gaspres,
                                            int_currents, int_current_areas,
                                            int_current_surfaces);
                                    }
                                } else
                                {
                                    for (int specid = captured_startspec;
                                         specid < (captured_startspec +
                                                   captured_numspec);
                                         specid++)
                                    {
                                        plasmachem_transport::species_bc(
                                            i, j, k, idim, +1, specid,
                                            specid - captured_startspec, sb_arr,
                                            bc_arr, robin_a_arr, robin_b_arr,
                                            robin_f_arr, prob_lo, prob_hi, dx,
                                            time, *localprobparm,
                                            captured_gastemp, captured_gaspres,
                                            int_currents, int_current_areas,
                                            int_current_surfaces);
                                    }
                                }
                            });
                    }
                }
            }
        }

        if (using_ib)
        {
            null_bcoeff_at_ib(ilev, face_bcoeff, Sborder[ilev], numspec);
            // FIXME: may be can be inverted for performance
            for (int specid = startspec; specid < (startspec + numspec);
                 specid++)
            {
                set_explicit_fluxes_at_ib(
                    ilev, rhs[ilev], acoeff[ilev], bcoeff[ilev], 
                    Sborder[ilev], current_time,
                    specid, specid - startspec);
            }
        }

        acoeff[ilev].mult(vidyut_timescale, 0, 1, num_grow);
        linsolve_ptr->setACoeffs(ilev, acoeff[ilev]);

        // set b with diffusivities
        linsolve_ptr->setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff));

        // scaling
        if (!electron_energy_flag)
        {
            for (int sp = 0; sp < numspec; sp++)
            {
                specdata[ilev].mult(
                    1.0 / vidyut_specscales[startspec + sp], sp, 1, num_grow);
                robin_f[ilev].mult(
                    1.0 / vidyut_specscales[startspec + sp], sp, 1, num_grow);
                rhs[ilev].mult(
                    vidyut_timescale / vidyut_specscales[startspec + sp], sp, 1,
                    num_grow);
            }
        } else
        {
            // only 1 spec
            specdata[ilev].mult(1.0 / vidyut_eescale, 0, 1, num_grow);
            robin_f[ilev].mult(1.0 / vidyut_eescale, 0, 1, num_grow);
            rhs[ilev].mult(vidyut_timescale / vidyut_eescale, 0, 1, num_grow);
        }
        solution[ilev].setVal(0.0);
        amrex::MultiFab::Copy(solution[ilev], specdata[ilev], 0, 0, numspec, 0);

        // bc's are stored in the ghost cells
        if (mixedbc)
        {
            linsolve_ptr->setLevelBC(
                ilev, &(specdata[ilev]), &(robin_a[ilev]), &(robin_b[ilev]),
                &(robin_f[ilev]));
        } else
        {
            linsolve_ptr->setLevelBC(ilev, &(specdata[ilev]));
        }
    }

    MLMG mlmg(*linsolve_ptr);
    mlmg.setMaxIter(linsolve_maxiter);
    mlmg.setVerbose(linsolve_verbose);

#ifdef AMREX_USE_HYPRE
    if (use_hypre)
    {
        mlmg.setHypreOptionsNamespace("vidyut.hypre");
        mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
    }
#endif

    mlmg.solve(
        GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), tol_rel, tol_abs);

    if (electron_flag)
    {
        mlmg.getGradSolution(GetVecOfArrOfPtrs(grad_fc));

        for (int ilev = 0; ilev <= finest_level; ilev++)
        {
            grad_fc[ilev][0].mult(vidyut_specscales[eidx], 0, 1, 0);
#if AMREX_SPACEDIM > 1
            grad_fc[ilev][1].mult(vidyut_specscales[eidx], 0, 1, 0);
#if AMREX_SPACEDIM == 3
            grad_fc[ilev][2].mult(vidyut_specscales[eidx], 0, 1, 0);
#endif
#endif
        }
    }

    // copy solution back to phi_new
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        // scaling back
        if (!electron_energy_flag)
        {
            for (int sp = 0; sp < numspec; sp++)
            {
                solution[ilev].mult(
                    vidyut_specscales[startspec + sp], sp, 1, num_grow);
            }
        } else
        {
            // only 1 spec
            solution[ilev].mult(vidyut_eescale, 0, 1, num_grow);
        }

        // bound species density
        if (bound_specden && !electron_energy_flag)
        {
            amrex::Real minelecden = min_electron_density;
            amrex::Real minspecden = min_species_density;
            auto soln_arrays = solution[ilev].arrays();

            amrex::ParallelFor(
                phi_new[ilev],
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                    auto soln_arr = soln_arrays[nbx];
                    if (electron_flag)
                    {
                        // FIXME: when electrons are solved
                        // there will only be 1 component
                        if (soln_arr(i, j, k, 0) < minelecden)
                        {
                            soln_arr(i, j, k, 0) = minelecden;
                        }
                    } else
                    {
                        for (int specid = startspec;
                             specid < (startspec + numspec); specid++)
                        {
                            if (soln_arr(i, j, k, specid - startspec) <
                                minspecden)
                            {
                                soln_arr(i, j, k, specid - startspec) =
                                    minspecden;
                            }
                        }
                    }
                });
        }
        amrex::MultiFab::Copy(
            phi_new[ilev], solution[ilev], 0, startspec, numspec, 0);
    }

    Print() << "Solved species:";
    for (int sp = startspec; sp < (startspec + numspec); sp++)
    {
        Print() << allvarnames[sp] << "\t";
    }
    Print() << "\n";

    if (electron_energy_flag)
    {
        /*for(int ilev=0; ilev <= finest_level; ilev++)
          {
          phi_new[ilev].setVal(1.0,ETEMP_ID,1);
          amrex::MultiFab::Multiply(phi_new[ilev],solution[ilev],EEN_ID,
          ETEMP_ID, 1, 0);
          amrex::MultiFab::Divide(phi_new[ilev],phi_new[ilev],EDN_ID, ETEMP_ID,
          1, 0); phi_new[ilev].mult(twothird/K_B, ETEMP_ID, 1);
          }*/

        for (int ilev = 0; ilev <= finest_level; ilev++)
        {
            amrex::Real minetemp = min_electron_temp;
            auto phi_arrays = phi_new[ilev].arrays();
            auto sborder_arrays = Sborder[ilev].const_arrays();
            amrex::ParallelFor(
                phi_new[ilev],
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                    auto phi_arr = phi_arrays[nbx];
                    auto sb_arr = sborder_arrays[nbx];
                    phi_arr(i, j, k, ETEMP_ID) = twothird / K_B *
                                                 phi_arr(i, j, k, EEN_ID) /
                                                 sb_arr(i, j, k, eidx);
                    if (phi_arr(i, j, k, ETEMP_ID) < minetemp)
                    {
                        phi_arr(i, j, k, ETEMP_ID) = minetemp;
                        phi_arr(i, j, k, EEN_ID) =
                            1.5 * K_B * phi_arr(i, j, k, eidx) * minetemp;
                    }
                });
        }
    }

    // clean-up
    specdata.clear();
    acoeff.clear();
    bcoeff.clear();
    solution.clear();
    rhs.clear();
    solvemask.clear();

    robin_a.clear();
    robin_b.clear();
    robin_f.clear();
}
