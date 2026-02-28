#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLTensorOp.H>

#include <Vidyut.H>
#include <UserSources.H>
#include <Chemistry.H>
#include <UserFunctions.H>
#include <PlasmaChem.H>
#include <ProbParm.H>
#include <AMReX_MLABecLaplacian.H>
#include <HelperFuncs.H>

void Vidyut::solve_photoionization(
    Real current_time,
    Vector<MultiFab>& Sborder,
    amrex::Vector<int>& bc_lo,
    amrex::Vector<int>& bc_hi,
    Vector<MultiFab>& photoionization_src,
    int sph_id)
{
    BL_PROFILE("Vidyut::solve_photoionization()");

    // FIXME: add these as inputs
    int max_coarsening_level = linsolve_max_coarsening_level;
    int max_iter = linsolve_maxiter;
    Real ascalar = 1.0;
    Real bscalar = 1.0;
    ProbParm const* localprobparm = d_prob_parm;
    int linsolve_verbose = solver_verbose;
    int pterm = sph_id;

    // First initialization of MLMG solver
    LPInfo info;
    info.setAgglomeration(true);
    info.setConsolidation(true);
    info.setMaxCoarseningLevel(max_coarsening_level);

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

    // FIXME: had to adjust for constant coefficent,
    // this could be due to missing terms in the
    // intercalation reaction or sign mistakes...
    const Real tol_rel = linsolve_reltol;
    const Real tol_abs = linsolve_abstol;
    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;
    int userdefphotoion = user_defined_photoionization;

    // Photoionization specific variables
    const amrex::Real cm_to_m = 0.01;
    const amrex::Real Torr_to_Pa = 133.322;

    amrex::Real A_j[3];
    amrex::Real lambda_j[3];

    amrex::Real pq = 30.0 * Torr_to_Pa; // From Bourdon et al., 2007 Plasma
                                        // Sources Sci. Technol. 16 656
    amrex::Real pO2 =
        captured_gaspres * 0.21; // assuming air - update this as per need
    amrex::Real quenching_fact = pq / (pq + captured_gaspres);
    amrex::Real photoion_eff = 0.05; // From Bouwman et al., 2022 Plasma Sources
                                     // Sci. Technol. 31 045023

    A_j[0] = 1.986e-4 / ((cm_to_m * cm_to_m) * (Torr_to_Pa * Torr_to_Pa));
    lambda_j[0] = 0.0553 / ((cm_to_m) * (Torr_to_Pa));

    A_j[1] = 0.0051 / ((cm_to_m * cm_to_m) * (Torr_to_Pa * Torr_to_Pa));
    lambda_j[1] = 0.1460 / ((cm_to_m) * (Torr_to_Pa));

    A_j[2] = 0.4886 / ((cm_to_m * cm_to_m) * (Torr_to_Pa * Torr_to_Pa));
    lambda_j[2] = 0.89 / ((cm_to_m) * (Torr_to_Pa));

#ifdef AMREX_USE_HYPRE
    if (use_hypre && linsolve_verbose)
    {
        amrex::Print() << "using hypre\n";
    }
#endif

    // default to inhomogNeumann since it is defaulted to flux = 0.0 anyways
    std::array<LinOpBCType, AMREX_SPACEDIM> bc_photoionizationsolve_lo = {
        AMREX_D_DECL(
            LinOpBCType::Robin, LinOpBCType::Robin, LinOpBCType::Robin)};

    std::array<LinOpBCType, AMREX_SPACEDIM> bc_photoionizationsolve_hi = {
        AMREX_D_DECL(
            LinOpBCType::Robin, LinOpBCType::Robin, LinOpBCType::Robin)};

    int mixedbc = 0;
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        // lower side bcs
        if (bc_lo[idim] == PERBC)
        {
            bc_photoionizationsolve_lo[idim] = LinOpBCType::Periodic;
        }
        if (bc_lo[idim] == DIRCBC)
        {
            bc_photoionizationsolve_lo[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_lo[idim] == HNEUBC)
        {
            bc_photoionizationsolve_lo[idim] = LinOpBCType::Neumann;
        }
        if (bc_lo[idim] == IHNEUBC)
        {
            bc_photoionizationsolve_lo[idim] = LinOpBCType::inhomogNeumann;
        }
        if (bc_lo[idim] == ROBINBC)
        {
            bc_photoionizationsolve_lo[idim] = LinOpBCType::Robin;
            mixedbc = 1;
        }
        if (bc_lo[idim] == AXISBC)
        {
            bc_photoionizationsolve_lo[idim] = LinOpBCType::Neumann;
        }

        // higher side bcs
        if (bc_hi[idim] == PERBC)
        {
            bc_photoionizationsolve_hi[idim] = LinOpBCType::Periodic;
        }
        if (bc_hi[idim] == DIRCBC)
        {
            bc_photoionizationsolve_hi[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_hi[idim] == HNEUBC)
        {
            bc_photoionizationsolve_hi[idim] = LinOpBCType::Neumann;
        }
        if (bc_hi[idim] == IHNEUBC)
        {
            bc_photoionizationsolve_hi[idim] = LinOpBCType::inhomogNeumann;
        }
        if (bc_hi[idim] == ROBINBC)
        {
            bc_photoionizationsolve_hi[idim] = LinOpBCType::Robin;
            mixedbc = 1;
        }
        if (bc_hi[idim] == AXISBC)
        {
            bc_photoionizationsolve_hi[idim] = LinOpBCType::Neumann;
        }
    }

    // Vector<MultiFab> photoionization_src(finest_level+1);
    Vector<MultiFab> acoeff(finest_level + 1);
    Vector<MultiFab> bcoeff(finest_level + 1);
    Vector<MultiFab> solution(finest_level + 1);
    Vector<MultiFab> rhs(finest_level + 1);

    Vector<MultiFab> robin_a(finest_level + 1);
    Vector<MultiFab> robin_b(finest_level + 1);
    Vector<MultiFab> robin_f(finest_level + 1);
    Vector<iMultiFab> solvemask(finest_level + 1);

    const int num_grow = 1;

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        photoionization_src[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        acoeff[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        bcoeff[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        solution[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);

        robin_a[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        robin_b[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        robin_f[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);

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
            info));
    } else
    {
        linsolve_ptr.reset(new MLABecLaplacian(
            Geom(0, finest_level), boxArray(0, finest_level),
            DistributionMap(0, finest_level), info));
    }

    linsolve_ptr->setMaxOrder(2);
    linsolve_ptr->setDomainBC(
        bc_photoionizationsolve_lo, bc_photoionizationsolve_hi);
    linsolve_ptr->setScalars(ascalar, bscalar);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        photoionization_src[ilev].setVal(0.0);

        // Copy (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp,
        // int dstcomp, int numcomp, const IntVect& nghost)
        amrex::Copy(
            photoionization_src[ilev], Sborder[ilev], PHOTO_ION_SRC_ID, 0, 1,
            0);

        solution[ilev].setVal(0.0);

        rhs[ilev].setVal(0.0);

        //FIXME:need to make this general
        acoeff[ilev].setVal(amrex::Math::powi<2>(lambda_j[sph_id] * pO2));
        bcoeff[ilev].setVal(1.0);

        // default to homogenous Nuemann // Dirichlet
        robin_a[ilev].setVal(0.0);
        robin_b[ilev].setVal(1.0);
        robin_f[ilev].setVal(0.0);

        // Get the boundary ids
        const int* domlo_arr = geom[ilev].Domain().loVect();
        const int* domhi_arr = geom[ilev].Domain().hiVect();

        GpuArray<int, AMREX_SPACEDIM> domlo = {
            AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
        GpuArray<int, AMREX_SPACEDIM> domhi = {
            AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

        // calculate and fill rhs
        const auto dx = geom[ilev].CellSizeArray();
        auto prob_lo = geom[ilev].ProbLoArray();
        auto prob_hi = geom[ilev].ProbHiArray();
        const Box& domain = geom[ilev].Domain();

        Real time = current_time; // for GPU capture

        auto phi_arrays = Sborder[ilev].arrays();
        auto rhs_arrays = rhs[ilev].arrays();
        auto acoeff_arrays = acoeff[ilev].arrays();
        amrex::ParallelFor(
            rhs[ilev],
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                auto phi_arr = phi_arrays[nbx];
                auto rhs_arr = rhs_arrays[nbx];
                auto acoeff_arr = acoeff_arrays[nbx];

                // default
                acoeff_arr(i, j, k) = 0.0;

                user_transport::get_photoion_acoeff(
                    i, j, k, pterm, phi_arr, acoeff_arr, prob_lo, prob_hi, dx,
                    time, *localprobparm, captured_gastemp, captured_gaspres);

                rhs_arr(i, j, k) = 0.0;

                /*amrex::Real e_num_density = phi_arr(i,j,k,E_ID);
                  amrex::Real Te = phi_arr(i,j,k,ETEMP_ID);
                  amrex::Real O2_num_density = phi_arr(i,j,k,O2_ID);
                  amrex::Real N2_num_density = phi_arr(i,j,k,N2_ID);
                  std::vector<amrex::Real> Fit1(7, 0.0);
                  Fit1 = {-3.36229396e+01,  2.98924694e-01, -2.65909178e+05,
                0.0, 0.0, 0.0, 0.0}; amrex::Real k_N2_ion = std::exp(Fit1[0] +
                Fit1[1]*log(Te) + Fit1[2]/Te
                  + Fit1[3]/amrex::Math::powi<2>(Te)
                  + Fit1[4]/amrex::Math::powi<3>(Te)
                  + Fit1[5]/amrex::Math::powi<4>(Te) +
                Fit1[6]/amrex::Math::powi<5>(Te)); amrex::Real rate_N2_ion =
                k_N2_ion*e_num_density*N2_num_density; amrex::Real k_N2_exc =
                std::exp(-1.67067355e+01 + (-1.32208241e+00)*std::log(Te) +
                  (-2.17286625e+05)/Te + (2.14505360e+09)/std::pow(Te,2) +
                  (-9.57567162e+12)/std::pow(Te,3)) + std::exp(-1.67067355e+01
                  + (-1.32208241e+00)*std::log(Te) +
                  (-2.17286625e+05)/Te + (2.14505360e+09)/std::pow(Te,2) +
                  (-9.57567162e+12)/std::pow(Te,3)) + std::exp(-1.67067355e+01
                  + (-1.32208241e+00)*std::log(Te) +
                  (-2.17286625e+05)/Te + (2.14505360e+09)/std::pow(Te,2) +
                  (-9.57567162e+12)/std::pow(Te,3));
                // Update these to b1Piu, b1'Sg+u and c41'Sg+u
                //Aj * pO2 * I(r) where I(r) = (pq/(pq+p))*Xi*nu_u/nu_i*Si(r)
                //nu_u / nu_i is assumed to be 1 for now, as is also done in
                Breden et al. i
                //- A numerical study of high-pressure non-equilibrium streamers
                for combustion ignition application
                // Si(r) = electron impact ionization rate of photon emitting
                species only, i.e. N2
                // -1 multiplied on both sides of equation 8 in Bourdon et al.'s
                work rhs_arr(i,j,k) =
                (A_j[sph_id]*pO2*pO2)*(quenching_fact*photoion_eff*rate_N2_ion);*/

                user_transport::get_photoion_rhs(
                    i, j, k, pterm, phi_arr, rhs_arr, prob_lo, prob_hi, dx,
                    time, *localprobparm, captured_gastemp, captured_gaspres);
            });

        // average cell coefficients to faces, this includes boundary faces
        Array<MultiFab, AMREX_SPACEDIM> face_bcoeff;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(
                acoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
            face_bcoeff[idim].define(ba, acoeff[ilev].DistributionMap(), 1, 0);
            face_bcoeff[idim].setVal(1.0);
        }

        // set boundary conditions
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Array4<Real> phi_arr = Sborder[ilev].array(mfi);
            Array4<Real> bc_arr = photoionization_src[ilev].array(mfi);

            Array4<Real> robin_a_arr = robin_a[ilev].array(mfi);
            Array4<Real> robin_b_arr = robin_b[ilev].array(mfi);
            Array4<Real> robin_f_arr = robin_f[ilev].array(mfi);

            Real time = current_time; // for GPU capture

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                // note: bdryLo/bdryHi grabs the face indices from bx that are
                // the boundary since they are face indices, the bdry normal
                // index is 0/n+1, n is number of cells so the ghost cell index
                // at left side is i-1 while it is i on the right
                if (bx.smallEnd(idim) == domain.smallEnd(idim))
                {
                    amrex::ParallelFor(
                        amrex::bdryLo(bx, idim),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            int domend = -1;

                            if (userdefphotoion == 1)
                            {
                                user_transport::photoionization_bc(
                                    i, j, k, idim, -1, phi_arr, bc_arr,
                                    robin_a_arr, robin_b_arr, robin_f_arr,
                                    prob_lo, prob_hi, dx, time, *localprobparm,
                                    captured_gastemp, captured_gaspres);
                            } else
                            {
                                plasmachem_transport::photoionization_bc(
                                    i, j, k, idim, -1, phi_arr, bc_arr,
                                    robin_a_arr, robin_b_arr, robin_f_arr,
                                    prob_lo, prob_hi, dx, time, *localprobparm,
                                    captured_gastemp, captured_gaspres);
                            }
                        });
                }
                if (bx.bigEnd(idim) == domain.bigEnd(idim))
                {
                    amrex::ParallelFor(
                        amrex::bdryHi(bx, idim),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            int domend = 1;

                            if (userdefphotoion == 1)
                            {
                                user_transport::photoionization_bc(
                                    i, j, k, idim, +1, phi_arr, bc_arr,
                                    robin_a_arr, robin_b_arr, robin_f_arr,
                                    prob_lo, prob_hi, dx, time, *localprobparm,
                                    captured_gastemp, captured_gaspres);
                            } else
                            {
                                plasmachem_transport::photoionization_bc(
                                    i, j, k, idim, +1, phi_arr, bc_arr,
                                    robin_a_arr, robin_b_arr, robin_f_arr,
                                    prob_lo, prob_hi, dx, time, *localprobparm,
                                    captured_gastemp, captured_gaspres);
                            }
                        });
                }
            }
        }

        if (using_ib)
        {
            null_bcoeff_at_ib(ilev, face_bcoeff, Sborder[ilev], 1);
            set_explicit_fluxes_at_ib(
                ilev, rhs[ilev], acoeff[ilev], bcoeff[ilev],
                Sborder[ilev], current_time,
                PHOTO_ION_SRC_ID, 0);
        }

        linsolve_ptr->setACoeffs(ilev, acoeff[ilev]);

        // set b with diffusivities
        linsolve_ptr->setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff));

        // bc's are stored in the ghost cells of potential
        if (mixedbc)
        {
            linsolve_ptr->setLevelBC(
                ilev, &photoionization_src[ilev], &(robin_a[ilev]),
                &(robin_b[ilev]), &(robin_f[ilev]));
        } else
        {
            linsolve_ptr->setLevelBC(ilev, &photoionization_src[ilev]);
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

    amrex::Print() << "Solved Photoionization\n";

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        amrex::MultiFab::Copy(
            phi_new[ilev], solution[ilev], 0, PHOTO_ION_SRC_ID, 1, 0);
    }

    // clean-up
    //  Commented since this MF is added to rxn_src MF
    //  photoionization_src.clear();

    acoeff.clear();
    bcoeff.clear();
    solution.clear();
    rhs.clear();
    solvemask.clear();

    robin_a.clear();
    robin_b.clear();
    robin_f.clear();
}
