#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <Prob.H>
#include <Vidyut.H>
#include <Tagging.H>
#include <Chemistry.H>
#include <PlasmaChem.H>
#include <ProbParm.H>
#include <stdio.h>
#include <VarDefines.H>
#include <UserFunctions.H>
#include <HelperFuncs.H>

using namespace amrex;

ProbParm* Vidyut::h_prob_parm = nullptr;
ProbParm* Vidyut::d_prob_parm = nullptr;

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
Vidyut::Vidyut()
{
    ReadParameters();
    h_prob_parm = new ProbParm{};
    d_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));
    amrex_probinit(*h_prob_parm, *d_prob_parm);

    plasma_param_names.resize(NUM_PLASMAVARS);
    plasma_param_names[0] = "Electron_energy";
    plasma_param_names[1] = "Electron_Temp";
    plasma_param_names[2] = "Potential";
    plasma_param_names[3] = "Efieldx";
    plasma_param_names[4] = "Efieldy";
    plasma_param_names[5] = "Efieldz";
    plasma_param_names[6] = "Electron_Jheat";
    plasma_param_names[7] = "Electron_inelasticHeat";
    plasma_param_names[8] = "Electron_elasticHeat";
    plasma_param_names[9] = "ReducedEF";
    plasma_param_names[10] = "SurfaceCharge";
    plasma_param_names[11] = "PhotoIon_Src";
    plasma_param_names[12] = "cellmask";
    plasma_param_names[13] = "Ecurden_X";
    plasma_param_names[14] = "Ecurden_Y";
    plasma_param_names[15] = "Ecurden_Z";
    plasma_param_names[16] = "Icurden_X";
    plasma_param_names[17] = "Icurden_Y";
    plasma_param_names[18] = "Icurden_Z";
    plasma_param_names[19] = "Dcurden_X";
    plasma_param_names[20] = "Dcurden_Y";
    plasma_param_names[21] = "Dcurden_Z";

    allvarnames.resize(NVAR);
    for (int i = 0; i < NUM_SPECIES; i++)
    {
        allvarnames[i] = plasmachem::specnames[i];
    }
    for (int i = 0; i < NUM_PLASMAVARS; i++)
    {
        allvarnames[i + NUM_SPECIES] = plasma_param_names[i];
    }

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    for (int lev = 1; lev <= max_level; ++lev)
    {
        nsubsteps[lev] = MaxRefRatio(lev - 1);
    }

    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);

    for (int lev = 0; lev < nlevs_max; lev++)
    {
        dt[lev] = fixed_dt;
    }

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    ParmParse pp("vidyut");
    pp.queryarr("pot_bc_lo", pot_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("pot_bc_hi", pot_bc_hi, 0, AMREX_SPACEDIM);

    pp.queryarr("eden_bc_lo", eden_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("eden_bc_hi", eden_bc_hi, 0, AMREX_SPACEDIM);

    pp.queryarr("eenrg_bc_lo", eenrg_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("eenrg_bc_hi", eenrg_bc_hi, 0, AMREX_SPACEDIM);

    pp.queryarr("ion_bc_lo", ion_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("ion_bc_hi", ion_bc_hi, 0, AMREX_SPACEDIM);

    pp.queryarr("neutral_bc_lo", neutral_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("neutral_bc_hi", neutral_bc_hi, 0, AMREX_SPACEDIM);

    pp.queryarr("photoion_bc_lo", photoion_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("photoion_bc_hi", photoion_bc_hi, 0, AMREX_SPACEDIM);

    // foextrap all states as bcs imposed
    // through linear solver
    bcspec.resize(NVAR);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {

        int bctype =
            (geom[0].isPeriodic(idim)) ? BCType::int_dir : BCType::foextrap;

        for (int sp = 0; sp < NVAR; sp++)
        {
            bcspec[sp].setLo(idim, bctype);
            bcspec[sp].setHi(idim, bctype);
        }
    }

    // Find the electron index, throw error if not found
    int E_idx = (plasmachem::find_id("E") != -1)    ? plasmachem::find_id("E")
                : (plasmachem::find_id("E-") != -1) ? plasmachem::find_id("E-")
                : (plasmachem::find_id("e") != -1)  ? plasmachem::find_id("e")
                                                    : plasmachem::find_id("e-");

    if (E_idx != -1)
    {
        E_IDX = E_idx;
    } else
    {
        amrex::Abort("Electron not found in chemistry mechanism!\n");
    }

    if (multicompsolves)
    {
        if (using_ib)
        {
            amrex::Print() << "cannot solve multiple components together "
                              "because of single component acoeff**\n";
            amrex::Print()
                << "comp_ion_chunks and comp_neutral_chunks set to 1***\n";
            comp_ion_chunks = 1;
            comp_neutral_chunks = 1;
        }

        if (ion_bc_lo[0] == ROBINBC || ion_bc_hi[0] == ROBINBC
#if AMREX_SPACEDIM > 1
            || ion_bc_lo[1] == ROBINBC || ion_bc_lo[1] == ROBINBC
#if AMREX_SPACEDIM == 3
            || ion_bc_lo[2] == ROBINBC || ion_bc_lo[2] == ROBINBC
#endif
#endif
        )
        {
            amrex::Print()
                << "cannot do multicomponent solves with Robin BC for ions**\n";
            amrex::Print() << "comp_ion_chunks set to 1***\n";
            comp_ion_chunks = 1;
        }

        if (neutral_bc_lo[0] == ROBINBC || neutral_bc_hi[0] == ROBINBC
#if AMREX_SPACEDIM > 1
            || neutral_bc_lo[1] == ROBINBC || neutral_bc_lo[1] == ROBINBC
#if AMREX_SPACEDIM == 3
            || neutral_bc_lo[2] == ROBINBC || neutral_bc_lo[2] == ROBINBC
#endif
#endif
        )
        {
            amrex::Print() << "cannot do multicomponent solves with Robin BC "
                              "for neutrals**\n";
            amrex::Print() << "comp_neutral_chunks set to 1***\n";
            comp_neutral_chunks = 1;
        }
    }

    if (using_ib && track_surf_charge)
    {
        amrex::Print()
            << "**Warning: Surface charge on IB not implemented***\n";
        amrex::Print()
            << "**Surface charge on physical boundaries will be tracked***\n";
    }

    // Check inputs for axisymmetric geometry
    // only needed if one boundary is at r=0 and
    // the user will set the condition accordingly
    /*if(geom[0].IsRZ()){
        if(AMREX_SPACEDIM != 2) amrex::Abort("AMREX_SPACEDIM should be 2 for
    axisymmetric coordinates");
        // Axisymmetric implementation assumes x-low boundary is the axis of
    symmatry if(pot_bc_lo[0] != HNEUBC || eden_bc_lo[0] != HNEUBC ||
    ion_bc_lo[0] != HNEUBC || neutral_bc_lo[0] != HNEUBC
           || eenrg_bc_lo[0] != HNEUBC || photoion_bc_lo[0] != HNEUBC)
        {
            if(pot_bc_lo[0] != AXISBC || eden_bc_lo[0] != AXISBC || ion_bc_lo[0]
    != AXISBC
               || neutral_bc_lo[0] != AXISBC || eenrg_bc_lo[0] != AXISBC ||
    photoion_bc_lo[0] != AXISBC)
            {
                amrex::Abort("All x_lo boundaries must be Homogenous Neumann
    (equal to 2) or axis (equal to 5)");
            }
        }
    }*/
}

Vidyut::~Vidyut()
{
    delete h_prob_parm;
    The_Arena()->free(d_prob_parm);
}
// initializes multilevel data
void Vidyut::InitData()
{
    BL_PROFILE("Vidyut::InitData()");
    ProbParm* localprobparm = d_prob_parm;

    if (restart_chkfile == "")
    {
        // start simulation from the beginning
        const Real time = 0.0;
        InitFromScratch(time);
        AverageDown();

        if (chk_int > 0 || chk_time > 0.0)
        {
            WriteCheckpointFile(0);
        }

    } else
    {
        // restart from a checkpoint
        ReadCheckpointFile();
    }

    if (plot_int > 0 || plot_time > 0.0)
    {
        WritePlotFile(
            amrex::Math::floor(amrex::Real(istep[0]) / amrex::Real(plot_int)));
    }
    if (monitor_file_int > 0)
    {
        WriteMonitorFile(0.0);
    }
}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void Vidyut::ErrorEst(int lev, TagBoxArray& tags, Real time, int ngrow)
{
    BL_PROFILE("Vidyut::ErrorEst()");
    static bool first = true;

    // only do this during the first call to ErrorEst
    if (first)
    {
        first = false;
        ParmParse pp("vidyut");
        if (pp.contains("tagged_vars"))
        {
            int nvars = pp.countval("tagged_vars");
            refine_phi.resize(nvars);
            refine_phigrad.resize(nvars);
            refine_phi_comps.resize(nvars);
            std::string varname;
            for (int i = 0; i < nvars; i++)
            {
                pp.get("tagged_vars", varname, i);
                pp.get((varname + "_refine").c_str(), refine_phi[i]);
                pp.get((varname + "_refinegrad").c_str(), refine_phigrad[i]);

                int spec_id = plasmachem::find_id(varname);
                if (spec_id == -1)
                {
                    int varname_id = -1;
                    auto it = std::find(
                        plasma_param_names.begin(), plasma_param_names.end(),
                        varname);
                    if (it != plasma_param_names.end())
                    {
                        varname_id = it - plasma_param_names.begin();
                    }

                    if (varname_id == -1)
                    {
                        Print() << "Variable name:" << varname
                                << " not found for tagging\n";
                        amrex::Abort("Invalid tagging variable");
                    } else
                    {
                        refine_phi_comps[i] = varname_id + NUM_SPECIES;
                    }
                } else
                {
                    refine_phi_comps[i] = spec_id;
                }
            }
        }
    }

    if (refine_phi.size() == 0) return;

    //    const int clearval = TagBox::CLEAR;
    const int tagval = TagBox::SET;

    const MultiFab& state = phi_new[lev];
    MultiFab Sborder(grids[lev], dmap[lev], state.nComp(), 1);
    FillPatch(lev, time, Sborder, 0, Sborder.nComp());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        auto sb_arrays = Sborder.arrays();
        auto tags_arrays = tags.arrays();

        amrex::Real* refine_phi_dat = refine_phi.data();
        amrex::Real* refine_phigrad_dat = refine_phigrad.data();
        int* refine_phi_comps_dat = refine_phi_comps.data();
        int ntagged_comps = refine_phi_comps.size();

        amrex::ParallelFor(
            Sborder,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                auto statefab = sb_arrays[nbx];
                auto tagfab = tags_arrays[nbx];
                state_based_refinement(
                    i, j, k, tagfab, statefab, refine_phi_dat,
                    refine_phi_comps_dat, ntagged_comps, tagval);
            });

        amrex::ParallelFor(
            Sborder,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                auto statefab = sb_arrays[nbx];
                auto tagfab = tags_arrays[nbx];
                stategrad_based_refinement(
                    i, j, k, tagfab, statefab, refine_phigrad_dat,
                    refine_phi_comps_dat, ntagged_comps, tagval);
            });
    }
}

// read in some parameters from inputs file
void Vidyut::ReadParameters()
{
    BL_PROFILE("Vidyut::ReadParameters()");
    {
        ParmParse
            pp; // Traditionally, max_step and stop_time do not have prefix.
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);
    }

    {
        ParmParse pp("amr"); // Traditionally, these have prefix, amr.
        pp.query("regrid_int", regrid_int);
        pp.query("plot_file", plot_file);
        pp.query("plot_int", plot_int);
        plot_int_old = plot_int;
        pp.query("plot_int_old", plot_int_old);
        pp.query("plot_time", plot_time);
        pp.query("chk_file", chk_file);
        pp.query("chk_int", chk_int);
        chk_int_old = chk_int;
        pp.query("chk_int_old", chk_int_old);
        pp.query("chk_time", chk_time);
        pp.query("restart", restart_chkfile);
    }

    {
        ParmParse pp("vidyut");

        pp.query("dt", fixed_dt);
        pp.query("adaptive_dt", adaptive_dt);
        if (adaptive_dt)
        {
            pp.query("advective_cfl", advective_cfl);
            pp.query("diffusive_cfl", diffusive_cfl);
            pp.query("dielectric_cfl", dielectric_cfl);
            pp.query("dt_min", dt_min);
            pp.query("dt_max", dt_max);
            pp.query("adaptive_dt_delay", adaptive_dt_delay);
            pp.query("dt_stretch", dt_stretch);
        }

        pp.query("linsolve_reltol", linsolve_reltol);
        pp.query("linsolve_abstol", linsolve_abstol);
        pp.query("linsolve_bot_reltol", linsolve_bot_reltol);
        pp.query("linsolve_bot_abstol", linsolve_bot_abstol);

        pp.query("linsolve_num_pre_smooth", linsolve_num_pre_smooth);
        pp.query("linsolve_num_post_smooth", linsolve_num_post_smooth);
        pp.query("linsolve_num_final_smooth", linsolve_num_final_smooth);
        pp.query("linsolve_num_bottom_smooth", linsolve_num_bottom_smooth);

        pp.query("linsolve_maxiter", linsolve_maxiter);
        pp.query(
            "linsolve_max_coarsening_level", linsolve_max_coarsening_level);
        pp.query("bound_specden", bound_specden);
        pp.query("min_species_density", min_species_density);
        pp.query("min_electron_density", min_electron_density);
        pp.query("min_electron_temp", min_electron_temp);
        pp.query("elecenergy_solve", elecenergy_solve);
        pp.query("using_LFA", using_LFA);
        pp.query("hyp_order", hyp_order);
        pp.query("do_reactions", do_reactions);
        pp.query("do_transport", do_transport);
        pp.query("do_spacechrg", do_spacechrg);
        pp.query("user_defined_potential", user_defined_potential);
        pp.query("user_defined_species", user_defined_species);
        pp.query("user_defined_vel", user_defined_vel);
        pp.query("do_bg_reactions", do_bg_reactions);
        pp.query("do_photoionization", do_photoionization);
        pp.query("photoion_ID", photoion_ID);
        pp.query("multicompsolves", multicompsolves);
        pp.query("comp_ion_chunks", comp_ion_chunks);
        pp.query("comp_neutral_chunks", comp_neutral_chunks);

        pp.query("gas_temperature", gas_temperature);
        pp.query("gas_pressure", gas_pressure);
        bg_specid_list.resize(0);
        pp.queryarr("bg_species_ids", bg_specid_list);

        pp.query("weno_scheme", weno_scheme);
        pp.query("track_surf_charge", track_surf_charge);
        pp.query("solver_verbose", solver_verbose);
        pp.query("evolve_verbose", evolve_verbose);
        pp.query("track_current_den", track_current_den);
        pp.query("int_current_filename", intcurrentfilename);

        if (hyp_order == 1) // first order upwind
        {
            ngrow_for_fillpatch = 1;
        } else if (hyp_order == 2) // second-order flux limited
        {
            ngrow_for_fillpatch = 2;
        } else if (hyp_order == 5) // weno 5
        {
            ngrow_for_fillpatch = 3;
            // amrex::Abort("hyp_order 5 not implemented yet");
        } else
        {
            amrex::Abort("Specified hyp_order not implemented yet");
        }

        // Voltage options
        pp.query("voltage_profile", voltage_profile);
        pp.queryarr("voltage_amp_lo", voltage_amp_lo, 0, AMREX_SPACEDIM);
        pp.queryarr("voltage_amp_hi", voltage_amp_hi, 0, AMREX_SPACEDIM);
        if (voltage_profile == 1)
        {
            pp.get("voltage_freq", voltage_freq);
        } else if (voltage_profile == 2)
        {
            pp.get("voltage_dur", voltage_dur);
            pp.get("voltage_center", voltage_center);
        }

        pp.query("monitor_file_int", monitor_file_int);
        pp.query("num_timestep_correctors", num_timestep_correctors);
        pp.query("floor_jh", floor_jh);

        Vector<int> scaling_specid_list;
        Vector<amrex::Real> scale_value_list;
        pp.queryarr("scaling_species_ids", scaling_specid_list);
        pp.queryarr("scale_values", scale_value_list);
        pp.query("potential_scale", vidyut_potscale);
        pp.query("elecenergy_scale", vidyut_eescale);
        pp.query("time_scale", vidyut_timescale);

        if (scaling_specid_list.size() != scale_value_list.size())
        {
            amrex::Print() << "scaling lists dont have the same length\n";
            amrex::Abort();
        }
        for (int i = 0; i < NUM_SPECIES; i++)
        {
            vidyut_specscales[i] = 1.0;
        }
        for (int i = 0; i < scaling_specid_list.size(); i++)
        {
            vidyut_specscales[scaling_specid_list[i]] = scale_value_list[i];
        }

#ifdef AMREX_USE_HYPRE
        pp.query("use_hypre", use_hypre);
#endif
        pp.query("using_ib", using_ib);

        if (using_ib)
        {
            ngrow_for_fillpatch = 3;
        }

        pp.query("track_integrated_currents", track_integrated_currents);
        if (track_integrated_currents)
        {

            print_current_int = plot_int;
            pp.query("print_current_int", print_current_int);
            Vector<int> current_loc_surfaces_vec;
            pp.queryarr("current_loc_surfaces", current_loc_surfaces_vec);
            ncurrent_locs = current_loc_surfaces_vec.size();

            if (ncurrent_locs > MAX_CURRENT_LOCS)
            {
                amrex::Print()
                    << "ncurrent locs is greater than maximum allowed ("
                    << MAX_CURRENT_LOCS << ")\n";
                amrex::Abort("Reduce the number of ncurrent locs\n");
            }

            for (int i = 0; i < ncurrent_locs; i++)
            {
                current_loc_surfaces[i] = current_loc_surfaces_vec[i];
            }
        }
    }
}

// utility to copy in data from phi_old and/or phi_new into another multifab
void Vidyut::GetData(
    int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    BL_PROFILE("Vidyut::GetData()");
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_new[lev]);
    } else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
        data.push_back(&phi_old[lev]);
        datatime.push_back(t_old[lev]);
    } else
    {
        data.push_back(&phi_old[lev]);
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}

// IB functions
void Vidyut::null_bcoeff_at_ib(
    int ilev,
    Array<MultiFab, AMREX_SPACEDIM>& face_bcoeff,
    MultiFab& Sborder,
    int numcomps)
{
    int captured_ncomps = numcomps;
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

        Array4<Real> sb_arr = Sborder.array(mfi);
        GpuArray<Array4<Real>, AMREX_SPACEDIM> face_bcoeff_arr{AMREX_D_DECL(
            face_bcoeff[0].array(mfi), face_bcoeff[1].array(mfi),
            face_bcoeff[2].array(mfi))};

        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            amrex::ParallelFor(
                face_boxes[idim],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    IntVect face{AMREX_D_DECL(i, j, k)};
                    IntVect lcell{AMREX_D_DECL(i, j, k)};
                    IntVect rcell{AMREX_D_DECL(i, j, k)};

                    lcell[idim] -= 1;
                    int mask_L = int(sb_arr(lcell, CMASK_ID));
                    int mask_R = int(sb_arr(rcell, CMASK_ID));

                    // 1 when both mask_L and mask_R are 0
                    int covered_interface = (!mask_L) * (!mask_R);
                    // 1 when both mask_L and mask_R are 1
                    int regular_interface = (mask_L) * (mask_R);
                    // 1-0 or 0-1 interface
                    int cov_uncov_interface =
                        (mask_L) * (!mask_R) + (!mask_L) * (mask_R);

                    if (cov_uncov_interface) // 1*0 0*1 cases
                    {
                        for (int sp = 0; sp < captured_ncomps; sp++)
                        {
                            face_bcoeff_arr[idim](face, sp) = 0.0;
                        }
                    } else if (covered_interface) // 0*0 case
                    {
                        // keeping bcoeff non zero in dead cells just in case
                        for (int sp = 0; sp < captured_ncomps; sp++)
                        {
                            face_bcoeff_arr[idim](face, sp) = 1.0;
                        }
                    } else
                    {
                        // do nothing
                    }
                });
        }
    }
}

void Vidyut::set_explicit_fluxes_at_ib(
    int ilev,
    MultiFab& rhs,
    MultiFab& acoeff,
    MultiFab& Sborder,
    Real time,
    int compid,
    int rhscompid)
{
    Real captured_gastemp = gas_temperature;
    Real captured_gaspres = gas_pressure;
    Real captured_time = time;
    int solved_comp = compid;
    int rhs_comp = rhscompid;
    ProbParm const* localprobparm = d_prob_parm;

    for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto dx = geom[ilev].CellSizeArray();
        auto prob_lo = geom[ilev].ProbLoArray();
        auto prob_hi = geom[ilev].ProbHiArray();
        const Box& domain = geom[ilev].Domain();
        const int* domlo_arr = geom[ilev].Domain().loVect();
        const int* domhi_arr = geom[ilev].Domain().hiVect();

        GpuArray<int, AMREX_SPACEDIM> domlo = {
            AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
        GpuArray<int, AMREX_SPACEDIM> domhi = {
            AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

        Array4<Real> sb_arr = Sborder.array(mfi);
        Array4<Real> rhs_arr = rhs.array(mfi);
        Array4<Real> acoeff_arr = acoeff.array(mfi);

        Array<Box, AMREX_SPACEDIM> face_boxes;
        face_boxes[0] = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
        face_boxes[1] = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
        face_boxes[2] = mfi.nodaltilebox(2);
#endif
#endif
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (sb_arr(i, j, k, CMASK_ID) < 1.0)
            {
                rhs_arr(i, j, k, rhscompid) = 0.0;
            }
        });

        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {

            amrex::ParallelFor(
                face_boxes[idim], [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    IntVect face{AMREX_D_DECL(i, j, k)};
                    IntVect lcell{AMREX_D_DECL(i, j, k)};
                    IntVect rcell{AMREX_D_DECL(i, j, k)};
                    lcell[idim] -= 1;

                    int mask_L = int(sb_arr(lcell, CMASK_ID));
                    int mask_R = int(sb_arr(rcell, CMASK_ID));

                    int cov_uncov_interface =
                        (mask_L) * (!mask_R) + (!mask_L) * (mask_R);

                    if (cov_uncov_interface) // 1-0 or 0-1 interface
                    {
                        int sgn = (int(sb_arr(lcell, CMASK_ID)) == 1) ? 1 : -1;
                        IntVect intcell = (sgn == 1) ? lcell : rcell;

                        user_transport::bc_ib(
                            face, idim, sgn, solved_comp, rhs_comp, sb_arr,
                            acoeff_arr, rhs_arr, domlo, domhi, prob_lo, prob_hi,
                            dx, captured_time, *localprobparm, captured_gastemp,
                            captured_gaspres);
                    }
                });
        }
    }
}

void Vidyut::set_solver_mask(
    Vector<iMultiFab>& solvermask, Vector<MultiFab>& Sborder)
{
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        for (MFIter mfi(Sborder[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Array4<Real> sb_arr = Sborder[ilev].array(mfi);
            Array4<int> smask_arr = solvermask[ilev].array(mfi);
            const Box& bx = mfi.tilebox();

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    smask_arr(i, j, k) = int(sb_arr(i, j, k, CMASK_ID));
                });
        }
    }
}

void Vidyut::correct_efields_ib(
    Vector<MultiFab>& Sborder,
    Vector<Array<MultiFab, AMREX_SPACEDIM>>& efield_fc,
    Real time)
{
    Real captured_time = time;
    ProbParm const* localprobparm = d_prob_parm;
    Real captured_gastemp = gas_temperature;
    Real captured_gaspres = gas_pressure;

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        for (MFIter mfi(Sborder[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Array4<Real> sb_arr = Sborder[ilev].array(mfi);
            const Box& bx = mfi.tilebox();
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();
            const int* domlo_arr = geom[ilev].Domain().loVect();
            const int* domhi_arr = geom[ilev].Domain().hiVect();

            GpuArray<int, AMREX_SPACEDIM> domlo = {
                AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
            GpuArray<int, AMREX_SPACEDIM> domhi = {
                AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

            Array<Box, AMREX_SPACEDIM> face_boxes;
            face_boxes[0] = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
            face_boxes[1] = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
            face_boxes[2] = mfi.nodaltilebox(2);
#endif
#endif
            GpuArray<Array4<Real>, AMREX_SPACEDIM> efield_fc_arr{AMREX_D_DECL(
                efield_fc[ilev][0].array(mfi), efield_fc[ilev][1].array(mfi),
                efield_fc[ilev][2].array(mfi))};

            for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
            {
                amrex::ParallelFor(
                    face_boxes[idim],
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        IntVect face{AMREX_D_DECL(i, j, k)};
                        IntVect lcell{AMREX_D_DECL(i, j, k)};
                        IntVect rcell{AMREX_D_DECL(i, j, k)};
                        lcell[idim] -= 1;

                        int mask_L = int(sb_arr(lcell, CMASK_ID));
                        int mask_R = int(sb_arr(rcell, CMASK_ID));

                        // 1 when both mask_L and mask_R are 0
                        int covered_interface = (!mask_L) * (!mask_R);

                        // 1-0 or 0-1 interface
                        int cov_uncov_interface =
                            (mask_L) * (!mask_R) + (!mask_L) * (mask_R);

                        if (covered_interface)
                        {
                            efield_fc_arr[idim](face) = 0.0;
                        } else if (cov_uncov_interface)
                        {
                            // FIXME/WARNING: this logic will work if there
                            // sufficient number of valid cells between two
                            // immersed boundaries. situation like
                            // covered|valid|covered will cause problems

                            int sgn = (mask_L == 1) ? 1 : -1;
                            amrex::Real pot_grad = get_onesided_grad(
                                face, sgn, idim, POT_ID, dx, sb_arr);
                            efield_fc_arr[idim](face) = -pot_grad;
                        } else
                        {
                            // do nothing
                        }
                    });
            }
        }
    }

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        const Array<const MultiFab*, AMREX_SPACEDIM> allefieldcomps = {
            AMREX_D_DECL(
                &efield_fc[ilev][0], &efield_fc[ilev][1], &efield_fc[ilev][2])};

        average_face_to_cellcenter(phi_new[ilev], EFX_ID, allefieldcomps);

        // Calculate the reduced electric field
        auto phi_arrays = phi_new[ilev].arrays();
        amrex::ParallelFor(
            phi_new[ilev],
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                auto s_arr = phi_arrays[nbx];
                RealVect Evect{AMREX_D_DECL(
                    s_arr(i, j, k, EFX_ID), s_arr(i, j, k, EFY_ID),
                    s_arr(i, j, k, EFZ_ID))};
                Real Esum = 0.0;
                amrex::Real ndens = 0.0;

                for (int sp = 0; sp < NUM_SPECIES; sp++)
                    ndens += s_arr(i, j, k, sp);
                ndens =
                    ndens -
                    s_arr(i, j, k, E_ID); // Only use heavy species densities

                for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
                    Esum += Evect[dim] * Evect[dim];
                s_arr(i, j, k, REF_ID) = (pow(Esum, 0.5) / ndens) / 1.0e-21;
            });
    }
}

#ifdef ENABLE_IB_FIELD_INTERPOLATION
void Vidyut::interpolate_fields_ib(
    Vector<MultiFab>& Sborder, int startcomp, int numcomp)
{
    for (int lev = 0; lev <= finest_level; lev++)
    {
        const auto& sb_arrays = Sborder[lev].arrays();
        const auto prob_lo = geom[lev].ProbLoArray();
        const auto prob_hi = geom[lev].ProbHiArray();
        const int* domlo_arr = geom[lev].Domain().loVect();
        const int* domhi_arr = geom[lev].Domain().hiVect();
        const GpuArray<int, AMREX_SPACEDIM> domlo = {
            AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
        const GpuArray<int, AMREX_SPACEDIM> domhi = {
            AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};
        const auto dx = geom[lev].CellSizeArray();
        ProbParm const* localprobparm = d_prob_parm;

        amrex::ParallelFor(
            Sborder[lev], Sborder[lev].nGrowVect(), numcomp,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) noexcept {
                auto& statefab = sb_arrays[nbx];
                const IntVect iv{AMREX_D_DECL(i, j, k)};
                if ((n != CMASK_ID))
                {
                    if ((statefab(iv, CMASK_ID) < (1 - 1e-16)) &&
                        (statefab(iv, CMASK_ID) > 1e-16))
                    {
                        amrex::Real xib[AMREX_SPACEDIM];
                        user_transport::get_surface_point(
                            iv, prob_lo, prob_hi, domlo, domhi, dx,
                            *localprobparm, xib);

                        const amrex::Real xi =
                            (xib[0] - (prob_lo[0] + iv[0] * dx[0])) / dx[0];
                        const amrex::Real yi =
                            (xib[1] - (prob_lo[1] + iv[1] * dx[1])) / dx[1];
#if (AMREX_SPACEDIM == 3)
                        const amrex::Real zi =
                            (xib[2] - (prob_lo[2] + iv[2] * dx[2])) / dx[2];
#endif

                        amrex::Real interp_val = 0.0;
                        amrex::Real weight_sum = 0.0;

#if (AMREX_SPACEDIM == 3)
                        for (int kk = -1; kk <= 1; kk++)
                        {
#endif
                            for (int jj = -1; jj <= 1; jj++)
                            {
                                for (int ii = -1; ii <= 1; ii++)
                                {
                                    const IntVect ivn =
                                        iv + IntVect(AMREX_D_DECL(ii, jj, kk));
                                    if (((ivn[0] >= domlo[0]) &&
                                         (ivn[0] <= domhi[0])) &&
                                        ((ivn[1] >= domlo[1]) &&
                                         (ivn[1] <= domhi[1])) &&
#if (AMREX_SPACEDIM == 3)
                                        ((ivn[2] >= domlo[2]) &&
                                         (ivn[2] <= domhi[2])) &&
#endif
                                        (statefab(ivn, CMASK_ID) >=
                                         (1 - 1e-16)))
                                    {
                                        const amrex::Real dx = xi - (ii + 0.5);
                                        const amrex::Real dy = yi - (jj + 0.5);
                                        const amrex::Real dz =
                                            AMREX_D_PICK(0, 0, zi - (kk + 0.5));
                                        const amrex::Real dist =
                                            sqrt(dx * dx + dy * dy + dz * dz);

                                        if (dist > 1e-16)
                                        {
                                            const amrex::Real weight =
                                                1.0 / dist;
                                            interp_val +=
                                                weight *
                                                statefab(ivn, startcomp + n);
                                            weight_sum += weight;
                                        }
                                    }
                                }
                            }
#if (AMREX_SPACEDIM == 3)
                        }
#endif

                        if (weight_sum > 1e-16)
                        {
                            statefab(i, j, k, startcomp + n) =
                                interp_val / weight_sum;
                        }
                    }
                }
            });
    }
}
#endif

void Vidyut::null_field_in_covered_cells(
    Vector<MultiFab>& fld,
    Vector<MultiFab>& Sborder,
    int startcomp,
    int numcomp)
{

    // multiply syntax
    // Multiply (FabArray<FAB>& dst, FabArray<FAB> const& src,
    // int srccomp, int dstcomp, int numcomp, int nghost)

    for (int lev = 0; lev <= finest_level; lev++)
    {
        for (int c = startcomp; c < (startcomp + numcomp); c++)
        {
            amrex::MultiFab::Multiply(
                fld[lev], Sborder[lev], CMASK_ID, c, 1, 0);
        }
    }
}
