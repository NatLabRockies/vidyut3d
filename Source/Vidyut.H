#ifndef Vidyut_H_
#define Vidyut_H_

#include <AMReX_MLMG.H>

#include <string>
#include <limits>
#include <memory>

#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <ProbParm.H>

#include <AMReX_MLABecLaplacian.H>
#include <UnivConstants.H>
#include <VarDefines.H>

using namespace amrex;

class Vidyut : public amrex::AmrCore
{
public:
    ////////////////
    // public member functions

    // constructor - reads in parameters from inputs file
    //             - sizes multilevel arrays and data structures
    Vidyut();
    virtual ~Vidyut();

    // advance solution to final time
    void Evolve();

    // initializes multilevel data
    void InitData();

    // Make a new level using provided BoxArray and DistributionMapping and
    // fill with interpolated coarse level data.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromCoarse(
        int lev,
        amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping
    // and fill with existing fine and coarse data. overrides the pure virtual
    // function in AmrCore
    virtual void RemakeLevel(
        int lev,
        amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    // Delete level data
    // overrides the pure virtual function in AmrCore
    virtual void ClearLevel(int lev) override;

    // Make a new level from scratch using provided BoxArray and
    // DistributionMapping. Only used during initialization. overrides the pure
    // virtual function in AmrCore
    virtual void MakeNewLevelFromScratch(
        int lev,
        amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    // tag all cells for refinement
    // overrides the pure virtual function in AmrCore
    virtual void
    ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
        override;

    void compute_scalar_transport_flux(
        int startspec,
        int numspec,
        int lev,
        MultiFab& Sborder,
        Array<MultiFab, AMREX_SPACEDIM>& flux,
        Vector<int>& bc_lo,
        Vector<int>& bc_hi,
        Real time);

    void fill_sborder(amrex::Vector<amrex::MultiFab> &Sborder, 
            amrex::Real cur_time, amrex::Real dt_common);

    void solve_three_term_photoionization(amrex::Vector<amrex::MultiFab> &Sborder,
                                          amrex::Vector<amrex::MultiFab> &rxn_src,
                                           amrex::Real cur_time,amrex::Real time_offset);

    void compute_axisym_correction(
        int startspec,
        int numspec,
        int lev,
        MultiFab& Sborder,
        MultiFab& dsdt,
        Real time);

    void compute_dsdt(
        int startspec,
        int numspec,
        int lev,
        Array<MultiFab, AMREX_SPACEDIM>& flux,
        MultiFab& rxn_src,
        MultiFab& dsdt,
        Real time,
        Real dt);

    void solve_potential(
            Real current_time,
            Vector<MultiFab>& Sborder,
            Vector<MultiFab>& Sborder_old,
            Vector<int>& bc_lo,
            Vector<int>& bc_hi,
            Vector<Array<MultiFab, 
            AMREX_SPACEDIM>>& efield_ec,
            int electron_poisson_coupling,amrex::Real dt_global,
            amrex::Vector<MultiFab>& ephi_src);

    void update_explsrc_at_all_levels(
            int startspec,
            int numspec,
            Vector<MultiFab>& Sborder,
            Vector<MultiFab>& rxn_src,
            Vector<MultiFab>& expl_src,
            Vector<int>& bc_lo,
            Vector<int>& bc_hi,
            amrex::Real cur_time);

    void update_rxnsrc_at_all_levels(
            Vector<MultiFab>& Sborder,
            Vector<MultiFab>& rxn_src,
            amrex::Real cur_time);

    void implicit_solve_scalar(
            Real current_time,
            Real dt,
            int startspec,
            int numspec,
            Vector<MultiFab>& Sborder,
            Vector<MultiFab>& Sborder_old,
            Vector<MultiFab>& dsdt_expl,
            Vector<int>& bc_lo,
            Vector<int>& bc_hi,
            Vector<Array<MultiFab, AMREX_SPACEDIM>>& grad_fc);

    void solve_photoionization(
            Real current_time,
            Vector<MultiFab>& Sborder,
            Vector<int>& bc_lo,
            Vector<int>& bc_hi,
            Vector<MultiFab>& photoion_src,
            int sph_id);

    void compute_elecenergy_source(
            int lev,
            MultiFab& Sborder,
            MultiFab& rxnsrc,
            Array<MultiFab, AMREX_SPACEDIM>& efield,
            Array<MultiFab, AMREX_SPACEDIM>& gradne,
            MultiFab& dsdt,
            Real time,
            Real dt,
            int floor_jh);

    void compute_electemp_lfa(int lev, MultiFab& Sborder, amrex::Real time);

    void compute_current_den(Vector<MultiFab>& Sborder);
    void compute_disp_current_den(amrex::Real time, amrex::Real tstep);
    void compute_integrated_currents();

    void compute_current_density_at_level(
            int lev,
            MultiFab& Sborder,
            Array<MultiFab, AMREX_SPACEDIM>& ecurrden,
            Array<MultiFab, AMREX_SPACEDIM>& icurrden);

    // compute dt from CFL considerations
    void find_time_scales(
            int lev,
            amrex::Real& dt_edrift,
            amrex::Real& dt_ediff,
            amrex::Real& dt_diel_relax);

    static ProbParm* h_prob_parm;
    static ProbParm* d_prob_parm;

    std::unique_ptr<amrex::MLABecLaplacian> linsolve_ptr;

    ////////////////
    amrex::Gpu::ManagedVector<Real> refine_phi;
    amrex::Gpu::ManagedVector<Real> refine_phigrad;
    amrex::Gpu::ManagedVector<int> refine_phi_comps;

    amrex::Vector<int> istep;     // which step?
    amrex::Vector<int> nsubsteps; // how many substeps on each level?

    // keep track of old time, new time, and time step at each level
    amrex::Vector<amrex::Real> t_new;
    amrex::Vector<amrex::Real> t_old;
    amrex::Vector<amrex::Real> dt;

    // array of multifabs to store the solution at each level of refinement
    // after advancing a level we use "swap".
    amrex::Vector<amrex::MultiFab> phi_new;
    amrex::Vector<amrex::MultiFab> phi_old;

    // this is essentially a 2*DIM integer array storing the physical boundary
    // condition types at the lo/hi walls in each direction
    amrex::Vector<BCRec> bcspec; // 1-component

    // maximum number of steps and stop time
    int max_step = std::numeric_limits<int>::max();
    amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();

    amrex::Real vidyut_specscales[NUM_SPECIES] = {1.0};
    amrex::Real vidyut_potscale = 1.0;
    amrex::Real vidyut_eescale = 1.0;
    amrex::Real vidyut_timescale = 1.0;

    // if >= 0 we restart from a checkpoint
    std::string restart_chkfile = "";

    amrex::Vector<int> pot_bc_lo{ROBINBC, ROBINBC, ROBINBC};
    amrex::Vector<int> pot_bc_hi{ROBINBC, ROBINBC, ROBINBC};

    amrex::Vector<int> eden_bc_lo{ROBINBC, ROBINBC, ROBINBC};
    amrex::Vector<int> eden_bc_hi{ROBINBC, ROBINBC, ROBINBC};

    amrex::Vector<int> eenrg_bc_lo{ROBINBC, ROBINBC, ROBINBC};
    amrex::Vector<int> eenrg_bc_hi{ROBINBC, ROBINBC, ROBINBC};

    amrex::Vector<int> ion_bc_lo{ROBINBC, ROBINBC, ROBINBC};
    amrex::Vector<int> ion_bc_hi{ROBINBC, ROBINBC, ROBINBC};

    amrex::Vector<int> neutral_bc_lo{ROBINBC, ROBINBC, ROBINBC};
    amrex::Vector<int> neutral_bc_hi{ROBINBC, ROBINBC, ROBINBC};

    amrex::Vector<int> photoion_bc_lo{ROBINBC, ROBINBC, ROBINBC};
    amrex::Vector<int> photoion_bc_hi{ROBINBC, ROBINBC, ROBINBC};

    // advective cfl number - dt = cfl*dx/umax
    amrex::Real advective_cfl = 0.7;
    amrex::Real diffusive_cfl = 1.0;
    amrex::Real dielectric_cfl = 1.0;

    amrex::Real fixed_dt = 1e-12;
    int adaptive_dt = 0;
    amrex::Real dt_min = 1e-14;
    amrex::Real dt_max = 1e-9;
    amrex::Real adaptive_dt_delay = 0.0;
    amrex::Real dt_stretch = std::numeric_limits<amrex::Real>::max();
    int electron_poisson_coupling=0;
    int electron_poisson_maxiters=1;

    amrex::Real linsolve_reltol = 1e-10;
    amrex::Real linsolve_abstol = 1e-10;
    amrex::Real linsolve_bot_reltol = 1e-10;
    amrex::Real linsolve_bot_abstol = -1.0;
    amrex::Real min_species_density = 1e-20;
    amrex::Real min_electron_density = 1e12;
    amrex::Real min_electron_temp = 300.0;

    amrex::Real gas_pressure = P_NTP;
    amrex::Real gas_temperature = T_NTP;

    amrex::Vector<std::string> plasma_param_names;
    amrex::Vector<std::string> allvarnames;

    int linsolve_num_pre_smooth = 2;
    int linsolve_num_post_smooth = 2;
    int linsolve_num_final_smooth = 8;
    int linsolve_num_bottom_smooth = 0;

    int linsolve_maxiter = 100;
    int linsolve_max_coarsening_level = 10;
    int bound_specden = 0;
    int hyp_order = 1;
    int ngrow_for_fillpatch = 2;
    int weno_scheme = 3; // WENO-Z by default
    int track_surf_charge = 0;
    int track_current_den = 0;

    // Include physics
    int elecenergy_solve = 0;
    int do_reactions = 1;
    int do_transport = 1;
    int do_spacechrg = 1;
    int user_defined_potential = 0;
    int user_defined_species = 0;
    int user_defined_vel = 0;
    int user_defined_photoionization = 0;
    int do_bg_reactions = 0;
    int do_photoionization = 0;
    int photoion_ID = -1;
    int multicompsolves = 0;
    int comp_ion_chunks = NUM_IONS;
    int comp_neutral_chunks = NUM_NEUTRALS;

    // Voltage options
    int voltage_profile = 0;
    amrex::Vector<amrex::Real> voltage_amp_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> voltage_amp_hi{0.0, 0.0, 0.0};
    amrex::Real voltage_freq = 1.0;
    amrex::Real voltage_dur = 0.0;
    amrex::Real voltage_center = 0.0;

    int floor_jh = 0;

    void update_cc_efields(amrex::Vector<MultiFab>& Sborder);

    void
    update_surf_charge(Vector<MultiFab>& Sborder, Real current_time, Real dt);

    // IB functions
    void
    set_solver_mask(Vector<iMultiFab>& solvermask, Vector<MultiFab>& Sborder);

    void set_explicit_fluxes_at_ib(
        int lev,
        amrex::Real ascalar,
        amrex::Real bscalar,
        MultiFab& rhs,
        MultiFab& acoeff,
        MultiFab& bcoeff,
        MultiFab& Sborder,
        Real time,
        int compid,
        int rhscompid);

    void null_bcoeff_at_ib(
        int lev,
        Array<MultiFab, AMREX_SPACEDIM>& face_bcoeff,
        MultiFab& Sborder,
        int numcomps);

    void correct_efields_ib(
        Vector<MultiFab>& Sborder,
        Vector<Array<MultiFab, AMREX_SPACEDIM>>& efield_fc,
        Real time);

#ifdef ENABLE_IB_FIELD_INTERPOLATION
    void interpolate_fields_ib(
        Vector<MultiFab>& Sborder, int startcomp, int numcomp);
#endif

    void null_field_in_covered_cells(
        Vector<MultiFab>& fld,
        Vector<MultiFab>& Sborder,
        int startcomp,
        int numcomp);

    // how often each level regrids the higher levels of refinement
    // (after a level advances that many time steps)
    int regrid_int = 2;
    int num_timestep_correctors = 1;

#ifdef AMREX_USE_HYPRE
    int use_hypre = 0;
#endif

    // plotfile prefix and frequency
    std::string plot_file{"plt"};
    int plot_int = -1;
    int plot_int_old = -1;
    amrex::Real plot_time = -1.0;

    // checkpoint prefix and frequency
    std::string chk_file{"chk"};
    int chk_int = -1;
    int chk_int_old = -1;
    amrex::Real chk_time = -1.0;

    // Electron species index
    int E_IDX = 0;

    int using_ib = 0;

    int using_LFA = 0;

    // Monitor file options
    int monitor_file_int = -1;

    int solver_verbose = 1;
    int evolve_verbose = 1;

    int track_integrated_currents = 0;
    amrex::Real integrated_conduction_currents[MAX_CURRENT_LOCS] = {0.0};
    amrex::Real integrated_current_areas[MAX_CURRENT_LOCS] = {0.0};
    amrex::Real integrated_displacement_currents[MAX_CURRENT_LOCS] = {0.0};
    int ncurrent_locs = 0;
    int current_loc_surfaces[MAX_CURRENT_LOCS] = {-1};
    int print_current_int = -1;

private:
    ////////////////
    // private member functions

    // read in some parameters from inputs file
    void ReadParameters();

    // set covered coarse cells to be the average of overlying fine cells
    void AverageDown();

    // more flexible version of AverageDown() that lets you average down across
    // multiple levels
    void AverageDownTo(int crse_lev);

    // compute a new multifab by coping in phi from valid region and filling
    // ghost cells works for single level and 2-level cases (fill fine grid
    // ghost by interpolating from coarse)
    void FillPatch(
        int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // fill an entire multifab by interpolating from the coarser level
    // this comes into play when a new level of refinement appears
    void FillCoarsePatch(
        int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // utility to copy in data from phi_old and/or phi_new into another multifab
    void GetData(
        int lev,
        amrex::Real time,
        amrex::Vector<amrex::MultiFab*>& data,
        amrex::Vector<amrex::Real>& datatime);

    // a wrapper for EstTimeStep(0
    void ComputeDt(
        amrex::Real cur_time,
        amrex::Real dt_delay,
        amrex::Real dt_edrift,
        amrex::Real dt_ediff,
        amrex::Real dt_diel_relax);

    // put together an array of multifabs for writing
    amrex::Vector<const amrex::MultiFab*> PlotFileMF() const;

    std::string intcurrentfilename = "integrated_currents";

    // write plotfile to disk
    void WritePlotFile(int plotfilenum) const;

    // write checkpoint file to disk
    void WriteCheckpointFile(int chkfilenum) const;

    // read checkpoint file from disk
    void ReadCheckpointFile();

    // utility to skip to next line in Header
    static void GotoNextLine(std::istream& is);

    Vector<int> bg_specid_list;

    // Creates monitor file showing the maximum domain value of all variables
    void WriteMonitorFile(amrex::Real time);
};

#endif
