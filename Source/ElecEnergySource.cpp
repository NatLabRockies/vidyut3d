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

void Vidyut::compute_electemp_lfa(int lev, MultiFab& Sborder, amrex::Real time)
{
    BL_PROFILE("vidyut::compute_elecenergy_source()");

    Real captured_time = time;
    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    amrex::Real captured_gastemp = gas_temperature;
    amrex::Real captured_gaspres = gas_pressure;
    amrex::Real minetemp = min_electron_temp;
    int eidx = E_IDX;

    const int* domlo_arr = geom[lev].Domain().loVect();
    const int* domhi_arr = geom[lev].Domain().hiVect();

    GpuArray<int, AMREX_SPACEDIM> domlo = {
        AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
    GpuArray<int, AMREX_SPACEDIM> domhi = {
        AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

    auto sborder_arrays = Sborder.arrays();
    auto phi_arrays = phi_new[lev].arrays();

    amrex::ParallelFor(
        Sborder, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
            auto sborder_arr = sborder_arrays[nbx];
            auto phi_arr = phi_arrays[nbx];

            int validcell = int(sborder_arr(i, j, k, CMASK_ID));
            if (validcell)
            {
                // Create array with species concentrations
                amrex::Real spec[NUM_SPECIES];
                amrex::Real Te = sborder_arr(i, j, k, ETEMP_ID);
                amrex::Real EN = sborder_arr(i, j, k, REF_ID);
                for (int sp = 0; sp < NUM_SPECIES; sp++)
                {
                    spec[sp] = sborder_arr(i, j, k, sp);
                }
                // Get molar production rates
                phi_arr(i, j, k, ETEMP_ID) =
                    etempLFA(captured_gastemp, spec, EN);
                if (phi_arr(i, j, k, ETEMP_ID) < minetemp)
                {
                    phi_arr(i, j, k, ETEMP_ID) = minetemp;
                }
                phi_arr(i, j, k, EEN_ID) =
                    1.5 * K_B * phi_arr(i, j, k, eidx) * minetemp;
            }
        });
}

void Vidyut::compute_elecenergy_source(
    int lev,
    MultiFab& Sborder,
    MultiFab& rxnsrc,
    Array<MultiFab, AMREX_SPACEDIM>& efield,
    Array<MultiFab, AMREX_SPACEDIM>& gradne,
    MultiFab& dsdt,
    Real time,
    Real dt,
    int floor_jh)
{

    BL_PROFILE("vidyut::compute_elecenergy_source()");

    Real captured_time = time;
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

    auto sborder_arrays = Sborder.arrays();
    auto rxn_arrays = rxnsrc.const_arrays();
    auto phi_arrays = phi_new[lev].arrays();
    auto dsdt_arrays = dsdt.arrays();

    auto gradnex_arrays = gradne[0].arrays();
#if AMREX_SPACEDIM > 1
    auto gradney_arrays = gradne[1].arrays();
#if AMREX_SPACEDIM == 3
    auto gradnez_arrays = gradne[2].arrays();
#endif
#endif

    auto efx_arrays = efield[0].arrays();
#if AMREX_SPACEDIM > 1
    auto efy_arrays = efield[1].arrays();
#if AMREX_SPACEDIM == 3
    auto efz_arrays = efield[2].arrays();
#endif
#endif

    amrex::ParallelFor(
        dsdt, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
            auto sborder_arr = sborder_arrays[nbx];
            auto rxn_arr = rxn_arrays[nbx];
            auto phi_arr = phi_arrays[nbx];
            auto dsdt_arr = dsdt_arrays[nbx];

            GpuArray<Array4<Real>, AMREX_SPACEDIM> gradne_arr{AMREX_D_DECL(
                gradnex_arrays[nbx], gradney_arrays[nbx], gradnez_arrays[nbx])};

            GpuArray<Array4<Real>, AMREX_SPACEDIM> ef_arr{AMREX_D_DECL(
                efx_arrays[nbx], efy_arrays[nbx], efz_arrays[nbx])};

            // Joule heating
            amrex::Real mu, dcoeff, etemp, ne;
            amrex::Real efield_x, efield_y, efield_z, efield_face, gradne_face;
            amrex::Real charge = plasmachem::get_charge(eidx) * ECHARGE;
            amrex::Real current_density;
            amrex::Real elec_jheat = 0.0;

            // FIXME: This can be done more efficiently sweeping over
            // faces
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
            {
                // left,right,top,bottom,front,back
                for (int f = 0; f < 2; f++)
                {
                    IntVect face{AMREX_D_DECL(i, j, k)};
                    face[idim] += f;

                    IntVect lcell{AMREX_D_DECL(face[0], face[1], face[2])};
                    IntVect rcell{AMREX_D_DECL(face[0], face[1], face[2])};
                    lcell[idim] -= 1;

                    int mask_L = int(sborder_arr(lcell, CMASK_ID));
                    int mask_R = int(sborder_arr(rcell, CMASK_ID));

                    // 1 when both mask_L and mask_R are 0
                    int covered_interface = (!mask_L) * (!mask_R);
                    // 1 when both mask_L and mask_R are 1
                    int regular_interface = (mask_L) * (mask_R);
                    // 1-0 or 0-1 interface
                    int cov_uncov_interface =
                        (mask_L) * (!mask_R) + (!mask_L) * (mask_R);
                    amrex::Real mask_tot = amrex::Real(mask_L + mask_R);

                    if (!covered_interface)
                    {
                        // by 2 for regular and 1 for cov/uncov
                        etemp = (sborder_arr(lcell, ETEMP_ID) * mask_L +
                                 sborder_arr(rcell, ETEMP_ID) * mask_R) /
                                mask_tot;

                        // FIXME:use face centered updated efields here?
                        Real Esum = 0.0;
                        Real efieldvec_face[AMREX_SPACEDIM];
                        efieldvec_face[0] =
                            (sborder_arr(lcell, EFX_ID) * mask_L +
                             sborder_arr(rcell, EFX_ID) * mask_R) /
                            mask_tot;

                        Esum += amrex::Math::powi<2>(efieldvec_face[0]);
#if AMREX_SPACEDIM > 1
                        efieldvec_face[1] =
                            (sborder_arr(lcell, EFY_ID) * mask_L +
                             sborder_arr(rcell, EFY_ID) * mask_R) /
                            mask_tot;

                        Esum += amrex::Math::powi<2>(efieldvec_face[1]);
#if AMREX_SPACEDIM == 3
                        efieldvec_face[2] =
                            (sborder_arr(lcell, EFZ_ID) * mask_L +
                             sborder_arr(rcell, EFZ_ID) * mask_R) /
                            mask_tot;

                        Esum += amrex::Math::powi<2>(efieldvec_face[2]);
#endif
#endif
                        amrex::Real efield_mag = std::sqrt(Esum);

                        ne = (sborder_arr(lcell, eidx) * mask_L +
                              sborder_arr(rcell, eidx) * mask_R) /
                             mask_tot;

                        efield_face = ef_arr[idim](face);
                        // efield_face = efieldvec_face[idim];

                        amrex::Real ndens = 0.0;
                        for (int sp = 0; sp < NUM_SPECIES; sp++)
                        {
                            ndens += (sborder_arr(lcell, sp) * mask_L +
                                      sborder_arr(rcell, sp) * mask_R) /
                                     mask_tot;
                        }
                        mu = specMob(
                            eidx, etemp, ndens, efield_mag, captured_gastemp);
                        dcoeff = specDiff(
                            eidx, etemp, ndens, efield_mag, captured_gastemp);

                        if (regular_interface)
                        {
                            int sgn = (face[idim] == domlo[idim]) ? -1 : 1;

                            if (face[idim] == domlo[idim] ||
                                face[idim] == domhi[idim] + 1)
                            {
                                gradne_face = get_onesided_grad(
                                    face, sgn, idim, eidx, dx, sborder_arr);
                                current_density =
                                    charge * (mu * ne * efield_face -
                                              dcoeff * gradne_face);
                            } else
                            {
                                gradne_face = gradne_arr[idim](face);
                                current_density =
                                    charge * (mu * ne * efield_face -
                                              dcoeff * gradne_face);
                            }
                        } else // cov-uncov interface
                        {
                            // get_onesided_grad_ib falls back to a shorter
                            // stencil when the channel between two immersed
                            // boundaries is one or two gas cells wide
                            int sgn = (mask_L == 1) ? 1 : -1;
                            gradne_face = get_onesided_grad_ib(
                                face, sgn, idim, eidx, dx, sborder_arr);
                            current_density = charge * (mu * ne * efield_face -
                                                        dcoeff * gradne_face);
                        }

                        elec_jheat += current_density * efield_face;
                    }
                }
            }

            // why only 0.5?
            // because left and right contribute to JxEx,
            // top and bottom contribute to JyEy,
            // front and back contribute to JzEz,
            // so we are averaging each component, so it is 1/2 and not 1/6
            // read Deconinck, T, S. Mahadevan, and L. L. Raja.
            //"Discretization of the Joule heating term for plasma discharge
            // fluid models in unstructured meshes."
            // Journal of computational physics 228.12 (2009): 4435-4443.

            elec_jheat *= 0.5;

            // a switch to make sure joule heating is
            // heating the electrons and not cooling them
            if (floor_jh)
            {
                if (elec_jheat < 0.0)
                {
                    elec_jheat = 0.0;
                }
            }

            // inelastic term already added through reaction source
            dsdt_arr(i, j, k, NUM_SPECIES) += (elec_jheat);
            phi_arr(i, j, k, EJH_ID) = elec_jheat;
            phi_arr(i, j, k, EIH_ID) =
                rxn_arr(i, j, k, EEN_ID); // EEN_ID is same as NUM_SPECIES

            // TODO: Adjust reactive source calculations to split into
            // elastic/inelastic phi_arr(i,j,k,EEH_ID)=elec_elastic_coll_term;
        });
}
