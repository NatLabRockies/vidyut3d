#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>

#include <Prob.H>
#include <Tagging.H>
#include <BCFill.H>
#include <Vidyut.H>
#include <Chemistry.H>

// Split the state into contiguous runs of solution and geometry components.
// The geometry components are the IB cell mask and, when EB is on, the
// boundary centroids and normals it carries along with it.
Vector<Vidyut::CompRange> Vidyut::comp_ranges() const
{
    auto is_geometry = [](int c) {
#ifdef AMREX_USE_EB
        return (c == CMASK_ID) || (c >= CPX_ID && c <= NZ_ID);
#else
        return (c == CMASK_ID);
#endif
    };

    Vector<CompRange> ranges;
    int start = 0;
    for (int c = 1; c <= NVAR; c++)
    {
        if (c == NVAR || is_geometry(c) != is_geometry(start))
        {
            ranges.push_back({start, c - start, is_geometry(start)});
            start = c;
        }
    }
    return ranges;
}

// Recompute a level's geometry components from the geometry itself. A level
// that has just been built or rebuilt by regridding has those components
// filled by interpolation from the coarse level, which is meaningless for a
// volume fraction and worse than meaningless for a normal vector.
void Vidyut::rebuild_level_geometry(
    int lev,
    const BoxArray& ba,
    const DistributionMapping& dm,
    bool preserve_solution)
{
#ifdef AMREX_USE_EB
    // EB rebuilds the index space from the level's own Geometry, so the
    // boundary is resolved at this level's dx rather than inherited from the
    // coarse level. That sharpening is the whole point of refining here.
    if (h_prob_parm->enable_EB)
    {
        // A case's initdomaindata_eb is free to write solution components as
        // well as geometry, and several do: the Laplace cases set the
        // densities and the electron temperature there. On a fresh level that
        // is the intended initial condition. After a regrid it is not - the
        // solution has just been interpolated from the coarse level or copied
        // from the old grids, and overwriting it would silently discard the
        // time-evolved state every time the hierarchy changed. Keep the
        // non-geometry components across the call in that case.
        MultiFab saved;
        const int ncomp = phi_new[lev].nComp();
        const int nghost = phi_new[lev].nGrow();
        if (preserve_solution)
        {
            saved.define(ba, dm, ncomp, nghost);
            MultiFab::Copy(saved, phi_new[lev], 0, 0, ncomp, nghost);
        }

        init_level_with_eb(ba, dm, geom[lev], phi_new[lev], d_prob_parm);

        if (preserve_solution)
        {
            // Restore the old values only where the cell was ALREADY fluid.
            //
            // A cell that the coarse level saw as solid carries the decoupled
            // row's value there, normally zero, and FillPatch/FillCoarsePatch
            // hands that straight down. When the finer level resolves fluid
            // that the coarse mask did not have - which is the whole point of
            // refining across a gap narrower than a coarse cell - restoring it
            // would write that zero into a cell that is now part of the
            // solution. Those cells keep what init_level_with_eb just gave
            // them, which is the case's own initialisation, and the solve
            // takes it from there.
            //
            // The test uses the PRE-rebuild mask, held in `saved`. Geometry
            // components are injected rather than interpolated (see
            // comp_ranges), so a fine cell under a solid coarse cell has
            // exactly 0 there and the comparison is clean.
            auto ranges = comp_ranges();
            for (const auto& r : ranges)
            {
                if (r.is_geometry) continue;
                const int sc = r.scomp;
                const int ec = r.scomp + r.ncomp;
                auto const& sv = saved.const_arrays();
                auto const& ph = phi_new[lev].arrays();
                amrex::ParallelFor(
                    phi_new[lev], amrex::IntVect(nghost),
                    [=] AMREX_GPU_DEVICE(
                        int nbx, int i, int j, int k) noexcept {
                        if (int(sv[nbx](i, j, k, CMASK_ID)) != 1) return;
                        for (int n = sc; n < ec; n++)
                        {
                            ph[nbx](i, j, k, n) = sv[nbx](i, j, k, n);
                        }
                    });
            }
            amrex::Gpu::streamSynchronize();
        }
    }
#else
    // Without EB there is no per-level geometry to rebuild from: the mask is
    // written by initdomaindata, which would clobber the solution. The mask
    // was injected rather than interpolated (see comp_ranges), so it stays a
    // valid 0/1 field, but it is the coarse level's staircase, not a sharper
    // boundary. Use an EB case if you want refinement to improve the geometry.
    amrex::ignore_unused(lev, ba, dm);
#endif
}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void Vidyut::MakeNewLevelFromCoarse(
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    BL_PROFILE("vidyut::MakeNewLevelFromCoarse()");
    const int ncomp = phi_new[lev - 1].nComp();
    const int nghost = phi_new[lev - 1].nGrow();

    phi_new[lev].define(ba, dm, ncomp, nghost);
    phi_old[lev].define(ba, dm, ncomp, nghost);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    FillCoarsePatch(lev, time, phi_new[lev], 0, ncomp);

    rebuild_level_geometry(lev, ba, dm, /*preserve_solution=*/true);
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void Vidyut::RemakeLevel(
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    BL_PROFILE("vidyut::RemakeLevel()");
    const int ncomp = phi_new[lev].nComp();
    const int nghost = phi_new[lev].nGrow();

    MultiFab new_state(ba, dm, ncomp, nghost);
    MultiFab old_state(ba, dm, ncomp, nghost);

    FillPatch(lev, time, new_state, 0, ncomp);

    std::swap(new_state, phi_new[lev]);
    std::swap(old_state, phi_old[lev]);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    rebuild_level_geometry(lev, ba, dm, /*preserve_solution=*/true);
}

// Delete level data
// overrides the pure virtual function in AmrCore
void Vidyut::ClearLevel(int lev)
{
    BL_PROFILE("vidyut::ClearLevel()");
    phi_new[lev].clear();
    phi_old[lev].clear();
}

// Make a new level from scratch using provided BoxArray and
// DistributionMapping. Only used during initialization. overrides the pure
// virtual function in AmrCore
void Vidyut::MakeNewLevelFromScratch(
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    BL_PROFILE("vidyut::MakeNewLevelFromScratch()");

    const int nghost = 0;
    int ncomp = NVAR;

    phi_new[lev].define(ba, dm, ncomp, nghost);
    phi_old[lev].define(ba, dm, ncomp, nghost);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    MultiFab& state = phi_new[lev];
    state.setVal(0.0);

    // Problem parameters on device were initialized in amrex_probinit(...)
    ProbParm* localprobparm = d_prob_parm;

    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        Array4<Real> fab = state[mfi].array();
        GeometryData geomData = geom[lev].data();
        const Box& box = mfi.validbox();

        amrex::launch(box, [=] AMREX_GPU_DEVICE(Box const& tbx) {
            initdomaindata(tbx, fab, geomData, localprobparm);
        });
    }
    // Conditionally build EB depending on user request in the ProbParm.
    // Same call as after a regrid, so a level has the same geometry however
    // it came into existence.
    rebuild_level_geometry(lev, ba, dm);

    // copy new -> old
    amrex::MultiFab::Copy(phi_old[lev], phi_new[lev], 0, 0, ncomp, 0);
}

// set covered coarse cells to be the average of overlying fine cells
void Vidyut::AverageDown()
{
    BL_PROFILE("vidyut::AverageDown()");
    for (int lev = finest_level - 1; lev >= 0; --lev)
    {
        AverageDownTo(lev);
    }
}

// more flexible version of AverageDown() that lets you average down across
// multiple levels
void Vidyut::AverageDownTo(int crse_lev)
{
    // Geometry components are skipped: the coarse level already holds the
    // mask, centroids and normals of its own dx, and averaging the fine
    // level's into them turns unit normals into short ones and drags cells
    // that are wholly fluid at the coarse dx below the mask cutoff.
    for (const auto& r : comp_ranges())
    {
        if (r.is_geometry) continue;

        amrex::average_down(
            phi_new[crse_lev + 1], phi_new[crse_lev], geom[crse_lev + 1],
            geom[crse_lev], r.scomp, r.ncomp, refRatio(crse_lev));
    }
}

// compute a new multifab by coping in phi from valid region and filling ghost
// cells works for single level and 2-level cases (fill fine grid ghost by
// interpolating from coarse)
void Vidyut::FillPatch(int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    BL_PROFILE("vidyut::FillPatch()");
    if (lev == 0)
    {
        Vector<MultiFab*> smf;
        Vector<Real> stime;
        GetData(0, time, smf, stime);

        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(amrcore_fill_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> physbc(
            geom[lev], bcspec, gpu_bndry_func);

        amrex::FillPatchSingleLevel(
            mf, time, smf, stime, 0, icomp, ncomp, geom[lev], physbc, 0);

    } else
    {

        Vector<MultiFab*> cmf, fmf;
        Vector<Real> ctime, ftime;

        GetData(lev - 1, time, cmf, ctime);
        GetData(lev, time, fmf, ftime);

        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(amrcore_fill_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> cphysbc(
            geom[lev - 1], bcspec, gpu_bndry_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> fphysbc(
            geom[lev], bcspec, gpu_bndry_func);

        // Solution components are interpolated; geometry components are
        // injected, so a fine ghost cell inherits its covering coarse cell's
        // mask exactly instead of a conservative blend of it and its
        // neighbours, which would put fractional values into what the IB
        // code reads as a 0/1 flag.
        for (const auto& r : comp_ranges())
        {
            if (r.scomp + r.ncomp <= icomp || r.scomp >= icomp + ncomp)
                continue;

            const int scomp = amrex::max(r.scomp, icomp);
            const int ecomp = amrex::min(r.scomp + r.ncomp, icomp + ncomp);

            Interpolater* mapper = r.is_geometry
                                       ? (Interpolater*)&pc_interp
                                       : (Interpolater*)&cell_cons_interp;

            amrex::FillPatchTwoLevels(
                mf, time, cmf, ctime, fmf, ftime, scomp, scomp, ecomp - scomp,
                geom[lev - 1], geom[lev], cphysbc, scomp, fphysbc, scomp,
                refRatio(lev - 1), mapper, bcspec, scomp);
        }
    }
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void Vidyut::FillCoarsePatch(
    int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    BL_PROFILE("vidyut::FillCoarsePatch()");
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev - 1, time, cmf, ctime);

    if (cmf.size() != 1)
    {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(amrcore_fill_func);
    PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> cphysbc(
        geom[lev - 1], bcspec, gpu_bndry_func);
    PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> fphysbc(
        geom[lev], bcspec, gpu_bndry_func);

    // Geometry components are injected rather than interpolated. For an EB
    // case they are overwritten straight after by rebuild_level_geometry();
    // filling them here anyway keeps the state fully defined for the cases
    // that have no geometry to rebuild from.
    for (const auto& r : comp_ranges())
    {
        if (r.scomp + r.ncomp <= icomp || r.scomp >= icomp + ncomp) continue;

        const int scomp = amrex::max(r.scomp, icomp);
        const int ecomp = amrex::min(r.scomp + r.ncomp, icomp + ncomp);

        Interpolater* mapper = r.is_geometry ? (Interpolater*)&pc_interp
                                             : (Interpolater*)&cell_cons_interp;

        amrex::InterpFromCoarseLevel(
            mf, time, *cmf[0], scomp, scomp, ecomp - scomp, geom[lev - 1],
            geom[lev], cphysbc, scomp, fphysbc, scomp, refRatio(lev - 1),
            mapper, bcspec, scomp);
    }
}