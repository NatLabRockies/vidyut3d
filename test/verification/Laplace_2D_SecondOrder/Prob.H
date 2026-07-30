#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>

#include <Chemistry.H>
#include <ProbParm.H>
#include <UnivConstants.H>
#include <VarDefines.H>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_MultiCutFab.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit(ProbParm& h_prob_parm, ProbParm& d_prob_parm)
{
    amrex::ParmParse pp("prob");

    pp.query("inner_radius", h_prob_parm.inner_radius);
    pp.query("outer_radius", h_prob_parm.outer_radius);
    pp.query("inner_voltage", h_prob_parm.inner_voltage);
    pp.query("outer_voltage", h_prob_parm.outer_voltage);
    pp.query("inner_dirichlet", h_prob_parm.inner_dirichlet);
    pp.query("outer_dirichlet", h_prob_parm.outer_dirichlet);
    pp.query("a_inner_dir", h_prob_parm.a_inner_dir);
    pp.query("b_inner_neumann", h_prob_parm.b_inner_neumann);
    pp.query("a_outer_dir", h_prob_parm.a_outer_dir);
    pp.query("b_outer_neumann", h_prob_parm.b_outer_neumann);

    AMREX_ALWAYS_ASSERT(!(
        h_prob_parm.inner_dirichlet == 0 && h_prob_parm.outer_dirichlet == 0));

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif
}

AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(
    Box const& bx,
    Array4<Real> const& phi,
    GeometryData const& geomdata,
    ProbParm* prob_parm)
{}

AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata_eb(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real const> const& vfrac,
    Array4<Real const> const& cp,
    Array4<Real const> const& nrm,
    GeometryData const& geomdata,
    ProbParm* prob_parm)
{
    for (int sp = 0; sp < NVAR; sp++) phi(i, j, k, sp) = 0.0;

    phi(i, j, k, CMASK_ID) = vfrac(i, j, k);

    phi(i, j, k, CPX_ID) = cp(i, j, k, 0);
    phi(i, j, k, CPY_ID) = cp(i, j, k, 1);
#if (AMREX_SPACEDIM == 3)
    phi(i, j, k, CPZ_ID) = cp(i, j, k, 2);
#endif

    phi(i, j, k, NX_ID) = nrm(i, j, k, 0);
    phi(i, j, k, NY_ID) = nrm(i, j, k, 1);
#if (AMREX_SPACEDIM == 3)
    phi(i, j, k, NZ_ID) = nrm(i, j, k, 2);
#endif

    phi(i, j, k, E_ID) = 1e12;
    phi(i, j, k, S1_ID) = 1e12;
    phi(i, j, k, S2_ID) = 1e12;
    phi(i, j, k, ETEMP_ID) = 0.5 * eV;
}

AMREX_INLINE
void init_level_with_eb(
    const BoxArray& ba,
    const DistributionMapping& dm,
    const Geometry& gm,
    MultiFab& state,
    ProbParm* prob_parm)
{
    using namespace amrex;

    const Real inner_r = prob_parm->inner_radius;
    const Real outer_r = prob_parm->outer_radius;

    if (!EB2::IndexSpace::empty()) EB2::IndexSpace::clear();
    RealArray center{
        0.0, 0.0
#if (AMREX_SPACEDIM == 3)
        ,
        0.0
#endif
    };

#if (AMREX_SPACEDIM == 2)
    EB2::CylinderIF inner_cyl(inner_r, 20, 2, center, false);
    EB2::CylinderIF outer_cyl(outer_r, 20, 2, center, true);
    auto annulus_solid = EB2::makeUnion(inner_cyl, outer_cyl);
    auto gshop = EB2::makeShop(annulus_solid);
    EB2::Build(gshop, gm, 0, 0);
#else
    EB2::CylinderIF inner_cyl(inner_r, 20, 2, center, false);
    EB2::CylinderIF outer_cyl(outer_r, 20, 2, center, true);
    auto annulus_solid = EB2::makeUnion(inner_cyl, outer_cyl);
    auto gshop = EB2::makeShop(annulus_solid);
    EB2::Build(gshop, gm, 0, 0);
#endif

    // 1 Ghost cell is sufficient for immediate neighbor search
    int nghost = 1;
    std::unique_ptr<EBFArrayBoxFactory> ebfactory =
        makeEBFabFactory(gm, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    const auto& vfrac_mf = ebfactory->getVolFrac();
    const auto& flags = ebfactory->getMultiEBCellFlagFab();
    const auto& bcent = ebfactory->getBndryCent();
    const auto& bnorm = ebfactory->getBndryNormal();

    MultiFab cp_mf(ba, dm, AMREX_SPACEDIM, nghost);
    MultiFab nrm_mf(ba, dm, AMREX_SPACEDIM, nghost);
    cp_mf.setVal(0.0);
    nrm_mf.setVal(0.0);

    const Real* plo = gm.ProbLo();
    const Real* dx = gm.CellSize();

    // PASS 1: Fill Cut Cells with Absolute Data
    for (MFIter mfi(cp_mf); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();
        auto const& flagfab = flags[mfi].const_array();
        auto cp_arr = cp_mf.array(mfi);
        auto nrm_arr = nrm_mf.array(mfi);
        auto bc_arr = bcent.const_array(mfi);
        auto bn_arr = bnorm.const_array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (flagfab(i, j, k).isSingleValued())
                {
                    cp_arr(i, j, k, 0) =
                        plo[0] + (i + 0.5 + bc_arr(i, j, k, 0)) * dx[0];
                    cp_arr(i, j, k, 1) =
                        plo[1] + (j + 0.5 + bc_arr(i, j, k, 1)) * dx[1];
#if (AMREX_SPACEDIM == 3)
                    cp_arr(i, j, k, 2) =
                        plo[2] + (k + 0.5 + bc_arr(i, j, k, 2)) * dx[2];
#endif
                    for (int n = 0; n < AMREX_SPACEDIM; ++n)
                        nrm_arr(i, j, k, n) = bn_arr(i, j, k, n);
                }
            });
    }
    cp_mf.FillBoundary(gm.periodicity());
    nrm_mf.FillBoundary(gm.periodicity());

    // PASS 2: Fluid neighbors inherit geometric data
    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& flagfab = flags[mfi].const_array();
        auto cp_arr = cp_mf.array(mfi);
        auto nrm_arr = nrm_mf.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (flagfab(i, j, k).isRegular())
                {
                    Real cell_center[3] = {
                        plo[0] + (i + 0.5) * dx[0], plo[1] + (j + 0.5) * dx[1],
                        0.0};
#if (AMREX_SPACEDIM == 3)
                    cell_center[2] = plo[2] + (k + 0.5) * dx[2];
#endif
                    Real best_d2 = 1e30;
                    for (int nk = -1; nk <= 1; ++nk)
                    {
                        for (int nj = -1; nj <= 1; ++nj)
                        {
                            for (int ni = -1; ni <= 1; ++ni)
                            {
                                if (flagfab(i + ni, j + nj, k + nk)
                                        .isSingleValued())
                                {
                                    Real d2 = 0;
                                    for (int n = 0; n < AMREX_SPACEDIM; ++n)
                                    {
                                        Real diff =
                                            cp_arr(i + ni, j + nj, k + nk, n) -
                                            cell_center[n];
                                        d2 += diff * diff;
                                    }
                                    if (d2 < best_d2)
                                    {
                                        best_d2 = d2;
                                        for (int n = 0; n < AMREX_SPACEDIM; ++n)
                                        {
                                            cp_arr(i, j, k, n) = cp_arr(
                                                i + ni, j + nj, k + nk, n);
                                            nrm_arr(i, j, k, n) = nrm_arr(
                                                i + ni, j + nj, k + nk, n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
    }

    // Final Initialization
    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto fab = state[mfi].array();
        auto vfrac_arr = vfrac_mf[mfi].const_array();
        auto cp_arr = cp_mf[mfi].const_array();
        auto nrm_arr = nrm_mf[mfi].const_array();
        auto geomData = gm.data();

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                initdomaindata_eb(
                    i, j, k, fab, vfrac_arr, cp_arr, nrm_arr, geomData,
                    prob_parm);
            });
    }
    amrex::Gpu::streamSynchronize();
    ebfactory.reset();
    EB2::IndexSpace::clear();
}

#endif