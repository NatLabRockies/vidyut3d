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
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("f", h_prob_parm.f);
    pp.query("dgap", h_prob_parm.dgap);
    pp.query("R1", h_prob_parm.R1);
    pp.query("V1", h_prob_parm.V1);
    pp.query("V2", h_prob_parm.V2);
    pp.query("ionden_init", h_prob_parm.ionden_init);
    pp.query("gama", h_prob_parm.gama);

    amrex::ParmParse pp_solver("vidyut");
    amrex::Real Pg, Tg;
    pp_solver.query("gas_pressure", Pg);
    pp_solver.query("gas_temperature", Tg);
    h_prob_parm.bg_gasden = Pg / K_B / Tg;

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
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx = geomdata.CellSize();

    amrex::Real halfgap = 0.5 * prob_parm->dgap;

    for (int k = lo.z; k <= hi.z; ++k)
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {

            for (int i = lo.x; i <= hi.x; ++i)
            {
                Real z = prob_lo[2] + (0.5 + k) * dx[2];
                Real y = prob_lo[1] + (0.5 + j) * dx[1];
                Real x = prob_lo[0] + (0.5 + i) * dx[0];

                for (int c = 0; c < NVAR; c++)
                {
                    phi(i, j, k, c) = 0.0;
                }

                phi(i, j, k, CMASK_ID) = 1.0;
                for (int sp = 0; sp < NUM_SPECIES; sp++)
                {
                    phi(i, j, k, sp) = 1e12;
                }

                phi(i, j, k, AR_ID) = prob_parm->bg_gasden;
                phi(i, j, k, ARp_ID) = prob_parm->ionden_init;
                phi(i, j, k, E_ID) =
                    phi(i, j, k, ARp_ID) + phi(i, j, k, AR2p_ID);
                phi(i, j, k, ETEMP_ID) = 0.5 * eV;
                phi(i, j, k, EEN_ID) =
                    1.5 * phi(i, j, k, E_ID) * K_B * phi(i, j, k, ETEMP_ID);
            }
        }
    }
}
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
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

    phi(i, j, k, CMASK_ID) = vfrac(i, j, k); //(vfrac(i, j, k)>0.1)? 1.0 : 0.0;
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
    if (!EB2::IndexSpace::empty()) EB2::IndexSpace::clear();
    ParmParse pp_vidyut("vidyut");
    bool primitive_shape = false;
    pp_vidyut.query("primitive_shape", primitive_shape);
    if (primitive_shape)
    {
        amrex::Print()
            << "[EB] Calling EB2::Build with eb2.geom_type from inputs...\n";
        EB2::Build(gm, 0, 10);
        amrex::Print() << "[EB] EB2::Build complete. IndexSpace empty? "
                       << (EB2::IndexSpace::empty() ? "YES (problem!)"
                                                    : "NO (ok)")
                       << "\n";
    } else
    {
        ParmParse pp("eb2");
        std::string geom;
        pp.query("geom_type", geom);
        amrex::Print() << "[EB] Building geometry with geom_type = " << geom
                       << "\n";
        if (geom == "gec")
        {
            amrex::Real R1 = prob_parm->R1;
            amrex::Real halfgap = 0.5 * prob_parm->dgap;
            const Real* prob_lo = gm.ProbLo();
            const Real* prob_hi = gm.ProbHi();
            EB2::BoxIF lower_electrode(
                {AMREX_D_DECL(prob_lo[0], prob_lo[1], prob_lo[2])},
                {AMREX_D_DECL(R1, -halfgap, prob_hi[2])}, false);
            EB2::BoxIF upper_electrode(
                {AMREX_D_DECL(prob_lo[0], halfgap, prob_lo[2])},
                {AMREX_D_DECL(R1, prob_hi[1], prob_hi[2])}, false);
            auto gec_solid = EB2::makeUnion(lower_electrode, upper_electrode);
            auto gshop = EB2::makeShop(gec_solid);
            EB2::Build(gshop, gm, 0, 10);
        } else if (geom == "gec_noaxisym")
        {
            amrex::Real R1 = prob_parm->R1;
            amrex::Real halfgap = 0.5 * prob_parm->dgap;
            const Real* prob_lo = gm.ProbLo();
            const Real* prob_hi = gm.ProbHi();
            const amrex::Real xmid = 0.5 * (prob_lo[0] + prob_hi[0]);
            amrex::Print() << xmid << "  " << xmid - R1 << "   " << xmid + R1
                           << std::endl;
            EB2::BoxIF lower_electrode(
                {AMREX_D_DECL(xmid - R1, prob_lo[1], prob_lo[2])},
                {AMREX_D_DECL(xmid + R1, -halfgap, prob_hi[2])}, false);
            EB2::BoxIF upper_electrode(
                {AMREX_D_DECL(xmid - R1, halfgap, prob_lo[2])},
                {AMREX_D_DECL(xmid + R1, prob_hi[1], prob_hi[2])}, false);
            auto gec_solid = EB2::makeUnion(lower_electrode, upper_electrode);
            auto gshop = EB2::makeShop(gec_solid);
            EB2::Build(gshop, gm, 0, 10);
        } else if (geom == "gec3")
        {
            amrex::Real R1 = prob_parm->R1;
            amrex::Real halfgap = 0.5 * prob_parm->dgap;
            const Real* prob_lo = gm.ProbLo();
            const Real* prob_hi = gm.ProbHi();
            // Height of the electrode extending from the domain boundary
            const amrex::Real height =
                (prob_hi[1] - prob_lo[1]) / 2.0 - halfgap;
            // Radius of the outer cylinder that covers the entire domain
            // (centered at x=0, z=0)
            amrex::Real outer_radius = 0.5 * (prob_hi[0] - prob_lo[0]);
            // Height of the outer cylinder (covers entire Y domain)
            amrex::Real total_y_length = prob_hi[1] - prob_lo[1];
            // Outer cylinder that covers the entire domain
            EB2::CylinderIF outer_cover(
                outer_radius, total_y_length, 1, {AMREX_D_DECL(0.0, 0.0, 0.0)},
                true);
            // Height of the top electrode's center (halfway between the domain
            // boundary and the inner edge of the electrode)
            amrex::Real top_electrode_center_y = prob_hi[1] - 0.5 * height;
            // Cylinder representing the top electrode, centered at (x=0,
            // y=top_electrode_center_y, z=0) with radius R1 and height
            // extending in both directions from the center
            EB2::CylinderIF top_electrode(
                R1, height, 1, {AMREX_D_DECL(0.0, top_electrode_center_y, 0.0)},
                false);
            // Cylinder representing the top electrode, centered at (x=0,
            // y=top_electrode_center_y, z=0) with radius R1 and height
            // extending in both directions from the center
            EB2::CylinderIF bottom_electrode(
                R1, height, 1,
                {AMREX_D_DECL(0.0, -top_electrode_center_y, 0.0)}, false);
            // Add Electrodes
            auto electrodes = EB2::makeUnion(top_electrode, bottom_electrode);
            // Subtract the electrodes from the outer cover to create a fluid
            // region that is the entire domain minus the electrodes
            auto fluid_region = EB2::makeUnion(outer_cover, electrodes);
            auto gshop = EB2::makeShop(fluid_region);
            EB2::Build(gshop, gm, 0, 10);
        } else
        {
            amrex::Abort("Unknown geometry type: " + geom);
        }
    }

    amrex::Print() << "[EB] EB2::Build complete. IndexSpace empty? "
                   << (EB2::IndexSpace::empty() ? "YES (problem!)" : "NO (ok)")
                   << "\n";

    // 1 Ghost cell is sufficient for immediate neighbor search
    int nghost = 1;
    std::unique_ptr<EBFArrayBoxFactory> ebfactory =
        makeEBFabFactory(gm, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    const auto& vfrac_mf = ebfactory->getVolFrac();

    Real vf_min = vfrac_mf.min(0);
    Real vf_max = vfrac_mf.max(0);
    amrex::Print() << "[EB] vfrac min/max: " << vf_min << " / " << vf_max
                   << "\n";

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
