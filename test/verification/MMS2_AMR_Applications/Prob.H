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
#include "AppGeom.H"
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
    pp.query("solnfrac", h_prob_parm.solnfrac);
    pp.query("corner_fallback", h_prob_parm.corner_fallback);
    pp.query("d_floor_frac", h_prob_parm.d_floor_frac);
    pp.query("proj_switch", h_prob_parm.proj_switch);
    pp.query("face_fallback", h_prob_parm.face_fallback);
    pp.query("ray_dirichlet", h_prob_parm.ray_dirichlet);
    pp.query("perface_fallback", h_prob_parm.perface_fallback);
    pp.query("cmask_vfrac", h_prob_parm.cmask_vfrac);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        h_prob_parm.cmask_vfrac == 0 || h_prob_parm.cmask_vfrac == 1,
        "prob.cmask_vfrac must be 0 (binary) or 1 (vfrac in cut cells)");

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
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
    for (int k = lo.z; k <= hi.z; ++k)
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i)
            {
                amrex::Real xloc[2];
                const amrex::Real frac = prob_parm->solnfrac;
                const amrex::Real alpha = ECHARGE / EPS0;
                xloc[0] = prob_lo[0] + (i + 0.5) * dx[0];
                xloc[1] = prob_lo[1] + (j + 0.5) * dx[1];
                const amrex::Real rad =
                    sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);
                phi(i, j, k, E_ID) =
                    frac * (rad * rad / alpha + prob_parm->n0);
                phi(i, j, k, HE_ID) = 1.0;
                phi(i, j, k, HEp_ID) =
                    frac * (rad * rad / (2.0 * alpha) + prob_parm->n0);

                phi(i, j, k, POT_ID) = frac * (rad * rad * rad * rad) / 32.0;
                phi(i, j, k, EFX_ID) = 0.0;
                phi(i, j, k, EFY_ID) = 0.0;
                phi(i, j, k, EFZ_ID) = 0.0;
                phi(i, j, k, EEN_ID) =
                    frac * (rad * rad / alpha + prob_parm->n0);
                phi(i, j, k, ETEMP_ID) = 1.0;
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

    /*phi(i, j, k, CMASK_ID) = vfrac(i, j, k);
    if (vfrac(i, j, k) < 0.5)
    {
        phi(i, j, k, CMASK_ID) = 0.0;
    }*/
    // prob.cmask_vfrac = 0: binary mask; 1: vfrac in cut cells, so that
    // vidyut.refine_cutcells can tag them. The solver reads int(CMASK), so
    // both give the same fluid set.
    const amrex::Real vf = vfrac(i, j, k);
    phi(i, j, k, CMASK_ID) = (vf > 1.0 - 1e-8) ? 1.0
                             : (prob_parm->cmask_vfrac ? vf : 0.0);
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

    const Real inner_r_pp = prob_parm->inner_radius;
    const Real outer_r_pp = prob_parm->outer_radius;

    if (!EB2::IndexSpace::empty()) EB2::IndexSpace::clear();
    ParmParse pp_vidyut("vidyut");
    bool primitive_shape = false;
    pp_vidyut.query("primitive_shape", primitive_shape);

    if (primitive_shape)
    {
        EB2::Build(gm, 0, 10);
    } else
    {
        ParmParse pp("eb2");
        std::string geom;
        pp.query("geom_type", geom);

        if (geom == "annulus")
        {
            amrex::Real inner_r, outer_r, height;
            int axis;
            amrex::Vector<amrex::Real> ctr_vec;
            amrex::RealArray ctr;
            pp.get("inner_r", inner_r);
            pp.get("outer_r", outer_r);
            pp.get("height", height);
            pp.get("axis", axis);
            pp.getarr("center", ctr_vec);
            if (pp.queryarr("center", ctr_vec))
            {
                for (int i = 0; i < AMREX_SPACEDIM; ++i)
                {
                    ctr[i] = ctr_vec[i];
                }
            }
            EB2::CylinderIF inner_cyl(inner_r, height, axis, ctr, false);
            EB2::CylinderIF outer_cyl(outer_r, height, axis, ctr, true);

            auto shape = EB2::makeUnion(inner_cyl, outer_cyl);
            auto shop = EB2::makeShop(shape);

            EB2::Build(shop, gm, 0, 10);
        } else if (geom == "gecmms")
        {
            // GEC-like 2D stress geometry (all solid bodies, fluid elsewhere):
            //  chamber : circle of radius Ro about (cox,coy), fluid inside
            //  dome    : half disk radius R1, flat face on y = +gap_half
            //  stem    : box joining the dome to the chamber wall
            //  rect    : fully grid-aligned box electrode, top on y = -gap_half
            amrex::Real Ro = 2.6, cox = 0.05, coy = -0.03;
            amrex::Real R1 = 1.125, gap_half = 0.5625, dome_xc = 0.04;
            amrex::Real stem_hw = 0.45, stem_y0 = 1.3625;
            amrex::Real shiftx = 0.0, shifty = 0.0;
            pp.query("chamber_r", Ro);
            pp.query("chamber_cx", cox);
            pp.query("chamber_cy", coy);
            pp.query("electrode_r", R1);
            pp.query("gap_half", gap_half);
            pp.query("dome_xc", dome_xc);
            pp.query("stem_halfwidth", stem_hw);
            pp.query("stem_y0", stem_y0);
            pp.query("shiftx", shiftx); // rigid shift of the whole geometry
            pp.query("shifty", shifty);
            const amrex::Real big = 50.0;
            amrex::RealArray c_out{AMREX_D_DECL(cox + shiftx, coy + shifty, 0.0)};
            amrex::RealArray c_dom{AMREX_D_DECL(dome_xc + shiftx, gap_half + shifty, 0.0)};
            EB2::CylinderIF chamber(Ro, 2.0 * big, 2, c_out, true);
            EB2::CylinderIF disk(R1, 2.0 * big, 2, c_dom, false);
            EB2::BoxIF upper_half(
                {AMREX_D_DECL(-big, gap_half + shifty, -big)},
                {AMREX_D_DECL(big, big, big)}, false);
            auto dome = EB2::makeIntersection(disk, upper_half);
            EB2::BoxIF stem(
                {AMREX_D_DECL(dome_xc - stem_hw + shiftx, stem_y0 + shifty, -big)},
                {AMREX_D_DECL(dome_xc + stem_hw + shiftx, big, big)}, false);
            EB2::BoxIF rect(
                {AMREX_D_DECL(-R1 + shiftx, -big, -big)},
                {AMREX_D_DECL(R1 + shiftx, -gap_half + shifty, big)}, false);
            // Flat top and bottom plates (GEC-like): electrodes meet them at
            // 90 degrees, the curved chamber wall meets them at an obtuse angle.
            // plate_half <= 0 disables them (electrodes then meet the curved
            // wall in acute wedges, which the field stencil cannot resolve).
            amrex::Real plate_half = 2.0625;
            pp.query("plate_half", plate_half);
            const amrex::Real ph = (plate_half > 0.0) ? plate_half : 2.0 * big;
            EB2::BoxIF top_plate(
                {AMREX_D_DECL(-big, ph + shifty, -big)},
                {AMREX_D_DECL(big, 4.0 * big, big)}, false);
            EB2::BoxIF bot_plate(
                {AMREX_D_DECL(-big, -4.0 * big, -big)},
                {AMREX_D_DECL(big, -ph + shifty, big)}, false);
            auto shape = EB2::makeUnion(chamber, dome, stem, rect, top_plate, bot_plate);
            auto shop = EB2::makeShop(shape);
            EB2::Build(shop, gm, 0, 10);
        } else if (geom == "applications")
        {
            // streamer needle, pin array or barrier-discharge strips, see AppGeom.H
            appgeom::AppIF shape(appgeom::read_params());
            auto shop = EB2::makeShop(shape);
            EB2::Build(shop, gm, 0, 10);
        } else
        {
            amrex::Abort("Unknown geometry type");
        }
    }

    // Analytical annulus geometry parameters used to set the exact wall
    // foot / normal for every regular fluid cell adjacent to a cut cell.
    // This is the "Issue 1" fix: eliminates the O(dx) error introduced
    // by inheriting a neighbor cut cell's centroid/normal directly.
    ParmParse pp_eb("eb2");
    amrex::Real inner_r_geom = inner_r_pp;
    amrex::Real outer_r_geom = outer_r_pp;
    amrex::Vector<amrex::Real> ctr_vec_geom(AMREX_SPACEDIM, 0.0);
    pp_eb.query("inner_r", inner_r_geom);
    pp_eb.query("outer_r", outer_r_geom);
    pp_eb.queryarr("center", ctr_vec_geom);
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ctr_geom{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        ctr_geom[d] = (d < static_cast<int>(ctr_vec_geom.size()))
                          ? ctr_vec_geom[d]
                          : 0.0;
    }
    const amrex::Real r_mid = 0.5 * (inner_r_geom + outer_r_geom);

    // 1 Ghost cell is sufficient for immediate neighbor search
    int nghost = 3;
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

    // -----------------------------------------------------------------
    // PASS 1: Fill cut cells with AMReX EB centroid + normal.
    // -----------------------------------------------------------------
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
    const int method = 4; // scaled quadric + Newton closest point (only method in this case)
    // prob.eb_quadric_offset: 1 fit w = c0 + c1 u + a u^2/2, 0 purely quadratic fit
    // through the centroid. Default in this folder: 1. Must be used together with
    // prob.corner_fallback = 1: without it the offset fit fails at unresolved corners.
    int quadric_offset = 1;
    {
        ParmParse pp_prob("prob");
        pp_prob.query("eb_quadric_offset", quadric_offset);
    }
    const bool use_quadric_offset = (quadric_offset != 0);
    amrex::Print() << "IB reconstruction: scaled quadric + Newton, prob.eb_quadric_offset = "
                   << quadric_offset << ", prob.corner_fallback = "
                   << prob_parm->corner_fallback << "\n";
    const bool use_corner = (prob_parm->corner_fallback != 0);
 if (method == 4)
{
// Method 4 - Weighted least-squares plane fit followed by quadric refinement
// with h-independent conditioning, PLUS corner-aware fallback.
// -----------------------------------------------------------------
// PASS 2 (surface-agnostic, h-independent conditioning, corner-safe):
//   (0) CORNER DETECTION: uses the NEAREST single-valued neighbor's
//       normal (ref_n) as the reference -- NOT the first one found in
//       scan order (which is arbitrary and noisy). If other nearby
//       raw normals disagree with ref_n beyond a threshold, the
//       stencil straddles two distinct surfaces meeting at a sharp
//       corner; skip the fit and fall back to a direction-aware
//       nearest-cut-cell copy.
//   (a) Otherwise, weighted least-squares PLANE fit of neighbouring
//       cut-cell boundary centroids -> O(h^2) normal + plane foot.
//       A flatness self-consistency check compares the fitted normal
//       against ref_n; if they disagree beyond tolerance, treat the
//       cell as corner-like too (catches cases where the pairwise
//       corner test barely passes but the fit is still skewed).
//   (b) QUADRIC refinement of the foot in the plane tangent frame,
//       followed by 1-2 Newton closest-point iterations -> O(h^3) foot.
//
//   The corner fallback picks the single-valued neighbor whose normal
//   best matches ref_n, breaking ties by distance -- instead of pure
//   nearest-distance, which can grab the wrong face right at a corner.
//
//   Stored per regular fluid cell adjacent to the EB:
//     cp_arr  -> curvature-corrected foot-of-normal (physical coords)
//     nrm_arr -> curvature-corrected unit normal
// -----------------------------------------------------------------
{
    constexpr int RAD = 2;                      // fit stencil (requires nghost >= 2)
    constexpr int RAD_CORNER = 3;                // corner-detection window (requires nghost >= 3)
    constexpr Real corner_cos_thresh = 0.85;     // ~32 deg; tune per geometry
    constexpr Real flatness_dot_thresh = 0.995;  // fit-vs-ref_n agreement

    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& flagfab = flags[mfi].const_array();
        auto cp_arr = cp_mf.array(mfi);
        auto nrm_arr = nrm_mf.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (!flagfab(i, j, k).isRegular()) return;

                Real cc[3] = {
                    plo[0] + (i + 0.5) * dx[0],
                    plo[1] + (j + 0.5) * dx[1],
                    0.0};
#if (AMREX_SPACEDIM == 3)
                cc[2] = plo[2] + (k + 0.5) * dx[2];
#endif

                // =========================================================
                // (0) Find ref_n: the NEAREST single-valued neighbor's
                // normal, used as the stable reference for corner
                // detection and the flatness check (replaces the old
                // scan-order-dependent "first_n").
                // =========================================================
                Real ref_n[3] = {0.0, 0.0, 0.0};
                Real ref_d2 = 1.0e30;
                bool have_ref = false;

#if (AMREX_SPACEDIM == 3)
                for (int nk0 = -RAD_CORNER; nk0 <= RAD_CORNER; ++nk0)
#else
                const int nk0 = 0;
#endif
                for (int nj0 = -RAD_CORNER; nj0 <= RAD_CORNER; ++nj0)
                for (int ni0 = -RAD_CORNER; ni0 <= RAD_CORNER; ++ni0)
                {
                    if (!flagfab(i + ni0, j + nj0, k + nk0).isSingleValued())
                        continue;

                    Real p[3] = {
                        cp_arr(i + ni0, j + nj0, k + nk0, 0),
                        cp_arr(i + ni0, j + nj0, k + nk0, 1),
                        0.0};
#if (AMREX_SPACEDIM == 3)
                    p[2] = cp_arr(i + ni0, j + nj0, k + nk0, 2);
#endif
                    Real d2 = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        Real diff = p[d] - cc[d];
                        d2 += diff * diff;
                    }

                    if (d2 < ref_d2)
                    {
                        Real ncand[3] = {
                            nrm_arr(i + ni0, j + nj0, k + nk0, 0),
                            nrm_arr(i + ni0, j + nj0, k + nk0, 1),
                            0.0};
#if (AMREX_SPACEDIM == 3)
                        ncand[2] = nrm_arr(i + ni0, j + nj0, k + nk0, 2);
#endif
                        Real nm = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            nm += ncand[d] * ncand[d];
                        nm = std::sqrt(nm);
                        if (nm > 1e-14)
                        {
                            ref_d2 = d2;
                            have_ref = true;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                ref_n[d] = ncand[d] / nm;
                        }
                    }
                }

                if (!have_ref) return; // no cut-cell neighbor at all

                // =========================================================
                // Pairwise corner check: does any OTHER nearby raw normal
                // disagree with ref_n beyond threshold?
                // =========================================================
                Real min_dot = 1.0;

#if (AMREX_SPACEDIM == 3)
                for (int nk1 = -RAD_CORNER; nk1 <= RAD_CORNER; ++nk1)
#else
                const int nk1 = 0;
#endif
                for (int nj1 = -RAD_CORNER; nj1 <= RAD_CORNER; ++nj1)
                for (int ni1 = -RAD_CORNER; ni1 <= RAD_CORNER; ++ni1)
                {
                    if (!flagfab(i + ni1, j + nj1, k + nk1).isSingleValued())
                        continue;

                    Real ncand[3] = {
                        nrm_arr(i + ni1, j + nj1, k + nk1, 0),
                        nrm_arr(i + ni1, j + nj1, k + nk1, 1),
                        0.0};
#if (AMREX_SPACEDIM == 3)
                    ncand[2] = nrm_arr(i + ni1, j + nj1, k + nk1, 2);
#endif
                    Real nm = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        nm += ncand[d] * ncand[d];
                    nm = std::sqrt(nm);
                    if (nm < 1e-14) continue;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) ncand[d] /= nm;

                    Real dotc = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        dotc += ncand[d] * ref_n[d];
                    min_dot = amrex::min(min_dot, dotc);
                }

                bool corner_like = use_corner && (min_dot < corner_cos_thresh);

                if (corner_like)
                {
                    // -----------------------------------------------------
                    // Direction-aware fallback -- prefer the single-valued
                    // neighbor whose normal best matches ref_n, breaking
                    // ties by distance. Pure nearest-distance can grab the
                    // wrong face right at a corner.
                    // -----------------------------------------------------
                    Real best_score = -1.0e30;
                    Real best_cp[3] = {0.0, 0.0, 0.0};
                    Real best_nrm[3] = {0.0, 0.0, 0.0};
                    const Real dist_norm =
                        RAD_CORNER * RAD_CORNER * dx[0] * dx[0];

#if (AMREX_SPACEDIM == 3)
                    for (int nk2 = -RAD_CORNER; nk2 <= RAD_CORNER; ++nk2)
#else
                    const int nk2 = 0;
#endif
                    for (int nj2 = -RAD_CORNER; nj2 <= RAD_CORNER; ++nj2)
                    for (int ni2 = -RAD_CORNER; ni2 <= RAD_CORNER; ++ni2)
                    {
                        if (!flagfab(i + ni2, j + nj2, k + nk2)
                                 .isSingleValued())
                            continue;

                        Real p[3] = {
                            cp_arr(i + ni2, j + nj2, k + nk2, 0),
                            cp_arr(i + ni2, j + nj2, k + nk2, 1),
                            0.0};
#if (AMREX_SPACEDIM == 3)
                        p[2] = cp_arr(i + ni2, j + nj2, k + nk2, 2);
#endif
                        Real cand_n[3] = {
                            nrm_arr(i + ni2, j + nj2, k + nk2, 0),
                            nrm_arr(i + ni2, j + nj2, k + nk2, 1),
                            0.0};
#if (AMREX_SPACEDIM == 3)
                        cand_n[2] = nrm_arr(i + ni2, j + nj2, k + nk2, 2);
#endif
                        Real cn_mag = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            cn_mag += cand_n[d] * cand_n[d];
                        cn_mag = std::sqrt(cn_mag);
                        if (cn_mag < 1e-14) continue;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            cand_n[d] /= cn_mag;

                        Real d2 = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            Real diff = p[d] - cc[d];
                            d2 += diff * diff;
                        }

                        Real align = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            align += cand_n[d] * ref_n[d];

                        // prioritize normal alignment; distance is tiebreak
                        Real score = align - 0.01 * (d2 / dist_norm);

                        if (score > best_score)
                        {
                            best_score = score;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                best_cp[d] = p[d];
                                best_nrm[d] = cand_n[d];
                            }
                        }
                    }

                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        cp_arr(i, j, k, d) = best_cp[d];
                        nrm_arr(i, j, k, d) = best_nrm[d];
                    }
                    return; // skip the plane/quadric fit for this cell
                }

                // =========================================================
                // Not corner-like (yet) -- proceed with the normal
                // h-independent plane + quadric fit (unchanged math from
                // the original method 4), with a flatness self-consistency
                // check applied right after the plane fit, using ref_n.
                // =========================================================

                // Length scale for non-dimensionalization.
                const Real hscale = dx[0];
                const Real inv_h  = 1.0 / hscale;
                // Gaussian weight sigma ~ RAD*dx (in physical units).
                const Real sig2 =
                    2.0 * (RAD * hscale) * (RAD * hscale);

                // =========================================================
                // (a) PLANE FIT  (all math in units of dx)
                // =========================================================

                Real wsum = 0.0;
                Real x0[3] = {0.0, 0.0, 0.0};   // weighted centroid (phys)
                Real nref[3] = {0.0, 0.0, 0.0}; // nearest-cut normal (sign)
                Real nearest_d2 = 1.0e30;
                int  ncut = 0;

#if (AMREX_SPACEDIM == 3)
                for (int nk = -RAD; nk <= RAD; ++nk)
#else
                const int nk = 0;
#endif
                for (int nj = -RAD; nj <= RAD; ++nj)
                for (int ni = -RAD; ni <= RAD; ++ni)
                {
                    if (!flagfab(i + ni, j + nj, k + nk).isSingleValued())
                        continue;

                    Real p[3] = {
                        cp_arr(i + ni, j + nj, k + nk, 0),
                        cp_arr(i + ni, j + nj, k + nk, 1),
                        0.0};
#if (AMREX_SPACEDIM == 3)
                    p[2] = cp_arr(i + ni, j + nj, k + nk, 2);
#endif
                    Real d2 = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        Real diff = p[d] - cc[d];
                        d2 += diff * diff;
                    }
                    Real w = std::exp(-d2 / sig2);   // smooth, dx-scaled
                    wsum += w;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        x0[d] += w * p[d];
                    ++ncut;

                    if (d2 < nearest_d2) {
                        nearest_d2 = d2;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            nref[d] = nrm_arr(i + ni, j + nj, k + nk, d);
                    }
                }

                if (ncut == 0 || wsum <= 0.0) return;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) x0[d] /= wsum;

                // Weighted covariance C in NON-DIMENSIONAL coords q=(p-x0)/h
                Real C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
#if (AMREX_SPACEDIM == 3)
                for (int nk3 = -RAD; nk3 <= RAD; ++nk3)
#else
                const int nk3 = 0;
#endif
                for (int nj3 = -RAD; nj3 <= RAD; ++nj3)
                for (int ni3 = -RAD; ni3 <= RAD; ++ni3)
                {
                    if (!flagfab(i + ni3, j + nj3, k + nk3).isSingleValued())
                        continue;

                    Real p[3] = {
                        cp_arr(i + ni3, j + nj3, k + nk3, 0),
                        cp_arr(i + ni3, j + nj3, k + nk3, 1),
                        0.0};
#if (AMREX_SPACEDIM == 3)
                    p[2] = cp_arr(i + ni3, j + nj3, k + nk3, 2);
#endif
                    Real q[3] = {0,0,0};
                    Real d2 = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        q[d]  = (p[d] - x0[d]) * inv_h;  // non-dim
                        Real diff = p[d] - cc[d];
                        d2 += diff * diff;
                    }
                    Real w = std::exp(-d2 / sig2);
                    for (int a = 0; a < AMREX_SPACEDIM; ++a)
                    for (int b = 0; b < AMREX_SPACEDIM; ++b)
                        C[a][b] += w * q[a] * q[b];
                }

                // Smallest-eigenvalue eigenvector of C = plane normal.
                Real nfit[3] = {0,0,0};
#if (AMREX_SPACEDIM == 2)
                if (ncut < 2) {
                    for (int d = 0; d < 2; ++d) nfit[d] = nref[d];
                } else {
                    Real a = C[0][0], b = C[0][1], c = C[1][1];
                    Real tr = a + c;
                    Real det = a * c - b * b;
                    Real disc = std::sqrt(std::max(0.0, tr*tr*0.25 - det));
                    Real lam_min = tr * 0.5 - disc;
                    Real e0[2] = { b, lam_min - a };
                    Real e1[2] = { lam_min - c, b };
                    Real m0 = e0[0]*e0[0] + e0[1]*e0[1];
                    Real m1 = e1[0]*e1[0] + e1[1]*e1[1];
                    if (m0 >= m1 && m0 > 1e-24) { nfit[0]=e0[0]; nfit[1]=e0[1]; }
                    else if (m1 > 1e-24)        { nfit[0]=e1[0]; nfit[1]=e1[1]; }
                    else { nfit[0]=nref[0]; nfit[1]=nref[1]; }
                }
#else
                if (ncut < 3) {
                    for (int d = 0; d < 3; ++d) nfit[d] = nref[d];
                } else {
                    Real A[3][3];
                    for (int a=0;a<3;++a) for (int b=0;b<3;++b) A[a][b]=C[a][b];
                    Real V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
                    for (int sweep = 0; sweep < 8; ++sweep) {
                        Real off = A[0][1]*A[0][1] + A[0][2]*A[0][2]
                                 + A[1][2]*A[1][2];
                        if (off < 1e-24) break;
                        for (int p2 = 0; p2 < 3; ++p2)
                        for (int q2 = p2+1; q2 < 3; ++q2) {
                            Real apq = A[p2][q2];
                            if (std::abs(apq) < 1e-24) continue;
                            Real app = A[p2][p2], aqq = A[q2][q2];
                            Real phi = 0.5 * std::atan2(2.0*apq, aqq - app);
                            Real cph = std::cos(phi), sph = std::sin(phi);
                            for (int m = 0; m < 3; ++m) {
                                Real amp = A[m][p2], amq = A[m][q2];
                                A[m][p2] = cph*amp - sph*amq;
                                A[m][q2] = sph*amp + cph*amq;
                            }
                            for (int m = 0; m < 3; ++m) {
                                Real apm = A[p2][m], aqm = A[q2][m];
                                A[p2][m] = cph*apm - sph*aqm;
                                A[q2][m] = sph*apm + cph*aqm;
                            }
                            for (int m = 0; m < 3; ++m) {
                                Real vmp = V[m][p2], vmq = V[m][q2];
                                V[m][p2] = cph*vmp - sph*vmq;
                                V[m][q2] = sph*vmp + cph*vmq;
                            }
                        }
                    }
                    int imin = 0;
                    Real emin = A[0][0];
                    if (A[1][1] < emin) { emin = A[1][1]; imin = 1; }
                    if (A[2][2] < emin) { emin = A[2][2]; imin = 2; }
                    nfit[0] = V[0][imin];
                    nfit[1] = V[1][imin];
                    nfit[2] = V[2][imin];
                }
#endif

                // Normalize plane normal (fallback to nref if degenerate).
                Real nmag = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += nfit[d]*nfit[d];
                nmag = std::sqrt(nmag);
                if (nmag < 1e-14) {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) nfit[d] = nref[d];
                    nmag = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += nfit[d]*nfit[d];
                    nmag = std::sqrt(nmag);
                    if (nmag < 1e-14) return;
                }
                for (int d = 0; d < AMREX_SPACEDIM; ++d) nfit[d] /= nmag;

                // Orient plane normal to agree with nearest cut-cell normal.
                Real dotref = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) dotref += nfit[d]*nref[d];
                if (dotref < 0.0)
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) nfit[d] = -nfit[d];

                // =========================================================
                // Flatness self-consistency check (uses ref_n, the robust
                // nearest-neighbor reference, not scan-order-dependent).
                // If the fitted plane normal disagrees meaningfully with
                // ref_n, treat this cell as corner-like too and fall back
                // to the direction-aware nearest-cut-cell copy.
                // =========================================================
                Real dot_dominant = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    dot_dominant += nfit[d] * ref_n[d];

                if (use_corner && (dot_dominant < flatness_dot_thresh))
                {
                    Real best_score = -1.0e30;
                    Real best_cp[3] = {0.0, 0.0, 0.0};
                    Real best_nrm[3] = {0.0, 0.0, 0.0};
                    const Real dist_norm =
                        RAD_CORNER * RAD_CORNER * dx[0] * dx[0];

#if (AMREX_SPACEDIM == 3)
                    for (int nk4 = -RAD_CORNER; nk4 <= RAD_CORNER; ++nk4)
#else
                    const int nk4 = 0;
#endif
                    for (int nj4 = -RAD_CORNER; nj4 <= RAD_CORNER; ++nj4)
                    for (int ni4 = -RAD_CORNER; ni4 <= RAD_CORNER; ++ni4)
                    {
                        if (!flagfab(i + ni4, j + nj4, k + nk4)
                                 .isSingleValued())
                            continue;

                        Real p[3] = {
                            cp_arr(i + ni4, j + nj4, k + nk4, 0),
                            cp_arr(i + ni4, j + nj4, k + nk4, 1),
                            0.0};
#if (AMREX_SPACEDIM == 3)
                        p[2] = cp_arr(i + ni4, j + nj4, k + nk4, 2);
#endif
                        Real cand_n[3] = {
                            nrm_arr(i + ni4, j + nj4, k + nk4, 0),
                            nrm_arr(i + ni4, j + nj4, k + nk4, 1),
                            0.0};
#if (AMREX_SPACEDIM == 3)
                        cand_n[2] = nrm_arr(i + ni4, j + nj4, k + nk4, 2);
#endif
                        Real cn_mag = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            cn_mag += cand_n[d] * cand_n[d];
                        cn_mag = std::sqrt(cn_mag);
                        if (cn_mag < 1e-14) continue;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            cand_n[d] /= cn_mag;

                        Real d2 = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            Real diff = p[d] - cc[d];
                            d2 += diff * diff;
                        }

                        Real align = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            align += cand_n[d] * ref_n[d];

                        Real score = align - 0.01 * (d2 / dist_norm);

                        if (score > best_score)
                        {
                            best_score = score;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                best_cp[d] = p[d];
                                best_nrm[d] = cand_n[d];
                            }
                        }
                    }

                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        cp_arr(i, j, k, d) = best_cp[d];
                        nrm_arr(i, j, k, d) = best_nrm[d];
                    }
                    return; // skip the quadric refinement for this cell
                }

                // =========================================================
                // (b) QUADRIC REFINEMENT (non-dimensional fit) + NEWTON foot
                // =========================================================

                // Tangent basis (t1, t2) orthonormal to nfit.
                Real t1[3] = {0,0,0};
                Real t2[3] = {0,0,0};
                {
                    Real ax[3] = {1,0,0};
                    if (std::abs(nfit[0]) >= std::abs(nfit[1]) &&
                        std::abs(nfit[0]) >= std::abs(nfit[2]))
                    { ax[0]=0; ax[1]=1; ax[2]=0; }
                    Real dotan = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) dotan += ax[d]*nfit[d];
                    Real t1n = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    { t1[d] = ax[d] - dotan*nfit[d]; t1n += t1[d]*t1[d]; }
                    t1n = std::sqrt(t1n);
                    if (t1n < 1e-14) {
                        // fallback: plane foot (non-dim distn back to phys)
                        Real distn = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            distn += (cc[d] - x0[d]) * nfit[d];
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            cp_arr(i, j, k, d)  = cc[d] - distn * nfit[d];
                            nrm_arr(i, j, k, d) = nfit[d];
                        }
                        return;
                    }
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) t1[d] /= t1n;
                    t2[0] = nfit[1]*t1[2] - nfit[2]*t1[1];
                    t2[1] = nfit[2]*t1[0] - nfit[0]*t1[2];
                    t2[2] = nfit[0]*t1[1] - nfit[1]*t1[0];
                }

                // Fit quadric  w = 0.5 a u^2 + b u v + 0.5 c v^2  in NON-DIM
                // (u,v,w) = (rel.t1, rel.t2, rel.nfit)/h. Curvature coeffs
                // a,b,c are then in units of 1/h; rescale on use below.
                // c0_nd, c1_nd: offset and slope of the surface relative to the
                // plane through the weighted centroid x0 (the centroid of points on
                // a curved wall lies off the wall by ~0.5*kappa*var(u) = O(h^2)).
                Real a_nd = 0.0, b_nd = 0.0, c_nd = 0.0;
                Real c0_nd = 0.0, c1_nd = 0.0;
#if (AMREX_SPACEDIM == 2)
                {
                    Real S_bw = 0.0, S_bb = 0.0;
                    Real Mq[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                    Real rq3[3] = {0,0,0};
                    int nq = 0;
                    for (int nj5 = -RAD; nj5 <= RAD; ++nj5)
                    for (int ni5 = -RAD; ni5 <= RAD; ++ni5)
                    {
                        if (!flagfab(i + ni5, j + nj5, k).isSingleValued())
                            continue;
                        Real p[2] = {
                            cp_arr(i + ni5, j + nj5, k, 0),
                            cp_arr(i + ni5, j + nj5, k, 1)};
                        Real rel[2] = {(p[0]-x0[0])*inv_h,
                                       (p[1]-x0[1])*inv_h};
                        Real u = rel[0]*t1[0] + rel[1]*t1[1];
                        Real w = rel[0]*nfit[0] + rel[1]*nfit[1];
                        Real basis = 0.5*u*u;
                        Real d2 = 0.0;
                        { Real dxp0=p[0]-cc[0], dxp1=p[1]-cc[1];
                          d2 = dxp0*dxp0 + dxp1*dxp1; }
                        Real wt = std::exp(-d2 / sig2);
                        S_bw += wt*basis*w;
                        S_bb += wt*basis*basis;
                        const Real g3[3] = {1.0, u, basis};
                        for (int a = 0; a < 3; ++a) {
                            rq3[a] += wt*g3[a]*w;
                            for (int b = 0; b < 3; ++b)
                                Mq[a][b] += wt*g3[a]*g3[b];
                        }
                        ++nq;
                    }
                    // full fit  w = c0 + c1 u + 0.5 a u^2  (Cramer)
                    const Real det3 =
                        Mq[0][0]*(Mq[1][1]*Mq[2][2]-Mq[1][2]*Mq[2][1])
                      - Mq[0][1]*(Mq[1][0]*Mq[2][2]-Mq[1][2]*Mq[2][0])
                      + Mq[0][2]*(Mq[1][0]*Mq[2][1]-Mq[1][1]*Mq[2][0]);
                    if (use_quadric_offset && nq >= 4 && std::abs(det3) > 1e-12) {
                        const Real inv3 = 1.0/det3;
                        c0_nd = inv3*(
                            rq3[0]*(Mq[1][1]*Mq[2][2]-Mq[1][2]*Mq[2][1])
                          - Mq[0][1]*(rq3[1]*Mq[2][2]-Mq[1][2]*rq3[2])
                          + Mq[0][2]*(rq3[1]*Mq[2][1]-Mq[1][1]*rq3[2]));
                        c1_nd = inv3*(
                            Mq[0][0]*(rq3[1]*Mq[2][2]-Mq[1][2]*rq3[2])
                          - rq3[0]*(Mq[1][0]*Mq[2][2]-Mq[1][2]*Mq[2][0])
                          + Mq[0][2]*(Mq[1][0]*rq3[2]-rq3[1]*Mq[2][0]));
                        a_nd = inv3*(
                            Mq[0][0]*(Mq[1][1]*rq3[2]-rq3[1]*Mq[2][1])
                          - Mq[0][1]*(Mq[1][0]*rq3[2]-rq3[1]*Mq[2][0])
                          + rq3[0]*(Mq[1][0]*Mq[2][1]-Mq[1][1]*Mq[2][0]));
                    } else {
                        // Divide with safe operands: clang -O3 divides before it
                        // selects, so S_bb = 0 would trap as 0/0 (as in #71).
                        const Real S_bb_div = (S_bb > 1e-20) ? S_bb : 1.0;
                        a_nd = ((S_bb > 1e-20 && nq >= 2) ? S_bw : 0.0) / S_bb_div;
                    }
                }
#else
                {
                    Real M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                    Real rq[3] = {0,0,0};
                    int nq = 0;
                    for (int nk5 = -RAD; nk5 <= RAD; ++nk5)
                    for (int nj5 = -RAD; nj5 <= RAD; ++nj5)
                    for (int ni5 = -RAD; ni5 <= RAD; ++ni5)
                    {
                        if (!flagfab(i+ni5, j+nj5, k+nk5).isSingleValued())
                            continue;
                        Real p[3] = {
                            cp_arr(i+ni5, j+nj5, k+nk5, 0),
                            cp_arr(i+ni5, j+nj5, k+nk5, 1),
                            cp_arr(i+ni5, j+nj5, k+nk5, 2)};
                        Real rel[3] = {(p[0]-x0[0])*inv_h,
                                       (p[1]-x0[1])*inv_h,
                                       (p[2]-x0[2])*inv_h};
                        Real u = rel[0]*t1[0]+rel[1]*t1[1]+rel[2]*t1[2];
                        Real v = rel[0]*t2[0]+rel[1]*t2[1]+rel[2]*t2[2];
                        Real w = rel[0]*nfit[0]+rel[1]*nfit[1]+rel[2]*nfit[2];
                        Real g[3] = {0.5*u*u, u*v, 0.5*v*v};
                        Real d2 = 0.0;
                        { Real dxp0=p[0]-cc[0], dxp1=p[1]-cc[1], dxp2=p[2]-cc[2];
                          d2 = dxp0*dxp0 + dxp1*dxp1 + dxp2*dxp2; }
                        Real wt = std::exp(-d2 / sig2);
                        for (int a=0;a<3;++a){
                            rq[a] += wt*g[a]*w;
                            for (int b=0;b<3;++b) M[a][b] += wt*g[a]*g[b];
                        }
                        ++nq;
                    }
                    if (nq >= 6) {
                        Real det =
                            M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                          - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                          + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
                        if (std::abs(det) > 1e-20) {
                            Real inv = 1.0/det;
                            a_nd = inv*(
                                rq[0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                              - M[0][1]*(rq[1]*M[2][2]-M[1][2]*rq[2])
                              + M[0][2]*(rq[1]*M[2][1]-M[1][1]*rq[2]));
                            b_nd = inv*(
                                M[0][0]*(rq[1]*M[2][2]-M[1][2]*rq[2])
                              - rq[0]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                              + M[0][2]*(M[1][0]*rq[2]-rq[1]*M[2][0]));
                            c_nd = inv*(
                                M[0][0]*(M[1][1]*rq[2]-rq[1]*M[2][1])
                              - M[0][1]*(M[1][0]*rq[2]-rq[1]*M[2][0])
                              + rq[0]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]));
                        }
                    }
                }
#endif

                // Non-dim target of the cell center in the (u,v,w) frame.
                Real relc[3] = {0,0,0};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    relc[d] = (cc[d] - x0[d]) * inv_h;
                Real uc = 0.0, vc = 0.0, wc = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    uc += relc[d]*t1[d];
                    wc += relc[d]*nfit[d];
#if (AMREX_SPACEDIM == 3)
                    vc += relc[d]*t2[d];
#endif
                }

                // Newton closest-point on quadric w(u,v)=0.5a u^2 + b uv + 0.5c v^2
                Real us = uc, vs = vc;
                for (int it = 0; it < 2; ++it) {
                    Real wq = c0_nd + c1_nd*us
                            + 0.5*a_nd*us*us + b_nd*us*vs + 0.5*c_nd*vs*vs;
                    Real wu = c1_nd + a_nd*us + b_nd*vs;   // dw/du
                    Real wv = b_nd*us + c_nd*vs;   // dw/dv
                    Real gu  = (us - uc) + (wq - wc)*wu;
                    Real guu = 1.0 + wu*wu + (wq - wc)*a_nd;
                    if (std::abs(guu) > 1e-20) us -= gu/guu;
#if (AMREX_SPACEDIM == 3)
                    Real gv  = (vs - vc) + (wq - wc)*wv;
                    Real gvv = 1.0 + wv*wv + (wq - wc)*c_nd;
                    if (std::abs(gvv) > 1e-20) vs -= gv/gvv;
#endif
                }

                Real ws = c0_nd + c1_nd*us
                        + 0.5*a_nd*us*us + b_nd*us*vs + 0.5*c_nd*vs*vs;
                Real wu = c1_nd + a_nd*us + b_nd*vs;
                Real wv = b_nd*us + c_nd*vs;

                // Local surface normal in (t1,t2,nfit) frame: (-wu,-wv,1).
                Real nl[3] = { -wu, -wv, 1.0 };
                Real nlmag = std::sqrt(nl[0]*nl[0]+nl[1]*nl[1]+nl[2]*nl[2]);
                for (int d = 0; d < 3; ++d) nl[d] /= nlmag;

                // Map foot & normal back to PHYSICAL coords (rescale by h).
                Real foot_w[3], norm_w[3];
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    foot_w[d] = x0[d]
                              + hscale * ( us*t1[d] + ws*nfit[d]
#if (AMREX_SPACEDIM == 3)
                                         + vs*t2[d]
#endif
                                         );
                    norm_w[d] = nl[0]*t1[d] + nl[2]*nfit[d]
#if (AMREX_SPACEDIM == 3)
                              + nl[1]*t2[d]
#endif
                              ;
                }
                Real nwmag = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) nwmag += norm_w[d]*norm_w[d];
                nwmag = std::sqrt(nwmag);
                if (nwmag < 1e-14) {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) norm_w[d] = nfit[d];
                } else {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) norm_w[d] /= nwmag;
                }
                Real orient = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) orient += norm_w[d]*nfit[d];
                if (orient < 0.0)
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) norm_w[d] = -norm_w[d];

                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    cp_arr(i, j, k, d)  = foot_w[d];
                    nrm_arr(i, j, k, d) = norm_w[d];
                }
            });
    }
}
cp_mf.FillBoundary(gm.periodicity());
nrm_mf.FillBoundary(gm.periodicity());

 }

    // ============================================================
    // -----------------------------------------------------------------
    // Final: copy vfrac + (cp, nrm) into the state MultiFab components
    // consumed by UserFunctions.H:bc_ib via CMASK_ID, CP*_ID, N*_ID.
    // -----------------------------------------------------------------
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
