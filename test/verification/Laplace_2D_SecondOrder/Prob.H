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
    /*    // PASS 2: Weighted least-squares plane fit of neighboring cut-cell
    //         boundary centroids. Gives an O(h^2) normal on smooth curved
    //         surfaces (no analytic level-set required). We store, per
    //         regular fluid cell:
    //           cp_arr  -> foot-of-normal of the cell center on the fitted plane
    //           nrm_arr -> fitted plane unit normal (oriented outward-ish)
    {
        // Use a 2-cell search radius for robustness in thin gaps / high
        // curvature. (Requires nghost >= 2 above; if nghost==1, change RAD to 1.)
        constexpr int RAD = 1; // set to 2 if you bump nghost to 2

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

                    // --- Gather weighted centroid x0 and count neighbors ---
                    Real wsum = 0.0;
                    Real x0[3] = {0.0, 0.0, 0.0};
                    // reference normal (nearest cut cell) for sign orientation
                    Real nref[3] = {0.0, 0.0, 0.0};
                    Real nearest_d2 = 1.0e30;
                    int ncut = 0;

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
                        Real w = 1.0 / (d2 + 1.0e-30);
                        wsum += w;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) x0[d] += w * p[d];
                        ++ncut;

                        if (d2 < nearest_d2) {
                            nearest_d2 = d2;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                nref[d] = nrm_arr(i + ni, j + nj, k + nk, d);
                        }
                    }

                    if (ncut == 0 || wsum <= 0.0) return; // nothing to fit

                    for (int d = 0; d < AMREX_SPACEDIM; ++d) x0[d] /= wsum;

                    // --- Build weighted covariance C = Σ w (p-x0)(p-x0)^T ---
                    Real C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
#if (AMREX_SPACEDIM == 3)
                    for (int nk2 = -RAD; nk2 <= RAD; ++nk2)
#else
                    const int nk2 = 0;
#endif
                    for (int nj2 = -RAD; nj2 <= RAD; ++nj2)
                    for (int ni2 = -RAD; ni2 <= RAD; ++ni2)
                    {
                        if (!flagfab(i + ni2, j + nj2, k + nk2).isSingleValued())
                            continue;

                        Real p[3] = {
                            cp_arr(i + ni2, j + nj2, k + nk2, 0),
                            cp_arr(i + ni2, j + nj2, k + nk2, 1),
                            0.0};
#if (AMREX_SPACEDIM == 3)
                        p[2] = cp_arr(i + ni2, j + nj2, k + nk2, 2);
#endif
                        Real q[3] = {0,0,0};
                        Real d2 = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            q[d] = p[d] - x0[d];
                            Real diff = p[d] - cc[d];
                            d2 += diff * diff;
                        }
                        Real w = 1.0 / (d2 + 1.0e-30);
                        for (int a = 0; a < AMREX_SPACEDIM; ++a)
                        for (int b = 0; b < AMREX_SPACEDIM; ++b)
                            C[a][b] += w * q[a] * q[b];
                    }

                    // --- Smallest-eigenvalue eigenvector of C = plane normal ---
                    Real nfit[3] = {0,0,0};

#if (AMREX_SPACEDIM == 2)
                    // 2x2 symmetric: analytic smallest eigenvector.
                    // If we have <2 distinct points, fall back to nref.
                    if (ncut < 2) {
                        for (int d = 0; d < 2; ++d) nfit[d] = nref[d];
                    } else {
                        Real a = C[0][0], b = C[0][1], c = C[1][1];
                        Real tr = a + c;
                        Real det = a * c - b * b;
                        Real disc = std::sqrt(std::max(0.0, tr*tr*0.25 - det));
                        Real lam_min = tr * 0.5 - disc; // smallest eigenvalue
                        // eigenvector of (C - lam_min I) null space
                        Real e0[2] = { b, lam_min - a };
                        Real e1[2] = { lam_min - c, b };
                        Real m0 = e0[0]*e0[0] + e0[1]*e0[1];
                        Real m1 = e1[0]*e1[0] + e1[1]*e1[1];
                        if (m0 >= m1 && m0 > 1e-30) { nfit[0]=e0[0]; nfit[1]=e0[1]; }
                        else if (m1 > 1e-30)        { nfit[0]=e1[0]; nfit[1]=e1[1]; }
                        else { nfit[0]=nref[0]; nfit[1]=nref[1]; }
                    }
#else
                    // 3x3 symmetric: cyclic Jacobi eigen-decomposition.
                    if (ncut < 3) {
                        for (int d = 0; d < 3; ++d) nfit[d] = nref[d];
                    } else {
                        Real A[3][3];
                        for (int a=0;a<3;++a) for (int b=0;b<3;++b) A[a][b]=C[a][b];
                        Real V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
                        for (int sweep = 0; sweep < 8; ++sweep) {
                            // off-diagonal magnitude
                            Real off = A[0][1]*A[0][1] + A[0][2]*A[0][2]
                                     + A[1][2]*A[1][2];
                            if (off < 1e-30) break;
                            for (int p2 = 0; p2 < 3; ++p2)
                            for (int q2 = p2+1; q2 < 3; ++q2) {
                                Real apq = A[p2][q2];
                                if (std::abs(apq) < 1e-30) continue;
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
                        // smallest diagonal -> its eigenvector column
                        int imin = 0;
                        Real emin = A[0][0];
                        if (A[1][1] < emin) { emin = A[1][1]; imin = 1; }
                        if (A[2][2] < emin) { emin = A[2][2]; imin = 2; }
                        nfit[0] = V[0][imin];
                        nfit[1] = V[1][imin];
                        nfit[2] = V[2][imin];
                    }
#endif

                    // Normalize
                    Real nmag = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += nfit[d]*nfit[d];
                    nmag = std::sqrt(nmag);
                    if (nmag < 1e-14) { // degenerate fit -> fall back
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) nfit[d] = nref[d];
                        nmag = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) nmag += nfit[d]*nfit[d];
                        nmag = std::sqrt(nmag);
                        if (nmag < 1e-14) return;
                    }
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) nfit[d] /= nmag;

                    // Orient the fitted normal to agree with the nearest
                    // cut-cell normal (eigenvector sign is arbitrary).
                    Real dotref = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) dotref += nfit[d]*nref[d];
                    if (dotref < 0.0)
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) nfit[d] = -nfit[d];

                    // Foot-of-normal of the cell center onto the fitted plane:
                    //   xb = cc - ((cc - x0)·n) n
                    Real distn = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        distn += (cc[d] - x0[d]) * nfit[d];

                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        cp_arr(i, j, k, d)  = cc[d] - distn * nfit[d];
                        nrm_arr(i, j, k, d) = nfit[d];
                    }
                });
        }
    }*/
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