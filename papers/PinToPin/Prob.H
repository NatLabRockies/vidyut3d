#ifndef _PROB_H_
#define _PROB_H_

// Pin-to-pin discharge, electrode geometry generated with AMReX EB2.
//
// The former GeomUtils.H located the electrode surface by bisection inside
// bc_ib, i.e. for every wall face, every equation and every time step. Here
// EB2 builds the geometry once at initialisation (init_level_with_eb): the
// EB centroid and normal of every cut cell are read from the EBFArrayBoxFactory
// and the wall foot / normal of every fluid cell next to the wall is
// reconstructed from them (same reconstruction as the second-order MMS cases,
// prob.eb_geom_method, default 4). The result is stored in the state
// components EB_cp_{x,y,z} and EB_norm_{x,y,z}; bc_ib only reads them.
// Only centroids and normals are taken from EB; the EB flux machinery of
// AMReX is not used, so the axisymmetric (RZ) build is fine.
//
// Dimension generic: DIM=2 gives the r-z axisymmetric case (pin axis = y),
// DIM=3 the Cartesian case (pin axis = z), see PinGeom.H.

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <Chemistry.H>
#include <ProbParm.H>
#include <UnivConstants.H>
#include <VarDefines.H>
#include <PinGeom.H>

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
    pp.query("pulse_freq", h_prob_parm.pulse_freq);
    pp.query("dgap", h_prob_parm.dgap);
    pp.query("pinrad", h_prob_parm.pinrad);
    pp.query("V1", h_prob_parm.V1);
    pp.query("ionden_init", h_prob_parm.ionden_init);
    pp.query("gama", h_prob_parm.gama);
    pp.query("tpulse", h_prob_parm.tpulse);
    pp.query("ramptime", h_prob_parm.ramptime);
    pp.query("seedionden", h_prob_parm.seedionden);

    pp.query("pin_axis", h_prob_parm.pin_axis);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        h_prob_parm.pin_axis >= 0 && h_prob_parm.pin_axis < AMREX_SPACEDIM,
        "prob.pin_axis must be 0, 1 (or 2 in 3D)");
    amrex::Print() << "Pin axis: prob.pin_axis = " << h_prob_parm.pin_axis
                   << " (transverse centre coordinates in increasing index order)\n";
    // pin arrays: nbot/ntop, centres (nbot*(DIM-1) values), radii (1 or n)
    auto read_array = [&](const char* name, int& n, amrex::Real (*c)[2], amrex::Real* r) {
        pp.query((std::string("n") + name).c_str(), n);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n >= 1 && n <= MAX_PINS, "prob.nbot/ntop must be in [1, MAX_PINS]");
        amrex::Vector<amrex::Real> cv, rv;
        pp.queryarr((std::string(name) + "_centers").c_str(), cv);
        pp.queryarr((std::string(name) + "_pinrad").c_str(), rv);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            cv.empty() || static_cast<int>(cv.size()) == n * (AMREX_SPACEDIM - 1),
            "prob.<bot|top>_centers needs n*(DIM-1) values");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            rv.empty() || rv.size() == 1 || static_cast<int>(rv.size()) == n,
            "prob.<bot|top>_pinrad needs 1 or n values");
        for (int i = 0; i < MAX_PINS; ++i)
        {
            c[i][0] = 0.0;
            c[i][1] = 0.0;
            r[i] = h_prob_parm.pinrad;
            if (i < n && !cv.empty())
            {
                c[i][0] = cv[i * (AMREX_SPACEDIM - 1)];
#if (AMREX_SPACEDIM == 3)
                c[i][1] = cv[i * 2 + 1];
#endif
            }
            if (i < n && !rv.empty()) r[i] = (rv.size() == 1) ? rv[0] : rv[i];
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(r[i] > 0.0, "pin radius must be positive");
        }
        amrex::Print() << "Pin array " << name << ": " << n << " pin(s)";
        for (int i = 0; i < n; ++i)
        {
            amrex::Print() << "  [" << c[i][0];
#if (AMREX_SPACEDIM == 3)
            amrex::Print() << ", " << c[i][1];
#endif
            amrex::Print() << "; r=" << r[i] << "]";
        }
        amrex::Print() << "\n";
    };
    read_array("bot", h_prob_parm.nbot, h_prob_parm.bot_c, h_prob_parm.bot_r);
    read_array("top", h_prob_parm.ntop, h_prob_parm.top_c, h_prob_parm.top_r);
    pp.query("sharp_interface_scheme", h_prob_parm.sharp_interface_scheme);
    pp.query("perface_fallback", h_prob_parm.perface_fallback);
    pp.query("d_floor_frac", h_prob_parm.d_floor_frac);
    pp.query("wall_extrap", h_prob_parm.wall_extrap);
    pp.query("wall_floor_frac", h_prob_parm.wall_floor_frac);
    if (h_prob_parm.wall_extrap < 0 || h_prob_parm.wall_extrap > 2)
    {
        amrex::Abort("prob.wall_extrap must be 0, 1 or 2");
    }
    if (!(h_prob_parm.wall_floor_frac > 0.0 && h_prob_parm.wall_floor_frac <= 1.0))
    {
        amrex::Abort("prob.wall_floor_frac must be in (0, 1]");
    }
    amrex::Print() << "Pin-to-pin wall closure: prob.sharp_interface_scheme = "
                   << h_prob_parm.sharp_interface_scheme
                   << ", prob.perface_fallback = " << h_prob_parm.perface_fallback
                   << ", prob.d_floor_frac = " << h_prob_parm.d_floor_frac
                   << ", prob.wall_extrap = " << h_prob_parm.wall_extrap
                   << " (0 cell value, 1 linear, 2 logarithmic)"
                   << ", prob.wall_floor_frac = " << h_prob_parm.wall_floor_frac
                   << "\n";

    amrex::ParmParse pp_solver("vidyut");
    pp_solver.query("gas_pressure", h_prob_parm.pres);
    pp_solver.query("gas_temperature", h_prob_parm.temp);

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif
}

// Plasma initial state. The mask and the wall geometry are set afterwards by
// init_level_with_eb (called by the solver right after this function).
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

    amrex::Real floorden = 1e10;

    int chrgspec[NUM_SPECIES] = {0};
    CKCHRG(chrgspec);

    for (int k = lo.z; k <= hi.z; ++k)
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i)
            {
                for (int c = 0; c < NVAR; c++)
                {
                    phi(i, j, k, c) = 0.0;
                }
                for (int c = 0; c < NUM_SPECIES; c++)
                {
                    phi(i, j, k, c) = floorden;
                }

                phi(i, j, k, CMASK_ID) = 1.0;

                for (int sp = 0; sp < NUM_SPECIES; sp++)
                {
                    phi(i, j, k, sp) = 1e12;
                }

                phi(i, j, k, AR_ID) = prob_parm->pres / K_B / prob_parm->temp;
                amrex::Real seed_specden = prob_parm->seedionden * 0.01;
                phi(i, j, k, ARm_ID) = seed_specden;
                phi(i, j, k, ARp_ID) = prob_parm->seedionden;

                amrex::Real net_ion_charge = 0.0;
                for (int sp = 0; sp < NUM_SPECIES; sp++)
                {
                    if (sp != E_ID)
                    {
                        net_ion_charge += phi(i, j, k, sp) * chrgspec[sp];
                    }
                }
                phi(i, j, k, E_ID) = net_ion_charge;
                phi(i, j, k, ETEMP_ID) = 0.5 * eV;
                phi(i, j, k, EEN_ID) =
                    1.5 * phi(i, j, k, E_ID) * K_B * phi(i, j, k, ETEMP_ID);
            }
        }
    }
}

// Mask and wall geometry from the EB data. Every cut cell (0 < vfrac < 1) is
// masked, as in the former node-count test of GeomUtils: a cell is fluid only
// when it is entirely in the gas. The plasma state set by initdomaindata is
// left untouched.
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
    phi(i, j, k, CMASK_ID) = (vfrac(i, j, k) >= 1.0 - 1.0e-12) ? 1.0 : 0.0;

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

    // ---- EB2 geometry: two paraboloidal pins (PinGeom.H) ----
    if (!EB2::IndexSpace::empty()) EB2::IndexSpace::clear();
    const pin::Pins pc = pin::make_pins(gm.ProbLoArray(), gm.ProbHiArray(), *prob_parm);
    amrex::Print() << "Pin electrodes (EB2): axis = " << pc.ax
                   << ", " << pc.nbot << " bottom pin(s), tip plane ax = " << pc.c2b
                   << ", " << pc.ntop << " top pin(s), tip plane ax = " << pc.c2t
                   << ", smallest tip radius of curvature = "
                   << 1.0 / (2.0 * std::abs((gm.ProbLo(pc.ax) - pc.c2b) / (pc.rmin * pc.rmin)))
                   << "\n";
    pin::PinIF pinif(pc);
    auto gshop = EB2::makeShop(pinif);
    // extend_domain_face = false: the implicit function is evaluated as is in
    // the ghost region outside the domain (it is defined everywhere). The
    // default clamps the coordinates to the domain faces, which turns the pin
    // surface into a vertical wall beyond the end faces and into a flat
    // plateau across the axis, and biases the fit for the cells there.
    EB2::Build(gshop, gm, 0, 0, 4, true, false);

    // 2 ghost cells: the reconstruction uses a 2-cell search radius
    int nghost = 2;
    std::unique_ptr<EBFArrayBoxFactory> ebfactory =
        makeEBFabFactory(gm, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    const auto& vfrac_mf = ebfactory->getVolFrac();
    const auto& ebflags = ebfactory->getMultiEBCellFlagFab();

    // EB2 flags a cell as single-valued (cut) whenever the surface touches it,
    // including cells touched only at a node or along an edge, whose volume
    // fraction is 1 (or 0) and whose EB centroid / normal are degenerate
    // (centroid on the node, axis-aligned normal). The paraboloids pass
    // exactly through grid nodes at round dimensions. Such cells are re-flagged
    // as regular (or covered) in a private copy of the flags so that the
    // reconstruction treats them like every other fluid cell and never uses
    // their degenerate data. The mask below is set from the volume fraction,
    // consistent with this.
    FabArray<EBCellFlagFab> flags(ba, dm, 1, nghost);
    int n_refl_reg = 0, n_refl_cov = 0;
    for (MFIter mfi(flags); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.growntilebox();
        flags[mfi].template copy<RunOn::Host>(ebflags[mfi], gbx);
        flags[mfi].setType(ebflags[mfi].getType());
        auto fl = flags[mfi].array();
        auto vf = vfrac_mf[mfi].const_array();
        amrex::LoopOnCpu(gbx, [&](int i, int j, int k) {
            if (fl(i, j, k).isSingleValued())
            {
                if (vf(i, j, k) >= 1.0 - 1.0e-12)
                {
                    fl(i, j, k).setRegular();
                    ++n_refl_reg;
                } else if (vf(i, j, k) <= 1.0e-12)
                {
                    fl(i, j, k).setCovered();
                    ++n_refl_cov;
                }
            }
        });
    }
    ParallelDescriptor::ReduceIntSum(n_refl_reg);
    ParallelDescriptor::ReduceIntSum(n_refl_cov);
    amrex::Print() << "EB cells re-flagged (incl. ghost cells): " << n_refl_reg
                   << " full -> regular, " << n_refl_cov << " empty -> covered\n";
    const auto& bcent = ebfactory->getBndryCent();
    const auto& bnorm = ebfactory->getBndryNormal();

    MultiFab cp_mf(ba, dm, AMREX_SPACEDIM, nghost);
    MultiFab nrm_mf(ba, dm, AMREX_SPACEDIM, nghost);
    cp_mf.setVal(0.0);
    nrm_mf.setVal(0.0);

    const Real* plo = gm.ProbLo();
    const Real* dx = gm.CellSize();

    // PASS 1: cut cells get their EB centroid (absolute coordinates) and normal
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

    // ---------------------------------------------------------------------
    // Wall-foot / normal reconstruction for the fluid cells next to the wall,
    // selected at run time (identical to the MMS1 sphere / annulus cases):
    //   prob.eb_geom_method    1 PCA plane + quadric (1/d^2 weights)
    //                          2 PCA tangent plane
    //                          3 analytical paraboloid (PinGeom.H), reference
    //                          4 scaled quadric + Newton closest point
    //                          5 nearest cut cell copy (IB-NG)
    //   prob.eb_quadric_offset 1 fit w = c0 + c1 u + a u^2/2 in method 4
    //                            (3D: c0 + c1 u + c2 v + full quadric)
    //                          0 purely quadratic fit through the centroid
    // Defaults: eb_geom_method = 4, eb_quadric_offset = 1.
    // ---------------------------------------------------------------------
    int method = 4;
    int quadric_offset = 1;
    {
        ParmParse pp_prob("prob");
        pp_prob.query("eb_geom_method", method);
        pp_prob.query("eb_quadric_offset", quadric_offset);
    }
    AMREX_ALWAYS_ASSERT(method >= 1 && method <= 5);
    {
        const char* mname[] = {"", "PCA plane + quadric", "PCA tangent plane",
                               "analytical paraboloid",
                               "scaled quadric + Newton", "nearest cut cell (IB-NG)"};
        amrex::Print() << "IB reconstruction: prob.eb_geom_method = " << method
                       << " (" << mname[method] << ")";
        if (method == 4)
            amrex::Print() << ", prob.eb_quadric_offset = " << quadric_offset;
        amrex::Print() << "\n";
    }
    const bool use_quadric_offset = (quadric_offset != 0);
    if (method == 1)
    {
    // Method 1 - Weighted least-squares plane fit followed by quadric refinement
        // PASS 2: (a) Weighted least-squares plane fit of neighboring cut-cell
    //             boundary centroids -> O(h^2) normal + plane foot.
    //         (b) Quadric refinement of the foot-of-normal using a local
    //             curvature fit in the plane-fit tangent frame -> O(h^3) foot.
    //         We store, per regular fluid cell adjacent to the EB:
    //           cp_arr  -> curvature-corrected foot-of-normal
    //           nrm_arr -> curvature-corrected unit normal
    {
        // 2-cell search radius (requires nghost >= 2 in init_level_with_eb).
        constexpr int RAD = 2;

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
                    // (a) PLANE FIT
                    // =========================================================

                    // --- Weighted centroid x0 + nearest-cut reference normal ---
                    Real wsum = 0.0;
                    Real x0[3] = {0.0, 0.0, 0.0};
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

                    if (ncut == 0 || wsum <= 0.0) return;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) x0[d] /= wsum;

                    // --- Weighted covariance C ---
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
                        if (m0 >= m1 && m0 > 1e-30) { nfit[0]=e0[0]; nfit[1]=e0[1]; }
                        else if (m1 > 1e-30)        { nfit[0]=e1[0]; nfit[1]=e1[1]; }
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
                    // (b) QUADRIC REFINEMENT of the foot-of-normal.
                    // =========================================================

                    // --- Tangent basis (t1, t2) orthonormal to nfit ---
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
                            // fallback: plane foot
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

                    // --- Fit quadric  w = 0.5 a u^2 + b u v + 0.5 c v^2 ---
                    Real a_curv = 0.0, b_curv = 0.0, c_curv = 0.0;
#if (AMREX_SPACEDIM == 2)
                    {
                        Real S_bw = 0.0, S_bb = 0.0;
                        int nq = 0;
                        for (int nj2 = -RAD; nj2 <= RAD; ++nj2)
                        for (int ni2 = -RAD; ni2 <= RAD; ++ni2)
                        {
                            if (!flagfab(i + ni2, j + nj2, k).isSingleValued())
                                continue;
                            Real p[2] = {
                                cp_arr(i + ni2, j + nj2, k, 0),
                                cp_arr(i + ni2, j + nj2, k, 1)};
                            Real rel[2] = { p[0]-x0[0], p[1]-x0[1] };
                            Real u = rel[0]*t1[0] + rel[1]*t1[1];
                            Real w = rel[0]*nfit[0] + rel[1]*nfit[1];
                            Real basis = 0.5*u*u;
                            S_bw += basis*w;
                            S_bb += basis*basis;
                            ++nq;
                        }
                        a_curv = (S_bb > 1e-30 && nq >= 2) ? (S_bw / S_bb) : 0.0;
                    }
#else
                    {
                        Real M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                        Real rq[3] = {0,0,0};
                        int nq = 0;
                        for (int nk2 = -RAD; nk2 <= RAD; ++nk2)
                        for (int nj2 = -RAD; nj2 <= RAD; ++nj2)
                        for (int ni2 = -RAD; ni2 <= RAD; ++ni2)
                        {
                            if (!flagfab(i+ni2, j+nj2, k+nk2).isSingleValued())
                                continue;
                            Real p[3] = {
                                cp_arr(i+ni2, j+nj2, k+nk2, 0),
                                cp_arr(i+ni2, j+nj2, k+nk2, 1),
                                cp_arr(i+ni2, j+nj2, k+nk2, 2)};
                            Real rel[3] = { p[0]-x0[0], p[1]-x0[1], p[2]-x0[2] };
                            Real u = rel[0]*t1[0]+rel[1]*t1[1]+rel[2]*t1[2];
                            Real v = rel[0]*t2[0]+rel[1]*t2[1]+rel[2]*t2[2];
                            Real w = rel[0]*nfit[0]+rel[1]*nfit[1]+rel[2]*nfit[2];
                            Real g[3] = {0.5*u*u, u*v, 0.5*v*v};
                            for (int a=0;a<3;++a){
                                rq[a] += g[a]*w;
                                for (int b=0;b<3;++b) M[a][b] += g[a]*g[b];
                            }
                            ++nq;
                        }
                        if (nq >= 6) {
                            Real det =
                                M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                              - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                              + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
                            if (std::abs(det) > 1e-30) {
                                Real inv = 1.0/det;
                                a_curv = inv*(
                                    rq[0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                                  - M[0][1]*(rq[1]*M[2][2]-M[1][2]*rq[2])
                                  + M[0][2]*(rq[1]*M[2][1]-M[1][1]*rq[2]));
                                b_curv = inv*(
                                    M[0][0]*(rq[1]*M[2][2]-M[1][2]*rq[2])
                                  - rq[0]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                                  + M[0][2]*(M[1][0]*rq[2]-rq[1]*M[2][0]));
                                c_curv = inv*(
                                    M[0][0]*(M[1][1]*rq[2]-rq[1]*M[2][1])
                                  - M[0][1]*(M[1][0]*rq[2]-rq[1]*M[2][0])
                                  + rq[0]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]));
                            }
                        }
                    }
#endif

                    // --- Foot on quadric + curvature-corrected normal ---
                    Real relc[3] = {0,0,0};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) relc[d] = cc[d] - x0[d];
                    Real uc = 0.0, vc = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        uc += relc[d]*t1[d];
#if (AMREX_SPACEDIM == 3)
                        vc += relc[d]*t2[d];
#endif
                    }
                    Real us = uc, vs = vc;
                    Real ws = 0.5*a_curv*us*us + b_curv*us*vs + 0.5*c_curv*vs*vs;
                    Real wu = a_curv*us + b_curv*vs;
                    Real wv = b_curv*us + c_curv*vs;
                    Real nl[3] = { -wu, -wv, 1.0 };
                    Real nlmag = std::sqrt(nl[0]*nl[0]+nl[1]*nl[1]+nl[2]*nl[2]);
                    for (int d = 0; d < 3; ++d) nl[d] /= nlmag;

                    Real foot_w[3], norm_w[3];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        foot_w[d] = x0[d] + us*t1[d] + ws*nfit[d]
#if (AMREX_SPACEDIM == 3)
                                  + vs*t2[d]
#endif
                                  ;
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
else if (method == 2)
{
    // Method 2 - Weighted least-squares plane fit of neighboring cut-cell boundary centroids
       // PASS 2: Weighted least-squares plane fit of neighboring cut-cell
    //         boundary centroids. Gives an O(h^2) normal on smooth curved
    //         surfaces (no analytic level-set required). We store, per
    //         regular fluid cell:
    //           cp_arr  -> foot-of-normal of the cell center on the fitted plane
    //           nrm_arr -> fitted plane unit normal (oriented outward-ish)
        // Use a 2-cell search radius for robustness in thin gaps / high
        // curvature. (Requires nghost >= 2 above; if nghost==1, change RAD to 1.)
        constexpr int RAD = 2; // set to 2 if you bump nghost to 2

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
    cp_mf.FillBoundary(gm.periodicity());
    nrm_mf.FillBoundary(gm.periodicity());
    }
    else if (method == 3)
    {
    // Method 3 - analytical foot and normal on the paraboloidal pins for the
    // regular fluid cells that touch a cut cell (reference for methods 1-5).
    const pin::Pins pc_dev = pc;

    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& flagfab = flags[mfi].const_array();
        auto cp_arr = cp_mf.array(mfi);
        auto nrm_arr = nrm_mf.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (!flagfab(i, j, k).isRegular()) return;

                bool has_cut_neighbor = false;
#if (AMREX_SPACEDIM == 3)
                // in 2D the k index has a single valid value
#if (AMREX_SPACEDIM == 3)
                const int nk_lo3 = -1, nk_hi3 = 1;
#else
                const int nk_lo3 = 0, nk_hi3 = 0;
#endif
                for (int nk = nk_lo3; nk <= nk_hi3 && !has_cut_neighbor; ++nk)
#else
                const int nk = 0;
#endif
                for (int nj = -1; nj <= 1 && !has_cut_neighbor; ++nj)
                for (int ni = -1; ni <= 1 && !has_cut_neighbor; ++ni)
                {
                    if (flagfab(i + ni, j + nj, k + nk).isSingleValued())
                        has_cut_neighbor = true;
                }
                if (!has_cut_neighbor) return;

                Real cc[AMREX_SPACEDIM];
                cc[0] = plo[0] + (i + 0.5) * dx[0];
                cc[1] = plo[1] + (j + 0.5) * dx[1];
#if (AMREX_SPACEDIM == 3)
                cc[2] = plo[2] + (k + 0.5) * dx[2];
#endif
                Real xb[AMREX_SPACEDIM], n[AMREX_SPACEDIM];
                pin::closest_point(cc, pc_dev, xb, n);
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    cp_arr(i, j, k, d) = xb[d];
                    nrm_arr(i, j, k, d) = n[d];
                }
            });
    }
    cp_mf.FillBoundary(gm.periodicity());
    nrm_mf.FillBoundary(gm.periodicity());
    }
    else if (method == 4)
    {
    // Method 4 - Weighted least-squares plane fit followed by quadric refinement with h-independent conditioning
    // -----------------------------------------------------------------
    // PASS 2 (surface-agnostic, h-independent conditioning):
    //   (a) Weighted least-squares PLANE fit of neighbouring cut-cell
    //       boundary centroids -> O(h^2) normal + plane foot.
    //   (b) QUADRIC refinement of the foot in the plane tangent frame,
    //       followed by 1-2 Newton closest-point iterations -> O(h^3) foot.
    //
    //   Key robustness changes vs. the previous version:
    //     * All fit coordinates are NON-DIMENSIONALIZED by dx (units of dx),
    //       so the covariance / normal-equation conditioning is INDEPENDENT
    //       of h. Curvature coeffs are rescaled back to physical units after.
    //     * Weights are a smooth dx-scaled Gaussian (support ~ RAD*dx), not
    //       an inverse-distance with a fixed 1e-30 floor, so the stencil
    //       keeps effective support as h -> 0.
    //     * The foot is refined by Newton iteration of the closest-point
    //       condition on the fitted quadric (not a single projection).
    //
    //   Stored per regular fluid cell adjacent to the EB:
    //     cp_arr  -> curvature-corrected foot-of-normal (physical coords)
    //     nrm_arr -> curvature-corrected unit normal
    // -----------------------------------------------------------------
    {
        constexpr int RAD = 2; // requires nghost >= 2

        for (MFIter mfi(state); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto const& flagfab = flags[mfi].const_array();
            auto cp_arr = cp_mf.array(mfi);
            auto nrm_arr = nrm_mf.array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    if (!flagfab(i, j, k).isRegular()) return;

                    // Length scale for non-dimensionalization.
                    const Real hscale = dx[0];
                    const Real inv_h  = 1.0 / hscale;
                    // Gaussian weight sigma ~ RAD*dx (in physical units).
                    const Real sig2 =
                        2.0 * (RAD * hscale) * (RAD * hscale);

                    Real cc[3] = {
                        plo[0] + (i + 0.5) * dx[0],
                        plo[1] + (j + 0.5) * dx[1],
                        0.0};
#if (AMREX_SPACEDIM == 3)
                    cc[2] = plo[2] + (k + 0.5) * dx[2];
#endif

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
                    // c0_nd, c1_nd: offset and slope of the surface relative
                    // to the plane through the weighted centroid x0. Points on a
                    // curved wall have a centroid that lies OFF the wall by
                    // ~0.5*kappa*var(u) = O(h^2); without c0 the fitted foot
                    // inherits that bias.
                    Real a_nd = 0.0, b_nd = 0.0, c_nd = 0.0;
                    Real c0_nd = 0.0, c1_nd = 0.0, c2_nd = 0.0;
#if (AMREX_SPACEDIM == 2)
                    {
                        Real S_bw = 0.0, S_bb = 0.0;
                        Real Mq[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                        Real rq3[3] = {0,0,0};
                        int nq = 0;
                        for (int nj2 = -RAD; nj2 <= RAD; ++nj2)
                        for (int ni2 = -RAD; ni2 <= RAD; ++ni2)
                        {
                            if (!flagfab(i + ni2, j + nj2, k).isSingleValued())
                                continue;
                            Real p[2] = {
                                cp_arr(i + ni2, j + nj2, k, 0),
                                cp_arr(i + ni2, j + nj2, k, 1)};
                            Real rel[2] = {(p[0]-x0[0])*inv_h,
                                           (p[1]-x0[1])*inv_h};
                            Real u = rel[0]*t1[0] + rel[1]*t1[1];
                            Real w = rel[0]*nfit[0] + rel[1]*nfit[1];
                            Real basis = 0.5*u*u;
                            // Gaussian weight (recompute in phys distance)
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
                            a_nd = (S_bb > 1e-20 && nq >= 2) ? (S_bw / S_bb) : 0.0;
                        }
                    }
#else
                    {
                        // 3D: full fit
                        //   w = c0 + c1 u + c2 v + 0.5 a u^2 + b u v + 0.5 c v^2
                        // when prob.eb_quadric_offset = 1, otherwise the purely
                        // quadratic fit through the weighted centroid.
                        Real M6[6][7];
                        for (int a6 = 0; a6 < 6; ++a6)
                            for (int b6 = 0; b6 < 7; ++b6) M6[a6][b6] = 0.0;
                        Real M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                        Real rq[3] = {0,0,0};
                        int nq = 0;
                        for (int nk2 = -RAD; nk2 <= RAD; ++nk2)
                        for (int nj2 = -RAD; nj2 <= RAD; ++nj2)
                        for (int ni2 = -RAD; ni2 <= RAD; ++ni2)
                        {
                            if (!flagfab(i+ni2, j+nj2, k+nk2).isSingleValued())
                                continue;
                            Real p[3] = {
                                cp_arr(i+ni2, j+nj2, k+nk2, 0),
                                cp_arr(i+ni2, j+nj2, k+nk2, 1),
                                cp_arr(i+ni2, j+nj2, k+nk2, 2)};
                            Real rel[3] = {(p[0]-x0[0])*inv_h,
                                           (p[1]-x0[1])*inv_h,
                                           (p[2]-x0[2])*inv_h};
                            Real u = rel[0]*t1[0]+rel[1]*t1[1]+rel[2]*t1[2];
                            Real v = rel[0]*t2[0]+rel[1]*t2[1]+rel[2]*t2[2];
                            Real w = rel[0]*nfit[0]+rel[1]*nfit[1]+rel[2]*nfit[2];
                            Real g[3] = {0.5*u*u, u*v, 0.5*v*v};
                            Real g6[6] = {1.0, u, v, 0.5*u*u, u*v, 0.5*v*v};
                            Real d2 = 0.0;
                            { Real dxp0=p[0]-cc[0], dxp1=p[1]-cc[1], dxp2=p[2]-cc[2];
                              d2 = dxp0*dxp0 + dxp1*dxp1 + dxp2*dxp2; }
                            Real wt = std::exp(-d2 / sig2);
                            for (int a=0;a<3;++a){
                                rq[a] += wt*g[a]*w;
                                for (int b=0;b<3;++b) M[a][b] += wt*g[a]*g[b];
                            }
                            for (int a6 = 0; a6 < 6; ++a6) {
                                M6[a6][6] += wt*g6[a6]*w;
                                for (int b6 = 0; b6 < 6; ++b6)
                                    M6[a6][b6] += wt*g6[a6]*g6[b6];
                            }
                            ++nq;
                        }
                        bool solved6 = false;
                        if (use_quadric_offset && nq >= 10) {
                            // Gaussian elimination with partial pivoting
                            solved6 = true;
                            for (int c6 = 0; c6 < 6 && solved6; ++c6) {
                                int piv = c6;
                                for (int r6 = c6+1; r6 < 6; ++r6)
                                    if (std::abs(M6[r6][c6]) > std::abs(M6[piv][c6])) piv = r6;
                                if (std::abs(M6[piv][c6]) < 1e-12) { solved6 = false; break; }
                                if (piv != c6)
                                    for (int b6 = 0; b6 < 7; ++b6) {
                                        Real tmp = M6[c6][b6]; M6[c6][b6] = M6[piv][b6]; M6[piv][b6] = tmp;
                                    }
                                for (int r6 = c6+1; r6 < 6; ++r6) {
                                    Real f6 = M6[r6][c6] / M6[c6][c6];
                                    for (int b6 = c6; b6 < 7; ++b6) M6[r6][b6] -= f6*M6[c6][b6];
                                }
                            }
                            if (solved6) {
                                Real sol[6];
                                for (int r6 = 5; r6 >= 0; --r6) {
                                    Real acc = M6[r6][6];
                                    for (int b6 = r6+1; b6 < 6; ++b6) acc -= M6[r6][b6]*sol[b6];
                                    sol[r6] = acc / M6[r6][r6];
                                }
                                c0_nd = sol[0]; c1_nd = sol[1]; c2_nd = sol[2];
                                a_nd = sol[3]; b_nd = sol[4]; c_nd = sol[5];
                            }
                        }
                        if (!solved6 && nq >= 6) {
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
#if (AMREX_SPACEDIM == 2)
                    const int n_newton = 2;
#else
                    const int n_newton = 4;
#endif
                    for (int it = 0; it < n_newton; ++it) {
                        Real wq = c0_nd + c1_nd*us + c2_nd*vs
                                + 0.5*a_nd*us*us + b_nd*us*vs + 0.5*c_nd*vs*vs;
                        Real wu = c1_nd + a_nd*us + b_nd*vs;   // dw/du
                        Real wv = c2_nd + b_nd*us + c_nd*vs;   // dw/dv
                        // minimize |(us-uc, vs-vc, wq-wc)|^2 on the surface
                        Real gu  = (us - uc) + (wq - wc)*wu;
                        Real guu = 1.0 + wu*wu + (wq - wc)*a_nd;
#if (AMREX_SPACEDIM == 2)
                        amrex::ignore_unused(wv);
                        if (std::abs(guu) > 1e-20) us -= gu/guu;
#else
                        Real gv  = (vs - vc) + (wq - wc)*wv;
                        Real gvv = 1.0 + wv*wv + (wq - wc)*c_nd;
                        Real guv = wu*wv + (wq - wc)*b_nd;
                        Real deth = guu*gvv - guv*guv;
                        if (std::abs(deth) > 1e-20) {
                            us -= ( gvv*gu - guv*gv)/deth;
                            vs -= (-guv*gu + guu*gv)/deth;
                        }
#endif
                    }

                    Real ws = c0_nd + c1_nd*us + c2_nd*vs
                            + 0.5*a_nd*us*us + b_nd*us*vs + 0.5*c_nd*vs*vs;
                    Real wu = c1_nd + a_nd*us + b_nd*vs;
                    Real wv = c2_nd + b_nd*us + c_nd*vs;

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
    else if (method == 5)
    {
    // Method 5 -   EB Copy
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
                    // in 2D the k index has a single valid value
#if (AMREX_SPACEDIM == 3)
                    const int nk_lo = -1, nk_hi = 1;
#else
                    const int nk_lo = 0, nk_hi = 0;
#endif
                    for (int nk = nk_lo; nk <= nk_hi; ++nk)
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
    cp_mf.FillBoundary(gm.periodicity());
    nrm_mf.FillBoundary(gm.periodicity());
    }
    // Final initialisation: mask and geometry into the state
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
