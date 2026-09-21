#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_FArrayBox.H>
#include <ProbParm.H>
#include <Chemistry.H>
#include <VarDefines.H>
#include <UnivConstants.H>
#include <UserFunctions.H>
#include <HelperFuncs.H>
#include <compute_explicit_flux.H>
#include <UnitTest.H>

using namespace amrex;

namespace {
const Real tight_tol = 1.0e-11;
const int ncells = 8;

// f = a + b s + c s^2, s is the coordinate along a direction
const Real qa = 3.0;
const Real qb = -2.0;
const Real qc = 5.0;

Real quadratic(Real s) { return qa + qb * s + qc * s * s; }
Real quadratic_deriv(Real s) { return qb + 2.0 * qc * s; }

GpuArray<Real, AMREX_SPACEDIM> cell_sizes()
{
    return {AMREX_D_DECL(0.1, 0.05, 0.025)};
}

Box test_box() { return Box(IntVect::TheZeroVector(), IntVect(ncells - 1)); }

// fills component comp with the quadratic (or a linear function when
// linear_only is set) of the cell-center coordinate along dir
void fill_along_dir(
    FArrayBox& fab, int comp, int dir, Real dxdir, bool linear_only = false)
{
    auto arr = fab.array();
    const Box& bx = fab.box();
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
        IntVect iv{AMREX_D_DECL(i, j, k)};
        Real s = (iv[dir] + 0.5) * dxdir;
        arr(iv, comp) = linear_only ? (qa + qb * s) : quadratic(s);
    });
}
} // namespace

//=========================================================================
// HelperFuncs.H
//=========================================================================
UT_TEST(onesided_grad_exact_for_quadratic)
{
    auto dx = cell_sizes();
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        FArrayBox fab(test_box(), NVAR, The_Managed_Arena());
        fab.setVal<RunOn::Host>(0.0);
        fill_along_dir(fab, POT_ID, dir, dx[dir]);
        Array4<Real> arr = fab.array();

        // wall on the high side, outward normal +1, face index ncells
        IntVect face_hi(1);
        face_hi[dir] = ncells;
        UT_CHECK_CLOSE(
            get_onesided_grad(face_hi, 1, dir, POT_ID, dx, arr),
            quadratic_deriv(ncells * dx[dir]), tight_tol);

        // wall on the low side, outward normal -1, face index 0
        IntVect face_lo(1);
        face_lo[dir] = 0;
        UT_CHECK_CLOSE(
            get_onesided_grad(face_lo, -1, dir, POT_ID, dx, arr),
            quadratic_deriv(0.0), tight_tol);

        // a face in the middle of the box, from both sides
        IntVect face_mid(1);
        face_mid[dir] = 4;
        UT_CHECK_CLOSE(
            get_onesided_grad(face_mid, 1, dir, POT_ID, dx, arr),
            quadratic_deriv(4 * dx[dir]), tight_tol);
        UT_CHECK_CLOSE(
            get_onesided_grad(face_mid, -1, dir, POT_ID, dx, arr),
            quadratic_deriv(4 * dx[dir]), tight_tol);
    }
}

UT_TEST(efield_alongdir_exact_for_quadratic)
{
    auto dx = cell_sizes();
    GpuArray<int, AMREX_SPACEDIM> domlo = {AMREX_D_DECL(0, 0, 0)};
    GpuArray<int, AMREX_SPACEDIM> domhi = {
        AMREX_D_DECL(ncells - 1, ncells - 1, ncells - 1)};

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        FArrayBox fab(test_box(), NVAR, The_Managed_Arena());
        fab.setVal<RunOn::Host>(0.0);
        fill_along_dir(fab, POT_ID, dir, dx[dir]);
        Array4<Real> arr = fab.array();

        // low boundary cell, interior cell, high boundary cell
        const int ids[3] = {0, 3, ncells - 1};
        for (int id : ids)
        {
            IntVect iv(1);
            iv[dir] = id;
            Real efield = get_efield_alongdir(
                AMREX_D_DECL(iv[0], iv[1], iv[2]),
#if AMREX_SPACEDIM < 3
                0,
#if AMREX_SPACEDIM < 2
                0,
#endif
#endif
                dir, domlo, domhi, dx, arr);
            UT_CHECK_CLOSE(
                efield, -quadratic_deriv((id + 0.5) * dx[dir]), tight_tol);
        }
    }
}

UT_TEST(efield_alongdir_thin_domain_exact_for_linear)
{
    // with three or fewer cells along dir the boundary cells fall back to a
    // two-point difference
    auto dx = cell_sizes();
    GpuArray<int, AMREX_SPACEDIM> domlo = {AMREX_D_DECL(0, 0, 0)};
    GpuArray<int, AMREX_SPACEDIM> domhi = {AMREX_D_DECL(2, 2, 2)};
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        FArrayBox fab(
            Box(IntVect::TheZeroVector(), IntVect(2)), NVAR,
            The_Managed_Arena());
        fab.setVal<RunOn::Host>(0.0);
        fill_along_dir(fab, POT_ID, dir, dx[dir], true);
        Array4<Real> arr = fab.array();
        for (int id = 0; id < 3; id++)
        {
            IntVect iv(1);
            iv[dir] = id;
            Real efield = get_efield_alongdir(
                AMREX_D_DECL(iv[0], iv[1], iv[2]),
#if AMREX_SPACEDIM < 3
                0,
#if AMREX_SPACEDIM < 2
                0,
#endif
#endif
                dir, domlo, domhi, dx, arr);
            UT_CHECK_CLOSE(efield, -qb, tight_tol);
        }
    }
}

UT_TEST(applied_potential_profiles)
{
    const Real vlo = 100.0;
    const Real vhi = -40.0;
    const Real freq = 13.56e6;
    const Real vdur = 2.0e-9;
    const Real vcen = 5.0e-9;

    // constant, low and high side
    UT_CHECK_CLOSE(
        get_applied_potential(1.0e-9, -1, 0, vlo, vhi, freq, vdur, vcen), vlo,
        tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(1.0e-9, 1, 0, vlo, vhi, freq, vdur, vcen), vhi,
        tight_tol);

    // sine
    UT_CHECK_CLOSE(
        get_applied_potential(0.0, -1, 1, vlo, vhi, freq, vdur, vcen), 0.0,
        tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(0.25 / freq, -1, 1, vlo, vhi, freq, vdur, vcen),
        vlo, tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(0.75 / freq, 1, 1, vlo, vhi, freq, vdur, vcen),
        -vhi, tight_tol);

    // triangular pulse
    UT_CHECK_CLOSE(
        get_applied_potential(vcen, -1, 2, vlo, vhi, freq, vdur, vcen), vlo,
        tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(
            vcen - 0.25 * vdur, -1, 2, vlo, vhi, freq, vdur, vcen),
        0.5 * vlo, tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(
            vcen + 0.25 * vdur, -1, 2, vlo, vhi, freq, vdur, vcen),
        0.5 * vlo, tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(vcen - vdur, -1, 2, vlo, vhi, freq, vdur, vcen),
        0.0, tight_tol);
    UT_CHECK_CLOSE(
        get_applied_potential(vcen + vdur, -1, 2, vlo, vhi, freq, vdur, vcen),
        0.0, tight_tol);
}

UT_TEST(minmod_limiter_values)
{
    auto vals =
        unittest::device_eval(5, [=] AMREX_GPU_DEVICE(int n, Real* out) {
            if (n == 0) out[n] = minmod_limiter(0.5, 1.0);  // 0 < r < 1
            if (n == 1) out[n] = minmod_limiter(3.0, 1.0);  // r > 1
            if (n == 2) out[n] = minmod_limiter(-1.0, 1.0); // extremum
            if (n == 3) out[n] = minmod_limiter(1.0, 0.0);  // flat on the right
            if (n == 4) out[n] = minmod_limiter(0.0, 1.0);  // flat on the left
        });
    UT_CHECK_CLOSE(vals[0], 0.5, tight_tol);
    UT_CHECK_CLOSE(vals[1], 1.0, tight_tol);
    UT_CHECK_CLOSE(vals[2], 0.0, tight_tol);
    UT_CHECK_CLOSE(vals[3], 1.0, tight_tol);
    UT_CHECK_CLOSE(vals[4], 0.0, tight_tol);
}

UT_TEST(gradlimiter_is_one_for_linear_data)
{
    auto dx = cell_sizes();
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        FArrayBox fab(test_box(), NVAR, The_Managed_Arena());
        fab.setVal<RunOn::Host>(0.0);
        fill_along_dir(fab, 0, dir, dx[dir], true);
        Array4<Real> arr = fab.array();
        IntVect iv(2);
        iv[dir] = 3;
        auto vals =
            unittest::device_eval(1, [=] AMREX_GPU_DEVICE(int n, Real* out) {
                out[n] = get_gradlimiter(
                    AMREX_D_DECL(iv[0], iv[1], iv[2]),
#if AMREX_SPACEDIM < 3
                    0,
#if AMREX_SPACEDIM < 2
                    0,
#endif
#endif
                    0, dir, arr);
            });
        UT_CHECK_CLOSE(vals[0], 1.0, tight_tol);
    }
}

//=========================================================================
// compute_explicit_flux.H
//=========================================================================
UT_TEST(weno_constant_and_linear_data)
{
    // every candidate stencil is exact for linear data, so the result does
    // not depend on the nonlinear weights
    for (int scheme = 1; scheme <= 3; scheme++)
    {
        auto vals =
            unittest::device_eval(3, [=] AMREX_GPU_DEVICE(int n, Real* out) {
                if (n == 0) out[n] = weno(7.0, 7.0, 7.0, 7.0, 7.0, scheme);
                if (n == 1) out[n] = weno(1.0, 3.0, 5.0, 7.0, 9.0, scheme);
                if (n == 2) out[n] = weno(9.0, 7.0, 5.0, 3.0, 1.0, scheme);
            });
        UT_CHECK_CLOSE(vals[0], 7.0, tight_tol);
        UT_CHECK_CLOSE(vals[1], 6.0, tight_tol);
        UT_CHECK_CLOSE(vals[2], 4.0, tight_tol);
    }
}

UT_TEST(weno_smooth_data_fifth_order)
{
    // cell averages of sin(x), the error of the face value has to drop by
    // about 2^5 when the cell size is halved
    for (int scheme = 1; scheme <= 3; scheme++)
    {
        auto vals =
            unittest::device_eval(2, [=] AMREX_GPU_DEVICE(int n, Real* out) {
                const Real h = (n == 0) ? 0.1 : 0.05;
                const Real x0 = 0.4; // face location
                Real avg[5];
                for (int c = 0; c < 5; c++)
                {
                    // cells i-2..i+2, cell i ends at x0
                    const Real xl = x0 + (c - 3) * h;
                    const Real xr = xl + h;
                    avg[c] = (std::cos(xl) - std::cos(xr)) / h;
                }
                out[n] = std::abs(
                    weno(avg[0], avg[1], avg[2], avg[3], avg[4], scheme) -
                    std::sin(x0));
            });
        UT_CHECK(vals[0] < 1.0e-5);
        UT_CHECK(vals[0] / vals[1] > 20.0);
    }
}

UT_TEST(weno_step_data_stays_bounded)
{
    for (int scheme = 1; scheme <= 3; scheme++)
    {
        auto vals =
            unittest::device_eval(4, [=] AMREX_GPU_DEVICE(int n, Real* out) {
                if (n == 0) out[n] = weno(0.0, 0.0, 0.0, 1.0, 1.0, scheme);
                if (n == 1) out[n] = weno(0.0, 0.0, 1.0, 1.0, 1.0, scheme);
                if (n == 2) out[n] = weno(1.0, 1.0, 1.0, 0.0, 0.0, scheme);
                if (n == 3) out[n] = weno(1.0, 1.0, 0.0, 0.0, 0.0, scheme);
            });
        for (int n = 0; n < 4; n++)
        {
            UT_CHECK(vals[n] > -1.0e-3);
            UT_CHECK(vals[n] < 1.0 + 1.0e-3);
        }
    }
}

UT_TEST(weno_reconstruct_mirror_symmetry)
{
    // data symmetric about the i+1/2 face gives the same value from the
    // left and from the right
    for (int scheme = 1; scheme <= 3; scheme++)
    {
        auto vals = unittest::device_eval(
            2, [=] AMREX_GPU_DEVICE(int n, Real* out) {
                Real umhalf, uphalf;
                weno_reconstruct(
                    0.3, 1.1, 2.0, 2.0, 1.1, 0.3, umhalf, uphalf, scheme);
                out[n] = (n == 0) ? umhalf : uphalf;
            });
        UT_CHECK_CLOSE(vals[0], vals[1], tight_tol);
    }
}

UT_TEST(upwind_flux_direction)
{
    auto vals =
        unittest::device_eval(2, [=] AMREX_GPU_DEVICE(int n, Real* out) {
            if (n == 0) out[n] = get_firstorder_upwind_flux(2.0, 2.0, 3.0, 5.0);
            if (n == 1)
                out[n] = get_firstorder_upwind_flux(-2.0, -2.0, 3.0, 5.0);
        });
    UT_CHECK_CLOSE(vals[0], 6.0, tight_tol);
    UT_CHECK_CLOSE(vals[1], -10.0, tight_tol);
}

UT_TEST(waf_flux_limits)
{
    auto vals =
        unittest::device_eval(4, [=] AMREX_GPU_DEVICE(int n, Real* out) {
            const Real vel = 2.0;
            const Real dx = 0.1;
            const Real dt = 0.01; // c = 0.2
            // uniform state
            if (n == 0)
                out[n] = get_secondorder_WAF_flux(
                    vel, vel, 4.0, 4.0, 4.0, 4.0, dx, dt);
            // extremum on the upwind side, limiter = 1, upwind flux
            if (n == 1)
                out[n] = get_secondorder_WAF_flux(
                    vel, vel, 5.0, 3.0, 5.0, 7.0, dx, dt);
            // linear data, limiter = |c|, Lax-Wendroff flux
            if (n == 2)
                out[n] = get_secondorder_WAF_flux(
                    vel, vel, 1.0, 3.0, 5.0, 7.0, dx, dt);
            // same with negative velocity
            if (n == 3)
                out[n] = get_secondorder_WAF_flux(
                    -vel, -vel, 1.0, 3.0, 5.0, 7.0, dx, dt);
        });
    UT_CHECK_CLOSE(vals[0], 8.0, tight_tol);
    UT_CHECK_CLOSE(vals[1], 6.0, tight_tol);
    // 0.5*(fL+fR) - 0.5*c*(fR-fL) = 8 - 0.5*0.2*4
    UT_CHECK_CLOSE(vals[2], 7.6, tight_tol);
    // 0.5*(fL+fR) + 0.5*c*(fR-fL) = -8 + 0.5*0.2*(-4)
    UT_CHECK_CLOSE(vals[3], -8.4, tight_tol);
}

//=========================================================================
// PlasmaChem and the mechanism in Chemistry.H
//=========================================================================
UT_TEST(plasmachem_species_lookup)
{
    UT_CHECK(plasmachem::find_id("not_a_species") == -1);
    UT_CHECK(int(plasmachem::specnames.size()) == NUM_SPECIES);
    for (int sp = 0; sp < NUM_SPECIES; sp++)
    {
        UT_CHECK(plasmachem::find_id(plasmachem::specnames[sp]) == sp);
    }

    // the electron has to be found under one of the names Vidyut looks for
    int eidx = -1;
    for (const auto* name : {"E", "E-", "e", "e-"})
    {
        if (eidx == -1) eidx = plasmachem::find_id(name);
    }
    UT_CHECK(eidx == E_ID);
    UT_CHECK(plasmachem::get_charge(E_ID) == -1);

    int nions = 0;
    for (int sp = 0; sp < NUM_SPECIES; sp++)
    {
        if (sp != E_ID && plasmachem::get_charge(sp) != 0) nions++;
    }
    UT_CHECK(nions == NUM_IONS);
}

UT_TEST(mechanism_conserves_charge)
{
    Real conc[NUM_SPECIES];
    Real wdot[NUM_SPECIES];
    for (int sp = 0; sp < NUM_SPECIES; sp++)
    {
        conc[sp] = 1.0e15 * (1.0 + 0.37 * sp);
    }
    const Real tgas = 300.0;
    const Real efield_by_n = 100.0;
    for (Real te_in_ev : {0.5, 3.0, 10.0})
    {
        Real ener_exch = 0.0;
        CKWC(tgas, conc, wdot, te_in_ev * eV, efield_by_n, &ener_exch);
        Real net = 0.0;
        Real total = 0.0;
        for (int sp = 0; sp < NUM_SPECIES; sp++)
        {
            net += plasmachem::get_charge(sp) * wdot[sp];
            total += std::abs(plasmachem::get_charge(sp) * wdot[sp]);
        }
        UT_CHECK(total > 0.0);
        UT_CHECK(std::abs(net) <= 1.0e-10 * total);
    }
}

UT_TEST(transport_coefficient_signs)
{
    const Real ndens = 1.0e22;
    const Real efield_mag = 1.0e4;
    const Real tgas = 300.0;
    const Real etemp = 3.0 * eV;
    for (int sp = 0; sp < NUM_SPECIES; sp++)
    {
        const Real mu = specMob(sp, etemp, ndens, efield_mag, tgas);
        const Real dcoeff = specDiff(sp, etemp, ndens, efield_mag, tgas);
        UT_CHECK(std::isfinite(mu));
        UT_CHECK(std::isfinite(dcoeff));
        UT_CHECK(dcoeff >= 0.0);
        // drift is along q E
        UT_CHECK(mu * plasmachem::get_charge(sp) >= 0.0);
    }
    UT_CHECK(specMob(E_ID, etemp, ndens, efield_mag, tgas) < 0.0);
}

//=========================================================================
int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int nfailed = 0;
    {
        std::string filter;
        amrex::ParmParse pp("unittest");
        pp.query("filter", filter);
        plasmachem::init();
        nfailed = unittest::run_all(filter);
        plasmachem::close();
    }
    amrex::Finalize();
    return nfailed;
}
