#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace user_transport {
AMREX_GPU_DEVICE AMREX_INLINE amrex::Real current_collector_value(
    IntVect cellid,
    int locid,
    int surfaceloc,
    GpuArray<Real, AMREX_SPACEDIM> problo,
    GpuArray<Real, AMREX_SPACEDIM> probhi,
    GpuArray<int, AMREX_SPACEDIM> domlo,
    GpuArray<int, AMREX_SPACEDIM> domhi,
    GpuArray<Real, AMREX_SPACEDIM> dx)
{
    return (0.0);
}

AMREX_GPU_DEVICE AMREX_INLINE void get_photoion_acoeff(
    int i,
    int j,
    int k,
    int sph_id,
    Array4<Real> const& sb_arr,
    Array4<Real> const& acoeff,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{}

AMREX_GPU_DEVICE AMREX_INLINE void get_photoion_rhs(
    int i,
    int j,
    int k,
    int sph_id,
    Array4<Real> const& sb_arr,
    Array4<Real> const& rhs_arr,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{}



// ---------------------------------------------------------------------------
// Wall boundary condition data at a point xb on the immersed boundary,
// written with the OUTWARD (fluid -> solid) normal:
//      a * phi + b * dphi/dn_out = f
// ---------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_INLINE void ib_wall_bc(
    const Real* xb, ProbParm const& prob_parm, Real& a, Real& b, Real& f)
{
    const Real rb = std::sqrt(xb[0] * xb[0] + xb[1] * xb[1]);
    const Real Rmid = 0.5_rt * (prob_parm.inner_radius + prob_parm.outer_radius);
    // exact solution: dphi/dr = cgrad / r
    const Real cgrad = (prob_parm.outer_voltage - prob_parm.inner_voltage) /
                       std::log(prob_parm.outer_radius / prob_parm.inner_radius);
    if (rb > Rmid)
    {
        a = prob_parm.a_outer_dir;
        b = prob_parm.b_outer_neumann;
        f = a * prob_parm.outer_voltage + b * cgrad / prob_parm.outer_radius;
    } else
    {
        a = prob_parm.a_inner_dir;
        b = prob_parm.b_inner_neumann;
        f = a * prob_parm.inner_voltage - b * cgrad / prob_parm.inner_radius;
    }
}

// Wall foot and outward unit normal stored for a regular fluid cell.
// Returns false when the cell carries no wall information.
AMREX_GPU_DEVICE AMREX_INLINE bool ib_wall_foot(
    IntVect cell,
    Array4<Real> const& sb_arr,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    Real* xb,
    Real* nout,
    Real& dist)
{
    Real xc[AMREX_SPACEDIM], n[AMREX_SPACEDIM], seed[AMREX_SPACEDIM];
    Real nmag = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        xc[d] = prob_lo[d] + (cell[d] + 0.5_rt) * dx[d];
        n[d] = sb_arr(cell, NX_ID + d);
        seed[d] = sb_arr(cell, CPX_ID + d);
        nmag += n[d] * n[d];
    }
    nmag = std::sqrt(nmag);
    if (nmag < 1.0e-14_rt) return false;

    // project the cell center onto the tangent plane through the seed
    Real distn = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        n[d] /= nmag;
        distn += (xc[d] - seed[d]) * n[d];
    }
    // orient the normal from the fluid cell towards the wall
    const Real orient = (distn > 0.0_rt) ? -1.0_rt : 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        xb[d] = xc[d] - distn * n[d];
        nout[d] = orient * n[d];
    }
    dist = amrex::Math::abs(distn);
    return (dist > 1.0e-12_rt);
}

// ---------------------------------------------------------------------------
// Second-order gradient at an IB face center.
//
// A quadratic polynomial is fitted (weighted least squares) around the face
// center xf using
//   * the (lagged) cell-centered values of all regular fluid cells in a
//     (2*RAD+1)^D block around the interior cell, and
//   * one boundary-condition row  a*p + b*dp/dn_out = f  at the wall foot of
//     every such cell that carries wall information.
// The BC rows bring the wall position, normal AND curvature into the fit, so
// dp/dx_dir(xf) is an O(h^2) approximation of the face-normal gradient for
// Dirichlet, Neumann and Robin walls alike.
// ---------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_INLINE bool ib_lsq_face_gradient(
    IntVect intcell,
    const Real* xf,
    int dir,
    int solved_comp,
    Array4<Real> const& sb_arr,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    ProbParm const& prob_parm,
    Real& grad)
{
#if (AMREX_SPACEDIM == 2)
    constexpr int NC = 6; // 1 X Y X2 Y2 XY
#else
    constexpr int NC = 10; // 1 X Y Z X2 Y2 Z2 XY XZ YZ
#endif
    constexpr int RAD = 2;
    constexpr Real sig2 = 2.25_rt; // Gaussian weight width^2 (units of h^2)
    constexpr Real wbc = 2.0_rt;   // extra weight of the BC rows

    const Real h = dx[0];
    const Real inv_h = 1.0_rt / h;

    Real M[NC][NC];
    Real r[NC];
    for (int p = 0; p < NC; ++p)
    {
        r[p] = 0.0_rt;
        for (int q = 0; q < NC; ++q) M[p][q] = 0.0_rt;
    }

    auto basis = [] AMREX_GPU_DEVICE(const Real* X, Real* v) {
        int n = 0;
        v[n++] = 1.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) v[n++] = X[d];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) v[n++] = X[d] * X[d];
        v[n++] = X[0] * X[1];
#if (AMREX_SPACEDIM == 3)
        v[n++] = X[0] * X[2];
        v[n++] = X[1] * X[2];
#endif
    };
    // derivative of the basis along the unit vector nv
    auto dbasis = [] AMREX_GPU_DEVICE(const Real* X, const Real* nv, Real* v) {
        int n = 0;
        v[n++] = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) v[n++] = nv[d];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) v[n++] = 2.0_rt * X[d] * nv[d];
        v[n++] = X[1] * nv[0] + X[0] * nv[1];
#if (AMREX_SPACEDIM == 3)
        v[n++] = X[2] * nv[0] + X[0] * nv[2];
        v[n++] = X[2] * nv[1] + X[1] * nv[2];
#endif
    };
    auto add_row = [&](const Real* row, Real val, Real w) {
        for (int p = 0; p < NC; ++p)
        {
            r[p] += w * row[p] * val;
            for (int q = 0; q < NC; ++q) M[p][q] += w * row[p] * row[q];
        }
    };

    int ncells = 0;
    int nbc = 0;
#if (AMREX_SPACEDIM == 3)
    for (int nk = -RAD; nk <= RAD; ++nk)
#endif
        for (int nj = -RAD; nj <= RAD; ++nj)
            for (int ni = -RAD; ni <= RAD; ++ni)
            {
                IntVect cell = intcell;
                cell[0] += ni;
                cell[1] += nj;
#if (AMREX_SPACEDIM == 3)
                cell[2] += nk;
#endif
                if (sb_arr(cell, CMASK_ID) < 1.0_rt) continue;

                Real X[AMREX_SPACEDIM];
                Real rho2 = 0.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    X[d] = (prob_lo[d] + (cell[d] + 0.5_rt) * dx[d] - xf[d]) *
                           inv_h;
                    rho2 += X[d] * X[d];
                }
                Real row[NC];
                basis(X, row);
                add_row(row, sb_arr(cell, solved_comp), std::exp(-rho2 / sig2));
                ++ncells;

                // boundary-condition row at this cell's wall foot
                Real xb[AMREX_SPACEDIM], nout[AMREX_SPACEDIM], dist;
                if (!ib_wall_foot(cell, sb_arr, prob_lo, dx, xb, nout, dist))
                    continue;

                Real a, b, f;
                ib_wall_bc(xb, prob_parm, a, b, f);

                Real Xb[AMREX_SPACEDIM];
                Real rhob2 = 0.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    Xb[d] = (xb[d] - xf[d]) * inv_h;
                    rhob2 += Xb[d] * Xb[d];
                }
                Real vrow[NC], drow[NC];
                basis(Xb, vrow);
                dbasis(Xb, nout, drow);
                // scale so the row has O(1) entries for any a, b, h
                const Real scale =
                    1.0_rt / (amrex::Math::abs(a) + amrex::Math::abs(b) * inv_h);
                for (int p = 0; p < NC; ++p)
                    row[p] = scale * (a * vrow[p] + b * inv_h * drow[p]);
                add_row(row, scale * f, wbc * std::exp(-rhob2 / sig2));
                ++nbc;
            }

    if (ncells + nbc < NC + 2 || nbc == 0) return false;

    // Gaussian elimination with partial pivoting
    for (int c = 0; c < NC; ++c)
    {
        int piv = c;
        for (int p = c + 1; p < NC; ++p)
            if (amrex::Math::abs(M[p][c]) > amrex::Math::abs(M[piv][c])) piv = p;
        if (amrex::Math::abs(M[piv][c]) < 1.0e-12_rt) return false;
        if (piv != c)
        {
            for (int q = 0; q < NC; ++q)
            {
                Real tmp = M[c][q];
                M[c][q] = M[piv][q];
                M[piv][q] = tmp;
            }
            Real tmp = r[c];
            r[c] = r[piv];
            r[piv] = tmp;
        }
        for (int p = c + 1; p < NC; ++p)
        {
            const Real fac = M[p][c] / M[c][c];
            for (int q = c; q < NC; ++q) M[p][q] -= fac * M[c][q];
            r[p] -= fac * r[c];
        }
    }
    Real coef[NC];
    for (int c = NC - 1; c >= 0; --c)
    {
        Real sum = r[c];
        for (int q = c + 1; q < NC; ++q) sum -= M[c][q] * coef[q];
        coef[c] = sum / M[c][c];
    }

    grad = coef[1 + dir] * inv_h; // d/dx_dir at the face center (X = 0)
    return true;
}

AMREX_GPU_DEVICE AMREX_INLINE void bc_ib(
    IntVect face,
    int dir,
    int sgn,
    int solved_comp,
    int rhs_comp,
    Array4<Real> const& sb_arr,
    amrex::Real ascalar,
    amrex::Real bscalar,
    Array4<Real> const& acoeff,
    Array4<Real> const& bcoeff,
    Array4<Real> const& rhs,
    GpuArray<int, AMREX_SPACEDIM> domlo,
    GpuArray<int, AMREX_SPACEDIM> domhi,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    Real Tg,
    Real Pg)
{
    IntVect lcell = face;
    IntVect rcell = face;
    lcell[dir] -= 1;

    Real vf_l = sb_arr(lcell, CMASK_ID);
    Real vf_r = sb_arr(rcell, CMASK_ID);

    // Interior (solve) cell: the fluid side (higher volume fraction).
    const bool fluid_is_left = (vf_l >= vf_r);
    IntVect intcell = fluid_is_left ? lcell : rcell;
    // direction (along dir) pointing from the fluid cell into the body
    const Real sout = fluid_is_left ? 1.0_rt : -1.0_rt;

    // wall foot, outward unit normal and normal distance of the fluid cell
    Real xb[AMREX_SPACEDIM], nout[AMREX_SPACEDIM], dist;
    if (!ib_wall_foot(intcell, sb_arr, prob_lo, dx, xb, nout, dist)) return;

    const Real ndir = amrex::Math::abs(nout[dir]);
    if (ndir < 1.0e-14_rt) return;

    Real a, b, f;
    ib_wall_bc(xb, prob_parm, a, b, f);

    constexpr Real eps = 1.0e-14_rt;
    const Real B = bscalar * bcoeff(intcell, 0.0);

    // ---------------------------------------------------------------------
    // Pure Dirichlet:  a * phi = f
    // one-sided difference along the wall normal, projected on dir.
    // Already second-order accurate for the solution.
    // ---------------------------------------------------------------------
    if (amrex::Math::abs(b) < eps)
    {
        if (amrex::Math::abs(a) < eps) return;
        const Real coeff = ndir / (dist * dx[dir]);
        acoeff(intcell) += (B / ascalar) * coeff;
        rhs(intcell, rhs_comp) += B * coeff * (f / a);
        return;
    }

    // ---------------------------------------------------------------------
    // Neumann / Robin:  a * phi + b * dphi/dn_out = f
    //
    // G = outward gradient through this face (d phi / d x_dir * sout).
    //
    // Implicit, first-order part (wall-normal one-sided closure):
    //      G1(phi) = ndir * (f - a*phi_P) / (a*dist + b)
    //
    // Second-order part: G2 = least-squares face-center gradient using the
    // lagged potential. Applied as a deferred correction on the rhs,
    //      G = G1(phi^{n+1}) + relax * [ G2(phi^n) - G1(phi^n) ],
    // which converges to the second-order closure G = G2(phi).
    // ---------------------------------------------------------------------
    const Real denom = a * dist + b;
    if (amrex::Math::abs(denom) < eps) return;
    const Real c1 = ndir / denom;

    acoeff(intcell) += (B / ascalar) * c1 * a / dx[dir];
    rhs(intcell, rhs_comp) += B * c1 * f / dx[dir];

    // (only the potential is corrected; at the first step there is no lagged
    //  solution yet, so that solve is the plain first-order one)
    if (prob_parm.ib_second_order && solved_comp == POT_ID && time > 0.0_rt)
    {
        Real xf[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            xf[d] = prob_lo[d] + (intcell[d] + 0.5_rt) * dx[d];
        xf[dir] += 0.5_rt * sout * dx[dir];

        Real grad = 0.0_rt;
        if (ib_lsq_face_gradient(
                intcell, xf, dir, solved_comp, sb_arr, prob_lo, dx, prob_parm,
                grad))
        {
            const Real G2 = sout * grad;
            const Real G1 = c1 * (f - a * sb_arr(intcell, solved_comp));
            rhs(intcell, rhs_comp) +=
                B * prob_parm.ib_relax * (G2 - G1) / dx[dir];
        }
    }
}


AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
int is_dielectric(
    int i,
    int j,
    int k,
    int dir,
    int sign,
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    const GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    const GpuArray<Real, AMREX_SPACEDIM> dx,
    Real time,
    ProbParm const& prob_parm)
{
    return (0);
}

AMREX_GPU_DEVICE AMREX_INLINE void potential_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    Array4<Real> const& phi,
    Array4<Real> const& bc_arr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg,
    amrex::Real app_voltage,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)
{}

AMREX_GPU_DEVICE AMREX_INLINE void photoionization_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    Array4<Real> const& phi,
    Array4<Real> const& bc_arr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg)
{}

AMREX_GPU_DEVICE AMREX_INLINE void species_bc(
    int i,
    int j,
    int k,
    int dir,
    int sgn,
    int spec_id,
    int bcspec_id,
    Array4<Real> const& phi,
    Array4<Real> const& bc_arr,
    Array4<Real> const& robin_a,
    Array4<Real> const& robin_b,
    Array4<Real> const& robin_f,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg,
    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)

{}

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real compute_vel(
    IntVect iv,
    int dir,
    int specid,
    Array4<Real> const& phi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm,
    amrex::Real Tg,
    amrex::Real Pg)
{
    amrex::Real efieldmag = std::sqrt(
        amrex::Math::powi<2>(phi(iv, EFX_ID)) +
        amrex::Math::powi<2>(phi(iv, EFY_ID)) +
        amrex::Math::powi<2>(phi(iv, EFZ_ID)));

    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += phi(iv, sp);

    amrex::Real mu = specMob(specid, phi(iv, ETEMP_ID), ndens, efieldmag, Tg);

    return (mu * phi(iv, EFX_ID + dir));
}
} // namespace user_transport
#endif
