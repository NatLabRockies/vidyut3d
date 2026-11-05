#ifndef _PINGEOM_H_
#define _PINGEOM_H_

// Pin-array electrode geometry for AMReX EB2.
//
// Two independent arrays of paraboloidal pins: the bottom array (powered,
// pins pointing up from the low axial face) and the top array (grounded, pins
// pointing down from the high axial face). The pin axis is prob.pin_axis
// (0 x, 1 y, 2 z; default: the last coordinate, y in 2D and z in 3D); the
// transverse coordinates are the remaining ones in increasing order, which is
// also the order of the centre coordinates in the inputs.
// With
//     l     = length of the domain along the axis
//     d     = prob.dgap    (tip-to-tip gap, common to both arrays)
//     r_i   = radius of pin i at the domain face
//     c_i   = transverse centre of pin i
// the surfaces are
//     bottom pin i:  ax = c1_i rho_i^2 + c2_b,  c2_b = lo + (l-d)/2, c1_i = (lo - c2_b)/r_i^2
//     top pin i:     ax = c1_i rho_i^2 + c2_t,  c2_t = c2_b + d,     c1_i = (hi - c2_t)/r_i^2
// with rho_i the transverse distance from c_i. The default (one pin per array
// on the axis, radius prob.pinrad) is the former single-pin case.
//
// Inputs (all under prob.):
//     nbot, bot_centers (nbot x (DIM-1) values), bot_pinrad (1 or nbot values)
//     ntop, top_centers, top_pinrad     (same)
// In a 2D axisymmetric (RZ) run an off-axis "pin" is a ring.
//
// The implicit function follows the AMReX convention: > 0 inside the body
// (electrode), < 0 in the fluid, = 0 on the surface; the union of all pins is
// the maximum. EB2 evaluates it once at initialisation; nothing in this header
// is called during the time loop.

#include <AMReX_REAL.H>
#include <AMReX_Array.H>
#include <AMReX_EB2_IF_Base.H>
#include <ProbParm.H>

namespace pin
{
constexpr int NT = AMREX_SPACEDIM - 1; // number of transverse coordinates

// Fixed-size description of both arrays (copied into the implicit function
// and into device lambdas).
struct Pins
{
    int ax, t0, t1;                              // axis and transverse coordinate indices
    int nbot, ntop;
    amrex::Real c2b, c2t;                        // tip planes
    amrex::Real bc[MAX_PINS][2], bc1[MAX_PINS];  // bottom: centres, c1
    amrex::Real tc[MAX_PINS][2], tc1[MAX_PINS];  // top: centres, c1
    amrex::Real rmin;                            // smallest pin radius
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Pins make_pins(
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi,
    ProbParm const& pp)
{
    Pins p;
    p.ax = pp.pin_axis;
    p.t0 = (p.ax == 0) ? 1 : 0;
    p.t1 = (p.ax == 2) ? 1 : 2;
    const int AX = p.ax;
    const amrex::Real l = prob_hi[AX] - prob_lo[AX];
    p.c2b = prob_lo[AX] + 0.5 * (l - pp.dgap);
    p.c2t = p.c2b + pp.dgap;
    p.nbot = pp.nbot;
    p.ntop = pp.ntop;
    p.rmin = 1.0e300;
    for (int i = 0; i < MAX_PINS; ++i)
    {
        p.bc[i][0] = pp.bot_c[i][0];
        p.bc[i][1] = pp.bot_c[i][1];
        p.tc[i][0] = pp.top_c[i][0];
        p.tc[i][1] = pp.top_c[i][1];
        p.bc1[i] = (prob_lo[AX] - p.c2b) / (pp.bot_r[i] * pp.bot_r[i]);
        p.tc1[i] = (prob_hi[AX] - p.c2t) / (pp.top_r[i] * pp.top_r[i]);
        if (i < pp.nbot) p.rmin = amrex::min(p.rmin, pp.bot_r[i]);
        if (i < pp.ntop) p.rmin = amrex::min(p.rmin, pp.top_r[i]);
    }
    return p;
}

// squared transverse distance of the point q from centre c
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real rho2_from(
    const amrex::Real q[3], const Pins& p, const amrex::Real c[2])
{
#if (AMREX_SPACEDIM == 2)
    return (q[p.t0] - c[0]) * (q[p.t0] - c[0]);
#else
    return (q[p.t0] - c[0]) * (q[p.t0] - c[0]) + (q[p.t1] - c[1]) * (q[p.t1] - c[1]);
#endif
}

// Implicit function handed to EB2::makeShop. GPUable so that EB2 uses its
// ParallelFor path (works on CPU builds as well).
class PinIF : public amrex::GPUable
{
public:
    PinIF(const Pins& p) : m_p(p) {}

    [[nodiscard]] AMREX_GPU_HOST_DEVICE inline amrex::Real
    operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const noexcept
    {
        const amrex::Real q[3] = {x, y, AMREX_D_PICK(0.0, 0.0, z)};
        const amrex::Real ax = q[m_p.ax];
        amrex::Real f = -1.0e300;
        for (int i = 0; i < m_p.nbot; ++i)
        {
            const amrex::Real fb = (m_p.bc1[i] * rho2_from(q, m_p, m_p.bc[i]) + m_p.c2b) - ax;
            f = amrex::max(f, fb);
        }
        for (int i = 0; i < m_p.ntop; ++i)
        {
            const amrex::Real ft = ax - (m_p.tc1[i] * rho2_from(q, m_p, m_p.tc[i]) + m_p.c2t);
            f = amrex::max(f, ft);
        }
        return f;
    }

    [[nodiscard]] inline amrex::Real operator()(const amrex::RealArray& p) const noexcept
    {
        return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
    }

private:
    Pins m_p;
};

// Foot on the curve ax = c1 rho^2 + c2 from the point (rho0, ax0): the
// stationary condition of the squared distance is the cubic
//     g(rho) = 2 c1^2 rho^3 + (1 + 2 c1 (c2 - ax0)) rho - rho0 = 0.
// Fluid points lie on the convex side of the paraboloid, so g has a single
// real root; it is bracketed (g(0) = -rho0 <= 0) and found by bisection.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real foot_rho(
    amrex::Real c1, amrex::Real c2, amrex::Real rho0, amrex::Real ax0, amrex::Real rscale)
{
    const amrex::Real p = 1.0 + 2.0 * c1 * (c2 - ax0);
    auto g = [&](amrex::Real rho) {
        return 2.0 * c1 * c1 * rho * rho * rho + p * rho - rho0;
    };
    amrex::Real lo = 0.0;
    amrex::Real hi = amrex::max(rho0, rscale);
    for (int it = 0; it < 60 && g(hi) < 0.0; ++it) hi *= 2.0;
    for (int it = 0; it < 80; ++it)
    {
        const amrex::Real mid = 0.5 * (lo + hi);
        if (g(mid) < 0.0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Analytical foot of the normal and unit normal (pointing from the fluid into
// the electrode, the AMReX EB convention) for a point x in the fluid: the
// nearest point over all pins of both arrays. Used only by the analytical
// reconstruction (prob.eb_geom_method = 3) at initialisation and as a
// reference for the EB-based reconstruction. Where two pins overlap the
// nearest point of an individual paraboloid can lie inside the other pin;
// the EB-based methods do not have this limitation.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void closest_point(
    const amrex::Real x[AMREX_SPACEDIM],
    const Pins& p,
    amrex::Real xb[AMREX_SPACEDIM],
    amrex::Real n[AMREX_SPACEDIM])
{
    const int AX = p.ax;
    const amrex::Real q[3] = {x[0], x[1], AMREX_D_PICK(0.0, 0.0, x[2])};
    const amrex::Real ax0 = q[AX];
    amrex::Real best_d2 = 1.0e300;
    for (int arr = 0; arr < 2; ++arr)
    {
        const int np = (arr == 0) ? p.nbot : p.ntop;
        const amrex::Real c2 = (arr == 0) ? p.c2b : p.c2t;
        const amrex::Real s = (arr == 0) ? 1.0 : -1.0;
        for (int i = 0; i < np; ++i)
        {
            const amrex::Real* c = (arr == 0) ? p.bc[i] : p.tc[i];
            const amrex::Real c1 = (arr == 0) ? p.bc1[i] : p.tc1[i];
            const amrex::Real rho0 = std::sqrt(rho2_from(q, p, c));
            // transverse unit vector from the pin centre
            amrex::Real ex = 1.0, ey = 0.0;
            if (rho0 > 0.0)
            {
                ex = (q[p.t0] - c[0]) / rho0;
#if (AMREX_SPACEDIM == 3)
                ey = (q[p.t1] - c[1]) / rho0;
#endif
            }
            const amrex::Real rs = foot_rho(c1, c2, rho0, ax0, p.rmin);
            const amrex::Real as = c1 * rs * rs + c2;
            const amrex::Real d2 = (rs - rho0) * (rs - rho0) + (as - ax0) * (as - ax0);
            if (d2 < best_d2)
            {
                best_d2 = d2;
                // gradient of the implicit function (points into the electrode):
                // bottom (rho, ax) -> (2 c1 rho, -1), top -> (-2 c1 rho, +1)
                amrex::Real nr = s * 2.0 * c1 * rs;
                amrex::Real na = -s;
                const amrex::Real nm = std::sqrt(nr * nr + na * na);
                nr /= nm;
                na /= nm;
                xb[p.t0] = c[0] + ex * rs;
                n[p.t0] = ex * nr;
#if (AMREX_SPACEDIM == 3)
                xb[p.t1] = c[1] + ey * rs;
                n[p.t1] = ey * nr;
#endif
                xb[AX] = as;
                n[AX] = na;
            }
        }
    }
}
} // namespace pin
#endif
