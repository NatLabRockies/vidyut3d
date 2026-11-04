#ifndef APP_GEOM_H_
#define APP_GEOM_H_

// Planar 2D geometries for applications where local refinement near walls
// should help, used with the MMS2-type manufactured solution of this case
// (UserSources.H). All walls are Dirichlet from the exact solution; no
// dielectric or surface physics is modelled.
//
//   eb2.app_type = streamer : one needle above a grounded plane
//   eb2.app_type = pins     : up to MAXPIN needles above a plane
//   eb2.app_type = dbd      : two flat plates, thin strips on the lower one
//
// A needle is solid above y = yt + c (sqrt(a^2 + (x - xc)^2) - a), with
// c = cot(half angle) and a = c * tip radius, so the tip curvature radius is
// eb2.app_tip_r. Every case sits inside a box chamber (fluid inside), like
// the composite chamber, so no wall is a domain face.
//
// Sign convention as AMReX EB2: the function is > 0 in the solid.

#include <AMReX_EB2_IF_Base.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <string>

namespace appgeom {

using amrex::Real;

constexpr int MAXPIN = 8;
constexpr int MAXSTRIP = 8;

struct Params
{
    int type = 0; // 0 streamer, 1 pins, 2 dbd
    int npin = 0;
    Real pin_x[MAXPIN] = {}, pin_yt[MAXPIN] = {};
    Real tip_r = 0.0, cot_half = 0.0;
    Real plane_y = 0.0;                    // grounded plane (streamer, pins)
    Real plate_y0 = 0.0, plate_y1 = 0.0;   // dbd plates: solid below y0, above y1
    int nstrip = 0;
    Real strip_a[MAXSTRIP] = {}, strip_b[MAXSTRIP] = {};
    Real strip_h = 0.0;
    Real ch_x = 0.0, ch_y0 = 0.0, ch_y1 = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
boxv(Real x, Real y, Real x0, Real x1, Real y0, Real y1) noexcept
{
    return amrex::min(amrex::min(x - x0, x1 - x), amrex::min(y - y0, y1 - y));
}

class AppIF : public amrex::GPUable
{
public:
    explicit AppIF(const Params& p) : m(p) {}

    [[nodiscard]] AMREX_GPU_HOST_DEVICE inline Real
    operator()(AMREX_D_DECL(Real x, Real y, Real /*z*/)) const noexcept
    {
        Real f = -boxv(x, y, -m.ch_x, m.ch_x, m.ch_y0, m.ch_y1); // chamber wall
        if (m.type == 2)
        {
            f = amrex::max(f, m.plate_y0 - y);
            f = amrex::max(f, y - m.plate_y1);
            for (int s = 0; s < m.nstrip; ++s)
            {
                f = amrex::max(f, boxv(x, y, m.strip_a[s], m.strip_b[s], m.plate_y0 - Real(1.0),
                                       m.plate_y0 + m.strip_h));
            }
            return f;
        }
        f = amrex::max(f, m.plane_y - y);
        const Real c = m.cot_half, a = c * m.tip_r;
        for (int s = 0; s < m.npin; ++s)
        {
            const Real dx = x - m.pin_x[s];
            f = amrex::max(f, y - m.pin_yt[s] - c * (std::sqrt(a * a + dx * dx) - a));
        }
        return f;
    }

    [[nodiscard]] inline Real operator()(const amrex::RealArray& p) const noexcept
    {
        return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
    }

private:
    Params m;
};

// Read eb2.app_* and check the layout; every check aborts.
inline Params read_params()
{
    amrex::ParmParse pp("eb2");
    Params p;
    std::string t;
    pp.get("app_type", t);
    if (t == "streamer") { p.type = 0; }
    else if (t == "pins") { p.type = 1; }
    else if (t == "dbd") { p.type = 2; }
    else { amrex::Abort("eb2.app_type must be streamer, pins or dbd"); }

    pp.get("app_ch_x", p.ch_x);
    pp.get("app_ch_y0", p.ch_y0);
    pp.get("app_ch_y1", p.ch_y1);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p.ch_x > 0.0 && p.ch_y0 < p.ch_y1,
                                     "eb2.app_ch_x must be > 0 and eb2.app_ch_y0 < eb2.app_ch_y1");

    if (p.type == 2)
    {
        pp.get("app_plate_y0", p.plate_y0);
        pp.get("app_plate_y1", p.plate_y1);
        pp.get("app_strip_h", p.strip_h);
        amrex::Vector<Real> st;
        pp.getarr("app_strips", st); // a0 b0 a1 b1 ... (x intervals)
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            st.size() % 2 == 0 && static_cast<int>(st.size()) / 2 <= MAXSTRIP,
            "eb2.app_strips needs pairs a b, at most MAXSTRIP of them");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            p.ch_y0 < p.plate_y0 && p.plate_y0 + p.strip_h < p.plate_y1 && p.plate_y1 < p.ch_y1,
            "eb2.app_plate_y0/y1: need app_ch_y0 < plate_y0 < plate_y0 + strip_h < plate_y1 < app_ch_y1");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p.strip_h > 0.0, "eb2.app_strip_h must be > 0");
        p.nstrip = static_cast<int>(st.size()) / 2;
        for (int i = 0; i < p.nstrip; ++i)
        {
            p.strip_a[i] = st[2 * i];
            p.strip_b[i] = st[2 * i + 1];
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                p.strip_a[i] < p.strip_b[i] && p.strip_a[i] > -p.ch_x && p.strip_b[i] < p.ch_x,
                "eb2.app_strips: each strip needs a < b inside the chamber");
        }
    }
    else
    {
        pp.get("app_plane_y", p.plane_y);
        pp.get("app_tip_r", p.tip_r);
        Real half = 0.0;
        pp.get("app_half_angle", half); // degrees
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p.tip_r > 0.0, "eb2.app_tip_r must be > 0");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(half > 0.0 && half < 60.0,
                                         "eb2.app_half_angle must be in (0, 60) degrees");
        p.cot_half = Real(1.0) / std::tan(half * amrex::Math::pi<Real>() / Real(180.0));
        amrex::Vector<Real> px, py;
        pp.getarr("app_pin_x", px);
        pp.getarr("app_pin_ytip", py);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            px.size() == py.size() && !px.empty() && static_cast<int>(px.size()) <= MAXPIN,
            "eb2.app_pin_x and eb2.app_pin_ytip need the same number of values, 1 to MAXPIN");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p.type == 1 || px.size() == 1,
                                         "eb2.app_type = streamer takes one needle");
        p.npin = static_cast<int>(px.size());
        for (int i = 0; i < p.npin; ++i)
        {
            p.pin_x[i] = px[i];
            p.pin_yt[i] = py[i];
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(px[i]) < p.ch_x && p.plane_y < py[i] && py[i] < p.ch_y1,
                "eb2.app_pin_x/ytip: each tip must lie inside the chamber, above app_plane_y");
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p.ch_y0 < p.plane_y,
                                         "eb2.app_plane_y must lie above eb2.app_ch_y0");
    }
    amrex::Print() << "[applications] " << t << ": " << p.npin << " needles, " << p.nstrip
                   << " strips\n";
    return p;
}

} // namespace appgeom
#endif
