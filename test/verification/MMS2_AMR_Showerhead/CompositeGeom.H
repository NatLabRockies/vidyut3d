#ifndef COMPOSITE_GEOM_H_
#define COMPOSITE_GEOM_H_

// Planar 2D composite chamber (y = electrode axis): showerhead with a plenum
// and slots, pedestal with lift-pin holes, wafer, focus ring, confinement
// ring. The slots can open in stages during a run (eb2.comp_slot_phase).
//
//   top electrode    : solid for y >= hg, |x| <= R1, with a plenum
//                      (plenum_y0 < y < plenum_y1, |x| < plenum_hw) and
//                      slots of graded width through the faceplate
//                      (hg < y < plenum_y0)
//   pedestal         : solid for y <= -hg, |x| <= R1, with blind lift-pin
//                      holes (pin_bot < y < -hg)
//   wafer            : -hg < y < wafer_top, |x| <= wafer_hw (covers the pins)
//   focus ring       : -hg < y < ring_top, ring_in <= |x| <= ring_out
//   confinement ring : conf_y0 < y < conf_y1, |x| >= conf_in (pump gap
//                      conf_in - R1 next to the pedestal)
//   chamber          : optional box wall, fluid inside |x| < ch_x,
//                      ch_y0 < y < ch_y1, so no wall is a domain face
//
// Every length is multiplied by eb2.comp_scale, so the same deck values give
// the plasma geometry in metres and the MMS geometry in its own units. The
// deck values are placed at (m + 1/3) h of h = 0.2032/1024 (metres), so no
// wall falls on a grid line of any 0.2032/N grid or its refined levels.
//
// Sign convention as AMReX EB2: the function is > 0 in the solid. A box
// body is min of its four signed face distances; union = max,
// difference A - B = min(A, -B).

#include <AMReX_EB2_IF_Base.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <string>
#include <utility>

namespace compgeom {

using amrex::Real;

constexpr int MAXSLOT = 16;
constexpr int MAXPIN = 8;

struct Params
{
    Real hg = 0.0, R1 = 0.0;
    Real plenum_y0 = 0.0, plenum_y1 = 0.0, plenum_hw = 0.0;
    int nslot = 0;
    Real slot_a[MAXSLOT] = {}, slot_b[MAXSLOT] = {};
    int slot_open[MAXSLOT] = {};
    int npin = 0;
    Real pin_a[MAXPIN] = {}, pin_b[MAXPIN] = {};
    Real pin_bot = 0.0;
    Real wafer_top = 0.0, wafer_hw = 0.0;
    Real ring_top = 0.0, ring_in = 0.0, ring_out = 0.0;
    Real conf_y0 = 0.0, conf_y1 = 0.0, conf_in = 0.0;
    Real ch_x = 0.0, ch_y0 = 0.0, ch_y1 = 0.0;
    int showerhead = 1, pins = 1, wafer = 1, ring = 1, conf = 1, chamber = 0;
    Real sx = 0.0, sy = 0.0; // rigid shift of the whole geometry (sub-cell sweep)
    Real big = 1.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
boxv(Real x, Real y, Real x0, Real x1, Real y0, Real y1) noexcept
{
    return amrex::min(amrex::min(x - x0, x1 - x), amrex::min(y - y0, y1 - y));
}

class CompositeIF : public amrex::GPUable
{
public:
    explicit CompositeIF(const Params& p) : m(p) {}

    [[nodiscard]] AMREX_GPU_HOST_DEVICE inline Real
    operator()(AMREX_D_DECL(Real xin, Real yin, Real /*z*/)) const noexcept
    {
        const Real B = m.big;
        const Real x = xin - m.sx, y = yin - m.sy;

        // top electrode minus plenum and slots
        Real top = boxv(x, y, -m.R1, m.R1, m.hg, B);
        if (m.showerhead)
        {
            Real v = boxv(x, y, -m.plenum_hw, m.plenum_hw, m.plenum_y0, m.plenum_y1);
            const Real e = Real(0.25) * (m.plenum_y0 - m.hg);
            for (int s = 0; s < m.nslot; ++s)
            {
                if (m.slot_open[s])
                {
                    v = amrex::max(
                        v, boxv(x, y, m.slot_a[s], m.slot_b[s], m.hg - e, m.plenum_y0 + e));
                }
            }
            top = amrex::min(top, -v);
        }

        // pedestal minus lift-pin holes
        Real ped = boxv(x, y, -m.R1, m.R1, -B, -m.hg);
        if (m.pins)
        {
            Real v = -B;
            const Real e = Real(0.25) * (m.wafer_top + m.hg);
            for (int s = 0; s < m.npin; ++s)
            {
                v = amrex::max(v, boxv(x, y, m.pin_a[s], m.pin_b[s], m.pin_bot, -m.hg + e));
            }
            ped = amrex::min(ped, -v);
        }

        Real f = amrex::max(top, ped);
        if (m.wafer)
        {
            f = amrex::max(f, boxv(x, y, -m.wafer_hw, m.wafer_hw, -m.hg, m.wafer_top));
        }
        if (m.ring)
        {
            f = amrex::max(f, boxv(x, y, m.ring_in, m.ring_out, -m.hg, m.ring_top));
            f = amrex::max(f, boxv(x, y, -m.ring_out, -m.ring_in, -m.hg, m.ring_top));
        }
        if (m.conf)
        {
            f = amrex::max(f, boxv(x, y, m.conf_in, B, m.conf_y0, m.conf_y1));
            f = amrex::max(f, boxv(x, y, -B, -m.conf_in, m.conf_y0, m.conf_y1));
        }
        if (m.chamber)
        {
            f = amrex::max(f, -boxv(x, y, -m.ch_x, m.ch_x, m.ch_y0, m.ch_y1));
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

// Read eb2.comp_* (every length times eb2.comp_scale) and check the layout.
inline Params read_params()
{
    amrex::ParmParse pp("eb2");
    Params p;
    Real s = 1.0;
    pp.query("comp_scale", s);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s > 0.0, "eb2.comp_scale must be > 0");

    auto get = [&](const char* k, Real& v) {
        pp.get(k, v);
        v *= s;
    };
    get("comp_hg", p.hg);
    get("comp_R1", p.R1);
    get("comp_plenum_y0", p.plenum_y0);
    get("comp_plenum_y1", p.plenum_y1);
    get("comp_plenum_hw", p.plenum_hw);
    get("comp_pin_bot", p.pin_bot);
    get("comp_wafer_top", p.wafer_top);
    get("comp_wafer_hw", p.wafer_hw);
    get("comp_ring_top", p.ring_top);
    get("comp_ring_in", p.ring_in);
    get("comp_ring_out", p.ring_out);
    get("comp_conf_y0", p.conf_y0);
    get("comp_conf_y1", p.conf_y1);
    get("comp_conf_in", p.conf_in);

    amrex::Vector<Real> sl, pn;
    pp.getarr("comp_slots", sl); // a0 b0 a1 b1 ... (x intervals)
    pp.getarr("comp_pins", pn);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        sl.size() % 2 == 0 && static_cast<int>(sl.size()) / 2 <= MAXSLOT,
        "eb2.comp_slots needs pairs a b, at most MAXSLOT of them");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        pn.size() % 2 == 0 && static_cast<int>(pn.size()) / 2 <= MAXPIN,
        "eb2.comp_pins needs pairs a b, at most MAXPIN of them");
    p.nslot = static_cast<int>(sl.size()) / 2;
    for (int i = 0; i < p.nslot; ++i)
    {
        p.slot_a[i] = s * sl[2 * i];
        p.slot_b[i] = s * sl[2 * i + 1];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            p.slot_a[i] < p.slot_b[i] && p.slot_a[i] > -p.plenum_hw
                && p.slot_b[i] < p.plenum_hw,
            "eb2.comp_slots: each slot needs a < b and must open into the plenum");
    }
    p.npin = static_cast<int>(pn.size()) / 2;
    for (int i = 0; i < p.npin; ++i)
    {
        p.pin_a[i] = s * pn[2 * i];
        p.pin_b[i] = s * pn[2 * i + 1];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            p.pin_a[i] < p.pin_b[i] && p.pin_a[i] > -p.R1 && p.pin_b[i] < p.R1,
            "eb2.comp_pins: each pin hole needs a < b inside the pedestal");
    }

    pp.query("comp_showerhead", p.showerhead);
    pp.query("comp_pins_on", p.pins);
    pp.query("comp_wafer", p.wafer);
    pp.query("comp_ring", p.ring);
    pp.query("comp_conf", p.conf);
    pp.query("comp_chamber", p.chamber);
    {
        amrex::Vector<Real> sh;
        if (pp.queryarr("comp_shift", sh))
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                sh.size() == 2, "eb2.comp_shift needs two values (x y)");
            p.sx = s * sh[0];
            p.sy = s * sh[1];
        }
    }

    // Mid-run geometry change: the solver adds 1 to vidyut.geom_phase before
    // each step in vidyut.geom_switch_steps. From geom_phase >= 1 the
    // eb2.comp_after_* switches override the ones above.
    int phase = 0;
    {
        amrex::ParmParse ppv("vidyut");
        ppv.query("geom_phase", phase);
    }
    // Staged opening: slot s is open once vidyut.geom_phase >= eb2.comp_slot_phase[s].
    {
        amrex::Vector<int> sp(p.nslot, 0);
        pp.queryarr("comp_slot_phase", sp);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(sp.size()) == p.nslot,
            "eb2.comp_slot_phase needs one value per slot");
        for (int i = 0; i < p.nslot; ++i)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                sp[i] >= 0, "eb2.comp_slot_phase must be >= 0");
            p.slot_open[i] = (phase >= sp[i]) ? 1 : 0;
        }
    }
    // Read eb2.comp_after_* at start-up too, so a bad value aborts at once
    // rather than at the first switch.
    {
        int after[5] = {p.showerhead, p.pins, p.wafer, p.ring, p.conf};
        const char* key[5] = {"comp_after_showerhead", "comp_after_pins_on",
                              "comp_after_wafer", "comp_after_ring",
                              "comp_after_conf"};
        for (int i = 0; i < 5; ++i)
        {
            pp.query(key[i], after[i]);
            if (after[i] != 0 && after[i] != 1)
            {
                amrex::Abort(std::string("eb2.") + key[i] + " must be 0 or 1");
            }
        }
        if (phase >= 1)
        {
            p.showerhead = after[0];
            p.pins = after[1];
            p.wafer = after[2];
            p.ring = after[3];
            p.conf = after[4];
        }
    }
    {
        const std::pair<const char*, int> onoff[] = {
            {"eb2.comp_showerhead / comp_after_showerhead", p.showerhead},
            {"eb2.comp_pins_on / comp_after_pins_on", p.pins},
            {"eb2.comp_wafer / comp_after_wafer", p.wafer},
            {"eb2.comp_ring / comp_after_ring", p.ring},
            {"eb2.comp_conf / comp_after_conf", p.conf},
            {"eb2.comp_chamber", p.chamber}};
        for (const auto& o : onoff)
        {
            if (o.second != 0 && o.second != 1)
            {
                amrex::Abort(std::string(o.first) + " must be 0 or 1");
            }
        }
    }
    if (p.chamber)
    {
        get("comp_ch_x", p.ch_x);
        get("comp_ch_y0", p.ch_y0);
        get("comp_ch_y1", p.ch_y1);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            p.ch_x > 0.0 && p.ch_y0 < p.ch_y1,
            "eb2.comp_ch_*: need comp_ch_x > 0 and comp_ch_y0 < comp_ch_y1");
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        p.hg > 0.0 && p.hg < p.plenum_y0 && p.plenum_y0 < p.plenum_y1,
        "eb2.comp: need 0 < hg < plenum_y0 < plenum_y1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        p.plenum_hw < p.R1, "eb2.comp: plenum must lie inside the electrode");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        p.pin_bot < -p.hg && -p.hg < p.wafer_top && p.wafer_hw < p.ring_in
            && p.ring_in < p.ring_out && -p.hg < p.ring_top,
        "eb2.comp: need pin_bot < -hg < wafer_top, wafer_hw < ring_in < ring_out");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        p.conf_y0 < p.conf_y1 && p.conf_y1 < -p.hg && p.conf_in > p.R1,
        "eb2.comp: confinement ring must sit below the pedestal top, outside R1");

    // larger than any coordinate of the domain and of the bodies
    p.big = Real(10.0) * amrex::max(amrex::max(p.plenum_y1, p.R1, p.conf_in, -p.pin_bot),
                                    amrex::max(p.ch_x, -p.ch_y0, p.ch_y1));

    amrex::Print() << "[composite] phase " << phase << ", scale " << s << ", " << p.nslot << " slots, "
                   << p.npin << " pin holes, showerhead " << p.showerhead
                   << ", pins " << p.pins << ", wafer " << p.wafer << ", ring "
                   << p.ring << ", confinement " << p.conf << ", chamber box "
                   << p.chamber << "\n";
    return p;
}

} // namespace compgeom
#endif
