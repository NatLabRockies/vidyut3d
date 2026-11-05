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
    amrex::Print() << "GEC wall fluxes: prob.wall_extrap = "
                   << h_prob_parm.wall_extrap
                   << " (0 cell value, 1 linear, 2 logarithmic), "
                   << "prob.wall_floor_frac = " << h_prob_parm.wall_floor_frac
                   << "\n";

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
                if (x * (x - prob_parm->R1) < 0.0 &&
                    (y * y - halfgap * halfgap) > 0.0)
                {
                    phi(i, j, k, CMASK_ID) = 0.0;
                }

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
#endif
