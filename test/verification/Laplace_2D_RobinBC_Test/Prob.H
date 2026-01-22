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
    //AMREX_ALWAYS_ASSERT((
    //    h_prob_parm.a_inner_dir == 1.0 || h_prob_parm.a_outer_dir == 1.0));

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

    for (int k = lo.z; k <= hi.z; ++k)
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {

            for (int i = lo.x; i <= hi.x; ++i)
            {
                for (int sp = 0; sp < NVAR; sp++)
                {
                    phi(i, j, k, sp) = 0.0;
                }

                // Real y = prob_lo[1] + (0.5 + j) * dx[1];
                // Real x = prob_lo[0] + (0.5 + i) * dx[0];
                // Real rad = sqrt(x * x + y * y);
                // if ((prob_parm->inner_radius < rad) &&
                //     (rad < prob_parm->outer_radius))
                // {
                //     phi(i, j, k, CMASK_ID) = 1;
                // } else
                // {
                //     phi(i, j, k, CMASK_ID) = 0;
                // }

                for (int jj = 0; jj < 2; jj++)
                {
                    for (int ii = 0; ii < 2; ii++)
                    {
                        Real xx = prob_lo[0] + (i + ii) * dx[0];
                        Real yy = prob_lo[1] + (j + jj) * dx[1];
                        Real rad = sqrt(xx * xx + yy * yy);

                        if ((prob_parm->inner_radius <= rad) &&
                            (rad <= prob_parm->outer_radius))
                        {
                            phi(i, j, k, CMASK_ID) += 1.0;
                        }
                    }
                }
                phi(i, j, k, CMASK_ID) *= 0.25;

                phi(i, j, k, E_ID) = 1e12;
                phi(i, j, k, S1_ID) = 1e12;
                phi(i, j, k, S2_ID) = 1e12;
                phi(i, j, k, ETEMP_ID) = 0.5 * eV;
            }
        }
    }
}
#endif
