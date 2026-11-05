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
    pp.query("r1", h_prob_parm.r1);
    pp.query("r2", h_prob_parm.r2);
    pp.query("field_sign", h_prob_parm.field_sign);
    pp.query("ib_face_flux", h_prob_parm.ib_face_flux);
    if (h_prob_parm.field_sign != 1.0 && h_prob_parm.field_sign != -1.0)
    {
        amrex::Abort("prob.field_sign must be 1 or -1");
    }
    if (h_prob_parm.ib_face_flux != 0 && h_prob_parm.ib_face_flux != 1)
    {
        amrex::Abort("prob.ib_face_flux must be 0 or 1");
    }

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
    Real x[3];

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k)
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            x[2] = prob_lo[2] + (0.5 + k) * dx[2];
            x[1] = prob_lo[1] + (0.5 + j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i)
            {
                x[0] = prob_lo[0] + (0.5 + i) * dx[0];
                for (int c = 0; c < NVAR; c++)
                {
                    phi(i, j, k, c) = 1e-12;
                }
                phi(i, j, k, CMASK_ID) = 1.0;
                if (x[0] > 1.0)
                {
                    phi(i, j, k, CMASK_ID) = 0.0;
                }
                phi(i, j, k, ETEMP_ID) = 0.5 * eV;
            }
        }
    }
}
#endif
