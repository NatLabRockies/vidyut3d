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
#include <GeomUtils.H>

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
    pp.query("sharp_interface_scheme", h_prob_parm.sharp_interface_scheme);

    amrex::ParmParse pp_solver("vidyut");
    amrex::Real Pg, Tg;
    pp_solver.query("gas_pressure", h_prob_parm.pres);
    pp_solver.query("gas_temperature", h_prob_parm.temp);

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

    GpuArray<Real, AMREX_SPACEDIM> ploArr;
    GpuArray<Real, AMREX_SPACEDIM> phiArr;

    ploArr[0] = prob_lo[0];
    ploArr[1] = prob_lo[1];

    phiArr[0] = prob_hi[0];
    phiArr[1] = prob_hi[1];

    amrex::Real floorden = 1e12;

    int chrgspec[NUM_SPECIES] = {0};
    CKCHRG(chrgspec);

    for (int k = lo.z; k <= hi.z; ++k)
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i)
            {
                Real y = prob_lo[1] + (0.5 + j) * dx[1];
                Real x = prob_lo[0] + (0.5 + i) * dx[0];

                for (int c = 0; c < NVAR; c++)
                {
                    phi(i, j, k, c) = 0.0;
                }
                for (int c = 0; c < NUM_SPECIES; c++)
                {
                    phi(i, j, k, c) = floorden;
                }

                amrex::Real gasvfrac = 0.0;
                for (int jj = 0; jj < 2; jj++)
                {
                    for (int ii = 0; ii < 2; ii++)
                    {
                        Real xnode = prob_lo[0] + (i + ii) * dx[0];
                        Real ynode = prob_lo[1] + (j + jj) * dx[1];

                        if (surfacefunc(
                                xnode, ynode, *prob_parm, ploArr, phiArr) > 0.0)
                        {
                            gasvfrac += 1.0;
                        }
                    }
                }
                gasvfrac *= 0.25;

                phi(i, j, k, CMASK_ID) = 1.0;
                if (gasvfrac < 1.0)
                {
                    phi(i, j, k, CMASK_ID) = 0.0;
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

#endif
