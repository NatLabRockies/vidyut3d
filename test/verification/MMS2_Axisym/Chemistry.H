#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <UnivConstants.H>
#include <VarDefines.H>

/* Elements
0  E
1  He
*/

// Species
#define E_ID 0
#define HE_ID 1
#define HEp_ID 2

#define NUM_ELEMENTS 2
#define NUM_SPECIES 3
#define FIRST_ION 2
#define NUM_IONS 1
#define FIRST_NEUTRAL 0
#define NUM_NEUTRALS 0
#define NUM_REACTIONS 1

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[3] = {
    1822.8884868472639482, // E
    0.2498374807187924,    // HE
    0.2498374807187924,    // HEp
};
#endif
const amrex::Real h_global_imw[3] = {
    1822.8884868472639482, // E
    0.2498374807187924,    // HE
    0.2498374807187924,    // HEp
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[3] = {
    0.000549, // E
    4.002602, // HE
    4.002602, // HEp
};
#endif
const amrex::Real h_global_mw[3] = {
    0.000549, // E
    4.002602, // HE
    4.002602, // HEp
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_imw(amrex::Real* imw_new)
{
    imw_new[0] = 1822.8884868472639482; // E
    imw_new[1] = 0.2498374807187924;    // HE
    imw_new[2] = 0.2498374807187924;    // HEp
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real imw(const int n)
{
#if AMREX_DEVICE_COMPILE
    return global_imw[n];
#else
    return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_mw(amrex::Real* mw_new)
{
    mw_new[0] = 0.000549; // E
    mw_new[1] = 4.002602; // HE
    mw_new[2] = 4.002602; // HEp
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real mw(const int n)
{
#if AMREX_DEVICE_COMPILE
    return global_mw[n];
#else
    return h_global_mw[n];
#endif
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real* tc)
{

    // temperature
    const amrex::Real T = tc[1];
    const amrex::Real invT = 1.0 / T;

    // species with midpoint at T=1000 kelvin
    if (T < 1000)
    {
        // species 0: E
        species[0] =
            -7.459783900000000e+02 * invT + 1.423710750000000e+01 -
            2.500251500000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 1: HE
        species[1] =
            -7.453750000000000e+02 * invT + 1.584651200000000e+00 -
            2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 2: HEp
        species[2] =
            -7.453750000000000e+02 * invT + 1.584651200000000e+00 -
            2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
    } else
    {
        // species 0: E
        species[0] =
            -7.459784500000000e+02 * invT + 1.423710750000000e+01 -
            2.500251500000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 1: HE
        species[1] =
            -7.453750000000000e+02 * invT + 1.584651100000000e+00 -
            2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 2: HEp
        species[2] =
            -7.453750000000000e+02 * invT + 1.584651100000000e+00 -
            2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
    }
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWT(amrex::Real wt[])
{
    get_mw(wt);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real specMob(
    const int specid,
    const amrex::Real Te,
    const amrex::Real ndens,
    const amrex::Real emag,
    const amrex::Real T)
{
    // mobility is scaled by charge
    amrex::Real mob = 0.0;
    amrex::Real elecmob = -1.0;
    amrex::Real ionmob = 0.5;
    if (specid == E_ID)
    {
        mob = elecmob;
    }
    if (specid == HEp_ID)
    {
        mob = ionmob;
    }
    if (specid == NUM_SPECIES)
    {
        mob = fivebythree * elecmob;
    }

    return (mob);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real specDiff(
    const int specid,
    const amrex::Real Te,
    const amrex::Real ndens,
    const amrex::Real emag,
    const amrex::Real T)
{
    amrex::Real dcoeff = 0.0;
    amrex::Real elecdcoeff = 1.0;
    amrex::Real iondcoeff = 0.5;
    if (specid == E_ID)
    {
        dcoeff = elecdcoeff;
    }
    if (specid == HEp_ID)
    {
        dcoeff = iondcoeff;
        // dcoeff=0.0;
    }
    if (specid == NUM_SPECIES)
    {
        dcoeff = fivebythree * elecdcoeff;
    }

    return (dcoeff);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void comp_ener_exch(
    amrex::Real qf,
    amrex::Real qr,
    const amrex::Real* sc,
    amrex::Real kf,
    int rxntype,
    amrex::Real eexci,
    int elidx,
    amrex::Real* enerExch,
    amrex::Real Ue,
    amrex::Real T,
    const amrex::Real Te)
{
    // (0)='no electron'
    // (1)='excitation/de-excitation'
    // (2)='ionization'
    // (3)='recombination'
    // (4)='attachment'
    // (5)='elastic exchange'
    // (6)='chemi-ioniz'
    // (7)='detachment'

    amrex::Real eV2J = 1.602176620800000e-19; // eV -> J
    amrex::Real NA = 6.02214085774e23;        // 1/mol
    amrex::Real kB = 1.380649e-23;            // J/K
    amrex::Real ne = sc[E_ID] * NA;           // 1/m3
    amrex::Real mee = Ue / ne;                // J

    // Molecular weight used to calculate elastic collision energy exchange
    amrex::Real wt[NUM_SPECIES];
    get_mw(wt);

    // Superelastic factor (keeps Te from dropping below gas temperarture)
    amrex::Real sefact = (tanh((Te - T) / 0.5) + 1.0) / 2.0;

    // *enerExch = 0.0;
    if (rxntype == 1 || rxntype == 2)
    {
        *enerExch -= (eexci * eV2J) * NA * (qf - qr) * sefact;
    }
    if (rxntype == 3)
    {
        *enerExch -= mee * NA * (qf - qr);
    }
    // if(rxntype[r] == 4){
    //   *enerExch -= mee*NA*(qf[r] - qr[r]);
    // }
    if (rxntype == 5)
    {
        amrex::Real qq = sc[E_ID] * sc[elidx] * kf;
        *enerExch -= 3.0 * kB * (wt[E_ID] / wt[elidx]) * (Te - T) * qq * NA;
    }
    // if(rxntype[r] == 7){
    //   *enerExch += mee*NA*(qf[r] - qr[r]);
    // }

    return;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void productionRate(
    amrex::Real* wdot,
    const amrex::Real* sc,
    const amrex::Real T,
    const amrex::Real Te,
    amrex::Real* enerExch)
{
    const amrex::Real tc[5] = {
        log(T), T, T * T, T * T * T, T * T * T * T}; // temperature cache
    const amrex::Real invT = 1.0 / tc[1];

    // reference concentration: P_atm / (RT) in inverse mol/m^3
    const amrex::Real refC = 101325 / 8.31446 * invT;
    const amrex::Real refCinv = 1 / refC;

    for (int i = 0; i < 3; ++i)
    {
        wdot[i] = 0.0;
    }
    *enerExch = 0.0;

    // compute the mixture concentration
    amrex::Real mixture = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        mixture += sc[i];
    }

    // compute the Gibbs free energy
    amrex::Real g_RT[3];
    gibbs(g_RT, tc);

    // Precalculating values for electron energy exchange evaluation
    amrex::Real ne = sc[E_ID] * 6.02214085774e23;
    amrex::Real Ue = 1.5 * Te * ne * 1.380649e-23;

    {
        // reaction 0:  HE + E => E + HEp + E
        // const amrex::Real k_f = 50.0*std::exp(-0.2/Te)*6.02214076e23;
        const amrex::Real Ta = 2.0 / K_B;
        const amrex::Real k_f = 5.0 * std::exp(-Ta / Te) * 6.02214076e23;
        const amrex::Real qf = k_f * (sc[0] * sc[1]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] += qdot;
        wdot[1] -= qdot;
        wdot[2] += qdot;
        int rxntype = 2;
        amrex::Real eV2J = 1.602176620800000e-19; // eV -> J
        amrex::Real eexci = 4.0 / eV2J;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    // Calculate electron energy loss via elastic collisions
    amrex::Real nu = 10000.0;
    amrex::Real elec_elastic_coll_term = 3.0 / 2.0 * sc[0] * N_A *
                                         (K_B * Te - K_B * tc[1]) * nu *
                                         (2.0 * ME / (4.0 * M_AMU));
    *enerExch -= elec_elastic_coll_term;
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWC(
    const amrex::Real T,
    amrex::Real C[],
    amrex::Real wdot[],
    const amrex::Real Te,
    const amrex::Real EN,
    amrex::Real* ener_exch)
{

    productionRate(wdot, C, T, Te, ener_exch);
}

// compute electron temperature from EbyN using local field approximation
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
etempLFA(const amrex::Real Tg, amrex::Real C[], const amrex::Real EN)
{
    // 0.5 eV
    return (5802.25);
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCHRG(int kcharge[])
{
    kcharge[0] = -1; // E
    kcharge[1] = 0;  // HE
    kcharge[2] = 1;  // HEp
}
#endif
