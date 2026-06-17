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
#define HEm_ID 3
#define HE2m_ID 4

// note: anyway HE_ID is skipped
// so we dont have to include it
// in neutrals
#define NUM_IONS 1
#define FIRST_ION 2
#define NUM_NEUTRALS 2
#define FIRST_NEUTRAL 3

#define NUM_ELEMENTS 2
#define NUM_SPECIES 5
#define NUM_REACTIONS 4

#define NUM_FIT 4

void atomicWeight(amrex::Real* awt);
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[5] = {
    1822.8884868472639482, // E
    0.2498374807187924,    // HE
    0.2498717270938452,    // HEp
    0.2498374807187924,    // HEm
    0.2498374807187924,    // HE2m
};
#endif
const amrex::Real h_global_imw[5] = {
    1822.8884868472639482, // E
    0.2498374807187924,    // HE
    0.2498717270938452,    // HEp
    0.2498374807187924,    // HEm
    0.2498374807187924,    // HE2m
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[5] = {
    0.000549, // E
    4.002602, // HE
    4.002053, // HEp
    4.002602, // HEm
    4.002602, // HE2m
};
#endif
const amrex::Real h_global_mw[5] = {
    0.000549, // E
    4.002602, // HE
    4.002053, // HEp
    4.002602, // HEm
    4.002602, // HE2m
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_imw(amrex::Real* imw_new)
{
    imw_new[0] = 1822.8884868472639482; // E
    imw_new[1] = 0.2498374807187924;    // HE
    imw_new[2] = 0.2498717270938452;    // HEp
    imw_new[3] = 0.2498374807187924;    // HEm
    imw_new[4] = 0.2498374807187924;    // HE2m
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
    mw_new[2] = 4.002053; // HEp
    mw_new[3] = 4.002602; // HEm
    mw_new[4] = 4.002602; // HE2m
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
        // species 3: HEm
        species[3] =
            -7.453750000000000e+02 * invT + 1.584651200000000e+00 -
            2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 4: HE2m
        species[4] =
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
        // species 3: HEm
        species[3] =
            -7.453750000000000e+02 * invT + 1.584651100000000e+00 -
            2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 4: HE2m
        species[4] =
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
    amrex::Real mob = 0.0;
    // mobility is scaled by charge

    // reduced mobility obtained from Yuan and Raja,
    // IEEE. Trans. Plasma Sci.,31,4,2003
    // they say these are 393K, may be temperature
    // scaling is necessary
    // amrex::Real Pres_ratio = (P_NTP/Pg);
    // amrex::Real elecmob=-0.1132*Pres_ratio;

    amrex::Real Pg = ndens * K_B * T;
    amrex::Real EbyN_cutoff = 0.1;
    amrex::Real EbyN = std::max(EbyN_cutoff, emag / ndens / 1e-21);
    amrex::Real meanE = 1.5 * Te / eV;

    // mobility from
    // Turner, Miles M., et al. "Simulation benchmarks for low-pressure
    // plasmas: Capacitive discharges."
    // Physics of Plasmas 20.1 (2013): 013507.
    amrex::Real Hepmob = 2.69 * std::pow(
                                    (1.0 + 1.2e-3 * amrex::Math::powi<2>(EbyN) +
                                     4.2e-8 * amrex::Math::powi<4>(EbyN)),
                                    -0.125);

    // computed by Taaresh Taneja (U Minnesota) using Turner's cross sections
    amrex::Real elecmob = (-1.0) *
                          std::exp(
                              55.0 + 0.3942 * std::log(meanE) + 2.134 / meanE -
                              0.6433 / amrex::Math::powi<2>(meanE) +
                              (0.7112e-1) / amrex::Math::powi<3>(meanE)) /
                          ndens;

    if (specid == E_ID)
    {
        mob = elecmob;
    }
    if (specid == HEp_ID)
    {
        // mob=0.001482*Pres_ratio;
        mob = Hepmob;
    }
    // for electron energy
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
    amrex::Real Pres_ratio = P_NTP / (ndens * K_B * T);
    amrex::Real Te_in_eV = Te / eV;
    amrex::Real Tg_in_eV = T / eV;
    amrex::Real mob;
    amrex::Real dcoeff = 1e-12; // dummy value
    if (specid == E_ID)
    {
        mob = specMob(specid, Te, ndens, emag, T);
        dcoeff = amrex::Math::abs(mob) * Te_in_eV;
    }
    if (specid == HEp_ID)
    {
        mob = specMob(specid, Te, ndens, emag, T);
        dcoeff = amrex::Math::abs(mob) * Tg_in_eV;
    }
    if (specid == HEm_ID)
    {
        dcoeff = 4.116e-4 * Pres_ratio;
    }
    if (specid == HE2m_ID)
    {
        dcoeff = 4.116e-4 * Pres_ratio;
    }
    // for electron energy
    if (specid == NUM_SPECIES)
    {
        mob = specMob(specid, Te, ndens, emag, T);
        dcoeff = amrex::Math::abs(mob) * Te_in_eV;
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

    for (int i = 0; i < 5; ++i)
    {
        wdot[i] = 0.0;
    }

    // compute the mixture concentration
    amrex::Real mixture = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        mixture += sc[i];
    }

    // compute the Gibbs free energy
    amrex::Real g_RT[5];
    gibbs(g_RT, tc);

    // Precalculating values for electron energy exchange evaluation
    amrex::Real ne = sc[E_ID] * 6.02214085774e23;  // number density in #/m3
    amrex::Real Ue = 1.5 * Te * ne * 1.380649e-23; // 3/2 nKT

    {
        // reaction 0:  HE + E => E + HEp + E
        const amrex::Real k_f =
            7e-14 * std::exp(-5.7248896e5 / Te) * 6.02214085774e23;
        const amrex::Real qf = k_f * (sc[0] * sc[1]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] += qdot;
        wdot[1] -= qdot;
        wdot[2] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 24.6;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 1:  HE + E => E + HEm
        const amrex::Real k_f =
            1e-14 * std::exp(-3.71344e5 / Te) * 6.02214085774e23;
        const amrex::Real qf = k_f * (sc[0] * sc[1]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[1] -= qdot;
        wdot[3] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 19.8;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 2:  HE + E => HE2m + E
        const amrex::Real k_f =
            1e-14 * std::exp(-4.02289e5 / Te) * 6.02214085774e23;
        const amrex::Real qf = k_f * (sc[0] * sc[1]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[1] -= qdot;
        wdot[4] += qdot;
        int rxntype = 1;
        amrex::Real eexci = 19.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }
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

    // Calculate electron energy loss via elastic collisions
    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += C[sp] * N_A;
    amrex::Real emag = EN * ndens * 1.0e-21; // electric field magnitude
    amrex::Real mu = specMob(E_ID, Te, ndens, emag, T);
    amrex::Real nu = -ECHARGE / ME / mu;
    amrex::Real elec_elastic_coll_term = 3.0 / 2.0 * K_B * C[0] * N_A *
                                         (Te - T) * nu *
                                         (2.0 * ME / (4.0 * M_AMU));
    *ener_exch -= elec_elastic_coll_term;
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCHRG(int kcharge[])
{
    kcharge[0] = -1; // E
    kcharge[1] = 0;  // HE
    kcharge[2] = 1;  // HEp
    kcharge[3] = 0;  // HEm
    kcharge[4] = 0;  // HE2m
}

// compute electron temperature from EbyN using local field approximation
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
etempLFA(const amrex::Real Tg, amrex::Real C[], const amrex::Real EN)
{
    // 0.5 eV
    return (5802.25);
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCHRGMASS(amrex::Real zk[])
{

    int kchrg[5];
    CKCHRG(kchrg);

    for (int id = 0; id < 5; ++id)
    {
        zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
    }
}
#endif
