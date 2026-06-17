#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <UnivConstants.H>
#include <VarDefines.H>

/* Elements
0  E
1  Ar
*/

// Species
#define E_ID 0
#define AR_ID 1
#define ARm_ID 2
#define ARp_ID 3

#define NUM_ELEMENTS 2
#define NUM_SPECIES 4
#define FIRST_ION 3
#define NUM_IONS 1
#define FIRST_NEUTRAL 2
#define NUM_NEUTRALS 1
#define NUM_REACTIONS 6

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
    mm = 2;
    kk = 4;
    ii = 6;
    nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[4] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.0250312891113892,    // ARm
    0.0250316328373182,    // ARp
};
#endif
const amrex::Real h_global_imw[4] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.0250312891113892,    // ARm
    0.0250316328373182,    // ARp
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[4] = {
    0.000549,  // E
    39.950000, // AR
    39.950000, // ARm
    39.949451, // ARp
};
#endif
const amrex::Real h_global_mw[4] = {
    0.000549,  // E
    39.950000, // AR
    39.950000, // ARm
    39.949451, // ARp
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_imw(amrex::Real* imw_new)
{
    imw_new[0] = 1822.8884868472639482; // E
    imw_new[1] = 0.0250312891113892;    // AR
    imw_new[2] = 0.0250312891113892;    // ARm
    imw_new[3] = 0.0250316328373182;    // ARp
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
    mw_new[0] = 0.000549;  // E
    mw_new[1] = 39.950000; // AR
    mw_new[2] = 39.950000; // ARm
    mw_new[3] = 39.949451; // ARp
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

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
    ru = 8.31446261815324e+07;
    ruc = 1.98721558317399615845;
    pa = 1.01325e+06;
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
        // species 3: ARp
        species[3] =
            +1.828792080000000e+05 * invT - 2.899129370000000e+00 -
            2.584996020000000e+00 * tc[0] + 6.355539600000000e-04 * tc[1] -
            8.544103316666667e-07 * tc[2] + 4.866947275000000e-10 * tc[3] -
            1.069662480000000e-13 * tc[4];
    } else
    {
        // species 0: E
        species[0] =
            -7.459784500000000e+02 * invT + 1.423710750000000e+01 -
            2.500251500000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
            0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
            0.000000000000000e+00 * tc[4];
        // species 3: ARp
        species[3] =
            +1.826983560000000e+05 * invT - 5.893438799999999e-01 -
            2.881122420000000e+00 * tc[0] + 8.072412650000000e-05 * tc[1] -
            3.140146533333333e-09 * tc[2] - 8.776421000000000e-14 * tc[3] +
            1.149512960000000e-17 * tc[4];
    }

    // species with no change at a midpoint T
    // species 1: AR
    species[1] = -7.453750000000000e+02 * invT - 1.866000000000000e+00 -
                 2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
                 0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
                 0.000000000000000e+00 * tc[4];
    // species 2: ARm
    species[2] = -7.453750000000000e+02 * invT - 1.866000000000000e+00 -
                 2.500000000000000e+00 * tc[0] - 0.000000000000000e+00 * tc[1] -
                 0.000000000000000e+00 * tc[2] - 0.000000000000000e+00 * tc[3] -
                 0.000000000000000e+00 * tc[4];
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

    amrex::Real Arp_cs = 120e-20; // A^2 (cross section)
    amrex::Real E_cs = 40e-20;    // A^2 (cross section)
    amrex::Real nu_elec = std::sqrt(8.0 * K_B * Te / PI / ME) * ndens * E_cs;
    amrex::Real elecmob = -ECHARGE / ME / nu_elec;

    amrex::Real mob = 0.0;
    if (specid == E_ID)
    {
        mob = elecmob;
    }
    if (specid == ARp_ID)
    {
        amrex::Real nu_Arp =
            std::sqrt(8.0 * K_B * T / PI / (39.948 * M_AMU)) * ndens * Arp_cs;
        mob = ECHARGE / (39.948 * M_AMU) / nu_Arp;
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
    // mu=e/(m nu) , D=kT/(m nu)
    //  m nu = e/mu  D=kT/(e/mu) = mu * T/(e/k)
    //  = mu * T_in_ev

    amrex::Real elecmob = specMob(E_ID, Te, ndens, emag, T);
    amrex::Real ionmob = specMob(ARp_ID, Te, ndens, emag, T);
    amrex::Real Te_in_ev = Te / eV;
    amrex::Real Tg_in_ev = T / eV;
    amrex::Real dcoeff = 1e-12; // dummy value

    if (specid == E_ID)
    {
        dcoeff = Te_in_ev * amrex::Math::abs(elecmob);
    }
    if (specid == ARp_ID)
    {
        dcoeff = Tg_in_ev * ionmob;
    }
    if (specid == ARm_ID)
    {
        dcoeff = Tg_in_ev * ionmob;
    }
    if (specid == AR_ID)
    {
        dcoeff = Tg_in_ev * ionmob;
    }
    if (specid == NUM_SPECIES)
    {
        dcoeff = fivebythree * Te_in_ev * amrex::Math::abs(elecmob);
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

    for (int i = 0; i < 4; ++i)
    {
        wdot[i] = 0.0;
    }

    // compute the mixture concentration
    amrex::Real mixture = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        mixture += sc[i];
    }

    // compute the Gibbs free energy
    amrex::Real g_RT[4];
    gibbs(g_RT, tc);

    // Precalculating values for electron energy exchange evaluation
    amrex::Real ne = sc[E_ID] * 6.02214085774e23;
    amrex::Real Ue = 1.5 * Te * ne * 1.380649e-23;

    {
        // reaction 1:  AR + E => ARp + E + E
        const amrex::Real k_f =
            42581353.6 * std::pow(Te, 0.60983) * std::exp(-1.8712e5 / Te);
        const amrex::Real qf = k_f * (sc[0] * sc[1]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] += qdot;
        wdot[1] -= qdot;
        wdot[3] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 16.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 2:  ARm + E => ARp + E + E
        const amrex::Real k_f =
            75011786500 * std::pow(Te, 0.050382) * std::exp(-6.0524e4 / Te);
        const amrex::Real qf = k_f * (sc[0] * sc[2]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] += qdot;
        wdot[2] -= qdot;
        wdot[3] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 4.43;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 0:  AR + E => ARm + E
        const amrex::Real k_f =
            7074811070 * std::pow(Te, 0.046639) * std::exp(-1.3856e5 / Te);
        const amrex::Real qf = k_f * (sc[0] * sc[1]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] -= qdot;
        wdot[0] += qdot;
        wdot[1] -= qdot;
        wdot[2] += qdot;
        int rxntype = 1;
        amrex::Real eexci = 11.56;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 3:  2 ARm => AR + ARp + E
        const amrex::Real k_f = 373372733;
        const amrex::Real qf = k_f * ((sc[2] * sc[2]));
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] += qdot;
        wdot[1] += qdot;
        wdot[2] -= 2.000000 * qdot;
        wdot[3] += qdot;
        int rxntype = 2;
        amrex::Real eexci = -7.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 4:  ARm + E => AR + E
        const amrex::Real k_f = 120442817000;
        const amrex::Real qf = k_f * (sc[0] * sc[2]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[0] -= qdot;
        wdot[0] += qdot;
        wdot[1] += qdot;
        wdot[2] -= qdot;
        int rxntype = 1;
        amrex::Real eexci = -11.56;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, tc[1], Te);
    }

    {
        // reaction 5:  AR + ARm => 2 AR
        const amrex::Real k_f = 1505.5352;
        const amrex::Real qf = k_f * (sc[1] * sc[2]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[1] -= qdot;
        wdot[1] += 2.000000 * qdot;
        wdot[2] -= qdot;
    }

    // Calculate electron energy loss via elastic collisions
    amrex::Real emag = 0.0;
    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sc[sp] * N_A;
    amrex::Real mu = specMob(E_ID, Te, ndens, emag, tc[1]);
    amrex::Real nu = -ECHARGE / ME / mu;
    amrex::Real elec_elastic_coll_term = 3.0 / 2.0 * K_B * sc[0] * N_A *
                                         (Te - tc[1]) * nu *
                                         (2.0 * ME / (39.948 * M_AMU));
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

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCHRG(int kcharge[])
{
    kcharge[0] = -1; // E
    kcharge[1] = 0;  // AR
    kcharge[2] = 0;  // ARm
    kcharge[3] = 1;  // ARp
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

    int kchrg[4];
    CKCHRG(kchrg);

    for (int id = 0; id < 4; ++id)
    {
        zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
    }
}

#endif
