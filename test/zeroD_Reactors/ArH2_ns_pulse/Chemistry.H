#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <UnivConstants.H>
#include <VarDefines.H>

/* Elements
   0  E
   1  Ar
   2 H
   */

// Species
#define E_ID 0
#define AR_ID 1
#define H2_ID 2
#define ARp_ID 3
#define AR2p_ID 4
#define Hp_ID 5
#define H2p_ID 6
#define H3p_ID 7
#define ARHp_ID 8
#define ARm_ID 9
#define AR2m_ID 10
#define H_ID 11
#define H2v1_ID 12
#define H2v2_ID 13
#define H2v3_ID 14

#define NUM_SPECIES 15
#define FIRST_ION 3
#define NUM_IONS 6
#define FIRST_NEUTRAL 9
#define NUM_NEUTRALS 6
#define LFA_NDATA 20

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

// note add a gpu constant array to
// work on gpu
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real redef_ee_array[LFA_NDATA][2] = {
    {2.4848, 1.1964},    {3.4393, 1.4586},   {4.6837, 1.9177},
    {6.2755, 2.5694},    {8.8284, 3.4426},   {12.0226, 4.1577},
    {16.3726, 4.6562},   {22.2964, 5.0213},  {30.8611, 5.2639},
    {42.7158, 5.6234},   {59.1241, 5.8397},  {81.8354, 6.1799},
    {111.4445, 6.6645},  {146.9125, 7.1197}, {203.3458, 7.678},
    {281.4568, 8.2023},  {389.5724, 9.1858}, {505.2769, 10.1905},
    {746.3477, 12.1917}, {968.0156, 15.0048}};
#endif
const amrex::Real h_redef_ee_array[LFA_NDATA][2] = {
    {2.4848, 1.1964},    {3.4393, 1.4586},   {4.6837, 1.9177},
    {6.2755, 2.5694},    {8.8284, 3.4426},   {12.0226, 4.1577},
    {16.3726, 4.6562},   {22.2964, 5.0213},  {30.8611, 5.2639},
    {42.7158, 5.6234},   {59.1241, 5.8397},  {81.8354, 6.1799},
    {111.4445, 6.6645},  {146.9125, 7.1197}, {203.3458, 7.678},
    {281.4568, 8.2023},  {389.5724, 9.1858}, {505.2769, 10.1905},
    {746.3477, 12.1917}, {968.0156, 15.0048}};

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[NUM_SPECIES] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.4960317460317460,    // H2
    0.0250312891113892,    // ARp
    0.0125156445556946,    // Ar2p
    0.9926036929001966,    // Hp
    0.4961667594819952,    // H2p
    0.3307478312219284,    // H3p
    0.0244155810000000,    // ARHp
    0.0250316328373182,    // ARm
    0.0125156445556946     // Ar2m
    0.9920634920634921,    // H
    0.4960317460317460,    // H2v1
    0.4960317460317460,    // H2v2
    0.4960317460317460     // H2v3
};
#endif
const amrex::Real h_global_imw[NUM_SPECIES] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.4960317460317460,    // H2
    0.0250312891113892,    // ARp
    0.0125156445556946,    // Ar2p
    0.9926036929001966,    // Hp
    0.4961667594819952,    // H2p
    0.3307478312219284,    // H3p
    0.0244155810000000,    // ARHp
    0.0250316328373182,    // ARm
    0.0125156445556946,    // Ar2m
    0.9920634920634921,    // H
    0.4960317460317460,    // H2v1
    0.4960317460317460,    // H2v2
    0.4960317460317460     // H2v3
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[NUM_SPECIES] = {
    0.000549,  // E
    39.950000, // AR
    2.0160000, // H2
    39.950000, // ARp
    79.900000, // AR2p
    1.007451,  // Hp
    2.015451,  // H2p
    3.023451,  // H3p
    40.957451, // ARHp
    39.949451, // ARm
    79.900000, // AR2m
    1.008000,  // H
    2.016000,  // H2v1
    2.016000,  // H2v2
    2.016000   // H2v3
};
#endif
const amrex::Real h_global_mw[NUM_SPECIES] = {
    0.000549,  // E
    39.950000, // AR
    2.0160000, // H2
    39.950000, // ARp
    79.900000, // AR2p
    1.007451,  // Hp
    2.015451,  // H2p
    3.023451,  // H3p
    40.957451, // ARHp
    39.949451, // ARm
    79.900000, // AR2m
    1.008000,  // H
    2.016000,  // H2v1
    2.016000,  // H2v2
    2.016000   // H2v3
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_imw(amrex::Real* imw_new)
{
    imw_new[0] = 1822.8884868472639482; // E
    imw_new[1] = 0.0250312891113892;    // AR
    imw_new[2] = 0.4960317460317460;    // H2
    imw_new[3] = 0.0250312891113892;    // ARp
    imw_new[4] = 0.0125156445556946;    // Ar2p
    imw_new[5] = 0.9926036929001966;    // Hp
    imw_new[6] = 0.4961667594819952;    // H2p
    imw_new[7] = 0.3307478312219284;    // H3p
    imw_new[8] = 0.0244155810000000;    // ARHp
    imw_new[9] = 0.0250316328373182;    // ARm
    imw_new[10] = 0.0125156445556946;   // Ar2m
    imw_new[11] = 0.9920634920634921;   // H
    imw_new[12] = 0.4960317460317460;   // H2v1
    imw_new[13] = 0.4960317460317460;   // H2v2
    imw_new[14] = 0.4960317460317460;   // H2v3
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
    mw_new[0] = 0.000549;   // E
    mw_new[1] = 39.950000;  // AR
    mw_new[2] = 2.0160000;  // H2
    mw_new[3] = 39.950000;  // ARp
    mw_new[4] = 79.900000;  // AR2p
    mw_new[5] = 1.007451;   // Hp
    mw_new[6] = 2.015451;   // H2p
    mw_new[7] = 3.023451;   // H3p
    mw_new[8] = 40.957451;  // ARHp
    mw_new[9] = 39.949451;  // ARm
    mw_new[10] = 79.900000; // AR2m
    mw_new[11] = 1.008000;  // H
    mw_new[12] = 2.0160000; // H2v1
    mw_new[13] = 2.0160000; // H2v2
    mw_new[14] = 2.0160000; // H2v3
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

    amrex::Real N0 = 2.686763e25;

    // see page 131, Transport properties of gaseous ions III
    amrex::Real Arpmob = 1.85e-4 * N0 / ndens;

    // see page 133, Transport properties of gaseous ions III
    amrex::Real Ar2pmob = 1.835e-4 * N0 / ndens;

    // see page 184, Transport properties of gaseous ions I
    // found in He, so scaling to Ar using molar mass
    amrex::Real Hpmob = 30e-4 * N0 / ndens * 4.0 / 40.0;

    // see page 186, Transport properties of gaseous ions III
    // found in He, so scaling to Ar using molar mass
    amrex::Real H2pmob = 25.0e-4 * N0 / ndens * 4.0 / 40.0;

    // see page 187, Transport properties of gaseous ions I
    // found in He, so scaling to Ar using molar mass
    amrex::Real H3pmob = 32e-4 * N0 / ndens * 4.0 / 40.0;

    // see page 187, Transport properties of gaseous ions I
    // found in He, so scaling to Ar using molar mass
    amrex::Real ARHpmob = 19.0e-4 * N0 / ndens * 4.0 / 40;

    amrex::Real EbyN_cutoff = 0.1;
    amrex::Real EbyN = std::max(EbyN_cutoff, emag / ndens / 1e-21);
    amrex::Real meanE = 1.5 * Te / eV;

    // fit from bolsig run
    amrex::Real elecmob =
        (-1.1296602970297028e25 *
         std::exp(
             0.12562012487329025 * 0.4347714833797336 -
             2.911157190192704 * std::tanh(
                                     0.2885129277383643 * std::log(EbyN) +
                                     0.5430100185063174))) /
        ndens;
    amrex::Real mob = 0.0;

    if (specid == E_ID)
    {
        mob = elecmob;
    }
    if (specid == ARp_ID)
    {
        mob = Arpmob;
    }
    if (specid == AR2p_ID)
    {
        mob = Ar2pmob;
    }
    if (specid == Hp_ID)
    {
        mob = Hpmob;
    }
    if (specid == H2p_ID)
    {
        mob = H2pmob;
    }
    if (specid == H3p_ID)
    {
        mob = H3pmob;
    }
    if (specid == ARHp_ID)
    {
        mob = ARHpmob;
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
    amrex::Real elecmob = specMob(E_ID, Te, ndens, emag, T);
    amrex::Real Te_in_ev = Te / eV;
    amrex::Real Tg_in_ev = T / eV;
    amrex::Real dcoeff = 0.0;

    // special case
    /*Tg_in_ev=P_NTP/ndens/K_B/eV;

    if(specid==E_ID)
    {
        dcoeff=Te_in_ev*amrex::Math::abs(elecmob);
    }
    if(specid==ARp_ID || specid==AR2p_ID || specid==Hp_ID || specid==H2p_ID ||
    specid==H3p_ID || specid==ARHp_ID)
    {
        amrex::Real ionmob=specMob(specid,Te,ndens,emag,T);
        dcoeff=Tg_in_ev*amrex::Math::abs(ionmob);
    }
    if(specid==ARm_ID || specid==AR_ID)
    {
        amrex::Real ionmob=specMob(ARp_ID,Te,ndens,emag,T);
        dcoeff=Tg_in_ev*amrex::Math::abs(ionmob);
    }
    if(specid==AR2m_ID)
    {
        amrex::Real ionmob=specMob(AR2p_ID,Te,ndens,emag,T);
        dcoeff=Tg_in_ev*amrex::Math::abs(ionmob);
    }
    if(specid==H_ID)
    {
        amrex::Real ionmob=specMob(Hp_ID,Te,ndens,emag,T);
        dcoeff=Tg_in_ev*amrex::Math::abs(ionmob);
    }
    if(specid==H2v1_ID || specid==H2v2_ID || specid==H2v3_ID || specid==H2_ID)
    {
        amrex::Real ionmob=specMob(H2p_ID,Te,ndens,emag,T);
        dcoeff=Tg_in_ev*amrex::Math::abs(ionmob);
    }
    if(specid==NUM_SPECIES)
    {
        dcoeff=fivebythree*Te_in_ev*amrex::Math::abs(elecmob);
    }
    else //shouldnt come to this ideally
    {
        amrex::Real ionmob=specMob(ARp_ID,Te,ndens,emag,T);
        dcoeff=Tg_in_ev*amrex::Math::abs(ionmob);
    }*/

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
    amrex::Real EN,
    amrex::Real* enerExch)
{
    for (int i = 0; i < NUM_SPECIES; ++i)
    {
        wdot[i] = 0.0;
    }

    // Precalculating values for electron energy exchange evaluation
    amrex::Real Te_critical = 200; // eV
    amrex::Real ne = sc[E_ID] * N_A;
    amrex::Real Ue = 1.5 * Te * ne * K_B;
    amrex::Real Te_eV = Te / eV;
    amrex::Real TeCache1[4] = {
        std::log(Te_eV), 1.0 / Te_eV, 1.0 / (Te_eV * Te_eV),
        1.0 / (Te_eV * Te_eV * Te_eV)};
    amrex::Real TeCache2[6] = {
        1.0 / Te,
        1.0 / (Te * Te),
        1.0 / (Te * Te * Te),
        1.0 / (Te * Te * Te * Te),
        1.0 / (Te * Te * Te * Te * Te),
        1.0 / (Te * Te * Te * Te * Te * Te)};

    // NOTE: units of JANEV fits for electron impact rxns are cm3/s and must be
    // convered to m3/mol-s Precalculating values
    amrex::Real Janev_sum;
    amrex::Real logTe =
        log(Te_eV); // Fits are performed assuming Te is eV rather than K
    amrex::Real Te_pow[] = {pow(logTe, 0), pow(logTe, 1), pow(logTe, 2),
                            pow(logTe, 3), pow(logTe, 4), pow(logTe, 5),
                            pow(logTe, 6), pow(logTe, 7), pow(logTe, 8)};
    amrex::Real Jfit_coefs[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    amrex::Real Ffit_coefs[] = {0.0, 0.0, 0.0, 0.0};

    // reaction rates taken from
    // 1) Levko and Raja, Self-pulsing of direct-current discharge
    // in planar and curved geometries, J. Phys. D: Appl. Phys. 54 235201

    // but Dima made an error where he continued scaling rates by 1e6
    // even for third order and 1st order reactions
    //
    // so I fixed it by using another paper:
    // Ashish Sharma; Dmitry Levko; Laxminarayan L. Raja, Effect of oxygen
    // impurities on atmospheric-pressure surface streamer
    // discharge in argon for large gap arc breakdown, Phys. Plasmas
    // 23, 103501 (2016)

    {
        // reaction 1:  AR + E => ARm + E
        amrex::Real meanE = 1.5 * Te_eV;
        const amrex::Real k_f =
            std::exp(
                -37.44 + 1.407 * std::log(meanE) + 50.34 / meanE -
                348.6 / (meanE * meanE) - 230.5 / (meanE * meanE * meanE)) *
            N_A;
        // const amrex::Real k_f =
        // std::exp(-37.44+1.407*TeCache1[0]+50.34*TeCache1[1]
        //                                  -348.6*TeCache1[2]-230.5*TeCache1[3])*N_A;
        // const amrex::Real k_f =
        // 7074811070*std::pow(Te,0.046639)*std::exp(-1.3856e5/Te);
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[AR_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR_ID] -= qdot;
        wdot[ARm_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 11.6;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 2:  AR + E => AR+ + 2E
        amrex::Real meanE = 1.5 * Te_eV;
        // const amrex::Real k_f =
        // std::exp(14.37-8.641*TeCache1[0]-622.9*TeCache1[1]+
        //                                  5341.0*TeCache1[2]
        //                                  - 0.1862e5*TeCache1[3])*N_A;
        const amrex::Real k_f =
            std::exp(
                14.37 - 8.641 * std::log(meanE) - 622.9 / meanE +
                5341.0 / (meanE * meanE) - 0.1862e5 / (meanE * meanE * meanE)) *
            N_A;
        // const amrex::Real k_f =
        // 42581353.6*std::pow(Te,0.60983)*std::exp(-1.8712e5/Te);
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[AR_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR_ID] -= qdot;
        wdot[ARp_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 15.759;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 3:  ARm + E => AR+ + E + E
        const amrex::Real k_f =
            std::exp(
                -5.5e28 * TeCache2[5] + 5.08e24 * TeCache2[4] -
                1.85e20 * TeCache2[3] + 3.38e15 * TeCache2[2] -
                3.34e10 * TeCache2[1] + 1.39e5 * TeCache2[0] - 29.7) *
            N_A;
        // const amrex::Real k_f =
        // 75011786500*std::pow(Te,0.050382)*std::exp(-6.0524e4/Te);
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[ARm_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARm_ID] -= qdot;
        wdot[ARp_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 4.159;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 4:  ARm + ARm => E + Ar + Ar+
        const amrex::Real k_f = 5.0e-16 * N_A;
        const amrex::Real qf = k_f * (sc[ARm_ID] * sc[ARm_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARm_ID] -= 2.0 * qdot;
        wdot[ARp_ID] += qdot;
        wdot[AR_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 5:  ARm + E => AR + E
        const amrex::Real k_f = std::exp((-4.82e4 / Te) - 32.46) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[ARm_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARm_ID] -= qdot;
        wdot[AR_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 6:  AR2m + E => AR2+ + 2E
        const amrex::Real k_f =
            1.29e-16 * std::pow(Te, 0.7) * std::exp(-0.42456e5 / Te) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[AR2m_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR2m_ID] -= qdot;
        wdot[AR2p_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 4.2;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 7:  AR2m + E => AR + AR + E
        const amrex::Real k_f = 1e-13 * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[AR2m_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR2m_ID] -= qdot;
        wdot[AR_ID] += 2.0 * qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 8:  ARm + 2Ar => AR2m + Ar
        const amrex::Real k_f = 1.14e-44 * N_A * N_A;
        const amrex::Real qf =
            k_f * (sc[ARm_ID] * sc[AR_ID] * sc[AR_ID]); // third order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARm_ID] -= qdot;
        wdot[AR_ID] -= qdot;
        wdot[AR2m_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 9:  AR+ + 2Ar => AR2+ + Ar
        const amrex::Real k_f = 2.5e-43 * N_A * N_A;
        const amrex::Real qf =
            k_f * (sc[ARp_ID] * sc[AR_ID] * sc[AR_ID]); // third order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARp_ID] -= qdot;
        wdot[AR_ID] -= qdot;
        wdot[AR2p_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 10:  AR2m => 2Ar
        const amrex::Real k_f = 6e7;
        const amrex::Real qf = k_f * (sc[AR2m_ID]); // first order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR2m_ID] -= qdot;
        wdot[AR_ID] += 2.0 * qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 11:  2Ar2m => AR2+ + 2Ar + E
        const amrex::Real k_f = 5e-16 * N_A;
        const amrex::Real qf = k_f * (sc[AR2m_ID] * sc[AR2m_ID]); // second
                                                                  // order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR2m_ID] -= 2.0 * qdot;
        wdot[AR2p_ID] += qdot;
        wdot[AR_ID] += 2.0 * qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 12:  AR+ + e => ARm
        const amrex::Real k_f = 43e-18 * std::pow(Te, -0.5) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[ARp_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARp_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[ARm_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 13:  AR+ + 2e => ARm + e
        const amrex::Real k_f = 9.75e-21 * std::pow(Te, -4.5) * N_A * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[ARp_ID] * sc[E_ID] * sc[E_ID])
                                   : 0.0; // third order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARp_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[ARm_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 14:  AR2+ + e => ARm + Ar
        const amrex::Real k_f = 25.9e-12 * std::pow(Te, -0.66) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[AR2p_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR2p_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[ARm_ID] += qdot;
        wdot[AR_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 15:  H2 + e => 2H + e
        const amrex::Real k_f =
            8.4e-14 * std::pow(Te_eV, -0.45) * std::exp(-11.18 / Te_eV) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H2_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2_ID] -= qdot;
        wdot[H_ID] += 2.0 * qdot;
        int rxntype = 2;
        amrex::Real eexci = 14.68;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 16:  H + e => H+ + 2e
        const amrex::Real k_f =
            1.1e-14 * std::pow(Te_eV, 0.29) * std::exp(-15.28 / Te_eV) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H_ID] -= qdot;
        wdot[Hp_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 13.6;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 17:  H2 + e => H2+ + 2e
        const amrex::Real k_f =
            2.3e-14 * std::pow(Te_eV, 0.19) * std::exp(-17.87 / Te_eV) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H2_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2_ID] -= qdot;
        wdot[H2p_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 15.4;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 18:  H2 + e => H+ + H + 2e
        const amrex::Real k_f =
            9.4e-16 * std::pow(Te_eV, 0.45) * std::exp(-29.94 / Te_eV) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H2_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2_ID] -= qdot;
        wdot[Hp_ID] += qdot;
        wdot[H_ID] += qdot;
        wdot[E_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 28.08;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 19:  H2p + e => H+ + H + e
        const amrex::Real k_f = 1.5e-13 * std::exp(-1.97 / Te_eV) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H2p_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2p_ID] -= qdot;
        wdot[Hp_ID] += qdot;
        wdot[H_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 12.68;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 20:  ArH+ + e => Ar + H
        const amrex::Real k_f = 1e-15 * N_A;
        const amrex::Real qf = k_f * (sc[ARHp_ID] * sc[E_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARHp_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[AR_ID] += qdot;
        wdot[H_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 21:  H3+ + e => 3H
        const amrex::Real k_f = 2.8e-15 * std::pow(Te_eV, 0.48) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H3p_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H3p_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[H_ID] += 3.0 * qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 22:  H3+ + e => H2 + H
        const amrex::Real k_f = 1.6e-15 * std::pow(Te_eV, 0.48) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H3p_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H3p_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[H2_ID] += qdot;
        wdot[H_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 23:  H2+ + e  => 2H
        const amrex::Real k_f = 1.4e-15 * std::pow(Te_eV, 0.43) * N_A;
        const amrex::Real qf = (Te_eV < Te_critical)
                                   ? k_f * (sc[H2p_ID] * sc[E_ID])
                                   : 0.0; // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2p_ID] -= qdot;
        wdot[E_ID] -= qdot;
        wdot[H_ID] += 2.0 * qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 24:  H2+ + H2  => H3+ + H
        const amrex::Real k_f = 20e-16 * N_A;
        const amrex::Real qf = k_f * (sc[H2p_ID] * sc[H2_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2p_ID] -= qdot;
        wdot[H2_ID] -= qdot;
        wdot[H3p_ID] += qdot;
        wdot[H_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 25:  ArH+ + H2  => H3+ + Ar
        const amrex::Real k_f = 6.3e-16 * N_A;
        const amrex::Real qf = k_f * (sc[ARHp_ID] * sc[H2_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARHp_ID] -= qdot;
        wdot[H2_ID] -= qdot;
        wdot[H3p_ID] += qdot;
        wdot[AR_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 26:  H2+ + AR  => ARH+ + H
        const amrex::Real k_f = 21.0e-16 * N_A;
        const amrex::Real qf = k_f * (sc[H2p_ID] * sc[AR_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2p_ID] -= qdot;
        wdot[AR_ID] -= qdot;
        wdot[ARHp_ID] += qdot;
        wdot[H_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 27:  H2+ + AR  => AR+ + H2
        const amrex::Real k_f = 2.0e-16 * N_A;
        const amrex::Real qf = k_f * (sc[H2p_ID] * sc[AR_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2p_ID] -= qdot;
        wdot[AR_ID] -= qdot;
        wdot[ARp_ID] += qdot;
        wdot[H2_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 28:  H3+ + AR  => ARH+ + H2
        const amrex::Real k_f = 3.7e-16 * N_A;
        const amrex::Real qf = k_f * (sc[H3p_ID] * sc[AR_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H3p_ID] -= qdot;
        wdot[AR_ID] -= qdot;
        wdot[ARHp_ID] += qdot;
        wdot[H2_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 29:  AR+ + H2  => ARH+ + H
        const amrex::Real k_f = 8.7e-16 * N_A;
        const amrex::Real qf = k_f * (sc[ARp_ID] * sc[H2_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARp_ID] -= qdot;
        wdot[H2_ID] -= qdot;
        wdot[ARHp_ID] += qdot;
        wdot[H_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 30:  AR+ + H2  => H2+ + AR
        const amrex::Real k_f = 0.2e-16 * N_A;
        const amrex::Real qf = k_f * (sc[ARp_ID] * sc[H2_ID]); // second order
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[ARp_ID] -= qdot;
        wdot[H2_ID] -= qdot;
        wdot[H2p_ID] += qdot;
        wdot[AR_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 0.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 31:  E + H2 => E + H2v1
        Janev_sum = 0.0;
        amrex::Real k_f = 0;
        amrex::Real Jfit_coefs[] = {
            -0.714316367883707,   1.38885188323777,     -0.896682059878988,
            0.205691155420593,    -0.00126119079861499, -0.0175689637140865,
            0.000166269542345062, 0.00200527749681444,  -0.000311801565498550};
        amrex::Real Jfit_A = 3.08600000000000e-15;
        for (int j = 0; j < 9; j++) Janev_sum += Jfit_coefs[j] * Te_pow[j];
        k_f = Jfit_A * exp(Janev_sum) * 6.02214085774e23;
        const amrex::Real qf =
            (Te_eV < Te_critical) ? k_f * (sc[H2_ID] * sc[E_ID]) : 0.0;
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2_ID] -= qdot;
        wdot[H2v1_ID] += qdot;
        int rxntype = 1;
        amrex::Real eexci = 0.52;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 32:  E + H2 => E + H2v2
        Janev_sum = 0.0;
        amrex::Real k_f = 0;
        amrex::Real Jfit_coefs[] = {
            -1.37864601209690,   2.63046538866803,   -1.83776941997664,
            0.549377188986794,   -0.106802078818864, 0.0965961309765532,
            -0.0731567659573803, 0.0210105098161069, -0.00203785023617388};
        amrex::Real Jfit_A = 2.66000000000000e-16;
        for (int j = 0; j < 9; j++) Janev_sum += Jfit_coefs[j] * Te_pow[j];
        k_f = Jfit_A * exp(Janev_sum) * 6.02214085774e23;
        const amrex::Real qf =
            (Te_eV < Te_critical) ? k_f * (sc[H2_ID] * sc[E_ID]) : 0.0;
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2_ID] -= qdot;
        wdot[H2v2_ID] += qdot;
        int rxntype = 1;
        amrex::Real eexci = 1.0;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 33:  E + H2 => E + H2v3
        Janev_sum = 0.0;
        amrex::Real k_f = 0;
        amrex::Real Jfit_coefs[] = {
            -1.39618682450319,  2.93664468501587,    -2.57302798290394,
            1.20616796599908,   -0.262729536638480,  -0.0184867105308846,
            0.0101128565840887, 0.00147049502521815, -0.000430172779036224};
        amrex::Real Jfit_A = 2.55200000000000e-17;
        for (int j = 0; j < 9; j++) Janev_sum += Jfit_coefs[j] * Te_pow[j];
        k_f = Jfit_A * exp(Janev_sum) * 6.02214085774e23;
        const amrex::Real qf =
            (Te_eV < Te_critical) ? k_f * (sc[H2_ID] * sc[E_ID]) : 0.0;
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[H2_ID] -= qdot;
        wdot[H2v3_ID] += qdot;
        int rxntype = 1;
        amrex::Real eexci = 1.5;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    // Calculate electron energy loss via elastic collisions
    amrex::Real emag = 0.0;
    amrex::Real ndens = 0.0;
    for (int sp = 0; sp < NUM_SPECIES; sp++) ndens += sc[sp] * N_A;
    amrex::Real mu = specMob(E_ID, Te, ndens, emag, T);
    amrex::Real nu = -ECHARGE / ME / mu;
    amrex::Real elec_elastic_coll_term = 3.0 / 2.0 * K_B * sc[E_ID] * N_A *
                                         (Te - T) * nu *
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
    productionRate(wdot, C, T, Te, EN, ener_exch);
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCHRG(int kcharge[])
{
    kcharge[0] = -1; // E
    kcharge[1] = 0;  // AR
    kcharge[2] = 0;  // H2
    kcharge[3] = 1;  // ARp
    kcharge[4] = 1;  // AR2p
    kcharge[5] = 1;  // Hp
    kcharge[6] = 1;  // H2p
    kcharge[7] = 1;  // H3p
    kcharge[8] = 1;  // ARHp
    kcharge[9] = 0;  // ARm
    kcharge[10] = 0; // AR2m
    kcharge[11] = 0; // H
    kcharge[12] = 0; // H2v1
    kcharge[13] = 0; // H2v2
    kcharge[14] = 0; // H2v3
}

// compute electron temperature from EbyN using local field approximation
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
etempLFA(const amrex::Real Tg, amrex::Real C[], const amrex::Real EN)
{
    int i = 0;
    int loc = 0;
    int found = 0;
    amrex::Real EE_interp = 0.5 * eV;
    amrex::Real H2frac = (C[H2_ID] * 100) / (C[H2_ID] + C[AR_ID]);
    amrex::Real Te_interp = 0.5 * eV;
    amrex::Real Te_0 =
        ((EN <= 58) * ((1988.4 * std::log(EN)) + 4484.5) +
         (EN > 58) * ((0.557 * pow(EN, 2)) - (229.53 * EN) + 37527));
    amrex::Real Te_1 = ((EN <= 57) * ((12544 * std::log(EN)) + 27070)) +
                       ((EN > 57) * ((111.59 * EN) + 66328));
    amrex::Real Te_10 =
        (EN <= 75) * ((0.4521 * pow(EN, 3)) - (71.365 * pow(EN, 2)) +
                      (3734.8 * EN) + 2813.3) +
        ((EN > 75) * ((123.67 * EN) + 63197));
    amrex::Real Te_20 =
        ((EN <= 78) * ((0.1442 * pow(EN, 3)) - (33.62 * pow(EN, 2)) +
                       (2622.4 * EN) + 2046.5)) +
        ((EN > 78) * ((140.41 * EN) + 59398));
    amrex::Real Te_30 =
        ((EN <= 80.5) * ((-11.878 * pow(EN, 2)) + (1736.6 * EN) + 1867.7)) +
        ((EN > 80.5) * ((141.17 * EN) + 55543));
    amrex::Real Te_40 =
        ((EN <= 80) * ((-8.51 * pow(EN, 2)) + (1437.2 * EN) + 1681.8)) +
        ((EN > 80) * ((158.44 * EN) + 51161));
    amrex::Real Te_50 =
        ((EN <= 80) * ((-5.0614 * pow(EN, 2)) + (1159.5 * EN) + 1619.2)) +
        ((EN > 80) * ((0.0292 * pow(EN, 2)) + (145.56 * EN) + 52720));
    amrex::Real Te_60 =
        ((EN <= 80) * ((-3.14 * pow(EN, 2)) + (970.65 * EN) + 1667)) +
        ((EN > 80) * ((186.1 * EN) + 43274));
    amrex::Real Te_70 =
        ((EN <= 77) * ((-0.091 * pow(EN, 3)) + (8.4139 * pow(EN, 2)) +
                       (577.31 * EN) + 2200.7)) +
        ((EN > 77) * ((223.27 * EN) + 36056));
    amrex::Real Te_80 =
        ((EN <= 150) * ((-2.2332 * pow(EN, 2)) + (800.33 * EN) + 1497.6)) +
        ((EN > 150) * ((268.61 * EN) + 22420));

    if (H2frac >= 0 && H2frac <= 1)
    {
        Te_interp = Te_0 + (H2frac) * (Te_1 - Te_0);
    }
    if (H2frac > 1 && H2frac <= 10)
    {
        Te_interp = Te_1 + ((H2frac - 1) * (Te_10 - Te_1) / 9);
    }
    if (H2frac > 10 && H2frac <= 20)
    {
        Te_interp = Te_10 + ((H2frac - 10) * (Te_20 - Te_10) / 10);
    }
    if (H2frac > 20 && H2frac <= 30)
    {
        Te_interp = Te_20 + ((H2frac - 20) * (Te_30 - Te_20) / 10);
    }
    if (H2frac > 30 && H2frac <= 40)
    {
        Te_interp = Te_30 + ((H2frac - 30) * (Te_40 - Te_30) / 10);
    }
    if (H2frac > 40 && H2frac <= 50)
    {
        Te_interp = Te_40 + ((H2frac - 40) * (Te_50 - Te_40) / 10);
    }
    if (H2frac > 50 && H2frac <= 60)
    {
        Te_interp = Te_50 + ((H2frac - 50) * (Te_60 - Te_50) / 10);
    }
    if (H2frac > 60 && H2frac <= 70)
    {
        Te_interp = Te_60 + ((H2frac - 60) * (Te_70 - Te_60) / 10);
    }
    if (H2frac > 70 && H2frac <= 80)
    {
        Te_interp = Te_70 + ((H2frac - 70) * (Te_80 - Te_70) / 10);
    }
    if (H2frac > 80 && H2frac <= 100)
    {
        Te_interp = Te_80;
    }

    return (Te_interp * twothird);
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCHRGMASS(amrex::Real zk[])
{

    int kchrg[NUM_SPECIES];
    CKCHRG(kchrg);

    for (int id = 0; id < NUM_SPECIES; ++id)
    {
        zk[id] = N_A * ECHARGE * kchrg[id] * imw(id);
    }
}

#endif
