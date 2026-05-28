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
#define ARp_ID 2
#define AR2p_ID 3
#define ARm_ID 4
#define AR2m_ID 5

#define NUM_SPECIES 6
#define FIRST_ION 2
#define NUM_IONS 2
#define FIRST_NEUTRAL 4
#define NUM_NEUTRALS 2

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

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[NUM_SPECIES] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.0250312891113892,    // ARp
    0.0125156445556946,    // Ar2p
    0.0250316328373182,    // ARm
    0.0125156445556946     // Ar2m
};
#endif
const amrex::Real h_global_imw[NUM_SPECIES] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.0250312891113892,    // ARp
    0.0125156445556946,    // Ar2p
    0.0250316328373182,    // ARm
    0.0125156445556946     // Ar2m
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[NUM_SPECIES] = {
    0.000549,  // E
    39.950000, // AR
    39.950000, // ARp
    79.900000, // AR2p
    39.949451, // ARm
    79.900000  // AR2m
};
#endif
const amrex::Real h_global_mw[NUM_SPECIES] = {
    0.000549,  // E
    39.950000, // AR
    39.950000, // ARp
    79.900000, // AR2p
    39.949451, // ARm
    79.900000  // AR2m
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_imw(amrex::Real* imw_new)
{
    imw_new[0] = 1822.8884868472639482; // E
    imw_new[1] = 0.0250312891113892;    // AR
    imw_new[2] = 0.0250312891113892;    // ARp
    imw_new[3] = 0.0125156445556946;    // AR2p
    imw_new[4] = 0.0250312891113892;    // ARm
    imw_new[5] = 0.0125156445556946;    // AR2m
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
    mw_new[2] = 39.950000; // ARp
    mw_new[3] = 79.900000; // AR2p
    mw_new[4] = 39.950000; // ARm
    mw_new[5] = 79.900000; // AR2m
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
    amrex::Real EbyN_cutoff = 0.1;
    amrex::Real EbyN = std::max(EbyN_cutoff, emag / ndens / 1e-21);
    amrex::Real meanE = 1.5 * Te / eV;
    amrex::Real cm2_to_m2 = 1e-4;
    amrex::Real cm_to_m = 1e-2;
    int use_boeuf_pitchford_model = 0;
    amrex::Real elecmob, Arpmob, Ar2pmob;
    amrex::Real pres_in_torr = ndens * K_B * T / 133.3223;

    // from Boeuf and Pitchford paper
    // Boeuf, J. P., and L. C. Pitchford. "Two-dimensional model of a
    // capacitively coupled rf discharge and comparisons with experiments in the
    // Gaseous Electronics Conference reference reactor." Physical Review E 51.2
    // (1995): 1376.
    if (use_boeuf_pitchford_model)
    {
        elecmob = -3e5 / pres_in_torr * cm2_to_m2;
        amrex::Real Ebyp = emag / pres_in_torr * cm_to_m; // units of V/(cm
                                                          // torr)
        if (Ebyp < 60.0)
        {
            Arpmob = 1e3 * (1.0 - 2.22e-3 * Ebyp) / pres_in_torr * cm2_to_m2;
            Ar2pmob = Arpmob;
        } else
        {
            Arpmob = 8.25e3 / std::sqrt(Ebyp) *
                     (1 - 86.52 / std::pow(Ebyp, 1.5)) / pres_in_torr *
                     cm2_to_m2;
            Ar2pmob = Arpmob;
        }
    } else
    {
        // fit from bolsig run
        elecmob =
            (-1.1296602970297028e25 *
             std::exp(
                 0.12562012487329025 * 0.4347714833797336 -
                 2.911157190192704 * std::tanh(
                                         0.2885129277383643 * std::log(EbyN) +
                                         0.5430100185063174))) /
            ndens;

        // see page 131, Transport properties of gaseous ions III
        Arpmob = 1.85e-4 * N0 / ndens;
        // see page 133, Transport properties of gaseous ions III
        Ar2pmob = 1.835e-4 * N0 / ndens;
    }

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

    if (specid == E_ID)
    {
        dcoeff = Te_in_ev * amrex::Math::abs(elecmob);
    }
    if (specid == ARp_ID || specid == AR2p_ID)
    {
        amrex::Real ionmob = specMob(specid, Te, ndens, emag, T);
        dcoeff = Tg_in_ev * amrex::Math::abs(ionmob);
    }
    if (specid == ARm_ID)
    {
        amrex::Real ionmob = specMob(ARp_ID, Te, ndens, emag, T);
        dcoeff = Tg_in_ev * amrex::Math::abs(ionmob);
    }
    if (specid == AR2m_ID)
    {
        amrex::Real ionmob = specMob(AR2p_ID, Te, ndens, emag, T);
        dcoeff = Tg_in_ev * amrex::Math::abs(ionmob);
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
    amrex::Real EN,
    amrex::Real* enerExch)
{
    for (int i = 0; i < NUM_SPECIES; ++i)
    {
        wdot[i] = 0.0;
    }

    // Precalculating values for electron energy exchange evaluation
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
    amrex::Real invTe = (Te == 0) ? 1.0 : 1.0 / Te;
    amrex::Real TeeV = Te / 11595.0;
    amrex::Real logTe =
        log(TeeV); // Fits are performed assuming Te is eV rather than K
    amrex::Real invTeeV = (Te == 0) ? 1.0 : 1.0 / (TeeV);
    amrex::Real Te_pow[] = {
        1.0,
        logTe,
        amrex::Math::powi<2>(logTe),
        amrex::Math::powi<3>(logTe),
        amrex::Math::powi<4>(logTe),
        amrex::Math::powi<5>(logTe),
        amrex::Math::powi<6>(logTe),
        amrex::Math::powi<7>(logTe),
        amrex::Math::powi<8>(logTe)};
    amrex::Real invTe_pow[] = {
        invTeeV, amrex::Math::powi<2>(invTeeV), amrex::Math::powi<3>(invTeeV),
        amrex::Math::powi<4>(invTeeV)};

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
    //
    // Also, it looks like the paper wrongly used Te instead of
    // mean energy (1.5*Te) for ionization and excitation
    // The fits Dima had were off
    // probably because the fit was done with BOLSIG data
    // which gives rate constants as function of mean energy (1.5*Te)
    // rather than Te

    {
        // reaction 1:  AR + E => ARm + E
        amrex::Real meanE = 1.5 * Te_eV;
        const amrex::Real k_f =
            std::exp(
                -37.44 + 1.407 * std::log(meanE) + 50.34 / meanE -
                348.6 / (meanE * meanE) - 230.5 / (meanE * meanE * meanE)) *
            N_A;
        // const amrex::Real k_f =
        // 7074811070*std::pow(Te,0.046639)*std::exp(-1.3856e5/Te);
        const amrex::Real qf = k_f * (sc[AR_ID] * sc[E_ID]); // second order
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
        amrex::Real k_f;
        amrex::Real meanE = 1.5 * Te_eV;
        k_f =
            std::exp(
                14.37 - 8.641 * std::log(meanE) - 622.9 / meanE +
                5341.0 / (meanE * meanE) - 0.1862e5 / (meanE * meanE * meanE)) *
            N_A;
        // const amrex::Real k_f =
        // 42581353.6*std::pow(Te,0.60983)*std::exp(-1.8712e5/Te);
        /*amrex::Real Janev_sum=0.0;
        if(TeeV < 8.0)
        {
            Jfit_coefs = {-1951.40116682516, 9125.28938191463,
        -19064.2751720459, 22763.7019090326, -16855.4371729404,
        7904.47881474249, -2291.72587558317, 375.740475221758,
        -26.6926001068518}; amrex::Real Jfit_A = 1.42900000000000e-14; for(int j
        = 0; j<9; j++) Janev_sum += Jfit_coefs[j] * Te_pow[j]; k_f = Jfit_A *
        exp(Janev_sum) * 6.02214085774e23;
        }
        else
        {
            Ffit_coefs = {-77.9510931269820, 533.060249822779,
        -2091.33857389100, 1893.83145115372}; amrex::Real Ffit_A
        = 1.13390537981533e-12; for(int j = 0; j<4; j++) Janev_sum +=
        Ffit_coefs[j] * invTe_pow[j]; k_f = Ffit_A * exp(Janev_sum)
        * 6.02214085774e23;
        }*/
        // k_f=1.5e-14*std::exp(-5.0/Te_eV-100.0/(Te_eV*Te_eV)-325.0/(Te_eV*Te_eV*Te_eV))*std::pow(Te_eV,0.75)*N_A;
        // k_f=1.2e-13*std::exp(-5.0/Te_eV-125.0/(Te_eV*Te_eV)-500.0/(Te_eV*Te_eV*Te_eV))*N_A;
        const amrex::Real qf = k_f * (sc[AR_ID] * sc[E_ID]); // second order
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
        const amrex::Real qf = k_f * (sc[ARm_ID] * sc[E_ID]); // second order
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
        const amrex::Real qf = k_f * (sc[ARm_ID] * sc[E_ID]); // second order
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
        const amrex::Real qf = k_f * (sc[AR2m_ID] * sc[E_ID]); // second order
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
        const amrex::Real qf = k_f * (sc[AR2m_ID] * sc[E_ID]); // second order
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
        const amrex::Real qf = k_f * (sc[ARp_ID] * sc[E_ID]); // second order
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
        const amrex::Real qf =
            k_f * (sc[ARp_ID] * sc[E_ID] * sc[E_ID]); // third order
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
        const amrex::Real qf = k_f * (sc[AR2p_ID] * sc[E_ID]); // second order
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
    kcharge[2] = 1;  // ARp
    kcharge[3] = 1;  // AR2p
    kcharge[4] = 0;  // ARm
    kcharge[5] = 0;  // AR2m
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

    int kchrg[NUM_SPECIES];
    CKCHRG(kchrg);

    for (int id = 0; id < NUM_SPECIES; ++id)
    {
        zk[id] = N_A * ECHARGE * kchrg[id] * imw(id);
    }
}

#endif
