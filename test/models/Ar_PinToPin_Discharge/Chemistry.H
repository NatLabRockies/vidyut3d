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
#define ARm_ID 3

#define NUM_SPECIES 4
#define FIRST_ION 2
#define NUM_IONS 1
#define FIRST_NEUTRAL 3
#define NUM_NEUTRALS 1
#define LFA_NDATA 101

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

// work on gpu
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real redef_ee_array_pureAr[LFA_NDATA, 2] = {
    {1.0, 2.407},    {1.089, 2.514},  {1.186, 2.627},  {1.291, 2.745},
    {1.406, 2.868},  {1.531, 2.996},  {1.667, 3.129},  {1.815, 3.267},
    {1.977, 3.409},  {2.152, 3.556},  {2.344, 3.708},  {2.552, 3.864},
    {2.779, 4.025},  {3.026, 4.19},   {3.295, 4.355},  {3.588, 4.516},
    {3.907, 4.668},  {4.254, 4.803},  {4.632, 4.919},  {5.044, 5.015},
    {5.493, 5.092},  {5.981, 5.153},  {6.513, 5.203},  {7.092, 5.243},
    {7.722, 5.279},  {8.409, 5.31},   {9.157, 5.339},  {9.971, 5.368},
    {10.86, 5.397},  {11.82, 5.426},  {12.87, 5.457},  {14.02, 5.489},
    {15.26, 5.523},  {16.62, 5.559},  {18.1, 5.597},   {19.71, 5.638},
    {21.46, 5.681},  {23.37, 5.727},  {25.45, 5.775},  {27.71, 5.826},
    {30.17, 5.88},   {32.85, 5.936},  {35.77, 5.996},  {38.95, 6.059},
    {42.42, 6.124},  {46.19, 6.192},  {50.3, 6.263},   {54.77, 6.337},
    {59.64, 6.413},  {64.94, 6.493},  {70.71, 6.575},  {77.0, 6.66},
    {83.84, 6.748},  {91.3, 6.84},    {99.41, 6.936},  {108.3, 7.035},
    {117.9, 7.138},  {128.4, 7.246},  {139.8, 7.359},  {152.2, 7.478},
    {165.7, 7.603},  {180.5, 7.735},  {196.5, 7.874},  {214.0, 8.023},
    {233.0, 8.182},  {253.7, 8.351},  {276.3, 8.535},  {300.8, 8.732},
    {327.6, 8.946},  {356.7, 9.18},   {388.4, 9.435},  {422.9, 9.715},
    {460.5, 10.02},  {501.5, 10.37},  {546.1, 10.74},  {594.6, 11.17},
    {647.5, 11.64},  {705.0, 12.16},  {767.7, 12.75},  {836.0, 13.42},
    {910.3, 14.16},  {991.2, 15.0},   {1079.0, 15.95}, {1175.0, 17.02},
    {1280.0, 18.22}, {1394.0, 19.57}, {1517.0, 21.1},  {1652.0, 22.81},
    {1799.0, 24.74}, {1959.0, 26.9},  {2133.0, 29.33}, {2323.0, 32.06},
    {2530.0, 35.13}, {2754.0, 38.58}, {2999.0, 42.47}, {3266.0, 46.86},
    {3556.0, 51.82}, {3873.0, 57.44}, {4217.0, 63.84}, {4592.0, 71.15},
    {5000.0, 79.53}};
#endif

const amrex::Real h_redef_ee_array_pureAr[LFA_NDATA][2] = {
    {1.0, 2.407},    {1.089, 2.514},  {1.186, 2.627},  {1.291, 2.745},
    {1.406, 2.868},  {1.531, 2.996},  {1.667, 3.129},  {1.815, 3.267},
    {1.977, 3.409},  {2.152, 3.556},  {2.344, 3.708},  {2.552, 3.864},
    {2.779, 4.025},  {3.026, 4.19},   {3.295, 4.355},  {3.588, 4.516},
    {3.907, 4.668},  {4.254, 4.803},  {4.632, 4.919},  {5.044, 5.015},
    {5.493, 5.092},  {5.981, 5.153},  {6.513, 5.203},  {7.092, 5.243},
    {7.722, 5.279},  {8.409, 5.31},   {9.157, 5.339},  {9.971, 5.368},
    {10.86, 5.397},  {11.82, 5.426},  {12.87, 5.457},  {14.02, 5.489},
    {15.26, 5.523},  {16.62, 5.559},  {18.1, 5.597},   {19.71, 5.638},
    {21.46, 5.681},  {23.37, 5.727},  {25.45, 5.775},  {27.71, 5.826},
    {30.17, 5.88},   {32.85, 5.936},  {35.77, 5.996},  {38.95, 6.059},
    {42.42, 6.124},  {46.19, 6.192},  {50.3, 6.263},   {54.77, 6.337},
    {59.64, 6.413},  {64.94, 6.493},  {70.71, 6.575},  {77.0, 6.66},
    {83.84, 6.748},  {91.3, 6.84},    {99.41, 6.936},  {108.3, 7.035},
    {117.9, 7.138},  {128.4, 7.246},  {139.8, 7.359},  {152.2, 7.478},
    {165.7, 7.603},  {180.5, 7.735},  {196.5, 7.874},  {214.0, 8.023},
    {233.0, 8.182},  {253.7, 8.351},  {276.3, 8.535},  {300.8, 8.732},
    {327.6, 8.946},  {356.7, 9.18},   {388.4, 9.435},  {422.9, 9.715},
    {460.5, 10.02},  {501.5, 10.37},  {546.1, 10.74},  {594.6, 11.17},
    {647.5, 11.64},  {705.0, 12.16},  {767.7, 12.75},  {836.0, 13.42},
    {910.3, 14.16},  {991.2, 15.0},   {1079.0, 15.95}, {1175.0, 17.02},
    {1280.0, 18.22}, {1394.0, 19.57}, {1517.0, 21.1},  {1652.0, 22.81},
    {1799.0, 24.74}, {1959.0, 26.9},  {2133.0, 29.33}, {2323.0, 32.06},
    {2530.0, 35.13}, {2754.0, 38.58}, {2999.0, 42.47}, {3266.0, 46.86},
    {3556.0, 51.82}, {3873.0, 57.44}, {4217.0, 63.84}, {4592.0, 71.15},
    {5000.0, 79.53}};

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[NUM_SPECIES] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.0250312891113892,    // ARp
    0.0250316328373182     // ARm
};
#endif
const amrex::Real h_global_imw[NUM_SPECIES] = {
    1822.8884868472639482, // E
    0.0250312891113892,    // AR
    0.0250312891113892,    // ARp
    0.0250316328373182,    // ARm
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[NUM_SPECIES] = {
    0.000549,  // E
    39.950000, // AR
    39.950000, // ARp
    39.949451  // ARm
};
#endif
const amrex::Real h_global_mw[NUM_SPECIES] = {
    0.000549,  // E
    39.950000, // AR
    39.950000, // ARp
    39.949451  // ARm
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void get_imw(amrex::Real* imw_new)
{
    imw_new[0] = 1822.8884868472639482; // E
    imw_new[1] = 0.0250312891113892;    // AR
    imw_new[2] = 0.0250312891113892;    // ARp
    imw_new[3] = 0.0250316328373182;    // ARm
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
    mw_new[3] = 39.949451; // ARm
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

    amrex::Real EbyN_cutoff = 1.0;
    amrex::Real EbyN = std::max(EbyN_cutoff, emag / ndens / 1e-21);
    amrex::Real meanE = 1.5 * Te / eV;
    amrex::Real ENmax = 1000.0;
    amrex::Real EN = std::min(EbyN, ENmax);

    amrex::Real elecmob =
        (9.706366336633663e+23 *
         (-0.11212465418903933 * std::log(EN) + 1.378598654615419 +
          -1.5695108175444188 / std::pow(EN, 1.0) +
          7.209283524569783 / std::pow(EN, 2.0) +
          -3.7925576113527955 / std::pow(EN, 3.0)));
    elecmob /= ndens;
    elecmob *= -1.0;

    amrex::Real mob = 0.0;

    if (specid == E_ID)
    {
        mob = elecmob;
    } else if (specid == ARp_ID)
    {
        mob = Arpmob;
    } else if (specid == NUM_SPECIES)
    {
        mob = fivebythree * elecmob;
    } else // neutrals
    {
        mob = 0.0;
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
    amrex::Real Te_in_ev = Te / eV;
    amrex::Real Tg_in_ev = T / eV;
    amrex::Real dcoeff = 0.0;
    amrex::Real elecdiff_fac = 100.0;
    amrex::Real EbyNcutoff = 500.0;

    // special case
    Tg_in_ev = P_NTP / ndens / K_B / eV;

    amrex::Real EbyN_cutoff = 0.1;
    amrex::Real EbyN = std::max(EbyN_cutoff, emag / ndens / 1e-21);
    amrex::Real ENmax = 1000.0;

    amrex::Real EN = std::min(EbyN, ENmax);
    amrex::Real elecdcoeff =
        (8.633653465346534e+24 *
         (-0.10198569959391364 * std::log(EN) + 1.2075008258799453 +
          0.00025677950087749026 * EN +
          5.578616426000813e-08 * std::pow(EN, 2.0) +
          -3.0983668588271214e-12 * std::pow(EN, 3.0)));
    elecdcoeff /= ndens;

    if (specid == E_ID)
    {
        dcoeff = elecdcoeff;
    } else if (specid == ARp_ID || specid == ARm_ID)
    {
        amrex::Real ionmob = specMob(specid, Te, ndens, emag, T);
        dcoeff = Tg_in_ev * amrex::Math::abs(ionmob);
    } else if (specid == NUM_SPECIES)
    {
        // dcoeff = fivebythree * Te_in_ev * amrex::Math::abs(elecmob);
        dcoeff = fivebythree * elecdcoeff;
    } else
    {
        // shouldnt come to this!
        dcoeff = 1e-10;
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
    amrex::Real rawEN,
    amrex::Real* enerExch)
{
    amrex::Real Te_critical = 200; // eV
    amrex::Real ne = sc[E_ID] * N_A;
    amrex::Real Ue = 1.5 * Te * ne * K_B;
    amrex::Real Te_eV = Te / eV;

    for (int i = 0; i < NUM_SPECIES; ++i)
    {
        wdot[i] = 0.0;
    }

    amrex::Real EN = std::max(1.0, rawEN);
    amrex::Real ENmax = 1000.0;
    EN = std::min(EN, ENmax);

    {
        // reaction 1:  AR + E => ARm + E
        const amrex::Real k_f =
            (6.224183655903354e-15 *
             (6.5108314064046695 * 0.5771372579711347 +
              3.7542423540822867 *
                  std::tanh(
                      0.7888478214798701 * std::log(EN) + -5.941773262593229)));

        const amrex::Real qf = k_f * N_A * (sc[AR_ID] * sc[E_ID]);
        const amrex::Real qr = 0.0;
        const amrex::Real qdot = qf - qr;
        wdot[AR_ID] -= qdot;
        wdot[ARm_ID] += qdot;
        int rxntype = 2;
        amrex::Real eexci = 11.50;
        int elidx = 0;
        comp_ener_exch(
            qf, qr, sc, k_f, rxntype, eexci, elidx, enerExch, Ue, T, Te);
    }

    {
        // reaction 2:  AR + E => AR+ + 2E
        const amrex::Real k_f =
            (1.1276223438961332e-14 *
             (-0.0064894826249676605 + 4.374690108699521e-05 * EN +
              1.6079679961847445e-06 * std::pow(EN, 2.0) +
              -4.192936668250674e-10 * std::pow(EN, 3.0) +
              3.453758772756147e-14 * std::pow(EN, 4.0)));
        const amrex::Real qf = k_f * N_A * (sc[AR_ID] * sc[E_ID]);
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
    kcharge[E_ID] = -1;  // E
    kcharge[AR_ID] = 0;  // AR
    kcharge[ARp_ID] = 1; // ARp
    kcharge[ARm_ID] = 0; // ARm
}

// compute electron temperature from EbyN using local field approximation
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real etempLFApure(
    const amrex::Real Tg,
    amrex::Real C[],
    const amrex::Real EN_in,
    const amrex::Real h_redef_ee_array[LFA_NDATA][2])
{
    int i = 0;
    int loc = 0;
    int found = 0;
    amrex::Real EE_interp = 0.5 * eV;
    amrex::Real ENmax = 1000.0;
    amrex::Real EN = std::min(amrex::Math::abs(EN_in), ENmax);

    for (i = 0; i < LFA_NDATA; i++)
    {
        if (h_redef_ee_array[i][0] > EN)
        {
            found = 1;
            break;
        }
    }

    loc = std::min(LFA_NDATA - 1, i);

    if (loc == 0)
    {
        EE_interp =
            h_redef_ee_array[loc][1] +
            (h_redef_ee_array[loc + 1][1] - h_redef_ee_array[loc][1]) /
                (h_redef_ee_array[loc + 1][0] - h_redef_ee_array[loc][0]) *
                (EN - h_redef_ee_array[loc][0]);
    } else
    {
        EE_interp =
            h_redef_ee_array[loc][1] +
            (h_redef_ee_array[loc][1] - h_redef_ee_array[loc - 1][1]) /
                (h_redef_ee_array[loc][0] - h_redef_ee_array[loc - 1][0]) *
                (EN - h_redef_ee_array[loc][0]);
    }

    return (EE_interp * twothird * eV);
}

// compute electron temperature from EbyN using local field approximation
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
etempLFA(const amrex::Real Tg, amrex::Real C[], const amrex::Real EN_in)
{
    amrex::Real Te_pureAr = etempLFApure(Tg, C, EN_in, h_redef_ee_array_pureAr);
    return (Te_pureAr);
    // return(Te_pureAr);
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
