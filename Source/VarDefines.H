#ifndef Vidyut_vardefines_H_
#define Vidyut_vardefines_H_
#include <Chemistry.H>

#define EEN_ID (NUM_SPECIES)
#define ETEMP_ID (NUM_SPECIES + 1)
#define POT_ID (NUM_SPECIES + 2)
#define EFX_ID (NUM_SPECIES + 3)
#define EFY_ID (NUM_SPECIES + 4)
#define EFZ_ID (NUM_SPECIES + 5)
#define EJH_ID (NUM_SPECIES + 6) // electron joule heating
#define EIH_ID (NUM_SPECIES + 7) // electron inelastic heating
#define EEH_ID (NUM_SPECIES + 8) // electron elastic heating
#define REF_ID (NUM_SPECIES + 9)
#define SRFCH_ID (NUM_SPECIES + 10)
#define PHOTO_ION_SRC_ID (NUM_SPECIES + 11)
#define CMASK_ID (NUM_SPECIES + 12)
#define ECURX_ID (NUM_SPECIES + 13)
#define ECURY_ID (NUM_SPECIES + 14)
#define ECURZ_ID (NUM_SPECIES + 15)
#define ICURX_ID (NUM_SPECIES + 16)
#define ICURY_ID (NUM_SPECIES + 17)
#define ICURZ_ID (NUM_SPECIES + 18)
#define DCURX_ID (NUM_SPECIES + 19)
#define DCURY_ID (NUM_SPECIES + 20)
#define DCURZ_ID (NUM_SPECIES + 21)
#ifdef AMREX_USE_EB
#define CPX_ID (NUM_SPECIES + 22)
#define CPY_ID (NUM_SPECIES + 23)
#define CPZ_ID (NUM_SPECIES + 24)
#define NX_ID (NUM_SPECIES + 25)
#define NY_ID (NUM_SPECIES + 26)
#define NZ_ID (NUM_SPECIES + 27)
#define NUM_PLASMAVARS (NZ_ID - NUM_SPECIES + 1)
#else
#define NUM_PLASMAVARS (DCURZ_ID - NUM_SPECIES + 1)
#endif
#define NVAR (NUM_SPECIES + NUM_PLASMAVARS)

#define PERBC 0
#define DIRCBC 1
#define HNEUBC 2
#define IHNEUBC 3
#define ROBINBC 4
#define AXISBC 5

#define POTEQ_TYPE 0
#define PHOTEQ_TYPE 1
#define SPECEQ_TYPE 2

#define MAX_CURRENT_LOCS 10

#endif
