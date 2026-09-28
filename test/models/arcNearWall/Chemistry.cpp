#include "Chemistry.H"

// save atomic weights into array
void atomicWeight(amrex::Real *  awt)
{
    awt[0] = 0.000549; // E
    awt[1] = 39.950000; // Ar
    awt[2] = 1.008; // Ar
}

// get atomic weight for all elements
void CKAWT( amrex::Real *  awt)
{
    atomicWeight(awt);
}

// Returns the vector of strings of element names
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(2);
    ename[0] = "E";
    ename[1] = "Ar";
    ename[2] = "H";
}

// Returns the vector of strings of species names
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(NUM_SPECIES);
    kname[E_ID] = "E";
    kname[AR_ID] = "AR";
    kname[H2_ID] = "H2";
    kname[ARp_ID] = "ARp";
    kname[AR2p_ID] = "AR2p";
    kname[Hp_ID] = "Hp";
    kname[H2p_ID] = "H2p";
    kname[H3p_ID] = "H3p";
    kname[ARHp_ID] = "ARHp";
    kname[ARm_ID] = "ARm";
    kname[AR2m_ID] = "AR2m";
    kname[H_ID] = "H";
    kname[H2v1_ID] = "H2v1";
    kname[H2v2_ID] = "H2v2";
    kname[H2v3_ID] = "H2v3";
}
