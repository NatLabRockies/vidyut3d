#include "Chemistry.H"

// save atomic weights into array
void atomicWeight(amrex::Real* awt)
{
    awt[0] = 0.000549;  // E
    awt[1] = 39.950000; // Ar
}

// get atomic weight for all elements
void CKAWT(amrex::Real* awt) { atomicWeight(awt); }

// Returns the vector of strings of element names
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(2);
    ename[0] = "E";
    ename[1] = "Ar";
}

// Returns the vector of strings of species names
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(6);
    kname[0] = "E";
    kname[1] = "AR";
    kname[2] = "ARp";
    kname[3] = "AR2p";
    kname[4] = "ARm";
    kname[5] = "AR2m";
}
