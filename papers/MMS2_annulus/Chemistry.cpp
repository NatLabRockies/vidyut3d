#include "Chemistry.H"

// save atomic weights into array
void atomicWeight(amrex::Real* awt)
{
    awt[0] = 0.000549; // E
    awt[1] = 4.002602; // He
}

// get atomic weight for all elements
void CKAWT(amrex::Real* awt) { atomicWeight(awt); }

// Returns the elemental composition
// of the speciesi (mdim is num of elements)
void CKNCF(int* ncf)
{
    int kd = 2;
    // Zero ncf
    for (int id = 0; id < kd * 3; ++id)
    {
        ncf[id] = 0;
    }

    // E
    ncf[0 * kd + 0] = 1; // E

    // HE
    ncf[1 * kd + 1] = 1; // He

    // HEp
    ncf[2 * kd + 1] = 1; // He
}

// Returns the vector of strings of element names
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(2);
    ename[0] = "E";
    ename[1] = "He";
}

// Returns the vector of strings of species names
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(3);
    kname[0] = "E";
    kname[1] = "HE";
    kname[2] = "HEp";
}
