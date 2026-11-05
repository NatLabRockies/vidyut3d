#include "Chemistry.H"
const int rmap[1] = {0};

// Returns 0-based map of reaction order
void GET_RMAP(int* _rmap)
{
    for (int j = 0; j < 1; ++j)
    {
        _rmap[j] = rmap[j];
    }
}

// Returns a count of species in a reaction, and their indices
// and stoichiometric coefficients. (Eq 50)
void CKINU(const int i, int& nspec, int ki[], int nu[])
{
    const int ns[1] = {4};
    const int kiv[4] = {1, 0, 1, 0};
    const int nuv[4] = {-1, -1, 1, 1};
    if (i < 1)
    {
        // Return max num species per reaction
        nspec = 4;
    } else
    {
        if (i > 1)
        {
            nspec = -1;
        } else
        {
            nspec = ns[i - 1];
            for (int j = 0; j < nspec; ++j)
            {
                ki[j] = kiv[(i - 1) * 4 + j] + 1;
                nu[j] = nuv[(i - 1) * 4 + j];
            }
        }
    }
}

// save atomic weights into array
void atomicWeight(amrex::Real* awt)
{
    awt[0] = 0.000549;  // E
    awt[1] = 39.950000; // Ar
}

// get atomic weight for all elements
void CKAWT(amrex::Real* awt) { atomicWeight(awt); }

// Returns the elemental composition
// of the speciesi (mdim is num of elements)
void CKNCF(int* ncf)
{
    int kd = 2;
    // Zero ncf
    for (int id = 0; id < kd * 2; ++id)
    {
        ncf[id] = 0;
    }

    // E
    ncf[0 * kd + 0] = 1; // E

    // NI
    ncf[1 * kd + 1] = 1; // Ar
}

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
    kname.resize(2);
    kname[0] = "E";
    kname[1] = "NI";
}
