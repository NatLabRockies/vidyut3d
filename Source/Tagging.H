#ifndef VIDYUT_TAGGING_H
#define VIDYUT_TAGGING_H

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void state_based_refinement(
    int i,
    int j,
    int k,
    amrex::Array4<char> const& tag,
    amrex::Array4<amrex::Real const> const& state,
    amrex::Real* refine_phi,
    int* refine_phi_comps,
    int ntagvars,
    char tagval)
{

    for (int c = 0; c < ntagvars; c++)
    {
        int comp = refine_phi_comps[c];
        if (std::abs(state(i, j, k, comp)) > refine_phi[c])
        {
            tag(i, j, k) = tagval;
        }
    }
}

// Tag the immersed-boundary cut cells, i.e. the cells the boundary passes
// through, identified by a volume fraction strictly between vfrac_lo and
// vfrac_hi. The geometry does not move, so the tagged set is the same at
// every regrid and the refinement is static. Widen the refined band into the
// fluid with amr.n_error_buf.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void cutcell_based_refinement(
    int i,
    int j,
    int k,
    amrex::Array4<char> const& tag,
    amrex::Array4<amrex::Real const> const& state,
    int cmask_comp,
    amrex::Real vfrac_lo,
    amrex::Real vfrac_hi,
    char tagval)
{
    const amrex::Real vfrac = state(i, j, k, cmask_comp);
    if (vfrac > vfrac_lo && vfrac < vfrac_hi)
    {
        tag(i, j, k) = tagval;
    }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void stategrad_based_refinement(
    int i,
    int j,
    int k,
    amrex::Array4<char> const& tag,
    amrex::Array4<amrex::Real const> const& state,
    amrex::Real* refine_phigrad,
    int* refine_phi_comps,
    int ntagvars,
    char tagval)
{

    for (int c = 0; c < ntagvars; c++)
    {
        int comp = refine_phi_comps[c];

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gradv = {{0.0}};

        amrex::Real gradxplus =
            std::abs(state(i + 1, j, k, comp) - state(i, j, k, comp));
        amrex::Real gradxminus =
            std::abs(state(i, j, k, comp) - state(i - 1, j, k, comp));
        gradv[0] = amrex::max(gradxplus, gradxminus);
#if AMREX_SPACEDIM > 1
        amrex::Real gradyplus =
            std::abs(state(i, j + 1, k, comp) - state(i, j, k, comp));
        amrex::Real gradyminus =
            std::abs(state(i, j, k, comp) - state(i, j - 1, k, comp));
        gradv[1] = amrex::max(gradyplus, gradyminus);
#if AMREX_SPACEDIM == 3
        amrex::Real gradzplus =
            std::abs(state(i, j, k + 1, comp) - state(i, j, k, comp));
        amrex::Real gradzminus =
            std::abs(state(i, j, k, comp) - state(i, j, k - 1, comp));
        gradv[2] = amrex::max(gradzplus, gradzminus);
#endif
#endif

#if AMREX_SPACEDIM > 1
        if (amrex::max(AMREX_D_DECL(gradv[0], gradv[1], gradv[2])) >
            refine_phigrad[c])
#else
        if (gradv[0] > refine_phigrad[c])
#endif
        {
            tag(i, j, k) = tagval;
        }
    }
}

#endif
