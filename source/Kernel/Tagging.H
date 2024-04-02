#ifndef TAGGING_H_
#define TAGGING_H_

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
state_error (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Real phierr, char tagval)
{
    if (state(i,j,k) > phierr)
        tag(i,j,k) = tagval;
}

#endif
