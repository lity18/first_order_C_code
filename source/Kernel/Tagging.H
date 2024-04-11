#ifndef TAGGING_H_
#define TAGGING_H_

#include <AMReX_Array4.H>
#include <iostream>
#include <cmath>


AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
state_error (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Real phierr, char tagval, amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    amrex::Real dphi = 0;
    dphi += pow((state(i+1,j,k,0) - state(i-1,j,k,0)) / 2 / dx[0], 2);
    dphi += pow((state(i,j+1,k,0) - state(i,j-1,k,0)) / 2 / dx[1], 2);
    dphi += pow((state(i,j,k+1,0) - state(i,j,k-1,0)) / 2 / dx[2], 2);
    dphi += pow((state(i+1,j,k,1) - state(i-1,j,k,1)) / 2 / dx[0], 2);
    dphi += pow((state(i,j+1,k,1) - state(i,j-1,k,1)) / 2 / dx[1], 2);
    dphi += pow((state(i,j,k+1,1) - state(i,j,k-1,1)) / 2 / dx[2], 2);
    if (dphi > phierr)
        tag(i,j,k) = tagval;
}

#endif
