#include <AMReX_AmrCore.H>

#include <AMReX_BCRec.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <utility>
#include <ostream>

#include<bits/stdc++.h>
#include <mpi.h>
#include <MainCore.H>
#include <Kernels.H>


using namespace std;
using namespace amrex;


// calculate txx, tyy,tzz,txy,txz,tyz
void MainCore::energy_momentum_tensor()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        partial_diag(lev);
    }
    AverageDownold();
    



}


// calculate txx, tyy,tzz, store in phi_old(0, 1, 2)
void MainCore::partial_diag(int lev)
{
    (phi_new[lev]).FillBoundary(Geom(lev).periodicity());

    MultiFab& state = phi_new[lev];
    MultiFab& tij_slice = phi_old[lev];

    const auto dx     = Geom(lev).CellSizeArray();
    Real dx2 = pow(2 * dx[0], 2);

    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> phi = state[mfi].array();
        Array4<Real> dphi = tij_slice[mfi].array();
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            const auto lo = lbound(tbx);
            const auto hi = ubound(tbx);
            for         (int k = lo.z; k <= hi.z; ++k) {
                for     (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        dphi(i,j,k,0) = (pow((phi(i+1,j,k,0) - phi(i-1,j,k,0)), 2) 
                                            + pow((phi(i+1,j,k,1) - phi(i-1,j,k,1)), 2) ) / dx2;
                        dphi(i,j,k,1) = (pow((phi(i,j+1,k,0) - phi(i,j-1,k,0)), 2) 
                                            + pow((phi(i,j+1,k,1) - phi(i,j-1,k,1)), 2) ) / dx2;
                        dphi(i,j,k,2) = (pow((phi(i,j,k+1,0) - phi(i,j,k-1,0)), 2) 
                                            + pow((phi(i,j,k+1,1) - phi(i,j,k-1,1)), 2) ) / dx2;
                    }
                }
            }
        });
    }
}



// calculate txy,txz,tyz, store in phi_old(0, 1, 2)
void MainCore::partial_offdiag(int lev)
{
    (phi_new[lev]).FillBoundary(Geom(lev).periodicity());

    MultiFab& state = phi_new[lev];
    MultiFab& tij_slice = phi_old[lev];

    const auto dx     = Geom(lev).CellSizeArray();
    Real dx2 = pow(2 * dx[0], 2);

    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> phi = state[mfi].array();
        Array4<Real> dphi = tij_slice[mfi].array();
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            const auto lo = lbound(tbx);
            const auto hi = ubound(tbx);
            for         (int k = lo.z; k <= hi.z; ++k) {
                for     (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        dphi(i,j,k,0) = ((phi(i+1,j,k,0) - phi(i-1,j,k,0)) * (phi(i,j+1,k,0) - phi(i,j-1,k,0))
                                            + (phi(i+1,j,k,1) - phi(i-1,j,k,1)) * (phi(i,j+1,k,1) - phi(i,j-1,k,1)) ) / dx2;
                        dphi(i,j,k,1) = ((phi(i+1,j,k,0) - phi(i-1,j,k,0)) * (phi(i,j,k+1,0) - phi(i,j,k-1,0))
                                            + (phi(i+1,j,k,1) - phi(i-1,j,k,1)) * (phi(i,j,k+1,1) - phi(i,j,k-1,1)) ) / dx2;
                        dphi(i,j,k,2) = ((phi(i,j,k+1,0) - phi(i,j,k-1,0)) * (phi(i,j+1,k,0) - phi(i,j-1,k,0))
                                            + (phi(i,j,k+1,1) - phi(i,j,k-1,1)) * (phi(i,j+1,k,1) - phi(i,j-1,k,1)) ) / dx2;
                    }
                }
            }
        });
    }
}