#ifndef init_H_
#define init_H_

#include <iostream>
#include <cmath>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

using namespace std;


// initialize the field, use trival value
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void initdata(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    using namespace amrex;

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    const int nf = phi.nComp();

    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int n = 0; n < nf; ++n) {
                    phi(i,j,k,n) = 0.0;
                }
            }
        }
    }


}











#endif
