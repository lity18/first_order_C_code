#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#include <vector>

#include <MainCore.H>
#include <Kernels.H>

using namespace amrex;
Real x0 = -1.7024143839193153;  // -2**(1/3)/(2-2**(1/3))
Real x1 = 1.3512071919596578; // 1/(2-2**(1/3))
Vector<Real> w = {x1 / 2, x1, (x1 + x0) / 2, x0, (x1 + x0) / 2, x1, x1 / 2};
int i_s = w.size();


// Advance a single level for a single time step, updates flux registers
void
MainCore::AdvancePhiAtLevel (int lev, Real time, Real dt_lev)
{
    MultiFab& state = phi_new[lev];

    const auto dx     = Geom(lev).CellSizeArray();

    a2[lev] = a_new[lev];
    Real dx_2 = pow(dx[0], 2);

    for(int i_step=0;i_step<i_s;++i_step){
        (phi_new[lev]).FillBoundary(Geom(lev).periodicity());
        if(i_step % 2 == 0){
            Real dt_w = (w[i_step] * dt_lev) / pow(a1[lev], 2);
            for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi) // then input bubbles
            {
                Array4<Real> phi = state[mfi].array();
                const Box& box = mfi.tilebox();

                amrex::launch(box,
                [=] AMREX_GPU_DEVICE (Box const& tbx)
                {
                    const auto lo = lbound(tbx);
                    const auto hi = ubound(tbx);
                    for         (int k = lo.z; k <= hi.z; ++k) {
                        for     (int j = lo.y; j <= hi.y; ++j) {
                            for (int i = lo.x; i <= hi.x; ++i) {
                                phi(i,j,k,0) += phi(i,j,k,2) * dt_w;
                                phi(i,j,k,1) += phi(i,j,k,3) * dt_w;
                            }
                        }
                    }
                });
            }
            a2[lev] += (w[i_step] * dt_lev) * H;
        }
        else{
            Real a_2 = pow(a2[lev], 2);
            Real dt_w = w[i_step] * dt_lev * a_2;
            for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi) // then input bubbles
            {
                Array4<Real> phi = state[mfi].array();
                const Box& box = mfi.tilebox();

                amrex::launch(box,
                [=] AMREX_GPU_DEVICE (Box const& tbx)
                {
                    const auto lo = lbound(tbx);
                    const auto hi = ubound(tbx);
                    for         (int k = lo.z; k <= hi.z; ++k) {
                        for     (int j = lo.y; j <= hi.y; ++j) {
                            for (int i = lo.x; i <= hi.x; ++i) {
                                Real phi_squ = pow(phi(i,j,k,0), 2) + pow(phi(i,j,k,1), 2);
                                Real V_phi = (1 - phi_squ * (1 - kappa * phi_squ)) * a_2;
                                Real laplace = (phi(i+1,j,k,0) + phi(i-1,j,k,0) + phi(i,j+1,k,0) + phi(i,j-1,k,0)
                                                + phi(i,j,k+1,0) + phi(i,j,k-1,0) - 6 * phi(i,j,k,0)) / dx_2;
                                phi(i,j,k,2) += (laplace - V_phi * phi(i,j,k,0)) * dt_w;
                                laplace = (phi(i+1,j,k,1) + phi(i-1,j,k,1) + phi(i,j+1,k,1) + phi(i,j-1,k,1)
                                                + phi(i,j,k+1,1) + phi(i,j,k-1,1) - 6 * phi(i,j,k,1)) / dx_2;
                                phi(i,j,k,3) += (laplace - V_phi * phi(i,j,k,1)) * dt_w;
                            }
                        }
                    }
                });
            }
            a1[lev] += (w[i_step] * dt_lev) * H;
        }
    }
    a_new[lev] = a1[lev];
}