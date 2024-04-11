#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <utility>
#include <ostream>
#include <random>
#include<bits/stdc++.h>
#include <mpi.h>
#include <MainCore.H>
#include <Kernels.H>

using namespace amrex;


// input bubbles into the field
void
MainCore::input_bubble(Real time)
{
    using namespace amrex;
    int new_finest;
    Vector<BoxArray> new_grids(max_level + 1);

    BoxArray level_grids = grids[0];
    DistributionMapping level_dmap = dmap[0];

    input_level_bubble(0, time, level_grids, level_dmap);
    MakeNewGrids(0, time, new_finest, new_grids);


    for (int lev = 1; lev <= new_finest; ++lev)
    {
        if (lev <= finest_level) // an old level
        {
            bool ba_changed = (new_grids[lev] != grids[lev]);
            level_grids = grids[lev];
            level_dmap = dmap[lev];
            if (ba_changed) {
                level_grids = new_grids[lev];
                level_dmap = DistributionMapping(level_grids);
            }
            const auto old_num_setdm = num_setdm;
            
            input_level_bubble(lev, time, level_grids, level_dmap);

            SetBoxArray(lev, level_grids);
            if (old_num_setdm == num_setdm) {
                SetDistributionMap(lev, level_dmap);
            }
        }
        else  // a new level
        {
            DistributionMapping new_dmap(new_grids[lev]);
            const auto old_num_setdm = num_setdm;
            input_newlevel_bubble(lev, time, new_grids[lev], new_dmap);
            SetBoxArray(lev, new_grids[lev]);
            if (old_num_setdm == num_setdm) {
                SetDistributionMap(lev, new_dmap);
            }
            finest_level = new_finest;
        }
        if(lev < max_level){
            MakeNewGrids(lev, time, new_finest, new_grids);
            }
              
    }

}


// imput bubbles at single level
void 
MainCore::input_level_bubble(int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    using namespace amrex;

    if(lev != 0){
        const int ncomp = phi_new[lev].nComp();
        const int nghost = phi_new[lev].nGrow();

        MultiFab new_state(ba, dm, ncomp, nghost);
        MultiFab old_state(ba, dm, ncomp, nghost);

        FillPatch(lev, time, new_state, 0, ncomp);  

        std::swap(new_state, phi_new[lev]);
        std::swap(old_state, phi_old[lev]);  //first remake the level 
    }

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    MultiFab& state = phi_new[lev];

    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();

    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi) // then input bubbles
    {
        Array4<Real> fab = state[mfi].array();
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            if(lev == 0){
                initbubble_0(tbx, fab, problo, dx, position_list_gather);
            }
            else{
                initbubble(tbx, fab, problo, dx);
            }
        });
    }

}


// imput bubbles on a new level
void 
MainCore::input_newlevel_bubble(int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    using namespace amrex;
    const int ncomp = phi_new[lev-1].nComp();
    const int nghost = phi_new[lev-1].nGrow();

    phi_new[lev].define(ba, dm, ncomp, nghost);
    phi_old[lev].define(ba, dm, ncomp, nghost);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    FillCoarsePatch(lev, time, phi_new[lev], 0, ncomp);  

    MultiFab& state = phi_new[lev];

    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();

    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi) // then input bubbles
    {
        Array4<Real> fab = state[mfi].array();
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            initbubble(tbx, fab, problo, dx);
        });
    }

}


// generate bubble coordinate
void MainCore::bubble_position()
{
    int myproc, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myproc);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    //int myproc = ParallelDescriptor::MyProc();  // rank
    //int nprocs = ParallelDescriptor::NProcs();  // number of process
    int list_size;
    
    std::vector<std::vector<double>> position_list; // the list of the positions of bubble center

    const auto box = Geom(0).Domain();
    const auto dx = Geom(0).CellSize();

    double L = box.length(0) * dx[0];   //length of box
    if(myproc == 0){
        std::random_device rd;
        std::mt19937 gen(rd());

        double probablity_bubble = dx[0] * dx[1] * dx[2] * dt_0 * pow(H, 4);
        
        std::binomial_distribution<> d(box.volume(), probablity_bubble);  //bubble generation
        std::uniform_real_distribution<> dis(0.0, L);  //bubble coordinate
        std::uniform_real_distribution<> dis_pi(0.0, 2 * M_PI);  //bubble phase

        // the number of bubble generated during this time interval
        int bubble_number = d(gen);

        for(int i = 0;i<bubble_number;++i){
            std::vector<double> position{dis(gen), dis(gen), dis(gen), dis_pi(gen)}; //generate bubble coordinate and phase
            int signal = 0;
            if(i > 0){
                for(int j = 0;j<position_list.size();++j){
                    double radius = 0;
                    double r_i = 0;
                    for(int di=0;di<3;++di){
                        r_i =  std::abs(position[di]-position_list[j][di]);
                        radius += pow(std::min(r_i, L - r_i),2);  //calculate the distance between the generated bubble and the bubble in lists
                    }                    
                    if(sqrt(radius) < 2 * wid_s / a_new[0]){   // if they are too close, drop the bubble
                        signal = 1;
                        break;                        
                    }
                }
                if(signal == 0){
                    position_list.push_back(position);
                }
            }
        }
        list_size = position_list.size();
    }
    MPI_Bcast(&list_size,1,MPI_INT,0,MPI_COMM_WORLD); // bcast the length of list,prepare for bcast the list
    if(myproc != 0){
        position_list.resize(list_size, {0, 0, 0, 0});
    }
    for(int i=0;i<list_size;++i){
        MPI_Bcast(position_list[i].data(),4,MPI_DOUBLE,0,MPI_COMM_WORLD); // bcast the list
    } 

    std::vector<int> signal_list_0(list_size, 1);   
    std::swap(signal_list_0, signal_list);

    for(int lev=0;lev<=finest_level;++lev){
        MultiFab& state = phi_new[lev];

        const auto problo = Geom(lev).ProbLoArray();
        const auto dx1     = Geom(lev).CellSizeArray();

        for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi) // then input bubbles
        {
            Array4<Real> fab = state[mfi].array();
            const Box& box1 = mfi.tilebox();

            amrex::launch(box1,
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {
                judgebubble(tbx, fab, problo, dx1, position_list, L); // decide whether the bubble can be generated
            });
        }
    }

    std::vector<int> signal_gather;
    std::vector<std::vector<double>> position_list_gather_0; // the list of the positions of bubble center after judgement

    signal_gather.resize(nprocs, 1);
    position_list_gather.resize(1, {0, 0, 0, 0});
    int signal_all=1;
    
    for(int i=0;i<list_size;++i){
        MPI_Gather(&signal_list[i], 1, MPI_INT, signal_gather.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(myproc == 0){
            signal_all = accumulate(signal_gather.begin(), signal_gather.end(), 1, multiplies<int>());
        }
        MPI_Bcast(&signal_all,1,MPI_INT,0,MPI_COMM_WORLD);
        if(signal_all == 1){
            position_list_gather_0.push_back(position_list[i]);
        }
    }
    //cout << position_list_gather_0.size() << endl;
    //position_list_gather.resize(1);
    std::swap(position_list_gather_0, position_list_gather);
    
    
}