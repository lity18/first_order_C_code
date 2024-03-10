#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

#include <MainCore.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        // wallclock time
        const auto strt_total = amrex::second();

        //read parameters from inputs
        MainCore maincore;




        // wallclock time
        auto end_total = amrex::second() - strt_total;
        ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << "\nTotal Time: " << end_total << '\n';
    }

    amrex::Finalize();
}

