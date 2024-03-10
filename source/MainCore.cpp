#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>


#include <MainCore.H>

using namespace amrex;

MainCore::MainCore()
{

}

MainCore::~MainCore()
{
}

void 
MainCore::MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm)
{

}

void
MainCore::RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm)
{

}

void 
MainCore::ClearLevel (int lev)
{

}

void
MainCore::MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                          const amrex::DistributionMapping& dm)
{

}

void
MainCore::ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
    
}