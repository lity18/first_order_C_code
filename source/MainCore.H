#ifndef MainCore_H_
#define MainCore_H_

#include <string>
#include <limits>
#include <memory>

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>

class MainCore
    : public amrex::AmrCore
{
public:
    //read parameters from inputs
    MainCore();
    virtual ~MainCore();

    // Make a new level using provided BoxArray and DistributionMapping and
    // fill with interpolated coarse level data.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping and
    // fill with existing fine and coarse data.
    // overrides the pure virtual function in AmrCore
    virtual void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm) override;

    // Delete level data
    // overrides the pure virtual function in AmrCore
    virtual void ClearLevel (int lev) override;

    // Make a new level from scratch using provided BoxArray and DistributionMapping.
    // Only used during initialization.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                          const amrex::DistributionMapping& dm) override;

    // tag all cells for refinement
    // overrides the pure virtual function in AmrCore
    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;



private:


};


#endif