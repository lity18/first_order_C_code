#ifndef MainCore_H_
#define MainCore_H_

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include <memory>
#include <boost/numeric/odeint.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>


typedef std::vector<double> state_type;
using namespace boost::math::interpolators;


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

    // initializes multilevel data
    void InitData();

    // evolution
    void Evolve();

    // input bubbles into the field, in <step.cpp>
    void input_bubble(amrex::Real time);

    // imput bubbles at single level , in <step.cpp>
    void input_level_bubble(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

    // imput bubbles on a new level
    void input_newlevel_bubble(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

    // bubble .cpp:
    

    // calculate the initial field value of the bounce solution
    double bubble_initial_condition();

    // get the bounce configuration and the bubble wall width
    void bubble_struc();

    // fillthe field value of bubble
    void initbubble(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx);
    
    // generate bubble coordinate
    void bubble_position();

    // decide whether a bubble can be generated
    void judgebubble(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx, 
         std::vector<std::vector<double>> const& position_list, double L);



    // the parameter of phase transition




private:
    // set covered coarse cells to be the average of overlying fine cells
    void AverageDown ();

    // read in some parameters from inputs file
    void ReadParameters();

    // compute a new multifab by coping in phi from valid region and filling ghost cells
    // works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
    void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // fill an entire multifab by interpolating from the coarser level
    // this comes into play when a new level of refinement appears
    void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // utility to copy in data from phi_old and/or phi_new into another multifab
    void GetData (int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data,
                  amrex::Vector<amrex::Real>& datatime);


    // keep track of old time, new time, and time step at each level
    amrex::Vector<amrex::Real> t_new;
    amrex::Vector<amrex::Real> t_old;
    amrex::Vector<amrex::Real> a_new;
    amrex::Vector<amrex::Real> dt;

    // array of multifabs to store the solution at each level of refinement
    // after advancing a level we use "swap".
    amrex::Vector<amrex::MultiFab> phi_new;
    amrex::Vector<amrex::MultiFab> phi_old;

    // this is essentially a 2*DIM integer array storing the physical boundary
    // condition types at the lo/hi walls in each direction
    amrex::Vector<amrex::BCRec> bcs;  // 1-component

    // Hubble parameter
    amrex::Real H;

    // time resolution at level 0
    amrex::Real dt_0;

    // bubble initial time
    amrex::Real dt_b = dt_0;

    // model parameter
    double kappa;

    // bounce solution
    state_type list_0={0.0, 1.0, 2.0, 3.0};
    barycentric_rational<double> bounce_x=barycentric_rational<double>(list_0.data(), list_0.data(), list_0.size());
    barycentric_rational<double> bounce_p=barycentric_rational<double>(list_0.data(), list_0.data(), list_0.size());

    // bubble size
    double wid_s, wid_l;

    // bounce center value
    double ini_phi;

    // store the information of whether the bubble is allowed
    amrex::Vector<int> signal_list;

    // the list of the positions of bubble center after judgement
    std::vector<std::vector<double>> position_list_gather; 




};

// bounce equation outside the bubble
// void bounce_outside(const state_type &x, state_type &dxdt, double t);

// bounce solution
//state_type empty_list1={0.0, 1.0, 2.0, 3.0};
//boost::math::interpolators::barycentric_rational<double> bounce_phi(empty_list1.data(), empty_list1.data(), empty_list1.size()); 
//boost::math::interpolators::barycentric_rational<double> bounce_momen(empty_list1.data(), empty_list1.data(), empty_list1.size());



#endif