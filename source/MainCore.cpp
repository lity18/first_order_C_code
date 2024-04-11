#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>


#include <MainCore.H>
#include <Kernels.H>
#include <mpi.h>

using namespace amrex;

MainCore::MainCore()
{
    ReadParameters();
    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.
    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);

    for (int lev = 1; lev <= max_level; ++lev) {
        nsubsteps[lev] = MaxRefRatio(lev-1);
    }

    t_new.resize(nlevs_max, 0.0);
    a_new.resize(nlevs_max, 1.0);
    a1.resize(nlevs_max, 1.0);
    a2.resize(nlevs_max, 1.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);
    dt[0] = dt_0;
    for (int lev = 1; lev < nlevs_max; ++lev) {
        dt[lev] = dt[lev-1] / 2;
    }

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    int bc_lo[AMREX_SPACEDIM];
    int bc_hi[AMREX_SPACEDIM];

    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        if (Geom(0).isPeriodic()[idim] == 1) {
            bc_lo[idim] = bc_hi[idim] = BCType::int_dir;  // periodic
        } else {
            bc_lo[idim] = bc_hi[idim] = BCType::foextrap;  // walls (Neumann)
        }
    }

    bcs.resize(1);     // Setup 1-component
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (bc_lo[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_lo[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_lo[idim] == BCType::ext_dir ) {  // external Dirichlet
            bcs[0].setLo(idim, bc_lo[idim]);
        }
        else {
            amrex::Abort("Invalid bc_lo");
        }

        // hi-side BCSs
        if (bc_hi[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_hi[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_hi[idim] == BCType::ext_dir ) {  // external Dirichlet
            bcs[0].setHi(idim, bc_hi[idim]);
        }
        else {
            amrex::Abort("Invalid bc_hi");
        }
    }


}

MainCore::~MainCore()
{
}


// initializes multilevel data
void
MainCore::InitData()
{
    const Real time = 0.0;
    InitFromScratch(time);
    AverageDown();
    bubble_struc();

}

// evolution
void
MainCore::Evolve()
{
    int myproc = ParallelDescriptor::MyProc();  // rank
    Real cur_time = t_new[0];
    auto strt_total = amrex::second();
    bubble_position();

    auto end_total = amrex::second() - strt_total;

    ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << "\nTotal Time: " << end_total << '\n';
    strt_total = amrex::second();


    input_bubble(cur_time);

    
    end_total = amrex::second() - strt_total;

    ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << "\nTotal Time: " << end_total << '\n';
    strt_total = amrex::second();


    for(int step=0;step<100;step++){
        timeStep(0, cur_time);
        cur_time += dt[0];
        if(step % 10==0 && myproc == 0){cout << step << endl;}
        
    }
    end_total = amrex::second() - strt_total;

    ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << "\nTotal Time: " << end_total << '\n';

    WritePlotFile();
    
    



    //WritePlotFile();
    



}



// Advance a level by dt
// (includes a recursive call for finer levels)
void
MainCore::timeStep (int lev, Real time)
{
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid
        if (lev < max_level && istep[lev] > last_regrid_step[lev])
        {
            if (istep[lev] % regrid_int == 0)
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;
                regrid(lev, time);

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }
            }
        }
    }

    // Advance a single level for a single time step, and update flux registers

    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    AdvancePhiAtLevel(lev, time, dt[lev]);

    ++istep[lev];

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            timeStep(lev+1, time+(i-1)*dt[lev+1]);
        }

        AverageDownTo(lev); // average lev+1 down to lev
    }

}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void 
MainCore::MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm)
{
    const int ncomp = phi_new[lev-1].nComp();
    const int nghost = phi_new[lev-1].nGrow();

    phi_new[lev].define(ba, dm, ncomp, nghost);
    phi_old[lev].define(ba, dm, ncomp, nghost);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    FillCoarsePatch(lev, time, phi_new[lev], 0, ncomp);
}


// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void
MainCore::RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm)
{
    const int ncomp = phi_new[lev].nComp();
    const int nghost = phi_new[lev].nGrow();

    MultiFab new_state(ba, dm, ncomp, nghost);
    MultiFab old_state(ba, dm, ncomp, nghost);

    FillPatch(lev, time, new_state, 0, ncomp);

    std::swap(new_state, phi_new[lev]);
    std::swap(old_state, phi_old[lev]);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;
}


// Delete level data
// overrides the pure virtual function in AmrCore
void
MainCore::ClearLevel (int lev)
{
    phi_new[lev].clear();
    phi_old[lev].clear();
}


// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization.
// overrides the pure virtual function in AmrCore
void
MainCore::MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                          const amrex::DistributionMapping& dm)
{
    const int ncomp = 4;   // number of components
    const int nghost = 2;   // number of ghost cells

    phi_new[lev].define(ba, dm, ncomp, nghost);
    phi_old[lev].define(ba, dm, ncomp, nghost);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    MultiFab& state = phi_new[lev];

    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();

    

    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> fab = state[mfi].array();
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            initdata(tbx, fab, problo, dx);
        });
    }

}

void
MainCore::ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
    static bool first = true;
    static Vector<Real> phierr;
    Real true_vac = sqrt((1 + sqrt(1 - 4 * kappa)) / 2 / kappa);


    // only do this during the first call to ErrorEst
    if (first)
    {
        first = false;
        // read in an array of "phierr", which is the tagging threshold
        ParmParse pp("core");
        int n = pp.countval("phierr");
        if (n > 0) {
            pp.getarr("phierr", phierr, 0, n);
        }
    }
    if (lev >= phierr.size()) return;

    //    const int clearval = TagBox::CLEAR;
    const int   tagval = TagBox::SET;
    const MultiFab& state = phi_new[lev];
    const auto dx     = Geom(lev).CellSizeArray();

    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx  = mfi.tilebox();
            const auto statefab = state.array(mfi);
            const auto tagfab  = tags.array(mfi);
            Real phierror = pow(phierr[lev] * true_vac * 2 * M_PI / dx[0], 2);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                state_error(i, j, k, tagfab, statefab, phierror, tagval, dx);
            });
        }
}

// read in some parameters from inputs file
void
MainCore::ReadParameters ()
{
    {
        ParmParse pp;
        pp.query("Hubble_constant", H);
        pp.query("dt", dt_0);
        pp.query("kappa", kappa);

    }

    {
        ParmParse pp("amr"); // Traditionally, these have prefix, amr.

        pp.query("regrid_int", regrid_int);
    }
}

// set covered coarse cells to be the average of overlying fine cells
void
MainCore::AverageDown ()
{
    for (int lev = finest_level-1; lev >= 0; --lev)
    {
        amrex::average_down(phi_new[lev+1], phi_new[lev],
                            geom[lev+1], geom[lev],
                            0, phi_new[lev].nComp(), refRatio(lev));
    }
}


void
MainCore::AverageDownold ()
{
    for (int lev = finest_level-1; lev >= 0; --lev)
    {
        amrex::average_down(phi_old[lev+1], phi_old[lev],
                            geom[lev+1], geom[lev],
                            0, phi_old[lev].nComp(), refRatio(lev));
    }
}

// more flexible version of AverageDown() that lets you average down across multiple levels
void
MainCore::AverageDownTo (int crse_lev)
{
    amrex::average_down(phi_new[crse_lev+1], phi_new[crse_lev],
                        geom[crse_lev+1], geom[crse_lev],
                        0, phi_new[crse_lev].nComp(), refRatio(crse_lev));
}


// compute a new multifab by coping in phi from valid region and filling ghost cells
// works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
void
MainCore::FillPatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    if (lev == 0)
    {
        Vector<MultiFab*> smf;
        Vector<Real> stime;
        GetData(0, time, smf, stime);

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbc(geom[lev],bcs,gpu_bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                        geom[lev], physbc, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev],bcs,bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                        geom[lev], physbc, 0);
        }
    }
    else
    {
        Vector<MultiFab*> cmf, fmf;
        Vector<Real> ctime, ftime;
        GetData(lev-1, time, cmf, ctime);
        GetData(lev  , time, fmf, ftime);

        Interpolater* mapper = &cell_cons_interp;

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, ncomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
            PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, ncomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        }
    }
}


// utility to copy in data from phi_old and/or phi_new into another multifab
void
MainCore::GetData (int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_new[lev]);
    }
    else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
        data.push_back(&phi_old[lev]);
        datatime.push_back(t_old[lev]);
    }
    else
    {
        data.push_back(&phi_old[lev]);
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}


// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void
MainCore::FillCoarsePatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev-1, time, cmf, ctime);
    Interpolater* mapper = &cell_cons_interp;

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    if(Gpu::inLaunchRegion())
    {
        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
    else
    {
        CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
        PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
        PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
}


// write plotfile to disk
void
MainCore::WritePlotFile () const
{
    const std::string& plotfilename = PlotFileName(istep[0]);
    const auto& mf = PlotFileMF();
    const auto& varnames = PlotFileVarNames();

    amrex::Print() << "Writing plotfile " << plotfilename << "\n";

    amrex::WriteMultiLevelPlotfile(plotfilename, finest_level+1, mf, varnames,
                                   Geom(), t_new[0], istep, refRatio());
}


// put together an array of multifabs for writing
amrex::Vector<const amrex::MultiFab*> MainCore::PlotFileMF () const
{
    amrex::Vector<const amrex::MultiFab*> r;
    for (int i = 0; i <= finest_level; ++i) {
        r.push_back(&phi_new[i]);
    }
    return r;
}

// get plotfile name
std::string
MainCore::PlotFileName (int lev) const
{
    return amrex::Concatenate(plot_file, lev, 5);
}

// set plotfile variable names
Vector<std::string>
MainCore::PlotFileVarNames () const
{
    return {"phi"};
}