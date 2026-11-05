#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <string>

// Error of NI against the exact n = x^2 over the fluid cells (cellmask = 1)
// of a single-level 1D plotfile from MMS1_HighPe.
//   l2err [-l2 tol] [-linf tol] plotfile
// Aborts when the L2 or Linf error is above the given tolerance.

using namespace amrex;

void main_main()
{
    const int narg = amrex::command_argument_count();
    std::string pltfile;
    Real tol_l2 = -1.0;
    Real tol_linf = -1.0;

    int farg = 1;
    while (farg <= narg)
    {
        const std::string& name = amrex::get_command_argument(farg);
        if (name == "-l2")
        {
            tol_l2 = std::stod(amrex::get_command_argument(++farg));
        } else if (name == "-linf")
        {
            tol_linf = std::stod(amrex::get_command_argument(++farg));
        } else
        {
            break;
        }
        ++farg;
    }
    if (farg <= narg)
    {
        pltfile = amrex::get_command_argument(farg);
    }
    if (pltfile.empty())
    {
        amrex::Abort("usage: l2err [-l2 tol] [-linf tol] plotfile");
    }

    PlotFileData pf(pltfile);
    if (pf.finestLevel() != 0)
    {
        amrex::Abort("l2err: single-level plotfiles only");
    }
    const Real xlo = pf.probLo()[0];
    const Real dx = pf.cellSize(0)[0];
    const MultiFab& ni = pf.get(0, "NI");
    const MultiFab& cm = pf.get(0, "cellmask");
    auto const& na = ni.const_arrays();
    auto const& ma = cm.const_arrays();

    auto r = ParReduce(
        TypeList<ReduceOpSum, ReduceOpSum, ReduceOpMax>{},
        TypeList<Real, Real, Real>{}, ni,
        [=] AMREX_GPU_DEVICE(
            int bno, int i, int j, int k) -> GpuTuple<Real, Real, Real> {
            if (int(ma[bno](i, j, k)) != 1)
            {
                return {0.0_rt, 0.0_rt, 0.0_rt};
            }
            const Real x = xlo + (i + 0.5_rt) * dx;
            const Real e = std::abs(na[bno](i, j, k) - x * x);
            return {e * e * dx, dx, e};
        });
    Real esq = amrex::get<0>(r);
    Real len = amrex::get<1>(r);
    Real emax = amrex::get<2>(r);
    ParallelDescriptor::ReduceRealSum(esq);
    ParallelDescriptor::ReduceRealSum(len);
    ParallelDescriptor::ReduceRealMax(emax);
    const Real l2 = std::sqrt(esq / len);
    const int nfluid = int(std::round(len / dx));

    amrex::Print() << "N_fluid " << nfluid << " L2 " << l2 << " Linf " << emax
                   << "\n";
    if ((tol_l2 > 0.0 && l2 > tol_l2) || (tol_linf > 0.0 && emax > tol_linf))
    {
        amrex::Abort("MMS1_HighPe: error above tolerance");
    }
    amrex::Print() << "MMS1_HighPe error within tolerance\n";
}

int main(int argc, char* argv[])
{
    amrex::SetVerbose(0);
    amrex::Initialize(argc, argv, false);
    main_main();
    amrex::Finalize();
}
