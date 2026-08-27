#ifndef _GEOMUTILS_H_
#define _GEOMUTILS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_INLINE
void surfacenorm(amrex::Real x,amrex::Real y,
                 GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                 GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                 ProbParm const& prob_parm,
                 amrex::Real n[AMREX_SPACEDIM])
{
    //normal pointing from fluid/plasma domain
    //to surface
    amrex::Real l=prob_hi[1]-prob_lo[1];
    amrex::Real d = prob_parm.dgap;
    amrex::Real r = prob_parm.pinrad;

    amrex::Real midy=0.5*(prob_lo[1]+prob_hi[1]);
    int bottom_half=(y<midy)?1:0;

    amrex::Real c2=(bottom_half)?prob_lo[1]+0.5*(l-d):prob_lo[1]+0.5*(l-d)+d;
    amrex::Real c1=(bottom_half)?(prob_lo[1]-c2)/(r*r):(prob_hi[1]-c2)/(r*r);

    n[0]=(bottom_half)?2.0*c1*x:-2.0*c1*x;
    n[1]=(bottom_half)?-1.0:1.0;
    Real nmag=std::sqrt(n[0]*n[0]+n[1]*n[1]);
    n[0]/=nmag;
    n[1]/=nmag;
}

AMREX_GPU_DEVICE AMREX_INLINE
amrex::Real surfaceval(amrex::Real x,amrex::Real y,
                       ProbParm const& prob_parm,
                       GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                       GpuArray<Real, AMREX_SPACEDIM> prob_hi)
{
    amrex::Real l=prob_hi[1]-prob_lo[1];
    amrex::Real d = prob_parm.dgap;
    amrex::Real r = prob_parm.pinrad;
    amrex::Real midy=0.5*(prob_lo[1]+prob_hi[1]);

    int bottom_half=(y<midy)?1:0;
    amrex::Real c2=(bottom_half)?prob_lo[1]+0.5*(l-d):prob_lo[1]+0.5*(l-d)+d;
    amrex::Real c1=(bottom_half)?(prob_lo[1]-c2)/(r*r):(prob_hi[1]-c2)/(r*r);

    return(y-(c1*x*x+c2)); 
}

AMREX_GPU_DEVICE AMREX_INLINE
amrex::Real surfacefunc(amrex::Real x,amrex::Real y,
                        ProbParm const& prob_parm,
                        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                        GpuArray<Real, AMREX_SPACEDIM> prob_hi)
{
    //the electrodes are parabolic of the form
    //y=c1*(x**2)+c2
    amrex::Real d = prob_parm.dgap;
    amrex::Real r = prob_parm.pinrad;

    amrex::Real l=prob_hi[1]-prob_lo[1];
    amrex::Real c2_b=prob_lo[1]+0.5*(l-d);
    amrex::Real c2_t=prob_lo[1]+0.5*(l-d)+d;

    amrex::Real c1_b=(prob_lo[1]-c2_b)/(r*r);
    amrex::Real c1_t=(prob_hi[1]-c2_t)/(r*r);

    amrex::Real yout=prob_lo[1]+l/2; 
    amrex::Real xout=0.0;

    amrex::Real sgn=1.0;
    if( (y-c1_b*x*x-c2_b)*(yout-c1_b*xout*xout-c2_b) < 0.0 ||
       (y-c1_t*x*x-c2_t)*(yout-c1_t*xout*xout-c2_t) < 0.0)
    {
        sgn=-1.0;
    }
    return(sgn); 
}

AMREX_GPU_DEVICE AMREX_INLINE
void get_surface_point(IntVect cutcell,
                       GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                       GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                       GpuArray<int, AMREX_SPACEDIM> domlo,
                       GpuArray<int, AMREX_SPACEDIM> domhi,
                       GpuArray<Real, AMREX_SPACEDIM> dx,
                       ProbParm const& prob_parm,
                       Real xib[AMREX_SPACEDIM])
{
    int niter=50;
    Real tol=1e-6;

    //guess
    xib[0]=prob_lo[0]+(cutcell[0]+0.5)*dx[0];
    xib[1]=prob_lo[1]+(cutcell[1]+0.5)*dx[1];

    amrex::Real sgn_ib=surfacefunc(xib[0],xib[1],prob_parm,
                                   prob_lo,prob_hi);

    //complementary point to ib
    //if xib is inside xib_c is outside
    //if xib is outside xib_c is inside
    Real xib_c[AMREX_SPACEDIM];
    Real xmid[AMREX_SPACEDIM];

    for(int jj=0;jj<AMREX_SPACEDIM;jj++)
    {
        int found=0;
        for(int ii=0;ii<AMREX_SPACEDIM;ii++)
        {
            xib_c[0]=prob_lo[0]+(cutcell[0]+ii)*dx[0];
            xib_c[1]=prob_lo[1]+(cutcell[1]+jj)*dx[1];

            if(surfacefunc(xib_c[0],xib_c[1],prob_parm,prob_lo,prob_hi)!=sgn_ib)
            {
                found=1;
                break;
            }
        }
        if(found)
        {
            break;
        }
    }

    int iter=0;
    for(iter=0;iter<niter;iter++)
    {
        xmid[0]=0.5*(xib_c[0]+xib[0]);
        xmid[1]=0.5*(xib_c[1]+xib[1]);

        if(amrex::Math::abs(surfaceval(xmid[0],xmid[1],prob_parm,prob_lo,prob_hi))<tol)
        {
            break;
        }

        Real sgnmid=surfacefunc(xmid[0],xmid[1],prob_parm,prob_lo,prob_hi);
        Real sgn1=surfacefunc(xib[0],xib[1],prob_parm,prob_lo,prob_hi);
        Real sgn2=surfacefunc(xib_c[0],xib_c[1],prob_parm,prob_lo,prob_hi);

        //amrex::Print()<<"sgn1,sgn2,sgnmid:"<<sgn1<<"\t"<<sgn2<<"\t"<<sgnmid<<"\n";
        //amrex::Print()<<"surfval at "<<xmid[0]<<","<<xmid[1]<<":"<<surfaceval(xmid[0],xmid[1],prob_parm,prob_lo,prob_hi)<<"\n";

        if(sgnmid==sgn1)
        {
            xib[0]=xmid[0];
            xib[1]=xmid[1];
        }
        else
        {
            xib_c[0]=xmid[0];
            xib_c[1]=xmid[1];
        }
    }

    //amrex::Print()<<"iter, tol:"<<iter<<"\t"<<tol<<"\n";

    //final value
    xib[0]=xmid[0];
    xib[1]=xmid[1];
}

#endif
