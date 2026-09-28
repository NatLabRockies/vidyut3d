#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <Chemistry.H>
#include <ProbParm.H>
#include <UnivConstants.H>
#include <VarDefines.H>

using namespace amrex;

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_arp_hp_h_conc(amrex::Real T,ProbParm *prob_parm,amrex::Real &arpconc,amrex::Real &hpconc,amrex::Real &hconc)
{
    int i;
    for(i=0;i<prob_parm->narr_pts;i++)
    {
        if(prob_parm->Tarr[i]>T)
        {
           break;
        }
    }

    amrex::Real frac=(T-prob_parm->Tarr[i-1])/(prob_parm->Tarr[i]-prob_parm->Tarr[i-1]);

    arpconc=(1-frac)*prob_parm->ARp_arr[i-1]+frac*prob_parm->ARp_arr[i];
    hpconc=(1-frac)*prob_parm->Hp_arr[i-1]+frac*prob_parm->Hp_arr[i];
    hconc=(1-frac)*prob_parm->H_arr[i-1]+frac*prob_parm->H_arr[i];
   
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_ar_h2_conc(amrex::Real T,ProbParm *prob_parm,amrex::Real &arconc,amrex::Real &h2conc,amrex::Real &hconc)
{
    int i;
    for(i=0;i<prob_parm->narr_pts;i++)
    {
        if(prob_parm->Tarr[i]>T)
        {
           break;
        }
    }

    amrex::Real frac=(T-prob_parm->Tarr[i-1])/(prob_parm->Tarr[i]-prob_parm->Tarr[i-1]);

    arconc=(1-frac)*prob_parm->Ar_arr[i-1]+frac*prob_parm->Ar_arr[i];
    h2conc=(1-frac)*prob_parm->H2_arr[i-1]+frac*prob_parm->H2_arr[i];
    hconc=(1-frac)*prob_parm->H_arr[i-1]+frac*prob_parm->H_arr[i];
}

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("wallpot",h_prob_parm.wallpot);
    pp.query("bulkplasmatemp",h_prob_parm.bulkplasmatemp);
    pp.query("bulkelectemp",h_prob_parm.bulkelectemp);
    pp.query("walltemp",h_prob_parm.walltemp);
    pp.query("Arp_arc",h_prob_parm.Arp_arc);
    pp.query("Hp_arc",h_prob_parm.Hp_arc);
    pp.query("H_arc",h_prob_parm.H_arc);
    pp.query("spec_arc",h_prob_parm.spec_arc);
    pp.query("beta_fe",h_prob_parm.beta_fe);
    pp.query("lamdar_te",h_prob_parm.lamdar_te);
    pp.query("h2frac",h_prob_parm.h2frac);
    pp.query("workfunc",h_prob_parm.workfunc);
    pp.query("gama_se",h_prob_parm.gama_se);
    pp.query("pecletnum",h_prob_parm.pecletnum);
    pp.query("arc_efield",h_prob_parm.arc_efield);
    pp.query("arc_edge_dirichlet",h_prob_parm.arc_edge_dirichlet);
    pp.query("stickfac",h_prob_parm.stickfac);


    amrex::ParmParse ppv("vidyut");
    ppv.query("gas_pressure",h_prob_parm.pres);

    amrex::Real arcspecies[NUM_SPECIES]={h_prob_parm.spec_arc};
    
    /*amrex::Real bulk_arpconc,bulk_hpconc,bulk_hconc;
    get_arp_hp_h_conc(h_prob_parm.bulkplasmatemp,&h_prob_parm,bulk_arpconc,
                      bulk_hpconc,bulk_hconc);

    h_prob_parm.Arp_arc=std::max(h_prob_parm.spec_arc,bulk_arpconc*N_A);
    h_prob_parm.Hp_arc=std::max(h_prob_parm.spec_arc,bulk_hpconc*N_A);
    h_prob_parm.H_arc=std::max(h_prob_parm.spec_arc,bulk_hconc*N_A);*/

    arcspecies[ARp_ID]=h_prob_parm.Arp_arc;
    arcspecies[Hp_ID]=h_prob_parm.Hp_arc;
    arcspecies[H_ID]=h_prob_parm.H_arc;

    int chrgspec[NUM_SPECIES]={0};
    CKCHRG(chrgspec);
    
    amrex::Real net_ion_charge=0.0;
    for(int sp=0;sp<NUM_SPECIES;sp++)
    {
        if(sp!=E_ID)
        {
            net_ion_charge += arcspecies[sp]*chrgspec[sp];
        }
    }
    h_prob_parm.E_arc=net_ion_charge;

    amrex::Print()<<"Arp,Hp,H,E:"<<h_prob_parm.Arp_arc<<"\t"<<
    h_prob_parm.Hp_arc<<"\t"<<h_prob_parm.H_arc<<"\t"<<h_prob_parm.E_arc<<"\n";

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

}


AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(Box const& bx, Array4<Real> const& phi, 
                    GeometryData const& geomdata,
                    ProbParm *prob_parm)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();

    amrex::Real L=prob_hi[0]-prob_lo[0];
    amrex::Real deltaTemp=prob_parm->bulkplasmatemp-prob_parm->walltemp;


    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j) 
        {

            for (int i = lo.x; i <= hi.x; ++i) 
            {
                Real z = prob_lo[2] + (0.5+k) * dx[2];
                Real y = prob_lo[1] + (0.5+j) * dx[1];
                Real x = prob_lo[0] + (0.5+i) * dx[0];

                int chrgspec[NUM_SPECIES]={0};
                CKCHRG(chrgspec);

                //linear
                //Real gastemp=prob_parm->bulkplasmatemp-deltaTemp/L*(x-prob_lo[0]);
                //boundary layer
                Real Pe=prob_parm->pecletnum;
                Real xx=x-prob_lo[0];
                Real gastemp=(prob_parm->bulkplasmatemp*(std::exp(Pe*xx)-std::exp(Pe*L)) +
                        prob_parm->walltemp*(1.0-std::exp(Pe*xx)) )/(1.0-std::exp(Pe*L));

                for(int c=0;c<NVAR;c++)
                {
                    phi(i,j,k,c)=0.0;
                }
                phi(i,j,k,CMASK_ID)=1.0;
                for(int sp=0;sp<NUM_SPECIES;sp++)
                {
                    phi(i,j,k,sp)=prob_parm->spec_arc;
                }

                amrex::Real arconc,h2conc,hconc;
                get_ar_h2_conc(gastemp,prob_parm,arconc,h2conc,hconc);
                amrex::Real totalnden=prob_parm->pres/K_B/gastemp;
                
                //phi(i,j,k,AR_ID)=(1.0-prob_parm->h2frac)*P_NTP/K_B/gastemp;
                //phi(i,j,k,H2_ID)=prob_parm->h2frac*P_NTP/K_B/gastemp;
                
                phi(i,j,k,AR_ID)=std::max(prob_parm->spec_arc,arconc*N_A);
                phi(i,j,k,H2_ID)=std::max(prob_parm->spec_arc,h2conc*N_A);
                phi(i,j,k,H_ID)=std::max(prob_parm->spec_arc,hconc*N_A);

                //amrex::Print()<<"totalnden:"<<totalnden<<"\t"<<phi(i,j,k,AR_ID)<<"\t"<<phi(i,j,k,H2_ID)<<"\n";
                //phi(i,j,k,H_ID)=std::max(prob_parm->spec_arc,totalnden-phi(i,j,k,AR_ID)-phi(i,j,k,H2_ID));
                //phi(i,j,k,H_ID)*=std::max(0.0,std::copysign(1.0,85e-6-x));
                //phi(i,j,k,H_ID)+=prob_parm->spec_arc;

                phi(i,j,k,ARp_ID)=prob_parm->Arp_arc;
                phi(i,j,k,Hp_ID)=prob_parm->Hp_arc;

                amrex::Real net_ion_charge=0.0;
                for(int sp=0;sp<NUM_SPECIES;sp++)
                {
                    if(sp!=E_ID)
                    {
                        net_ion_charge += phi(i,j,k,sp)*chrgspec[sp];
                    }
                }

                phi(i,j,k,E_ID)    = net_ion_charge;
                phi(i,j,k,ETEMP_ID) = prob_parm->bulkelectemp;
                phi(i,j,k,EEN_ID) = 1.5*phi(i,j,k,E_ID)*K_B*phi(i,j,k,ETEMP_ID);
            }
        }
    }
}
#endif
