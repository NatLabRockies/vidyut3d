#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <GeomUtils.H>

using namespace amrex;
namespace user_transport
{
    AMREX_GPU_HOST_DEVICE AMREX_INLINE amrex::Real current_collector_value(
        IntVect cellid,
        int locid,
        int surfaceloc,
        GpuArray<Real, AMREX_SPACEDIM> problo,
        GpuArray<Real, AMREX_SPACEDIM> probhi,
        GpuArray<int, AMREX_SPACEDIM> domlo,
        GpuArray<int, AMREX_SPACEDIM> domhi,
        GpuArray<Real, AMREX_SPACEDIM> dx)
    {
        return (0.0);
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void get_photoion_acoeff(int i,int j,int k,int sph_id,
                             Array4<Real> const& sb_arr,
                             Array4<Real> const& acoeff,
                             GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                             GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                             GpuArray<Real, AMREX_SPACEDIM> dx,
                             const Real time,
                             ProbParm const& prob_parm,
                             Real Tg, Real Pg)
    {

    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void get_photoion_rhs(int i,int j,int k,int sph_id,
                          Array4<Real> const& sb_arr,
                          Array4<Real> const& rhs_arr,
                          GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                          GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                          GpuArray<Real, AMREX_SPACEDIM> dx,
                          const Real time,
                          ProbParm const& prob_parm,
                          Real Tg, Real Pg)
    {

    }

    AMREX_GPU_HOST_DEVICE 
    AMREX_FORCE_INLINE
    amrex::Real get_applied_potential(amrex::Real current_time, 
                                      amrex::Real voltage_amp, 
                                      amrex::Real pulse_width, 
                                      amrex::Real pulse_ramp,
                                      amrex::Real pulse_freq)
    {
        amrex::Real voltage=0.0;
        amrex::Real pulse_reptime=1.0/pulse_freq;
        int pulsenum=int(current_time*pulse_freq);
        amrex::Real inpulse_time=current_time-pulsenum*pulse_reptime;
        int inside_pulse=(inpulse_time<pulse_width);

        if(inside_pulse)
        {
            if(inpulse_time < pulse_ramp) 
            {
                voltage = voltage_amp*(inpulse_time/pulse_ramp);
            } 
            else if(inpulse_time >= pulse_ramp && inpulse_time < (pulse_width-pulse_ramp))
            {
                voltage = voltage_amp;
            }
            else if(inpulse_time>=(pulse_width-pulse_ramp) && inpulse_time<pulse_width)
            {
                voltage = voltage_amp*(1-(inpulse_time-(pulse_width-pulse_ramp))/pulse_ramp);
            }
        }
        else
        {
            voltage=0.0;
        }
        //voltage=voltage_amp;
        return(voltage);
    }
    

    AMREX_GPU_DEVICE AMREX_INLINE
    void bc_ib(IntVect face,
               int dir, int sgn,
               int solved_comp,
               int rhs_comp,
               Array4<Real> const& sb_arr,
               amrex::Real ascalar,
               amrex::Real bscalar,
               Array4<Real> const& acoeff,
               Array4<Real> const& bcoeff,
               Array4<Real> const& rhs,
               GpuArray<int, AMREX_SPACEDIM> domlo,
               GpuArray<int, AMREX_SPACEDIM> domhi,
               GpuArray<Real, AMREX_SPACEDIM> prob_lo,
               GpuArray<Real, AMREX_SPACEDIM> prob_hi,
               GpuArray<Real, AMREX_SPACEDIM> dx,
               const Real time,
               ProbParm const& prob_parm,
               Real Tg, Real Pg)
    {
        IntVect lcell=face;
        IntVect rcell=face;
        lcell[dir]-=1;
        amrex::Real xloc[AMREX_SPACEDIM],xib[AMREX_SPACEDIM];
        xloc[0]=prob_lo[0]+(lcell[0]+0.5)*dx[0];
        xloc[1]=prob_lo[1]+(lcell[1]+0.5)*dx[1];
        xloc[dir]+=0.5*dx[dir];

        amrex::Real surface_normal[AMREX_SPACEDIM];
        amrex::Real ccibvec[AMREX_SPACEDIM];
        amrex::Real midht=0.5*(prob_lo[1]+prob_hi[1]);

        //interior cell
        IntVect intcell=(sgn==1)?lcell:rcell;
        amrex::Real outward_normal[AMREX_SPACEDIM]={0.0};
        outward_normal[dir]=sgn;
        amrex::Real dx2=dx[dir]*dx[dir];
        
        IntVect cutcell=(sgn==1)?rcell:lcell;
        
        if(prob_parm.sharp_interface_scheme)
        {
            get_surface_point(cutcell,prob_lo,prob_hi,domlo,domhi,dx,prob_parm,xib);
            surfacenorm(xib[0],xib[1],prob_lo,prob_hi,prob_parm,surface_normal);
        }
        amrex::Real ndens = 0.0;
        for(int sp=0; sp<NUM_SPECIES; sp++) ndens += sb_arr(intcell,sp);

        if(solved_comp==POT_ID)
        {
            //We are solving -B del.(beta * grad(phi))=source
            //in vidyut where B=1 and beta=-1
            //This is unlike the species equations where
            //B=1 and beta > 0
            //FVM for cell i at the boundary
            //(-B beta grad(phi).\hat{n}|internal 
            //-B beta grad(phi).\hat{n}|boundary)/h = source
            //now grad(phi).\hat{n} is always (phi_d-phi_i)/(h/2)
            //therefore, diagonal contribution is always +B beta/(0.5*h^2)
            //rhs contribution is always +B beta phi_d/(0.5*h^2)
            //remember B*beta=-1 for Poisson 
            //and B*beta=beta>0 for other equations

            //for both left or right being the interior cell 
            //Dirichlet condition works out to the same sign
            //
            //for sharp interface reconstruction
            //grad(phi).n = (phi_d - phi_i)/d_ccib \hat{d_ccib}.\hat{n}
            //grad(phi).n = (phi_d - phi_i)/d_ccib \vec{d_ccib}/d_ccib.\hat{n}
            //grad(phi).n = (phi_d - phi_i)/d_ccib^2 \vec{d_ccib}.\hat{n}

            amrex::Real phi_dirc=(xloc[1]>midht)?0.0:
            get_applied_potential(time,prob_parm.V1,prob_parm.tpulse,prob_parm.ramptime,prob_parm.pulse_freq);

            if(prob_parm.sharp_interface_scheme)
            {
                ccibvec[0]=xib[0]-(prob_lo[0]+(intcell[0]+0.5)*dx[0]);
                ccibvec[1]=xib[1]-(prob_lo[1]+(intcell[1]+0.5)*dx[1]);
                amrex::Real d_ccib2=ccibvec[0]*ccibvec[0]+ccibvec[1]*ccibvec[1];
                acoeff(intcell) += (bscalar/ascalar)*bcoeff(intcell,0)/d_ccib2*ccibvec[dir]*outward_normal[dir]/dx[dir];
                rhs(intcell,rhs_comp)   += bscalar*bcoeff(intcell,0)*phi_dirc/d_ccib2*ccibvec[dir]*outward_normal[dir]/dx[dir];
            }
            else
            {
                acoeff(intcell) += (bscalar/ascalar)*bcoeff(intcell,0)/(0.5*dx2);
                rhs(intcell,rhs_comp)   += bscalar*bcoeff(intcell,0)*phi_dirc/(0.5*dx2);
            }
        }
        else if(solved_comp==E_ID || solved_comp==EEN_ID) 
        {
            // |--x--|--x--|-----|
            //(f_{i+1/2}-f_{ib})/dx=rhs
            //
            //|-----|--x--|--x--|
            //(f_{ib}-f_{i-1/2})/dx=rhs
            amrex::Real ne=sb_arr(intcell,E_ID);
            amrex::Real Te=sb_arr(intcell,ETEMP_ID);
            amrex::Real cbar=sqrt(8.0*K_B*Te/PI/ME);
            amrex::Real flx = ne*cbar/4.0;
            if(solved_comp==EEN_ID)
            {
                flx*=(2.0*K_B*Te);
            }
            //flx=0.0;
            if(prob_parm.sharp_interface_scheme)
            {
                rhs(intcell,rhs_comp)   -= bscalar*flx/dx[dir]*amrex::Math::abs(surface_normal[dir]);
            }
            else
            {
                rhs(intcell,rhs_comp)   -= bscalar*flx/dx[dir];
            }
        }
        else if(solved_comp==ARp_ID)
        {
            amrex::Real efieldmag=std::sqrt(amrex::Math::powi<2>(sb_arr(intcell,EFX_ID))+
                                            amrex::Math::powi<2>(sb_arr(intcell,EFY_ID))+
                                            amrex::Math::powi<2>(sb_arr(intcell,EFZ_ID)));

            amrex::Real mu = specMob(solved_comp,sb_arr(intcell,ETEMP_ID),
                                     ndens, efieldmag,
                                     Tg);
            amrex::Real efield=sb_arr(intcell,EFX_ID+dir);
            amrex::Real ion_drift_flux = mu*sb_arr(intcell,solved_comp)*efield;
            amrex::Real ion_drift_flux_dotn = ion_drift_flux*outward_normal[dir];
            amrex::Real flx = (ion_drift_flux_dotn >= 0.0)?ion_drift_flux_dotn:0.0;

            //this remains the same 
            //even with sharp interface scheme
            rhs(intcell,rhs_comp) -= bscalar*flx/dx[dir];
        }
        else //any of the neutrals
        {
            amrex::Real cbar=sqrt(8.0*K_B*Tg/PI*imw(solved_comp));
            amrex::Real flx = sb_arr(intcell,solved_comp)*cbar/4.0;
            if(prob_parm.sharp_interface_scheme)
            {
                rhs(intcell,rhs_comp)   -= bscalar*flx/dx[dir]*amrex::Math::abs(surface_normal[dir]);
            }
            else
            {
                rhs(intcell,rhs_comp)   -= bscalar*flx/dx[dir];
            }
        }

    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    int is_dielectric(int i, int j, int k, int dir, int sgn, 
                      const GpuArray<Real,AMREX_SPACEDIM> prob_lo, 
                      const GpuArray<Real,AMREX_SPACEDIM> prob_hi, 
                      const GpuArray<Real,AMREX_SPACEDIM> dx,
                      Real time,
                      ProbParm const& prob_parm)
    {
        int val=0;
        return(val);
    } 

    AMREX_GPU_DEVICE AMREX_INLINE
    void potential_bc(int i, int j, int k,
                      int dir, int sgn,
                      Array4<Real> const& phi,
                      Array4<Real> const& bc_arr,
                      Array4<Real> const& robin_a,
                      Array4<Real> const& robin_b,
                      Array4<Real> const& robin_f,
                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                      GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                      GpuArray<Real, AMREX_SPACEDIM> dx,
                      const Real time,
                      ProbParm const& prob_parm,
                      amrex::Real Tg,amrex::Real Pg,
                      amrex::Real app_voltage,
                      GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
                      GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
                      GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)
    {
        IntVect cell_int{AMREX_D_DECL(i, j, k)};
        IntVect ghost_cell{AMREX_D_DECL(i, j, k)};
        amrex::Real outward_normal[AMREX_SPACEDIM] = {0.0};
        outward_normal[dir] = sgn;

        int gcell_adjust, cell_adjust;

        // ghost_cell is one behind
        gcell_adjust = (sgn == -1) ? -1 : 0;
        // cell is one behind
        cell_adjust = (sgn == 1) ? -1 : 0;

        ghost_cell[dir] += gcell_adjust;
        cell_int[dir] += cell_adjust;

        amrex::Real y = prob_lo[1] + (cell_int[1] + 0.5) * dx[1];

        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void photoionization_bc(int i, int j, int k,
                            int dir, int sgn,
                            Array4<Real> const& phi,
                            Array4<Real> const& bc_arr,
                            Array4<Real> const& robin_a,
                            Array4<Real> const& robin_b,
                            Array4<Real> const& robin_f,
                            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                            GpuArray<Real, AMREX_SPACEDIM> dx,
                            const Real time,
                            ProbParm const& prob_parm,
                            amrex::Real Tg,amrex::Real Pg)
    {
        IntVect cell_int{AMREX_D_DECL(i,j,k)};
        IntVect ghost_cell{AMREX_D_DECL(i,j,k)};
        amrex::Real outward_normal[AMREX_SPACEDIM]={0.0};
        outward_normal[dir]=sgn;

        int gcell_adjust,cell_adjust;

        //ghost_cell is one behind
        gcell_adjust= (sgn==-1)?-1:0;
        //cell is one behind
        cell_adjust = (sgn==1)?-1:0;

        ghost_cell[dir]+=gcell_adjust;
        cell_int[dir]+=cell_adjust;   

        robin_a(ghost_cell) = 0.0;
        robin_b(ghost_cell) = 1.0;
        robin_f(ghost_cell) = 0.0;
        bc_arr(ghost_cell) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void species_bc(int i,int j,int k, int dir, int sgn, 
                    int spec_id, int bcspec_id,
                    Array4<Real> const &phi, 
                    Array4<Real> const& bc_arr,
                    Array4<Real> const& robin_a,
                    Array4<Real> const& robin_b,
                    Array4<Real> const& robin_f,
                    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                    GpuArray<Real, AMREX_SPACEDIM> dx,
                    const Real time,
                    ProbParm const& prob_parm,
                    amrex::Real Tg,amrex::Real Pg,
                    GpuArray<Real, MAX_CURRENT_LOCS> int_currents,
                    GpuArray<Real, MAX_CURRENT_LOCS> int_current_areas,
                    GpuArray<int, MAX_CURRENT_LOCS> int_current_surfaces)

    {
        IntVect cell_int{AMREX_D_DECL(i,j,k)};
        IntVect ghost_cell{AMREX_D_DECL(i,j,k)};
        amrex::Real outward_normal[AMREX_SPACEDIM]={0.0};
        outward_normal[dir]=sgn;

        int gcell_adjust,cell_adjust;

        //ghost_cell is one behind
        gcell_adjust= (sgn==-1)?-1:0;
        //cell is one behind
        cell_adjust = (sgn==1)?-1:0;

        ghost_cell[dir]+=gcell_adjust;
        cell_int[dir]+=cell_adjust;

        amrex::Real y=prob_lo[1]+(cell_int[1]+0.5)*dx[1];
        int chrgspec[NUM_SPECIES+1]={0}; //include electron energy
        CKCHRG(chrgspec);

        //default
        robin_a(ghost_cell,bcspec_id) = 0.0;
        robin_b(ghost_cell,bcspec_id) = 1.0;
        robin_f(ghost_cell,bcspec_id) = 0.0;
        bc_arr(ghost_cell,bcspec_id) = 0.0;

    }

    AMREX_GPU_DEVICE AMREX_INLINE
    amrex::Real compute_vel(IntVect iv, int dir,
                            int specid,
                            Array4<Real> const& phi,
                            GpuArray<Real, AMREX_SPACEDIM> dx,
                            const Real time,
                            ProbParm const& prob_parm,
                            amrex::Real Tg,amrex::Real Pg)
    {
        amrex::Real efieldmag=std::sqrt( amrex::Math::powi<2>(phi(iv,EFX_ID))+
                                        amrex::Math::powi<2>(phi(iv,EFY_ID))+
                                        amrex::Math::powi<2>(phi(iv,EFZ_ID)) );
        amrex::Real ndens = 0.0;
        for(int sp=0; sp<NUM_SPECIES; sp++) ndens += phi(iv,sp);

        amrex::Real mu = specMob(specid,phi(iv,ETEMP_ID),
                                 ndens,efieldmag,Tg);

        return(mu*phi(iv,EFX_ID+dir));
    }
}
#endif
