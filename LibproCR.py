import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("text",usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import healpy as hp

mp=938.272e6 # eV -> Proton mass

#############################################################################
# Auxiliary functions
#############################################################################

# Compute the coefficients q_n
def compute_coefficients(f, zeta_n, R):
    coefficients = []
    for zeta in zeta_n:
        integrand = lambda r: r*f(r)*sp.special.j0(zeta*r/R)
        integral, _=sp.integrate.quad(integrand,0,R,epsabs=1.0e-11,epsrel=1.0e-11)
        coefficient=(2.0/(R**2*(sp.special.j1(zeta)**2)))*integral
        coefficients.append(coefficient)
    return np.array(coefficients)

# Reconstruct the function using the series expansion
def reconstruct_function(r, q_n, zeta_n, R):
    expansion=np.zeros_like(r)
    for q, zeta in zip(q_n, zeta_n):
        expansion+=q*sp.special.j0(zeta*r/R)
    return expansion

#############################################################################
# Functions for transport
#############################################################################

# Surface density of SNRs from Yusifov et al. 2004
def func_gSNR_YUK04(r):
# r (pc)

    r=np.array(r)*1.0e-3 # kpc
    gSNR=np.where(
        r<15.0,
        np.power((r+0.55)/9.05,1.64)*np.exp(-4.01*(r-8.5)/9.05)/5.95828e+8,
        0.0
    )    
    return gSNR # pc^-2

# Diffusion coefficient -> Rigidity dependence (Genolini et al. 2017)
def func_D_rigid(E):
# E (eV)
    
    pb=312.0e9
    p=np.sqrt((E+mp)**2-mp**2)  # eV
    vp=p/(E+mp)

    Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1
    
    return Diff # pc^2/yr

# Normalization for the injection spectrum
def func_Gam(alpha):

    xmin=np.sqrt((1.0e8+mp)**2-mp**2)/mp
    xmax=np.sqrt((1.0e14+mp)**2-mp**2)/mp
    
    Gam, _ =sp.integrate.quad(lambda x: x**(2.0-alpha)*(np.sqrt(x**2+1.0)-1.0),xmin,xmax)
    
    return Gam

# Source function from supernova remnants
def func_QSNR(alpha, xiSNR, E):
# E (eV)

    RSNR=0.03 # yr^-1 -> SNR rate
    ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
    p=np.sqrt((E+mp)**2-mp**2) # eV
    vp=p/(E+mp)
    
    Gam=func_Gam(alpha)
    Q0=xiSNR*ENSR/(mp**2*vp*Gam)
    Q=Q0*(p/mp)**(2.0-alpha)
    
    return RSNR*Q # eV^-1 yr^-1

# Compute the differential number density of protons
def func_jE(pars_prop, zeta_n, q_n, E, rg, zg):
# E (eV), rg (pc), and zg (pc)

    R=pars_prop[0] # pc
    L=pars_prop[1] # pc
    alpha=pars_prop[2] 
    xiSNR=pars_prop[3]
    u0=pars_prop[4]*365.0*86400.0/3.086e18 # km/s to pc/yr -> Advection speed

    QE=func_QSNR(alpha,xiSNR,E)*np.sqrt((E+mp)**2-mp**2)*3.0e10/(E+mp) # eV^-1 yr^-1 cm s^-1
    Diff=func_D_rigid(E)[np.newaxis,:,np.newaxis,np.newaxis] # pc^2/yr

    zg=zg[np.newaxis,np.newaxis,np.newaxis,:]
    rg=rg[np.newaxis,np.newaxis,:,np.newaxis]
    zeta_n=zeta_n[:,np.newaxis,np.newaxis,np.newaxis]
    q_n=q_n[:,np.newaxis,np.newaxis,np.newaxis]

    Sn=np.sqrt((u0/Diff)**2+4.0*(zeta_n/R)**2) # pc^-2
    fEn=sp.special.j0(zeta_n*rg/R)*q_n*np.exp(u0*zg/(2.0*Diff))*np.sinh(Sn*(L-zg)/2.0)/(np.sinh(Sn*L/2.0)*(u0+Sn*Diff*(np.cosh(Sn*L/2.0)/np.sinh(Sn*L/2.0))))
    fE=np.sum(fEn,axis=0) # eV^-1 pc^-3
    fE=np.where(fE<0.0,0.0,fE)

    jE=fE*QE[:,np.newaxis,np.newaxis]*1.0e9/(3.086e18)**3 

    return jE # GeV^-1 cm^-2 s^-1

#############################################################################
# Plots
#############################################################################

# Function to test the Bessel series
def plot_gSNR(zeta_n,q_n,r,R):

    fs=22

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    # Testing the Bessel series for the function of the surface density of SNRs
    gSNR_reconstructed=reconstruct_function(r,q_n,zeta_n,R)
    gSNR=func_gSNR_YUK04(r)

    ax.plot(r*1.0e-3,gSNR,label='Original',lw=2)
    ax.plot(r*1.0e-3,gSNR_reconstructed,label='Bessel series', lw=2, linestyle='--')

    ax.set_xlabel(r'$r\, ({\rm kpc})$',fontsize=fs)
    ax.set_ylabel(r'$g_{\rm SNR}(r)\,({\rm pc^{-2}})$', fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')
    
    plt.savefig('fg_gSNR.png')
    print('Saving: fg_gSNR.png')
    plt.close()

# Plot the local spectrum
def plot_jEp_LOC(pars_prop, zeta_n, q_n, Rsol):

    fs=22

    E=np.logspace(8.0,14.0,61)
    jE_loc=func_jE(pars_prop,zeta_n,q_n,E,np.array([Rsol]),np.array([0.0]))[:,0,0]

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    n=2

    ax.plot(E,1.0e-9*E**n*jE_loc/(4.0*np.pi),'k-',linewidth=3,label=r'${\rm Local\, Spectrum}$')

    # Read data of CRDB file
    # AMS data
    data=np.genfromtxt('crdb_import_p_R.csv', delimiter='","', dtype=str)
    pp=(data[:,14].astype(float))*1.0e9 # eV
    pp_lower=(data[:,15].astype(float))*1.0e9 # eV
    pp_upper=(data[:,16].astype(float))*1.0e9 # eV

    Ep=np.sqrt(pp**2+mp**2)-mp # eV
    Ep_lower=np.sqrt(pp_lower**2+mp**2)-mp # eV
    Ep_upper=np.sqrt(pp_upper**2+mp**2)-mp # eV

    vpp=pp/(Ep+mp)
    jE_data=(data[:,17].astype(float))*1.0e-13/vpp # eV^-1 cm^-2 s^-1 sr^-1
    err_jE_data=(data[:,20].astype(float))*1.0e-13/vpp # eV^-1 cm^-2 s^-1 sr^-1
    ax.errorbar(Ep,Ep**n*jE_data,Ep**n*err_jE_data,[Ep-Ep_lower,Ep_upper-Ep],'^',color='green',markersize=10.0,elinewidth=2.5,label=r'${\rm AMS}$')

    # DAMPE and CALET data
    data=np.genfromtxt('crdb_import_p_Ek.csv', delimiter='","', dtype=str)
    expname=data[:,0]
    expname=np.char.lstrip(expname,'"')
    Ep=(data[:,14].astype(float))*1.0e9 # eV
    Ep_lower=(data[:,15].astype(float))*1.0e9 # eV
    Ep_upper=(data[:,16].astype(float))*1.0e9 # eV

    jE_data=(data[:,17].astype(float))*1.0e-13 # eV^-1 cm^-2 s^-1 sr^-1
    err_jE_data=(data[:,20].astype(float))*1.0e-13 # eV^-1 cm^-2 s^-1 sr^-1

    expplot=np.array(['DAMPE', 'CALET'])
    expcolr=np.array(['lightsalmon', 'brown'])
    for i in range(len(expplot)):
        mask=(expname!=expplot[i])
        ax.errorbar(Ep[expname==expplot[i]],Ep[expname==expplot[i]]**n*jE_data[expname==expplot[i]],Ep[expname==expplot[i]]**n*err_jE_data[expname==expplot[i]],[Ep[expname==expplot[i]]-Ep_lower[expname==expplot[i]],Ep_upper[expname==expplot[i]]-Ep[expname==expplot[i]]],'^',color=expcolr[i],markersize=10.0,elinewidth=2.5,label=r'${\rm %s}$' % expplot[i])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E \,{\rm (eV)}$',fontsize=fs)
    ax.set_ylabel(r'$E^2\,j(E)\, {\rm (eV\,cm^{-2}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.set_xlim(E[0],E[-1])
    ax.set_ylim(1.0e5,1.0e9)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig("fg_jEp_LOC.png")
    print('Saving: fg_jEp_LOC.png')
    plt.close()

# Plot the spatial cosmic-ray distribution
def plot_jEp_GAL(jE, rg, zg):

    fs=22

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Spatial distribution over the entire grid
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(jE[0,:,:], origin='lower', extent=[rg[0]*1.0e-3, rg[-1]*1.0e-3, zg[0]*1.0e-3, zg[-1]*1.0e-3], cmap='magma')

    ## Colourbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cax.set_ylabel(r'$j(E) \, ({\rm GeV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}})$')  
    ax1.set_title(r'$E=10$\,{\rm GeV}')
    ax1.set_xlabel(r"$r_{G}\,{\rm [kpc]}$")
    ax1.set_ylabel(r"$z_{G}\,{\rm [kpc]}$")

    # Profile over r
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rg*1.0e-3, jE[0,:,0], 'r')
    ax2.set_title(r'$E=10$\,{\rm GeV}\, {\rm and}\, $z_G=0$\,{\rm kpc}')
    ax2.set_xlabel(r'$r_{G}\,{\rm [kpc]}$')
    ax2.set_ylabel(r'$j(E) \, ({\rm GeV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}})$')
    ax2.set_xlim(rg[0]*1.0e-3,rg[-1]*1.0e-3)

    # Profile over z
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(zg*1.0e-3, jE[0,0,:], 'r')
    # ax3.plot(zg*1.0e-3, fE[0,rg==4000.0,:][0], 'g')
    # ax3.plot(zg*1.0e-3, fE[0,rg==8000.0,:][0], 'k')
    ax3.set_title(r'$E=10$\,{\rm GeV}\, {\rm and}\, $r_G=0$\,{\rm kpc}')
    ax3.set_xlabel(r'$z_{G}\,{\rm [kpc]}$')
    ax3.set_ylabel(r'$f(E) \, ({\rm GeV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}})$')
    ax3.set_xlim(zg[0]*1.0e-3,zg[-1]*1.0e-3)

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)
    
    plt.savefig("fg_jEp_GAL.png", dpi=600)
    print('Saving: fg_jEp_GAL.png')
    plt.close()

# Plot the local gamma-ray emissivity
def plot_emi_LOC(qg, Eg, rg, zg, Rsol):

    fs=22

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    ax.plot(Eg,qg[:,np.abs(rg-Rsol)==np.amin(np.abs(rg-Rsol)),zg==0.0]/(4.0*np.pi),'k-',linewidth=3,label=r'${\rm Local\, Emissivity}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E \,{\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$\varepsilon(E)\, {\rm (GeV^{-1}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
    for label_axd in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_axd.set_fontsize(fs)
    # ax.set_xlim(1.0,100.0)
    # ax.set_ylim(1.0e-37,1.0e-36)
    ax.set_title(r'{\rm Emissivity at $r_G=%.1f$ pc}' % rg[np.abs(rg-Rsol)==np.amin(np.abs(rg-Rsol))], fontsize=fs)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig("fg_emissivity.png")
    print('Saving: fg_emissivity.png')
    plt.close()

#############################################################################
# Function to interpolate emissivity on healpix-r grid
#############################################################################

# Function to interpolate emissivity on healpix-r grid
def get_healpix_interp(qg, Eg, rg, zg, rs, NSIDE, Rsol):

    # Grid on which emissivity is calculated
    points=(rg,zg)

    # Grid on which emissivity is interpolated
    N_rs=len(rs)
    N_pix=12*NSIDE**2
    N_E=len(Eg)

    # Angles for all pixels
    thetas, phis = hp.pix2ang(NSIDE, np.arange(N_pix), nest=True, lonlat=False)

    # Points for interpolation
    ls=phis[np.newaxis,:]
    bs=np.pi/2.0-thetas[np.newaxis,:]
    rs=rs[:,np.newaxis]

    xs=-rs*np.cos(ls)*np.cos(bs)+Rsol
    ys=-rs*np.sin(ls)*np.cos(bs)
    zs=rs*np.sin(bs)

    points_intr=(np.sqrt(xs**2+ys**2), np.abs(zs))

    # Interpolator
    qg_healpix=np.zeros((N_E, N_rs, N_pix))
    for j in range(N_E):
        qg_healpix[j,:,:]=sp.interpolate.interpn(points, qg[j,:,:], points_intr, bounds_error=False, fill_value=0.0)

    return qg_healpix