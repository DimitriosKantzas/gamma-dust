import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("text",usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import numpy as np
import LibjaxCR as jCR
import jax.numpy as jnp

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
    plt.close()

# Plot the local spectrum
def plot_jEp_LOC(theta, pars_prop, zeta_n, Rsol):

    fs=22

    mp=938.272e6 # eV
    E=np.logspace(8.0,14.0,61)
    jE_loc=1.0e-9*jCR.func_jE_fit(theta,pars_prop,zeta_n,E,jnp.array([Rsol]),np.array([0.0]))[:,0,0]/(4.0*np.pi) # eV^-1 cm^-2 s^-1 sr^-1

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    n=2

    ax.plot(E,E**n*jE_loc,'k-',linewidth=3,label=r'${\rm Local\, Spectrum}$')
    ax.plot(E,E**n*jE_loc/1.3,'r-',linewidth=3,label=r'${\rm Local\, Spectrum}$')

    # Read data of CRDB file
    # AMS data
    data=np.genfromtxt('../crdb_import_p_R.csv', delimiter='","', dtype=str)
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
    data=np.genfromtxt('../crdb_import_p_Ek.csv', delimiter='","', dtype=str)
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
        ax.errorbar(Ep[expname==expplot[i]],Ep[expname==expplot[i]]**n*jE_data[expname==expplot[i]],Ep[expname==expplot[i]]**n*err_jE_data[expname==expplot[i]],[Ep[expname==expplot[i]]-Ep_lower[expname==expplot[i]],Ep_upper[expname==expplot[i]]-Ep[expname==expplot[i]]],'^',color=expcolr[i],markersize=10.0,elinewidth=2.5,label=r'${\rm %s}$' % expplot[i])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E \,{\rm (eV)}$',fontsize=fs)
    ax.set_ylabel(r'$j(E)\, {\rm (eV\,cm^{-2}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.set_xlim(E[0],E[-1])
    ax.set_ylim(1.0e5,1.0e9)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig("fg_jEp_LOC.png")
    plt.close()