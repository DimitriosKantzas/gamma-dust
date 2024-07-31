import numpy as np
import scipy as sp
import LibppGam as ppG
import LibproCR as pCR
import matplotlib.pyplot as plt
import time
from astropy.io import fits

# Record the starting time
start_time=time.time()

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)

# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

# Position of solar system from the gas map (see Soding et al. 2024)
Rsol=8178.0 # pc

# Parameters for injection spectrum
alpha=4.23 # -> Injection spectral index
xiSNR=0.065 # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0=7.0 # km/s -> Advection speed

# Combine all parameters for proagation
pars_prop=np.array([R, L, alpha, xiSNR, u0])

# Compute the coefficients
q_n=pCR.compute_coefficients(pCR.func_gSNR_YUK04,zeta_n,R)

# Define grid for cosmic-ray distribution
rg=np.linspace(0.0,R,501)    # pc
# rg=np.linspace(0.0,R,401)    # pc
zg=np.linspace(0.0,L,41)     # pc
# zg=np.linspace(0.0,L,201)     # pc
# E=np.logspace(10.0,14.0,81) # eV 
E=np.logspace(9.0,14.0,81) # eV 
# E=np.logspace(9.0,10.0,11) # eV 
jE=pCR.func_jE(pars_prop,zeta_n,q_n,E,rg,zg) # GeV^-1 cm^-2 s^-1

# Record the time finishing computing cosmic-ray distribution
CR_time=time.time()

# Compute the cross-section from Kafexhiu's code (numpy deos not work)
Eg=np.logspace(1,2,2)
dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
for i in range(len(E)):
    for j in range(len(Eg)):
        dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV

# Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
qg_Geant4=1.8*sp.integrate.trapezoid(jE[:,np.newaxis,:,:]*dXSdEg_Geant4[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 

# Load gas density
hdul=fits.open('samples_densities_hpixr.fits')
rs=(hdul[2].data)['radial pixel edges'].astype(np.float64) # kpc -> Edges of radial bins
drs=np.diff(rs)*3.086e21 # cm -> Radial bin width for line-of-sight integration
rs=(hdul[1].data)['radial pixel centres'].astype(np.float64)*1.0e3 # pc -> Centres of radial bins for interpolating cosmic-ray distribution
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()
ngas=2.0*samples_H2+samples_HI # cm^-3

# Interpolate gamma-ray emissivity on healpix-r grid as gas
N_sample, N_rs, N_pix=samples_HI.shape
NSIDE=int(np.sqrt(N_pix/12))
qg_Geant4_healpixr=pCR.get_healpix_interp(qg_Geant4,Eg,rg,zg,rs,NSIDE,Rsol) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity

# Compute the diffuse emission in all gas samples
gamma_map=np.sum(ngas[:,np.newaxis,:,:]*qg_Geant4_healpixr[np.newaxis,:,:,:]*drs[np.newaxis,np.newaxis,:,np.newaxis],axis=2,dtype=np.dtype(np.float32)) # GeV^-1 cm^-2 s^-1


# Record the time finishing computing cosmic-ray map
end_time=time.time()

# Calculate the computing time
elapsed_time_CR=CR_time-start_time
elapsed_time_gamma=end_time-CR_time

print("Cosmic-ray computing time:                 ", elapsed_time_CR, "seconds")
print("Gamma-ray computing time in %2d energy bin: " % len(Eg), elapsed_time_gamma, "seconds")

# Save the gamma-ray maps in a .npz file
np.savez('gamma_map.npz', Eg=Eg, gamma_map=gamma_map)

# # Plots
# pCR.plot_gSNR(zeta_n,q_n,rg,R)
pCR.plot_jEp_LOC(pars_prop,zeta_n,q_n,Rsol)
# pCR.plot_jEp_GAL(jE/(4.0*np.pi),rg,zg)
pCR.plot_emi_LOC(qg_Geant4,Eg,rg,zg,Rsol)


# # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
# dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
# dXSdEg_Pythia=np.zeros((len(E),len(Eg))) 
# dXSdEg_SIBYLL=np.zeros((len(E),len(Eg))) 
# dXSdEg_QGSJET=np.zeros((len(E),len(Eg))) 
# for i in range(len(E)):
#     for j in range(len(Eg)):
#         dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV
#         dXSdEg_Pythia[i,j]=ppG.dsigma_dEgamma_Pythia8(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV
#         dXSdEg_SIBYLL[i,j]=ppG.dsigma_dEgamma_SIBYLL(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV
#         dXSdEg_QGSJET[i,j]=ppG.dsigma_dEgamma_QGSJET(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV

# jE_loc=pCR.func_jE(pars_prop,zeta_n,q_n,E,np.array([Rsol]),np.array([0.0]))/(4.0*np.pi)
# qg_Geant4_loc=1.8*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_Geant4[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1
# qg_Pythia_loc=1.8*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_Pythia[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1
# qg_SIBYLL_loc=1.8*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_SIBYLL[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1
# qg_QGSJET_loc=1.8*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_QGSJET[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1

# fs=22

# fig=plt.figure(figsize=(10, 8))
# ax=plt.subplot(111)

# ax.plot(Eg,Eg**2.8*qg_Geant4_loc[:,0,0],'k-',linewidth=3,label=r'${\rm Geant\, 4}$')
# ax.plot(Eg,Eg**2.8*qg_Pythia_loc[:,0,0],'r--',linewidth=3,label=r'${\rm Pythia\, 8}$')
# ax.plot(Eg,Eg**2.8*qg_SIBYLL_loc[:,0,0],'g:',linewidth=3,label=r'${\rm SIBYLL}$')
# ax.plot(Eg,Eg**2.8*qg_QGSJET_loc[:,0,0],'m-.',linewidth=3,label=r'${\rm QGSJET}$')

# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel(r'$E \,{\rm (GeV)}$',fontsize=fs)
# ax.set_ylabel(r'$\varepsilon(E)\, {\rm (GeV^{-1}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
# for label_axd in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label_axd.set_fontsize(fs)
# ax.set_xlim(1.0,1.0e3)
# ax.set_ylim(1.0e-28,1.0e-26)
# ax.set_title(r'{\rm Local emissivity}' % rg[np.abs(rg-Rsol)==np.amin(np.abs(rg-Rsol))], fontsize=fs)
# ax.legend(loc='lower left', prop={"size":fs})
# ax.grid(linestyle='--')

# plt.savefig("fg_emissivity.png")
# plt.close()
# print('Plotting: ./fg_emissivity.png')
