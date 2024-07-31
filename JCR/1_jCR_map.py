import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import jax.numpy as jnp
import LibjaxCR as jCR
import LibpltCR as fCR
import time
import scipy as sp
import numpy as np

# Record the starting time
start_time=time.time()

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)
zeta_n=jnp.array(zeta_n)

# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

# Parameters for injection spectrum
alpha=4.23 # -> Injection spectral index
xiSNR=0.065 # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0=7.0 # km/s -> Advection speed

# Combine all parameters for proagation
pars_prop=jnp.array([R, L, alpha, xiSNR, u0])

# Define grid for cosmic-ray distribution
rg=jnp.linspace(0.0,R,501)    # pc
zg=jnp.linspace(0.0,L,41)     # pc
E=jnp.logspace(10.0,14.0,81) # eV 

# Define gamma-ray energy grid and compute the cross-section from Kafexhiu's code (numpy does not work)
Eg=jnp.logspace(1,2,2) # GeV
dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg) # cm^2 GeV^-1

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=jCR.load_gas('../samples_densities_hpixr.fits')

start_time=time.time()
N=1
for i in range(N):

    # Compute cosmic-ray flux
    jE=jCR.func_jE_YUK04(pars_prop,zeta_n,E,rg,zg)

    # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
    qg_Geant4=1.8*jnp.trapezoid(jE[:,jnp.newaxis,:,:]*dXSdEg_Geant4[:,:,jnp.newaxis,jnp.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 

    # Interpolate gamma-ray emissivity on healpix-r grid as gas
    qg_Geant4_healpixr=jCR.get_healpix_interp(qg_Geant4,rg,zg,points_intr) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity

    # Compute the diffuse emission in all gas samples
    gamma_map=jCR.func_gamma_map(ngas,qg_Geant4_healpixr,drs) # GeV^-1 cm^-2 s^-1

# Record the time finishing computing cosmic-ray map
end_time=time.time()

# Calculate the computing time
print("Computing time in %2d cosmic-ray parameters and %2d gamma-ray energy bin: " % (N,len(Eg)), end_time-start_time, "seconds")

# Save the gamma-ray maps in a .npz file
np.savez('gamma_map.npz', Eg=np.array(Eg), gamma_map=np.array(gamma_map))

fCR.plot_jEp_GAL(np.array(jE),rg,zg)