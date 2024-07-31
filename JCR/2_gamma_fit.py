import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import jax.numpy as jnp
from jax import grad
import LibjaxCR as jCR
import LibpltCR as fCR
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from healpy.newvisufunc import projview, newprojplot
import healpy as hp
import h5py
import optax
from jax import jit
import matplotlib.cm as cm

# Load diffuse gamma-ray map from Platz et al. 2023
with h5py.File('energy_bins.hdf5', 'r') as file:
    print("Keys: %s" % file.keys())
    Eg_data=file['geom_avg_bin_energy'][:]
    Eg_data_lower=file['lower_bin_boundaries'][:]
    Eg_data_upper=file['upper_bin_boundaries'][:]
    
dEg_data=Eg_data_upper-Eg_data_lower

with h5py.File('I_dust.hdf5', 'r') as file:
    print("Keys: %s" % file['stats'].keys())
    gamma_map_mean=file['stats']['mean'][:]
    gamma_map_std=file['stats']['standard deviation'][:]

gamma_map_mean*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_mean=hp.ud_grade(gamma_map_mean[5,:], nside_out=64)
gamma_map_mean=hp.reorder(gamma_map_mean, r2n=True)
gamma_map_mean=gamma_map_mean[np.newaxis,np.newaxis,:] # GeV^-1 cm^-2 s^-1
gamma_map_mean=jnp.array(gamma_map_mean)

gamma_map_std*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_std=hp.ud_grade(gamma_map_std[5,:], nside_out=64)
gamma_map_std=hp.reorder(gamma_map_std, r2n=True)
gamma_map_std=gamma_map_std[np.newaxis,np.newaxis,:] # GeV^-1 cm^-2 s^-1
gamma_map_std=jnp.array(gamma_map_mean)

# Plot gamma-ray sky maps
def plot_gamma_map(i, gamma_map_theta, gamma_map):

    fig=plt.figure(figsize=(18, 5))

    projview(
        np.log10(gamma_map_theta[0,0,:]), 
        title=r'Iteration %d gamma-ray map at $E_\gamma=10$ GeV' % i,
        coord=["G"], cmap='magma',
        min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\phi_{\rm fit}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=131
    )

    projview(
        np.log10(gamma_map[0,0,:]), 
        title=r'Mock data gamma-ray map at $E_\gamma=10$ GeV',
        coord=["G"], cmap='magma',
        min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\left[\phi_{\rm data}(E_\gamma)\right]\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=132
    )

    projview(
        np.abs(np.log10(gamma_map_theta[0,0,:]/gamma_map[0,0,:])), 
        title=r'Ratio',
        coord=["G"], cmap='magma',
        min=0.0, max=0.5,
        cbar_ticks=[0, 0.25, 0.5],
        nest=True, 
        unit=r'$\log_{10}\left[\phi_{\rm fit}(E_\gamma)/\phi_{\rm data}(E_\gamma)\right]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=133
    )

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

    plt.savefig('fg_gamma-map_FERMI5_%d.png' % i, dpi=300)
    plt.close()

fs=22

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

# Define cosmic-ray and gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
E=jnp.logspace(10.0,14.0,81) # eV 
# Eg=jnp.logspace(1,2,2) # GeV
Eg=jnp.logspace(np.log10(13.33521432163324),2,1) # GeV
dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=jCR.load_gas('../samples_densities_hpixr.fits')
ngas_mean=jnp.mean(ngas,axis=0)[jnp.newaxis,:,:]

# # Load data of gamma-ray map
# data=np.load('gamma_map.npz')
# gamma_map_mean=jnp.array(data['gamma_map'])

N=501

scan_pars=[]
theta=jnp.array([1.6e-9,0.6,1.6,4.0])
theta_min=jnp.array([0.5e-9,0.1,1.0,2.5])
theta_max=jnp.array([5.0e-9,4.0,2.0,4.5])

grad_init=jnp.abs(grad(jCR.loss_func_gamma_map)(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E,gamma_map_mean,gamma_map_std))
print('Initial gradient -> ', grad_init)

learning_rate=0.01*theta*(grad_init/(grad_init+1.0e-8))
optimizer=optax.adam(learning_rate)
opt_state=optimizer.init(theta)

@jit
def update_gamma_map(theta, opt_state, pars_prop, zeta_n, dXSdEg_Geant4, ngas, drs, points_intr, E, gamma_map_data, gamma_map_std):
    grads=grad(jCR.loss_func_gamma_map)(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E,gamma_map_data,gamma_map_std)
    updates, opt_state=optimizer.update(grads,opt_state)
    theta=optax.apply_updates(theta,updates)
    return theta, opt_state

color=cm.magma(np.linspace(0, 1, N))

for i in range(N):
    theta, opt_state=update_gamma_map(theta,opt_state,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E,gamma_map_mean,gamma_map_std)
    theta=jnp.clip(theta, theta_min, theta_max)
    current_loss=jCR.loss_func_gamma_map(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E,gamma_map_mean,gamma_map_std)
    scan_pars.append(np.concatenate([[current_loss], np.array(theta)]))

print('A_org=',1.0/5.95828e+8,'B_org=',0.55,'C_org=',1.64,'D_org=',4.01)

np.savetxt("scan_pars.txt", np.array(scan_pars), header="chi2 norm r_off index exp_arg_norm")

print('Runtime:',time.time()-start_time,'seconds')

