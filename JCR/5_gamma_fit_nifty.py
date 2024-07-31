# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Demonstration of the non-parametric correlated field model in NIFTy.re

# The Model
import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("text",usetex=True)

from jax import jit
import jax.random as jr
from jax import numpy as jnp
import jax.scipy as jsp

import LibjaxCR as jCR

import scipy as sp
import nifty8.re as jft

import healpy as hp
import h5py
import numpy as np
import time

from healpy.newvisufunc import projview, newprojplot

start_time = time.time()

mp = 938.272e6 # eV -> Proton mass

l, b=hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
l=np.where(l<0,l+360,l)

mask=(np.abs(b)<=30.0) 

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
gamma_map_std=jnp.array(gamma_map_std)

def plot_data(gamma_map_mean, gamma_map_std):
    fig=plt.figure(figsize=(18, 5))

    projview(
        np.log10(gamma_map_mean[0,0,:]), 
        title=r'Iteration gamma-ray map at $E_\gamma=10$ GeV',
        coord=["G"], cmap='magma',
        # min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\phi_{\rm fit}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=121
    )

    projview(
        np.log10(gamma_map_std[0,0,:]),
        title=r'Iteration gamma-ray map at $E_\gamma=10$ GeV',
        coord=["G"], cmap='magma',
        # min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\phi_{\rm fit}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=122
    )

    plt.savefig('Results_nifty/fg_gamma-map_nifty_2.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot((np.log(gamma_map_mean*1.0e8).ravel()), color='red', alpha=1)
    ax.plot((gamma_map_std/gamma_map_mean).ravel(), 'k:', alpha=0.2)

    # ax[0].set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
    # ax[0].set_xlabel(r'$R\, {\rm (pc)}$')

    # ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig("Results_nifty/data_test.png", dpi=400)

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=100
zeta_n=jnp.array(sp.special.jn_zeros(0, num_zeros))

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

# Define cosmic-ray grid and diffusion coefficient
E=jnp.logspace(10.0,14.0,81) # eV 
p=jnp.sqrt((E+mp)**2-mp**2)  # eV
vp=p/(E+mp)
Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/312.0e9)**2)**0.1 # pc^2/yr

# Injection spectrum of sources
xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)

RSNR=0.03 # yr^-1 -> SNR rate
ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
QE=RSNR*vp*3.0e10*(xiSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha)

# Bessel functions
r_int=jnp.linspace(0.0,R,25000)
j0_n_int=jCR.j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]/R)

# Define gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
Eg=jnp.logspace(np.log10(13.33521432163324),2,1) # GeV
dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=jCR.load_gas('../samples_densities_hpixr.fits')
ngas_mean=jnp.mean(ngas,axis=0)[jnp.newaxis,:,:]
_, _, Npix=ngas_mean.shape

seed = 42
key = jr.PRNGKey(seed)

R_SN = 15000.0 # pc -> SNR distribution extends up to R_SN

dims = (76, ) # Number of spatial grid points in the reconstructed gSNR 

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-5, 1e-2),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(1.0, 1.0, name="scaling", shape=(1,))

class Signal(jft.Model):
    def __init__(self, correlated_field, scaling, width=10.0):
        self.cf = correlated_field
        self.scaling = scaling
        self.rg_SN = jnp.linspace(0.0, R_SN, dims[0])

        self.zeta_n = zeta_n
        self.pars_prop = pars_prop
        self.Diff = Diff
        self.QE = QE

        self.r_int = r_int
        self.j0_n_int = j0_n_int

        self.mask=mask

        # Init methods of the Correlated Field model and any prior model in
        # NIFTy.re are aware that their input is standard normal a priori.
        # The `domain` of a model does not know this. Thus, tracking the `init`
        # methods should be preferred over tracking the `domain`.
        super().__init__(init=self.cf.init | self.scaling.init)

    # Spatial distribution of sources
    def func_gSNR(self, x):
        return self.scaling(x) * jnp.exp(self.cf(x) + jnp.log(jCR.func_gSNR_YUK04(self.rg_SN)))  

    # Coefficients for Bessel expansion
    def func_coef(self, x):
        fr_int=jnp.interp(self.r_int, self.rg_SN, self.func_gSNR(x), right=0.0)
        q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*self.j0_n_int,r_int)
        q_n*=(2.0/(self.pars_prop[0]**2*(jCR.j1(self.zeta_n)**2))) # pc^-2

        return q_n

    # Compute gamma-ray map
    def func_gamma_map(self, x):

        # Transport parameters
        R_CR=self.pars_prop[0] # pc
        L_CR=self.pars_prop[1] # pc
        u0_CR=self.pars_prop[4]*365.0*86400.0/3.086e18 # km/s to pc/yr -> Advection speed

        # Bessel expansion of source distribution
        # fr_int=jnp.interp(self.r_int, self.rg_SN, self.func_gSNR(x), right=0.0)
        # q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*self.j0_n_int,r_int)
        # q_n*=(2.0/(R_CR**2*(jCR.j1(self.zeta_n)**2))) # pc^-2

        q_n=self.func_coef(x)

        # Spatial grid for cosmic-ray
        rg=jnp.linspace(0.0,R_CR,41) # pc
        zg=jnp.linspace(0.0,L_CR,501) # pc

        Diff_CR=self.Diff[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
        zg_CR=zg[jnp.newaxis,jnp.newaxis,jnp.newaxis,:] # pc
        rg_CR=rg[jnp.newaxis,jnp.newaxis,:,jnp.newaxis] # pc
        zeta_n_CR=self.zeta_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]
        q_n_CR=q_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]

        Sn=jnp.sqrt((u0_CR/Diff_CR)**2+4.0*(zeta_n_CR/R_CR)**2) # pc^-2
        fEn=jCR.j0(zeta_n_CR*rg_CR/R_CR)*q_n_CR*jnp.exp(u0_CR*zg_CR/(2.0*Diff_CR))*jnp.sinh(Sn*(L_CR-zg_CR)/2.0)/(jnp.sinh(Sn*L_CR/2.0)*(u0_CR+Sn*Diff_CR*(jnp.cosh(Sn*L_CR/2.0)/jnp.sinh(Sn*L_CR/2.0))))
        fE=jnp.sum(fEn,axis=0) # eV^-1 pc^-3
        fE=jnp.where(fE<0.0,0.0,fE)

        jE=fE*self.QE[:,jnp.newaxis,jnp.newaxis]*1.0e9/(3.086e18)**3 # GeV^-1 cm^-2 s^-1

        # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
        qg_Geant4=1.88*jnp.trapezoid(jE[:,jnp.newaxis,:,:]*dXSdEg_Geant4[:,:,jnp.newaxis,jnp.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 -> Enhancement factor 1.88 from Kachelriess et al. 2014

        # Interpolate gamma-ray emissivity on healpix-r grid as gas
        qg_Geant4_healpixr=jCR.get_healpix_interp(qg_Geant4,rg,zg,points_intr) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity

        # Compute the diffuse emission in all gas samples
        gamma_map=jCR.func_gamma_map(ngas_mean,qg_Geant4_healpixr,drs) # GeV^-1 cm^-2 s^-1

        return gamma_map

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        return jnp.log(self.func_gamma_map(x)[0,0,mask]*1.0e8)#.ravel()

signal_response = Signal(correlated_field, scaling)

key, subkey = jr.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)

# signal_response_truth=signal_response(pos_truth)
# keys = jr.split(key, Npix)
# noise_truth=0.1*jnp.ones_like(signal_response_truth) # jnp.array([jr.normal(keys[i]) * 0.1 for i in range(Npix)])
# plot_data(signal_response.func_gamma_map(pos_truth),0.1*signal_response.func_gamma_map(pos_truth))

# # Plotting gSNR and power spectrum
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# # Plot mock data
# ax.plot(signal_response.rg_SN, signal_response.func_gSNR(pos_truth), c='C5', alpha=1.0, label='Mock data')
# ax.plot(signal_response.rg_SN, np.sum(signal_response.func_coef(pos_truth)[:,np.newaxis]*sp.special.j0(zeta_n[:,np.newaxis]*signal_response.rg_SN[np.newaxis,:]/20000.0), axis=0), 'g:')
# ax.plot(signal_response.rg_SN, jCR.func_gSNR_YUK04(signal_response.rg_SN), 'r-', alpha=1.0, label='YUK 04')

# for i in range(10):
#     key, subkey = jr.split(key)
#     pos_test = jft.random_like(subkey, signal_response.domain)
#     ax.plot(signal_response.rg_SN, signal_response.func_gSNR(pos_test), c='C5', alpha=1.0)
#     ax.plot(signal_response.rg_SN, np.sum(signal_response.func_coef(pos_test)[:,np.newaxis]*sp.special.j0(zeta_n[:,np.newaxis]*signal_response.rg_SN[np.newaxis,:]/20000.0), axis=0), 'g:')


# ax.set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
# ax.set_xlabel(r'$R\, {\rm (pc)}$')

# ax.legend(loc='upper left', prop={"size":22})

# fig.tight_layout()
# fig.savefig("Results_nifty/results_gSNR_test_test_5_2.png", dpi=400)


signal_response_truth = jnp.log(gamma_map_mean*1.0e8)[0,0,mask]
noise_truth = (gamma_map_std/gamma_map_mean)[0,0,mask]
plot_data(gamma_map_mean,gamma_map_std)

lh = jft.Gaussian(signal_response_truth, noise_cov_inv=1.0/noise_truth**2).amend(signal_response)

# Now lets run the main inference scheme:
n_vi_iterations = 6
delta = 1e-4
n_samples = 11

key, k_i, k_o = jr.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="Results_nifty",
    resume=False,
)

rG=jnp.linspace(0,R_SN,dims[0])
gSNR_sample = jnp.array(tuple(signal_response.func_gSNR(s) for s in samples))
spectrum_gSNR_sample = jnp.array(tuple(cfm.amplitude(s)[1:] for s in samples))
q_n_sample = jnp.array(tuple(signal_response.func_coef(s) for s in samples))
gamma_sample = jnp.array(tuple(signal_response.func_gamma_map(s) for s in samples))

# Save the gamma-ray maps in a .npz file
# np.savez('Results_nifty/gSNR_sample_3.npz', rG=np.array(rG), gSNR=np.array(gSNR_sample), spectrum_gSNR=np.array(spectrum_gSNR_sample), gSNR_truth=np.array(signal_response.func_gSNR(pos_truth)), q_n=np.array(q_n_sample), gamma=np.array(gamma_sample), gamma_truth=np.array(signal_response.func_gamma_map(pos_truth)))
np.savez('Results_nifty/gSNR_sample_4.npz', rG=np.array(rG), gSNR=np.array(gSNR_sample), spectrum_gSNR=np.array(spectrum_gSNR_sample), gSNR_truth=np.array(signal_response.func_gSNR(pos_truth)), q_n=np.array(q_n_sample), gamma=np.array(gamma_sample), gamma_truth=np.array(gamma_map_mean))

print('Runtime: ',time.time()-start_time,'s')