import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("text",usetex=True)

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

# Plot gamma-ray sky maps
def plot_gamma_map(gamma_map_theta, gamma_map_mean):

    fig=plt.figure(figsize=(18, 5))

    projview(
        np.log10(gamma_map_theta[0,0,:]), 
        title=r'Sample gamma-ray map at $E_\gamma=10$ GeV',
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
        np.log10(gamma_map_mean[0,0,:]), 
        title=r'Diffuse gamma-ray map at $E_\gamma=10$ GeV',
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
        np.abs(np.log10(gamma_map_theta[0,0,:]/gamma_map_mean[0,0,:])), 
        title=r'Ratio',
        coord=["G"], cmap='magma',
        min=0.0, max=0.05,
        cbar_ticks=[0, 0.05],
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

    # plt.savefig('fg_gamma-map_FERMIQGSJET_%d.png' % i, dpi=300)
    plt.savefig('Results_nn/fg_gamma-map_FERMIQGSJET_bestfit_1_3.png', dpi=300)
    plt.close()

# Load the .npz file
data = np.load('Results_nn/gSNR_nn.npz')

# Access arrays by their names
rG = data['rG']
gSNR = data['gSNR']
gSNR_samples =data['gSNR_samples']
gSNR_truth = data['gSNR_truth']
loss_history = data['loss_history']
gamma_sample = data['gamma']
gamma_truth = data['gamma_truth']


N_samples, _, _ = gSNR_samples.shape

plot_gamma_map(gamma_sample,gamma_truth)

fs=22

# Plotting gSNR 
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot mock data
ax.plot(rG, gSNR_truth, c='C5', alpha=1.0, label='Mock data')
ax.plot(rG, gSNR, c='red', alpha=1, label='Fit')

for i in range(N_samples):
    ax.plot(rG, gSNR_samples[i,:], 'k--')


ax.set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
ax.set_xlabel(r'$R\, {\rm (pc)}$')

ax.legend(loc='upper right', prop={"size":fs})

fig.tight_layout()
fig.savefig("Results_nn/results_gSNR_nn.png", dpi=400)

# Plot the loss function for each iteration
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot mock data
ax.plot(loss_history, c='C5', alpha=1.0, label='loss history')

# ax.set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
# ax.set_xlabel(r'$R\, {\rm (pc)}$')

ax.legend(loc='upper right', prop={"size":fs})
ax.set_yscale('log')

fig.tight_layout()
fig.savefig("Results_nn/results_loss_history_nn.png", dpi=400)