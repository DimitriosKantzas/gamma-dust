# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Demonstration of the non-parametric correlated field model in NIFTy.re

# The Model
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("text",usetex=True)
from jax import numpy as jnp
from jax import random
import LibjaxCR as jCR
import scipy as sp
import nifty8.re as jft

# Enable dtype64
jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

R_SN = 15000.0 # pc -> SNR distribution extends up to R_SN
R_CR = 20000.0 # pc -> CR halo extends up to R_CR

dims = (300, ) # Number of spatial grid points in the reconstructed gSNR 
Ndata = 150    # Number of points in the mock data

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-6, 1e-2),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(3.0, 1.0, name="scaling", shape=(1,))

class Signal(jft.Model):
    def __init__(self, correlated_field, scaling, width=10.0):
        self.cf = correlated_field
        self.scaling = scaling
        x_ = jnp.linspace(-dims[0]//2-1, dims[0]//2, dims[0])
        self.kernel = jnp.exp(-pow(x_/width, 2)/2)
        self.kernel /= jnp.sum(self.kernel)
        self.rG = jnp.linspace(0.0, R_SN, dims[0])

        # Init methods of the Correlated Field model and any prior model in
        # NIFTy.re are aware that their input is standard normal a priori.
        # The `domain` of a model does not know this. Thus, tracking the `init`
        # methods should be preferred over tracking the `domain`.
        super().__init__(init=self.cf.init | self.scaling.init)

    def func_gSNR(self, x):
        return self.scaling(x) * jnp.exp(self.cf(x) + jnp.log(jCR.func_gSNR_CAB98(self.rG)))  

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        return self.func_gSNR(x)[::int(dims[0]/Ndata)]


signal_response = Signal(correlated_field, scaling)

# Generate mock data for SNR distribution
rG_data=jnp.linspace(0.0, R_SN, Ndata)
signal_response_truth = jCR.func_gSNR_YUK04(rG_data)
keys = jax.random.split(key, dims[0])
noise_truth=jnp.array([jax.random.normal(keys[i]) * 0.05*signal_response_truth[i] for i in range(Ndata)])
data = signal_response_truth + noise_truth 

# This defines the Likelihood in Bayes' law. "Amend" glues your forward model to the input
lh = jft.Gaussian(data, noise_cov_inv=1.0/noise_truth**2).amend(signal_response)

# Now lets run the main inference scheme:
n_vi_iterations = 6
delta = 1e-4
n_samples = 5

key, k_i, k_o = random.split(key, 3)
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
spectrum_gSNR = jnp.array(tuple(cfm.amplitude(s)[1:] for s in samples))

# Testing Bessel expansion on one of the sample
num_zeros=200
zeta_n=jnp.array(sp.special.jn_zeros(0, num_zeros))
r_int=jnp.linspace(0.0,R_SN,200000)
fr_int=jnp.interp(r_int, rG, gSNR_sample[0,:])

j0_n=jCR.j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]/R_CR)
q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
q_n*=(2.0/(R_CR**2*(jCR.j1(zeta_n)**2))) # pc^-2

gSNR=jnp.sum(q_n[:,jnp.newaxis]*jCR.j0(zeta_n[:,jnp.newaxis]*rG[jnp.newaxis,:]/R_CR),axis=0)

# Plotting gSNR and power spectrum
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot mock data
ax[0].plot(rG_data, data, c='C5', alpha=1.0, label='Case 1998')

# Plot samples and mean 
for i in range(n_samples):
    ax[0].plot(rG, gSNR_sample[i,:], c='C1', alpha=0.2, label='Samples')
ax[0].plot(rG, gSNR_sample[0,:], c='C3', alpha=1)

# Plot Bessel expansion version
ax[0].plot(rG,gSNR, 'k--')

ax[0].set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
ax[0].set_xlabel(r'$R\, {\rm (pc)}$')

# Plot power spectra
for i in range(n_samples):
    ax[1].plot(spectrum_gSNR[i,:], 'k-')
ax[1].plot(jnp.mean(spectrum_gSNR, axis=0),'r-')

ax[1].set_xscale('log')
ax[1].set_yscale('log')

fig.tight_layout()
fig.savefig("results_gSNR.png", dpi=400)
