import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import LibjaxCR as jCR
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import healpy as hp
import time

l, b = hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
l = np.where(l < 0, l + 360, l)

mask = (np.abs(b) <= 10.0)

mp = 938.272e6  # eV

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros = 100
zeta_n = jnp.array(sp.special.jn_zeros(0, num_zeros))

# Size of the cosmic-ray halo
R = 20000.0  # pc -> Radius of halo
L = 4000.0  # pc -> Height of halo

# Parameters for injection spectrum
alpha = 4.23  # -> Injection spectral index
xiSNR = 0.065  # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0 = 7.0  # km/s -> Advection speed

# Combine all parameters for propagation
pars_prop = jnp.array([R, L, alpha, xiSNR, u0])

# Define cosmic-ray grid and diffusion coefficient
E = jnp.logspace(10.0, 14.0, 81)  # eV 
p = jnp.sqrt((E + mp)**2 - mp**2)  # eV
vp = p / (E + mp)
Diff = 1.1e28 * (365.0 * 86400.0 / (3.08567758e18)**2) * vp * (p / 1.0e9)**0.63 / (1.0 + (p / 312.0e9)**2)**0.1  # pc^2/yr

# Injection spectrum of sources
xmin = jnp.sqrt((1.0e8 + mp)**2 - mp**2) / mp
xmax = jnp.sqrt((1.0e14 + mp)**2 - mp**2) / mp
x = jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 5000)
Gam = jnp.trapezoid(x**(2.0 - alpha) * (jnp.sqrt(x**2 + 1.0) - 1.0), x)

RSNR = 0.03  # yr^-1 -> SNR rate
ENSR = 1.0e51 * 6.242e+11  # eV -> Average kinetic energy of SNRs
QE = RSNR * vp * 3.0e10 * (xiSNR * ENSR / (mp**2 * vp * Gam)) * (p / mp)**(2.0 - alpha)

# Bessel functions
r_int = jnp.linspace(0.0, R, 25000)
j0_n_int = jCR.j0(zeta_n[:, jnp.newaxis] * r_int[jnp.newaxis, :] / R)

# Define gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
Eg = jnp.logspace(np.log10(13.33521432163324), 2, 1)  # GeV
dXSdEg_Geant4 = jCR.func_dXSdEg(E * 1.0e-9, Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr = jCR.load_gas('../samples_densities_hpixr.fits')
ngas_mean = jnp.mean(ngas, axis=0)[jnp.newaxis, :, :]

# Define the properties of the neural network
N_SAMPLES = 51
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
N_EPOCHS = 10000
epoch_print = N_EPOCHS / 10

# Random key
key = jax.random.PRNGKey(42)

# Weight initialization
weight_matrices = []
bias_vectors = []
activation_functions = []

for (fan_in, fan_out) in zip(LAYERS[:-1], LAYERS[1:]):
    kernel_matrix_uniform_limit = jnp.sqrt(6 / (fan_in + fan_out))

    key, wkey = jax.random.split(key)

    W = jax.random.uniform(
        wkey,
        (fan_in, fan_out),
        minval=-kernel_matrix_uniform_limit,
        maxval=+kernel_matrix_uniform_limit,
    )

    b = jnp.zeros(fan_out)

    weight_matrices.append(W)
    bias_vectors.append(b)
    activation_functions.append(jax.nn.sigmoid)

activation_functions[-1] = lambda x: x

# Function to define the network 
def network_forward(x, weights, biases, activations):
    a = x
    for W, b, f in zip(weights, biases, activations):
        a = f(a @ W + b)
    return a

# Random key for mock data
key, ynoisekey = jax.random.split(key, 2)

# Spatial domain for the function gSNR to be inferred and renormalize for the neural network
x_samples_raw = jnp.linspace(0, 15.0, N_SAMPLES)[:, jnp.newaxis]
x_samples = x_samples_raw / x_samples_raw[-1]

# Mock data for testing
y_init = jnp.interp(8.178,
                        x_samples_raw.ravel(),
                        network_forward(x_samples_raw*1.0e3,weight_matrices,bias_vectors,activation_functions).ravel())

y_sol = 2.0e-9 # jCR.func_gSNR_YUK04(jnp.array([8178.0])) / jnp.exp(y_init)
y_grtruth = jnp.log((jCR.func_gSNR_YUK04(x_samples_raw * 1.0e3) + 1.0e-9*jnp.exp(-(x_samples_raw-10.0)**2/2.0) + 1.0e-9*jnp.exp(-(x_samples_raw-6.0)**2/0.5)) / (y_sol))
y_samples_raw = y_grtruth
y_samples = jnp.log(jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_samples_raw) * y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E))[0, 0, mask]

# Loss function to compare gamma-ray maps
def loss_forward(y_guess, y_ref):
    gamma_guess = jnp.log(jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_guess) * y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E))
    # gamma_ref = jnp.log(jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_ref) * y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E))
    delta = gamma_guess[0, 0, mask] - y_ref
    # delta = y_guess - y_ref

    return 100.0 * jnp.mean(delta**2)

# Function to find derivatives with respect to weight matrices and bias vectors
loss_and_grad_fun = jax.value_and_grad(
    lambda Ws, bs: loss_forward(
        network_forward(
            x_samples,
            Ws,
            bs,
            activation_functions,
        ),
        y_samples,
    ),
    argnums=(0, 1),
)

# Using optax.adam to optimize the learning rate
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init((weight_matrices, bias_vectors))

# Function for updating weights and biases with gradient descent 
@jax.jit
def update(opt_state, params):
    weights, biases = params
    loss, grads = loss_and_grad_fun(weights, biases)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss

# Training loop
start_time = time.time()
loss_history = []
gSNR_samples = []

params = (weight_matrices, bias_vectors)
for epoch in range(N_EPOCHS):
    opt_state, params, loss = update(opt_state, params)

    if (epoch + 1) % epoch_print == 0:
        print(f"epoch: {epoch+1}, loss: {loss}")
        weights, biases = params
        plt.scatter(x_samples_raw, jnp.exp(y_samples_raw) * y_sol, label='Samples')
        plt.scatter(x_samples_raw, jnp.exp(network_forward(x_samples, weights, biases, activation_functions)) * y_sol, label='Interation %d' % (epoch + 1))
        plt.plot(x_samples_raw, jnp.exp(y_grtruth) * y_sol, 'k--', label='Ground truth')
        plt.legend(loc='upper right')
        plt.savefig('nn_epoch_%d.png' % (epoch + 1))
        plt.close()
    
    loss_history.append(loss)
    gSNR_samples.append(jnp.exp(network_forward(x_samples, params[0], params[1], activation_functions)))

weights, biases = params
y_fit = network_forward(x_samples, weights, biases, activation_functions)

gamma_fit = jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_fit)*y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E)
gamma_grtruth = jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_grtruth)*y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E)

# Save best fit and samples
np.savez('Results_nn/gSNR_nn.npz', 
            rG=np.array(x_samples_raw.ravel()), 
            gSNR=np.array(jnp.exp(y_fit).ravel())*y_sol, 
            gSNR_truth=np.array(jnp.exp(y_grtruth))*y_sol, 
            gSNR_samples=np.array(gSNR_samples)*y_sol, 
            loss_history=np.array(loss_history), 
            gamma=np.array(gamma_fit), 
            gamma_truth=np.array(gamma_grtruth))

# Save weights and biases from training 
weights_dict = {f"weight_{i}": np.array(weights[i]) for i in range(len(weights))}
biases_dict = {f"bias_{i}": np.array(biases[i]) for i in range(len(biases))}
save_dict = {**weights_dict, **biases_dict}
np.savez('nn.npz', **save_dict)

print('Run time: %.2f seconds' % (time.time() - start_time))
