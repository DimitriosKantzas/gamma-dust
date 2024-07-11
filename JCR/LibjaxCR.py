import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax import grad
from jax import vmap
import numpy as np
import LibppGam_jax as ppG
from astropy.io import fits
import healpy as hp

# Zeroth order Bessel function of first kind
@jit
def j0(x):
    def small_x(x):
        z = x * x
        num = 57568490574.0 + z * (-13362590354.0 + z * (651619640.7 +
              z * (-11214424.18 + z * (77392.33017 + z * (-184.9052456)))))
        den = 57568490411.0 + z * (1029532985.0 + z * (9494680.718 +
              z * (59272.64853 + z * (267.8532712 + z * 1.0))))
        return num / den

    def large_x(x):
        y = 8.0 / x
        y2 = y * y
        ans1 = 1.0 + y2 * (-0.1098628627e-2 + y2 * (0.2734510407e-4 +
               y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y2 * (0.1430488765e-3 +
               y2 * (-0.6911147651e-5 + y2 * (0.7621095161e-6 -
               y2 * 0.934935152e-7)))
        return jnp.sqrt(0.636619772 / x) * (jnp.cos(x - 0.785398164) * ans1 - y * jnp.sin(x - 0.785398164) * ans2)

    return jnp.where(x < 5.0, small_x(x), large_x(x))

# First order Bessel function of first kind
@jit
def j1(x):
    def small_x(x):
        z = x * x
        num = x * (72362614232.0 + z * (-7895059235.0 + z * (242396853.1 +
              z * (-2972611.439 + z * (15704.48260 + z * (-30.16036606))))))
        den = 144725228442.0 + z * (2300535178.0 + z * (18583304.74 +
              z * (99447.43394 + z * (376.9991397 + z * 1.0))))
        return num / den

    def large_x(x):
        y = 8.0 / x
        y2 = y * y
        ans1 = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 +
               y2 * (0.2457520174e-5 - y2 * 0.240337019e-6)))
        ans2 = 0.04687499995 + y2 * (-0.2002690873e-3 +
               y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 +
               y2 * 0.105787412e-6)))
        return jnp.sqrt(0.636619772 / x) * (jnp.cos(x - 2.356194491) * ans1 - y * jnp.sin(x - 2.356194491) * ans2)

    return jnp.where(x < 5.0, small_x(x), large_x(x))

# Surface density of SNRs from Yusifov et al. 2004
@jit
def func_gSNR_YUK04(r):
# r (pc)

    r=jnp.array(r)*1.0e-3 # kpc
    gSNR=jnp.where(
        r<15.0,
        jnp.power((r+0.55)/9.05,1.64)*jnp.exp(-4.01*(r-8.5)/9.05)/5.95828e+8,
        0.0
    )    
    return gSNR # pc^-2

# Some functions to test jax on finding best fit with gradient descent
@jit
def jcompute_coefficients(zeta_n, R):

    r=jnp.linspace(0.0,R,250000)
    fr=func_gSNR_YUK04(r)
    j0_n=j0(zeta_n[:,jnp.newaxis]*r[jnp.newaxis,:]/R)
    coefficients=jnp.trapezoid(r[jnp.newaxis,:]*fr[jnp.newaxis,:]*j0_n,r)
    coefficients*=(2.0/(R**2*(j1(zeta_n)**2)))

    return coefficients

@jit
def func_gSNR_fit(theta, zeta_n, R, r):

    A, B, C=theta
    r_int=jnp.linspace(0.0,R,200000)
    fr_int=jnp.where(
        r_int<15.0,
        A*jnp.power((r_int+0.55)/9.05,B)*jnp.exp(-C*(r_int-8.5)/9.05),
        0.0
    ) 
    j0_n=j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]/R)
    coefficients=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
    coefficients*=(2.0/(R**2*(j1(zeta_n)**2)))

    gSNR=jnp.sum(coefficients[:,jnp.newaxis]*j0(zeta_n[:,jnp.newaxis]*r[jnp.newaxis,:]/R),axis=0)

    return gSNR

@jit
def loss_func(theta, zeta_n, R, r_data, gSNR_data):
    gSNR_fit=func_gSNR_fit(theta,zeta_n,R,r_data)

    return jnp.mean((gSNR_fit-gSNR_data)**2)

@jit
def update(theta, zeta_n, R, r_data, gSNR_data, lr=jnp.array([0.01,1.0e16,0.01])):
    return theta-lr*grad(loss_func)(theta,zeta_n,R,r_data,gSNR_data)

# Function to compute cosmic-ray flux for the YUK04 distribution of sources
@jit
def func_jE_YUK04(pars_prop, zeta_n, E, r, z):

    mp=938.272e6 # eV

    # Transsport parameters
    R=pars_prop[0] # pc
    L=pars_prop[1] # pc
    alpha=pars_prop[2] 
    xiSNR=pars_prop[3]
    u0=pars_prop[4]*365.0*86400.0/3.086e18 # km/s to pc/yr -> Advection speed

    p=jnp.sqrt((E+mp)**2-mp**2)  # eV
    vp=p/(E+mp)

    # Diffusion coefficient
    pb=312.0e9
    Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1 # pc^2/yr

    # Spatial distribution of sources
    r_int=jnp.linspace(0.0,R,200000)*1.0e-3
    fr_int=jnp.where(
        r_int<15.0,
        jnp.power((r_int+0.55)/9.05,1.64)*jnp.exp(-4.01*(r_int-8.5)/9.05)/5.95828e+8,
        0.0
    ) 
    j0_n=j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]*1.0e3/R)
    q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
    q_n*=1.0e6*(2.0/(R**2*(j1(zeta_n)**2))) # pc^-2

    # Injection spectrum of sources
    xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
    xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
    x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
    Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)
    # print('hj',xmin)

    RSNR=0.03 # yr^-1 -> SNR rate
    ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
    QE=(xiSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha)
    QE*=RSNR*vp*3.0e10

    Diff=Diff[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
    z=z[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]
    r=r[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]
    zeta_n=zeta_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]
    q_n=q_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]

    Sn=jnp.sqrt((u0/Diff)**2+4.0*(zeta_n/R)**2) # pc^-2
    fEn=j0(zeta_n*r/R)*q_n*jnp.exp(u0*z/(2.0*Diff))*jnp.sinh(Sn*(L-z)/2.0)/(jnp.sinh(Sn*L/2.0)*(u0+Sn*Diff*(jnp.cosh(Sn*L/2.0)/jnp.sinh(Sn*L/2.0))))
    fE=jnp.sum(fEn,axis=0) # eV^-1 pc^-3
    fE=jnp.where(fE<0.0,0.0,fE)

    jE=fE*QE[:,jnp.newaxis,jnp.newaxis]*1.0e9/(3.086e18)**3 

    return jE # GeV^-1 cm^-2 s^-1

# # Function to interpolate emissivity on healpix-r grid using JAX
# @jit
# def get_healpix_interp(qg, Eg, rg, zg, points_intr):
#     # Grid on which emissivity is calculated
#     points = (rg, zg)

#     # Grid on which emissivity is interpolated
#     N_rs, N_pix=points_intr[0].shape
#     N_E = len(Eg)

#     # Interpolator
#     qg_healpix = jnp.zeros((N_E, N_rs, N_pix))
#     for j in range(N_E):
#         interpolator = jsp.interpolate.RegularGridInterpolator(points, qg[j, :, :], method='linear', bounds_error=False, fill_value=0.0)
#         qg_healpix = qg_healpix.at[j, :, :].set(interpolator(points_intr))


#     return qg_healpix

# Function to interpolate emissivity on healpix-r grid using JAX
@jit
def interpolate_grid(qg_slice, points, points_intr):
    interpolator=jsp.interpolate.RegularGridInterpolator(points,qg_slice,method='linear',bounds_error=False,fill_value=0.0)
    return interpolator(points_intr)

@jit
def get_healpix_interp(qg, rg, zg, points_intr):
    points = (rg, zg)

    # Vectorize the interpolation across the energy levels
    vectorized_interpolation=vmap(interpolate_grid,in_axes=(0,None,None),out_axes=0)

    # Perform the vectorized interpolation
    qg_healpix=vectorized_interpolation(qg,points,points_intr)

    return qg_healpix

# Differential cross-section for gamma-ray production
def func_dXSdEg(E, Eg):
# E (GeV) and Eg (GeV)

    E=np.array(E)
    Eg=np.array(Eg)

    dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
    for i in range(len(E)):
        for j in range(len(Eg)):
            dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i],Eg[j])*1.0e-27
    
    return jnp.array(dXSdEg_Geant4) #  # cm^2 GeV^-1

# Function to calculate the gamma-ray map
@jit
def func_gamma_map(ngas, qg_Geant4_healpixr, drs):
    return jnp.sum(ngas[:,jnp.newaxis,:,:]*qg_Geant4_healpixr[jnp.newaxis,:,:,:]*drs[jnp.newaxis,jnp.newaxis,:,jnp.newaxis],axis=2) # GeV^-1 cm^-2 s^-1

# Function to load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
def load_gas(path_to_gas):

    # Position of solar system from the gas map (see Soding et al. 2024)
    Rsol=8178.0 # pc

    hdul=fits.open(path_to_gas)
    rs=(hdul[2].data)['radial pixel edges'].astype(np.float64) # kpc -> Edges of radial bins
    drs=np.diff(rs)*3.086e21 # cm -> Radial bin width for line-of-sight integration
    rs=(hdul[1].data)['radial pixel centres'].astype(np.float64)*1.0e3 # pc -> Centres of radial bins for interpolating cosmic-ray distribution
    samples_HI=(hdul[3].data).T # cm^-3
    samples_H2=(hdul[4].data).T # cm^-3
    hdul.close()
    ngas=2.0*samples_H2+samples_HI # cm^-3

    N_sample, N_rs, N_pix=samples_HI.shape
    NSIDE=int(np.sqrt(N_pix/12))

    # Angles for all pixels
    thetas, phis=hp.pix2ang(NSIDE,jnp.arange(N_pix),nest=True,lonlat=False)

    # Points for interpolation
    ls=phis[jnp.newaxis, :]
    bs=jnp.pi/2.0-thetas[jnp.newaxis, :]
    rs=rs[:, jnp.newaxis]

    xs=-rs*jnp.cos(ls)*jnp.cos(bs)+Rsol
    ys=-rs*jnp.sin(ls)*jnp.cos(bs)
    zs=rs*jnp.sin(bs)

    points_intr=(jnp.sqrt(xs**2+ys**2),jnp.abs(zs))

    return jnp.array(ngas), jnp.array(drs), points_intr

@jit
def func_gamma_map_fit(theta, pars_prop, zeta_n, dXSdEg_Geant4, ngas, drs, points_intr, E):
# E (eV) and Eg (GeV)

    A, B, C=theta

    mp=938.272e6 # eV

    # Transsport parameters
    R=pars_prop[0] # pc
    L=pars_prop[1] # pc
    alpha=pars_prop[2] 
    xiSNR=pars_prop[3]
    u0=pars_prop[4]*365.0*86400.0/3.086e18 # km/s to pc/yr -> Advection speed

    rg=jnp.linspace(0.0,R,501)    # pc
    zg=jnp.linspace(0.0,L,41)     # pc
    p=jnp.sqrt((E+mp)**2-mp**2)  # eV
    vp=p/(E+mp)

    # Diffusion coefficient
    pb=312.0e9
    Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1 # pc^2/yr

    # Spatial distribution of sources
    r_int=jnp.linspace(0.0,R,200000)*1.0e-3
    fr_int=jnp.where(
        r_int<15.0,
        A*jnp.power((r_int+0.55)/9.05,B)*jnp.exp(-C*(r_int-8.5)/9.05),
        0.0
    ) 
    j0_n=j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]*1.0e3/R)
    q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
    q_n*=1.0e6*(2.0/(R**2*(j1(zeta_n)**2))) # pc^-2

    # Injection spectrum of sources
    xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
    xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
    x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
    Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)

    RSNR=0.03 # yr^-1 -> SNR rate
    ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
    QE=(xiSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha)
    QE*=RSNR*vp*3.0e10

    Diff=Diff[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
    z=zg[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]
    r=rg[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]
    zeta_n=zeta_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]
    q_n=q_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]

    Sn=jnp.sqrt((u0/Diff)**2+4.0*(zeta_n/R)**2) # pc^-2
    fEn=j0(zeta_n*r/R)*q_n*jnp.exp(u0*z/(2.0*Diff))*jnp.sinh(Sn*(L-z)/2.0)/(jnp.sinh(Sn*L/2.0)*(u0+Sn*Diff*(jnp.cosh(Sn*L/2.0)/jnp.sinh(Sn*L/2.0))))
    fE=jnp.sum(fEn,axis=0) # eV^-1 pc^-3
    fE=jnp.where(fE<0.0,0.0,fE)

    jE=fE*QE[:,jnp.newaxis,jnp.newaxis]*1.0e9/(3.086e18)**3 # GeV^-1 cm^-2 s^-1

    # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
    qg_Geant4=1.8*jnp.trapezoid(jE[:,jnp.newaxis,:,:]*dXSdEg_Geant4[:,:,jnp.newaxis,jnp.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 

    # Interpolate gamma-ray emissivity on healpix-r grid as gas
    qg_Geant4_healpixr=get_healpix_interp(qg_Geant4,rg,zg,points_intr) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity

    # Compute the diffuse emission in all gas samples
    gamma_map=func_gamma_map(ngas,qg_Geant4_healpixr,drs) # GeV^-1 cm^-2 s^-1

    return gamma_map

@jit
def loss_func_gamma_map(theta, pars_prop, zeta_n, dXSdEg_Geant4, ngas, drs, points_intr, E, gamma_map_data):
    gamma_map_fit=func_gamma_map_fit(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E)

    return jnp.mean((gamma_map_fit-gamma_map_data)**2)

@jit
def update_gamma_map(theta, pars_prop, zeta_n, dXSdEg_Geant4, ngas, drs, points_intr, E, gamma_map_data, lr):
    return theta-lr*grad(loss_func_gamma_map)(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E,gamma_map_data)
