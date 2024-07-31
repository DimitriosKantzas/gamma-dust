import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import h5py

# Open the FITS file and get info on the data
hdul=fits.open('samples_densities_hpixr.fits')
print(hdul.info())

# Access data 
r=(hdul[2].data)['radial pixel edges'].astype(np.float64) # Edges of radial bins
dr=np.diff(r)*3.086e21 # cm -> Bin width for the integration along the line of sight
r_center=(hdul[1].data)['radial pixel centres'].astype(np.float64) # Edges of radial bins
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()

# Plot the mean column density map
NHImap=np.sum(samples_HI*dr[np.newaxis,:,np.newaxis],axis=1)
NH2map=np.sum(samples_H2*dr[np.newaxis,:,np.newaxis],axis=1)
N_sample, N_pix=NHImap.shape
print(samples_HI.shape)

# import h5py

# # Open the HDF5 file
# with h5py.File('JCR/energy_bins.hdf5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     Eg_data=file['geom_avg_bin_energy'][:]
#     Eg_data_lower=file['lower_bin_boundaries'][:]
#     Eg_data_upper=file['upper_bin_boundaries'][:]
    
# dEg_data=Eg_data_upper-Eg_data_lower

# with h5py.File('JCR/I_dust.hdf5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     gamma_map_data=file['stats']['mean'][:]

# gamma_map_data*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
# gamma_map_data=hp.ud_grade(gamma_map_data[5,:], nside_out=64)
# gamma_map_data=hp.reorder(gamma_map_data, r2n=True)

# emi_map=gamma_map_data/(np.mean(NHImap+2.0*NH2map,axis=0)*4.0*np.pi)
# print(emi_map.shape)

# l, b=hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
# l=np.where(l<0,l+360,l)

# mask=(np.abs(b)<=8.0) 
# emi_map[mask]=np.nan

# print(np.nanmean(emi_map))

fig=plt.figure(figsize=(12, 5))

projview(
    np.log10(np.mean(NHImap,axis=0)), 
    # np.log10(emi_map), 
    title=r'Mean HI',
    coord=["G"], cmap='viridis',
    min=20, max=23,
    nest=True, 
    unit=r'$log_{10}N({\rm HI})\, [{\rm cm}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=121
)

projview(
    np.log10(np.mean(NH2map,axis=0)), 
    title=r'Mean H2',
    coord=["G"], cmap='viridis',
    min=20, max=23,
    nest=True, 
    unit=r'$log_{10}N({\rm H}_2)\, [{\rm cm}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=122
)

fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

plt.savefig('fg_gas_map.png', dpi=150)
plt.close()

# Uncomment to get column density maps from samples
# for isample in range(N_sample):

#     fig=plt.figure(figsize=(12, 5))

#     projview(
#         np.log10(NHImap[isample,:]), 
#         title=r'HI sample %d' % (isample+1),
#         coord=["G"], cmap='viridis',
#         min=20, max=23,
#         nest=True, 
#         unit=r'$log_{10}N({\rm HI})\, [{\rm cm}^{-2}]$',
#         graticule=True, graticule_labels=True, 
#         # xlabel=r'longitude (deg)',
#         # ylabel=r'latitude (deg)',
#         projection_type="mollweide",
#         sub=121
#     )

#     projview(
#         np.log10(NH2map[isample,:]), 
#         title=r'H2 sample %d' % (isample+1),
#         coord=["G"], cmap='viridis',
#         min=20, max=23,
#         nest=True, 
#         unit=r'$log_{10}N({\rm H}_2)\, [{\rm cm}^{-2}]$',
#         graticule=True, graticule_labels=True, 
#         # xlabel=r'longitude (deg)',
#         # ylabel=r'latitude (deg)',
#         projection_type="mollweide",
#         sub=122
#     )

#     fig.tight_layout(pad=1.0)
#     fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

#     plt.savefig('fg_gas_sample_%d.png' % (isample), dpi=150)
#     plt.close()

# Load diffuse gamma-ray map from Platz et al. 2023
with h5py.File('JCR/energy_bins.hdf5', 'r') as file:
    print("Keys: %s" % file.keys())
    Eg_data=file['geom_avg_bin_energy'][:]
    Eg_data_lower=file['lower_bin_boundaries'][:]
    Eg_data_upper=file['upper_bin_boundaries'][:]
    
dEg_data=Eg_data_upper-Eg_data_lower

with h5py.File('JCR/I_dust.hdf5', 'r') as file:
    print("Keys: %s" % file['stats'].keys())
    gamma_map_mean=file['stats']['mean'][:]
    gamma_map_std=file['stats']['standard deviation'][:]

gamma_map_mean*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_mean=hp.ud_grade(gamma_map_mean[5,:], nside_out=64)
gamma_map_mean=hp.reorder(gamma_map_mean, r2n=True)
gamma_map_mean=gamma_map_mean[np.newaxis,np.newaxis,:] # GeV^-1 cm^-2 s^-1

fig=plt.figure(figsize=(6, 5))

# projview(
#     np.log10(np.mean(NHImap+2.0*NH2map,axis=0)), 
#     title=r'Mean gas column density',
#     coord=["G"], cmap='magma',
#     min=20, max=23,
#     nest=True, 
#     unit=r'$log_{10}N_{\rm H}\, [{\rm cm}^{-2}]$',
#     graticule=True, graticule_labels=True, 
#     # xlabel=r'longitude (deg)',
#     # ylabel=r'latitude (deg)',
#     projection_type="mollweide",
#     sub=111
# )

# projview(
#     np.log10(gamma_map_mean[0,0,:]), 
#     title=r'Diffuse gamma-ray at $E_\gamma=10$ GeV',
#     coord=["G"], cmap='magma',
#     min=-8.5, max=-4.5,
#     nest=True, 
#     unit=r'$log_{10}I\, [{\rm GeV^{-1}\, cm^{-2}\, s^{-1}}]$',
#     graticule=True, graticule_labels=True, 
#     # xlabel=r'longitude (deg)',
#     # ylabel=r'latitude (deg)',
#     projection_type="mollweide",
#     sub=111
# )

l, b=hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
l=np.where(l<0,l+360,l)

mask=(np.abs(b)<20.0) 
qg=gamma_map_mean[0,0,:]/(np.mean(NHImap+2.0*NH2map,axis=0)*4.0*np.pi)
qg[mask]=np.nan
print(np.mean(qg))

projview(
    np.log10(qg), 
    title=r'Local gamma-ray emissivity at $E_\gamma=10$ GeV',
    coord=["G"], cmap='magma',
    min=-30, max=-29,
    cbar_ticks=[-30, -29.5, -29],
    nest=True, 
    unit=r'$log_{10}q_\gamma\, [{\rm GeV^{-1}\, s^{-1}\, sr^{-1}}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=111
)

fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

plt.savefig('fg_gas_gamma_3.png', dpi=150)
plt.close()


import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Load or create a HEALPix map (for example, using a random map here)
nside=64
npix=hp.nside2npix(nside)

# Define the Cartesian grid
nlon, nlat=721, 121  # Resolution of the Cartesian grid
lon=np.linspace(-180,180,nlon)  # Longitude from -180 to 180 degrees
lat=np.linspace(30,-30,nlat)  # Latitude from -90 to 90 degrees
# lon_grid, lat_grid=np.meshgrid(lon,lat)
lat_grid, lon_grid=np.meshgrid(lat,lon)

# Convert Cartesian coordinates to theta and phi
theta=np.radians(90-lat_grid)  # Co-latitude in radians
phi=np.radians(lon_grid)  # Longitude in radians

# Get HEALPix pixel indices for each (theta, phi)
healpix_indices=hp.ang2pix(nside, theta, phi, nest=True)

# Create the new Cartesian skymap
NHImap_cart=np.mean(NHImap[:,healpix_indices][:,:,::-1], axis=0)
NH2map_cart=np.mean(NH2map[:,healpix_indices][:,:,::-1], axis=0)

samples_H2_cart=samples_H2[:,:,healpix_indices][:,:,:,::-1]
mean_H2_cart=np.mean(samples_H2_cart, axis=0)
NH2map_cart_test=np.sum(mean_H2_cart*dr[:,np.newaxis,np.newaxis],axis=0)

# Function to interpolate healpix map into Cartesian map
def interpolate_maps(r_values, maps, r_target):
    # Create an interpolator for each point in the map
    interpolator = sp.interpolate.interp1d(r_values, maps, axis=0, kind='linear', fill_value='extrapolate')
    # Get the interpolated map at the desired r value
    interpolated_map = interpolator(r_target)
    return interpolated_map

Nr=280
r_interp=np.linspace(0.05,27.95,Nr)
print(r_interp)
mean_H2_cart_intr=interpolate_maps(r_center,mean_H2_cart,r_interp)

dr_interp=np.append(np.diff(r_interp), 0.0)*3.086e21
NH2map_cart_intr=np.sum(mean_H2_cart_intr*dr_interp[:,np.newaxis,np.newaxis],axis=0)

# # Create headers for the data cubes
# header1 = fits.Header()
# header1['COMMENT'] = "Bins of heliocentric distances (kpc)"

# header2 = fits.Header()
# header2['COMMENT'] = "Cube gas density (cm^-3)"

# # Create FITS HDUs with data and headers
# hdu1 = fits.PrimaryHDU(data=r_interp, header=header1)
# hdu2 = fits.ImageHDU(data=mean_H2_cart_intr.T, header=header2)

# # Create an HDU list to contain both HDUs
# hdul = fits.HDUList([hdu1, hdu2])

# # Write the HDU list to a new FITS file
# hdul.writeto('H2_cube.fit', overwrite=True)

# # Verify by reading the FITS file
# with fits.open('H2_cube.fit') as hdulist:
#     print(hdulist.info())
#     data1 = hdulist[0].data
#     data2 = hdulist[1].data
#     print("Data cube 1 shape:", data1.shape)
#     print("Data cube 2 shape:", data2.shape)
#     print("Header for data cube 1:", repr(hdulist[0].header))
#     print("Header for data cube 2:", repr(hdulist[1].header))

# Create headers for the data cubes
header1 = fits.Header()
header1['COMMENT'] = "Bins of heliocentric distances (kpc)"

header2 = fits.Header()
header2['COMMENT'] = "Cube gas density (cm^-3)"

# mean_H2_cart_intr=mean_H2_cart_intr[:,:,::-1]#.astype(np.float64)

# Create FITS HDUs with data and headers
hdu1 = fits.PrimaryHDU(data=mean_H2_cart_intr.T, header=header1)
# hdu2 = fits.ImageHDU(data=mean_H2_cart_intr.T, header=header2)

# Create an HDU list to contain both HDUs
hdul = fits.HDUList([hdu1])

# Write the HDU list to a new FITS file
hdul.writeto('H2_cube_fixed2.fit', overwrite=True)

# Verify by reading the FITS file
with fits.open('H2_cube_fixed2.fit') as hdulist:
    print(hdulist.info())
    data1 = hdulist[0].data
    print("Data cube 1 shape:", data1.shape)
    print("Header for data cube 1:", repr(hdulist[0].header))

with fits.open('CO_3D_cube_all_sky_smooth.fit') as hdulist:
    print(hdulist.info())
    data2 = hdulist[0].data
    # data2 = hdulist[1].data
    # print("Data cube 1 shape:", data1.shape)
    # print("Data cube 2 shape:", data2.shape)
    # print("Header for data cube 1:", repr(hdulist[0].header))
    # print("Header for data cube 2:", repr(hdulist[1].header))

data2=data2.T
# dr_data=np.append(np.diff(data1), 0.0)*3.086e21
dr_data=np.append(np.diff(r_interp), 0.0)*3.086e21
mydata=np.sum(data2*dr_data[:,np.newaxis,np.newaxis],axis=0)

# print(NH2map_cart_intr/mydata)

fs=22 

fig, ax = plt.subplots(3, 1, figsize=(20, 21))

im1 = ax[0].imshow(np.log10(NH2map_cart).T, extent=(-180, 180, -30, 30), cmap='magma', origin='lower', vmin=19, vmax=24)
ax[0].set_xlabel(r'$l\,{\rm (degree)}$',fontsize=fs)
ax[0].set_ylabel(r'$b\,{\rm (degree)}$',fontsize=fs)
ax[0].set_title('H2 column from original grid',fontsize=fs)
ax[0].tick_params(axis='both', which='major', labelsize=fs)
ax[0].tick_params(axis='both', which='minor', labelsize=fs)
ax[0].set_ylim(-30, 30)

cbar1 = plt.colorbar(im1, ax=ax[0], orientation='horizontal', fraction=0.046, pad=0.15)
cbar1.set_label(r'$\log\left[N({\rm H}_2)\, {\rm cm}^{-2}\right]$', fontsize=fs)
cbar1.ax.tick_params(labelsize=fs)

im2 = ax[1].imshow(np.log10(mydata).T, extent=(-180, 180, -30, 30), cmap='magma', origin='lower', vmin=19, vmax=24)
ax[1].set_xlabel(r'$l\,{\rm (degree)}$',fontsize=fs)
ax[1].set_ylabel(r'$b\,{\rm (degree)}$',fontsize=fs)
ax[1].set_title('H2 column from interpolated grid',fontsize=fs)
ax[1].tick_params(axis='both', which='major', labelsize=fs)
ax[1].tick_params(axis='both', which='minor', labelsize=fs)
ax[1].set_ylim(-30, 30)

cbar2 = plt.colorbar(im2, ax=ax[1], orientation='horizontal', fraction=0.046, pad=0.15)
cbar2.set_label(r'$\log\left[N({\rm H}_2)\, {\rm cm}^{-2}\right]$', fontsize=fs)
cbar2.ax.tick_params(labelsize=fs)

im3 = ax[2].imshow(np.log10(NH2map_cart_intr/mydata).T, extent=(-180, 180, -30, 30), cmap='magma', origin='lower', vmin=-2, vmax=1)
ax[2].set_xlabel(r'$l\,{\rm (degree)}$',fontsize=fs)
ax[2].set_ylabel(r'$b\,{\rm (degree)}$',fontsize=fs)
ax[2].set_title('Ratio of H2 column density',fontsize=fs)
ax[2].tick_params(axis='both', which='major', labelsize=fs)
ax[2].tick_params(axis='both', which='minor', labelsize=fs)
ax[2].set_ylim(-30, 30)

cbar3 = plt.colorbar(im3, ax=ax[2], orientation='horizontal', fraction=0.046, pad=0.15)
cbar3.set_label(r'$\log\left[N^{\rm original}({\rm H}_2)/N^{\rm interp}({\rm H}_2)\right]$', fontsize=fs)
cbar3.ax.tick_params(labelsize=fs)

# Adjust layout to make room for color bars
plt.tight_layout()

# Save the figure
plt.savefig('combined_cartesian_projection.png')