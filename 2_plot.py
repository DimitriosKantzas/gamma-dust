import numpy as np
import matplotlib.pyplot as plt
# import healpy as hp
from healpy.newvisufunc import projview, newprojplot

# To load the data back from the .npz file
data=np.load('gamma_map.npz')
Eg=data['Eg']
gamma_map=data['gamma_map']

# Plot gamma-ray sky maps
fig=plt.figure(figsize=(12, 5))

projview(
    np.log10(np.mean(gamma_map,axis=0)[0,:]), 
    title=r'Mean gamma-ray map at $E_\gamma=%.1f$ GeV' % Eg[0],
    coord=["G"], cmap='magma',
    min=-8.5, max=-4.5,
    cbar_ticks=[-8.5, -6.5, -4.5],
    nest=True, 
    unit=r'$\log_{10}\phi_{\rm \gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=121
)

projview(
    np.log10(np.std(gamma_map,axis=0)[0,:]/np.mean(gamma_map,axis=0)[0,:]), 
    title=r'Uncertainty gamma-ray map at $E_\gamma=%.1f$ GeV' % Eg[0],
    coord=["G"], cmap='magma',
    min=-2, max=-0.4,
    cbar_ticks=[-2, -1, -0.4],
    nest=True, 
    unit=r'$\log_{10}\left[\Delta\phi_{\rm \gamma}(E_\gamma)/\phi_{\rm \gamma}(E_\gamma)\right]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=122
)

fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

plt.savefig('fg_gamma-map_jax.png', dpi=300)
plt.close()