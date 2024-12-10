import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


# set up the plotting
# set font size
plt.rcParams.update({'font.size': 18})
# set the figure size
plt.rcParams.update({'figure.figsize': (10, 7)})
# set the font to latex
plt.rcParams.update({'text.usetex': True})

# set the path to save plots
plots_path = '/Users/mncavieres/Documents/2024-2/HVS/Plots/new_cmd'

# load catalog
catalog = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/from_MIST/test_eep_propagated_phot_MIST_1e5.fits')#'/Users/mncavieres/Documents/2024-2/HVS/Data/importance_sampling/uniform_eep.fits')

# drop nan values
catalog = catalog.to_pandas().dropna(subset=['Gaia_BP_M', 'Gaia_RP_M', 'Gaia_G_M', 'dist'])

# select stars within the interpolation range for extinction correction
catalog = catalog[(catalog['Gaia_BP_M'] - catalog['Gaia_RP_M'] > -0.5) & (catalog['Gaia_BP_M'] - catalog['Gaia_RP_M'] < 3)]

# plot the CMD
plt.figure(figsize=(8, 10))
plt.scatter(catalog['Gaia_BP_M'] - catalog['Gaia_RP_M'], catalog['Gaia_G_M'] - 5*np.log10(catalog['dist']) + 5, s=10, c='k', alpha=0.1)
plt.xlabel('$G_{BP} - G_{RP}$')
plt.ylabel('$G$')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{plots_path}/uniform_eep_propagated_phot_cmd.png')
plt.show()

from matplotlib.colors import LogNorm

# plot the Hess diagram
plt.figure(figsize=(8, 10))
plt.hist2d(catalog['BP'] - catalog['RP'], catalog['G'], bins=100, cmap='viridis', norm=LogNorm())
plt.xlabel('$G_{BP} - G_{RP}$')
plt.ylabel('$G$')
plt.gca().invert_yaxis()
plt.colorbar(label='Counts')
plt.tight_layout()
plt.savefig(f'{plots_path}/uniform_eep_hess.png')
plt.show()