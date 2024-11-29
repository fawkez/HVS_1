"""
Using a speedystar catalog, mix it up with gaia data to construct a training catalog for classification.
"""

# imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.constants import kpc, au
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from numba import njit
import matplotlib.colors as colors

import random
import healpy as hp

from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize


# Add the path to the 'scripts' folder directly
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')


# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *
from scripts.CMD_selection import *

# set up the plotting
# set font size
plt.rcParams.update({'font.size': 18})
# set the figure size
plt.rcParams.update({'figure.figsize': (10, 7)})
# set the font to latex
plt.rcParams.update({'text.usetex': True})

#
def construct_training_catalog(data_gaia_big, simulated_catalog_f, subsample=100000):
    """ 
    Construct a training catalog for classification containing a mix of gaia data and speedystar HVS with labels
    
    input:
        data_gaia: dataframe containing gaia data
        simulated_catalog_f: astropy table containing speedystar data

    output:
        data_gaia: pandas dataframe containing gaia data

    """

    # drop nan values
    # drop nan values from astropy table
    #data_gaia = data_gaia.dropna(subset=['bp_rp', 'M_g'])

    # add color columns for future proecessing
    data_gaia_big['bp_rp'] = data_gaia_big['phot_bp_mean_mag'] - data_gaia_big['phot_rp_mean_mag']

    # simulated_catalog_f['bp_rp'] = simulated_catalog_f['Gaia_BP'] - simulated_catalog_f['Gaia_RP']
    # simulated_catalog_f['M_g'] = simulated_catalog_f['Gaia_G'] - 5*np.log10(simulated_catalog_f['dist']*1000) + 5

    # # rename the simulated catalog columns to match the gaia catalog
    simulated_catalog_gaia = simulated_catalog_f
    # simulated_catalog_gaia.rename_column('Gaia_G', 'phot_g_mean_mag')
    # simulated_catalog_gaia.rename_column('Gaia_BP', 'phot_bp_mean_mag')
    # simulated_catalog_gaia.rename_column('Gaia_RP', 'phot_rp_mean_mag')
    # simulated_catalog_gaia.rename_column('par', 'parallax')
    # simulated_catalog_gaia.rename_column('e_par', 'parallax_error')
    # simulated_catalog_gaia.rename_column('e_pmra', 'pmra_error')
    # simulated_catalog_gaia.rename_column('e_pmdec', 'pmdec_error')
    # simulated_catalog_gaia['parallax'] = 1/simulated_catalog_gaia['dist']
    # simulated_catalog_gaia['is_hvs'] = np.ones(len(simulated_catalog_gaia))

    # merge with a large gaia catalog
    simulated_catalog_gaia = simulated_catalog_gaia[['ra', 'dec', 'bp_rp', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'is_hvs']]
    data_gaia_big = data_gaia_big[['ra', 'dec', 'bp_rp', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']]
    data_gaia_big['is_hvs'] = np.zeros(len(data_gaia_big))

    # keep only a subsample of stars from the big catalog
    data_gaia_big = data_gaia_big.sample(n=subsample)

    data_gaia_big = data_gaia_big.dropna(subset=['bp_rp', 'phot_g_mean_mag'])
    simulated_catalog_gaia = simulated_catalog_gaia.to_pandas().dropna(subset=['bp_rp', 'phot_g_mean_mag'])

    # concatenate the catalogs
    data_gaia_big = pd.concat([data_gaia_big, simulated_catalog_gaia])

    # compute implied quantities 
    data_gaia_big = implied_calculations(data_gaia_big)

    # keep only positive implied parallaxes
    data_gaia_big = data_gaia_big.loc[data_gaia_big['implied_parallax'] > 0]

    # add extinction correction
    data_gaia_big = extinction_correction(data_gaia_big)

    # compute the implied absolute magnitude
    data_gaia_big['implied_M_g'] = data_gaia_big['phot_g_mean_mag'] - 5*np.log10(1000/data_gaia_big['implied_parallax']) + 5
    data_gaia_big['implied_M_g_corr'] = data_gaia_big['phot_g_mean_mag'] - 5*np.log10(1000/data_gaia_big['implied_parallax']) + 5

    return data_gaia_big


def construct_hvs_training( simulated_catalog_f, subsample=100000):
    """ 
    Construct a training catalog for classification containing a mix of gaia data and speedystar HVS with labels
    
    input:
        data_gaia: dataframe containing gaia data
        simulated_catalog_f: astropy table containing speedystar data

    output:
        data_gaia: pandas dataframe containing gaia data

    """

    # drop nan values
    # drop nan values from astropy table
    #data_gaia = data_gaia.dropna(subset=['bp_rp', 'M_g'])

    # add color columns for future proecessing
    simulated_catalog_f['bp_rp'] = simulated_catalog_f['Gaia_BP'] - simulated_catalog_f['Gaia_RP']
    simulated_catalog_f['M_g'] = simulated_catalog_f['Gaia_G'] - 5*np.log10(simulated_catalog_f['dist']*1000) + 5

    # rename the simulated catalog columns to match the gaia catalog
    simulated_catalog_gaia = simulated_catalog_f
    simulated_catalog_gaia.rename_column('Gaia_G', 'phot_g_mean_mag')
    simulated_catalog_gaia.rename_column('Gaia_BP', 'phot_bp_mean_mag')
    simulated_catalog_gaia.rename_column('Gaia_RP', 'phot_rp_mean_mag')
    simulated_catalog_gaia.rename_column('par', 'parallax')
    simulated_catalog_gaia.rename_column('e_par', 'parallax_error')
    simulated_catalog_gaia.rename_column('e_pmra', 'pmra_error')
    simulated_catalog_gaia.rename_column('e_pmdec', 'pmdec_error')
    #simulated_catalog_gaia['parallax'] = 1/simulated_catalog_gaia['dist']
    simulated_catalog_gaia['is_hvs'] = np.ones(len(simulated_catalog_gaia))

    # merge with a large gaia catalog
    simulated_catalog_gaia = simulated_catalog_gaia[['ra', 'dec', 'bp_rp', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'is_hvs']]

    # drop nans from the simulated catalog to avoid future errors in calculations
    # check if the simulated_catalog_gaia is a pandas dataframe or an astropy table
    if isinstance(simulated_catalog_gaia, pd.DataFrame):
        simulated_catalog_gaia = simulated_catalog_gaia.dropna(subset=['bp_rp', 'phot_g_mean_mag'])
    else:
        simulated_catalog_gaia = simulated_catalog_gaia.to_pandas().dropna(subset=['bp_rp', 'phot_g_mean_mag'])
    #simulated_catalog_gaia = simulated_catalog_gaia.to_pandas().dropna(subset=['bp_rp', 'phot_g_mean_mag'])

    # compute implied quantities 
    simulated_catalog_gaia = implied_calculations(simulated_catalog_gaia)

    # keep only positive implied parallaxes
    simulated_catalog_gaia = simulated_catalog_gaia.loc[simulated_catalog_gaia['implied_parallax'] > 0]

    # add extinction correction
    simulated_catalog_gaia = extinction_correction(simulated_catalog_gaia)

    # compute the implied absolute magnitude
    simulated_catalog_gaia['implied_M_g'] = simulated_catalog_gaia['phot_g_mean_mag'] - 5*np.log10(1000/simulated_catalog_gaia['implied_parallax']) + 5
    simulated_catalog_gaia['implied_M_g_corr'] = simulated_catalog_gaia['phot_g_mean_mag'] - 5*np.log10(1000/simulated_catalog_gaia['implied_parallax']) + 5

    return simulated_catalog_gaia

if __name__ == '__main__':

     # load the gaia data
    data_gaia_big = pd.read_feather('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/200pc/raw_gaia_catalog/3M_sources_goodruwe.feather')
    data_gaia_big['bp_rp'] = data_gaia_big['phot_bp_mean_mag'] - data_gaia_big['phot_rp_mean_mag']
    
    # load the speedystar simulation
    simulated_catalog_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/top_heavy_speedystar.fits'
    simulated_catalog_f =Table.read(simulated_catalog_path)#'/Users/mncavieres/Documents/2024-2/HVS/speedystar/simulated_catalogs/photometry/cat_ejection_kappa_1.7.fits')#('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/kappa_1.7_4.3_nocuts.fits')

    print(simulated_catalog_f.columns)
    # construct the training catalog
    training_catalog = construct_training_catalog(data_gaia_big, simulated_catalog_f, subsample=100000)

    # save the training catalog
    output_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs'
    training_catalog.to_csv(os.path.join(output_path, 'kappa_1.7_speedystar.csv'))

    # catalogs path
    speedy_star_catalogs = '/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/top_heavy_phot_prop'

    # # merge the catalogs into a single catalog
    # for i, catalog in enumerate(os.listdir(speedy_star_catalogs)):
    #     print(i)
    #     if i == 0:
    #         training_catalog = construct_hvs_training(Table.read(os.path.join(speedy_star_catalogs, catalog)), subsample=100000)
    #     else:
    #         training_catalog = pd.concat([training_catalog, construct_hvs_training(Table.read(os.path.join(speedy_star_catalogs, catalog)), subsample=100000)])
        
    # # save the training catalog
    # Table.from_pandas(training_catalog).write(os.path.join(output_path, 'top_heavy_speedystar.fits'), overwrite=True)
