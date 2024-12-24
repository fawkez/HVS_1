"""
This script prepares a Gaia DR3 catalog for the Bayesian KDE classifier.
It reads a Gaia DR3 catalog, and computes the implied quantities.
The script also adds extinction corrections and computes the implied 
absolute magnitude. The output is a pandas dataframe with the Gaia DR3 
catalog ready for the Bayesian KDE classifier.
"""
# Standard library imports
import os
import sys
import random

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

from scipy.optimize import minimize
import healpy as hp
from numba import njit

# Astropy imports
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
from astropy.constants import kpc, au
from astropy.coordinates import (
    SkyCoord,
    Galactocentric,
    ICRS,
    CartesianRepresentation,
    CartesianDifferential,
)
from astropy.coordinates.matrix_utilities import (
    rotation_matrix,
    matrix_product,
    matrix_transpose,
)

# Custom script imports
#sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')  # Add scripts folder to path
from implied_d_vr import *  # Import specific functions or classes as needed
from selections import *
#from CMD_selection import *

# Matplotlib configuration
plt.rcParams.update({
    'font.size': 18,          # Set font size
    'figure.figsize': (10, 7),  # Set figure size
    'text.usetex': True,      # Use LaTeX for text rendering
})

def prepare_gaia(data_gaia_big, subsample='all'):
    """ 
    Construct a training catalog for classification containing a mix of gaia data and speedystar HVS with labels
    
    input:
        data_gaia: dataframe containing gaia data
        simulated_catalog_f: astropy table containing speedystar data

    output:
        data_gaia: pandas dataframe containing gaia data

    """
    
    # Read the simulated catalog if a path is passed
    if isinstance(data_gaia_big, str):
        data_gaia_big = Table.read(data_gaia_big)
        data_gaia_big = data_gaia_big.to_pandas()
    
    # check if the data is an astropy table and convert it to a pandas dataframe
    if isinstance(data_gaia_big, Table):
        data_gaia_big = data_gaia_big.to_pandas()

    # check if the required columns are present in the data
    columns_required = ['ra', 'dec', 'bp_rp', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']
    if not all([col in data_gaia_big.columns for col in columns_required]):
        raise ValueError(f"Required columns missing in the data. Required columns are {columns_required}")

    # Select a random subsample of the simulated catalog
    if not subsample == 'all':
        data_gaia_big = data_gaia_big[np.random.choice(len(data_gaia_big), subsample)]
    
    
    # compute implied quantities 
    data_gaia_big = implied_calculations(data_gaia_big)

    # keep only positive implied parallaxes
    data_gaia_big = data_gaia_big.loc[data_gaia_big['implied_parallax'] > 0]

    # add extinction correction
    data_gaia_big = extinction_correction(data_gaia_big)

    # compute the implied absolute magnitude
    data_gaia_big['implied_M_g'] = data_gaia_big['phot_g_mean_mag'] - 5*np.log10(1000/data_gaia_big['implied_parallax']) + 5

    # correct the absolute magnitude for extinction
    data_gaia_big['implied_M_g_corr'] = data_gaia_big['implied_M_g'] - data_gaia_big['A_G']

     # compute the implied absolute magnitude error
    data_gaia_big['implied_M_g_corr_error'] = compute_absolute_magntiude(data_gaia_big['phot_g_mean_mag'], 1000/(data_gaia_big['implied_parallax']
                                                                    + data_gaia_big['implied_parallax_error']), [0])

    # compute bp_rp_corr_error
    #data_gaia_big['bp_rp_corr_error'] = 
    return data_gaia_big

def prepare_speedystar(simulated_catalog_gaia, subsample='all', filter_fast = False):
    """ 
    Prepare the speedystar catalog for classification by computing implied quantities and adding extinction corrections
    and setting the columns to match the Gaia DR3 catalog.
    
    input:
        simulated_catalog_f: astropy table containing speedystar data

    output:
        simulated_catalog: pandas dataframe containing speedystar data ready for classification

    """

    # Select a random subsample of the simulated catalog
    if not subsample == 'all':
        simulated_catalog_gaia = simulated_catalog_gaia[np.random.choice(len(simulated_catalog_gaia), subsample)]

    # add color columns for future proecessing
    simulated_catalog_gaia['bp_rp'] = simulated_catalog_gaia['Gaia_BP'] - simulated_catalog_gaia['Gaia_RP']
    simulated_catalog_gaia['M_g'] = simulated_catalog_gaia['Gaia_G'] - 5*np.log10(simulated_catalog_gaia['dist']*1000) + 5

    # rename the simulated catalog columns to match the gaia catalog
    simulated_catalog_gaia = simulated_catalog_gaia
    simulated_catalog_gaia.rename_column('Gaia_G', 'phot_g_mean_mag')
    simulated_catalog_gaia.rename_column('Gaia_BP', 'phot_bp_mean_mag')
    simulated_catalog_gaia.rename_column('Gaia_RP', 'phot_rp_mean_mag')
    simulated_catalog_gaia.rename_column('par', 'parallax')
    simulated_catalog_gaia.rename_column('e_par', 'parallax_error')
    simulated_catalog_gaia.rename_column('e_pmra', 'pmra_error')
    simulated_catalog_gaia.rename_column('e_pmdec', 'pmdec_error')

    # drop nans from the simulated catalog to avoid future errors in calculations
    if isinstance(simulated_catalog_gaia, pd.DataFrame):
        simulated_catalog_gaia = simulated_catalog_gaia.dropna(subset=['bp_rp', 'phot_g_mean_mag'])
    else:
        simulated_catalog_gaia = simulated_catalog_gaia.to_pandas().dropna(subset=['bp_rp', 'phot_g_mean_mag'])

    # compute implied quantities 
    simulated_catalog_gaia = implied_calculations(simulated_catalog_gaia)

    # keep only positive implied parallaxes since negative parallaxes are unphysical
    simulated_catalog_gaia = simulated_catalog_gaia.loc[simulated_catalog_gaia['implied_parallax'] > 0]

    # add extinction correction
    simulated_catalog_gaia = extinction_correction(simulated_catalog_gaia)

    # compute the implied absolute magnitude
    simulated_catalog_gaia['implied_M_g'] = simulated_catalog_gaia['phot_g_mean_mag'] - 5*np.log10(1000/simulated_catalog_gaia['implied_parallax']) + 5
   
    # correct the absolute magnitude for extinction
    simulated_catalog_gaia['implied_M_g_corr'] = simulated_catalog_gaia['implied_M_g'] - simulated_catalog_gaia['A_G']
    simulated_catalog_gaia['M_g_corr'] = simulated_catalog_gaia['phot_g_mean_mag'] - 5*np.log10(simulated_catalog_gaia['dist']*1000) + 5 - simulated_catalog_gaia['A_G']

    # compute the implied absolute magnitude error
    simulated_catalog_gaia['implied_M_g_corr_error'] = compute_absolute_magntiude(simulated_catalog_gaia['phot_g_mean_mag'], 1000/(simulated_catalog_gaia['implied_parallax']
                                                                    + simulated_catalog_gaia['implied_parallax_error']), [0])

    # limit the magntiude to make sure that the stars should be visible by Gaia
    simulated_catalog_gaia = simulated_catalog_gaia.loc[simulated_catalog_gaia['phot_g_mean_mag'] < 21]

    # keep only stars that are fast, this should actually be done after the preparations in a filtering section
    if filter_fast:
        simulated_catalog_gaia = simulated_catalog_gaia.loc[simulated_catalog_gaia['VGCR'] > 300]


    return simulated_catalog_gaia


if __name__ == '__main__':
    # just prepare a single speedystar catalog to test, why not??

    # load catalog
    speedycatalog_ini =  Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/stock_long.fits')

    speedy_ready= prepare_speedystar(speedycatalog_ini)

    # save as fits
    speedy_ready = Table.from_pandas(speedy_ready)
    speedy_ready.write('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/ready/stock_long_ready.fits')