"""
Compute a dictionary of prior probabilities for the HVSs based 
on the HEALPix grid of NSIDE 4 and the HVS catalog.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
from astropy.table import Table
import healpy as hp
from collections import Counter



def compute_healpix(l, b, nside=4):
    return hp.ang2pix(nside, l, b, lonlat=True)

def compute_prior(data, simulation_data):

    # compute the healpix pixel for the simulation
    simulation_data['healpix'] = compute_healpix(simulation_data['l'], simulation_data['b'])

    # select only stars in the same pixel as the data, this should be the same pixel for all the data
    healpix_data =compute_healpix(data['l'][0], data['b'][0])

    simulation_data = simulation_data[simulation_data['healpix'] == healpix_data]

    # the prior will be the number of stars in the simulation divided by the number of stars in the data
    prior = len(simulation_data)/len(data)

    return prior

def plot_healpix_prior(prior_dict, nside, title="HEALPix Prior Map"):
    """
    Plots a HEALPix map showing the probabilities computed for each pixel.

    Parameters:
    ----------
    prior_dict : dict
        Dictionary with HEALPix pixel indices as keys and probabilities as values.
    nside : int
        The HEALPix resolution parameter.
    title : str, optional
        Title of the plot. Default is "HEALPix Prior Map".
    """
    # Initialize a HEALPix map with zeros
    npix = hp.nside2npix(nside)
    healpix_map = np.zeros(npix)
    
    # Fill the map with the probabilities from the dictionary
    for pix, prob in prior_dict.items():
        healpix_map[pix] = prob
    
    # Plot the HEALPix map
    hp.mollview(healpix_map, title=title, cmap="viridis", unit="Probability", norm="hist")
    
    # Add a graticule
    hp.graticule()
    
    # Show the plot
    plt.show()

import numpy as np
import healpy as hp
from collections import Counter

def compute_prior_hvs(l, b, nside=4):
    """
    Computes the prior probability of finding a hypervelocity star (HVS) in a HEALPix pixel
    based on the fraction of HVS in each pixel relative to the total.

    Parameters:
    ----------
    l : array-like
        Galactic longitude of HVS in degrees (simulated).
    b : array-like
        Galactic latitude of HVS in degrees (simulated).
    nside : int
        The HEALPix resolution parameter.

    Returns:
    -------
    prior_dict : dict
        Dictionary with HEALPix pixel indices as keys and priors as values.
    """
    if not isinstance(l, np.ndarray):
        l = np.array(l)
    if not isinstance(b, np.ndarray):
        b = np.array(b)

    # Check for valid ranges of Galactic coordinates
    if np.any(l < 0) or np.any(l > 360):
        raise ValueError("Galactic longitude (l) must be in the range [0, 360] degrees.")
    if np.any(b < -90) or np.any(b > 90):
        raise ValueError("Galactic latitude (b) must be in the range [-90, 90] degrees.")
    
    # Filter out invalid or masked data
    valid_mask = (~np.isnan(l)) & (~np.isnan(b)) & (~np.ma.getmaskarray(l)) & (~np.ma.getmaskarray(b))
    l_valid = l[valid_mask]
    b_valid = b[valid_mask]
    
    # Compute HEALPix indices for valid HVS
    hvs_pix = hp.ang2pix(nside, l_valid, b_valid, lonlat=True)
    
    # Count HVS in each pixel
    hvs_counts = Counter(hvs_pix)
    n_hvs_total = sum(hvs_counts.values())  # Total number of HVS
    
    # Compute priors for each pixel
    prior_dict = {pix: count / n_hvs_total for pix, count in hvs_counts.items()}
    
    return prior_dict


if __name__ == '__main__':

    # Load the HVS catalog
    hvs_cat = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/from_MIST/test_eep_propagated_phot_MIST_1e5.fits')

    # select sources observable by Gaia
    hvs_cat = hvs_cat[hvs_cat['Gaia_G_M'] < 21]
    # compute the total nside = 4 healpix pixels
    nside = 4
    prior_dict = compute_prior_hvs(hvs_cat['l'], hvs_cat['b'], nside=nside)

    # Make a pandas dataframe from the dictionary
    prior_df = pd.DataFrame(prior_dict.items(), columns=['healpix', 'prior'])

    # Save the prior dictionary as a csv file

    prior_dict_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/priors'
    os.makedirs(prior_dict_path, exist_ok=True)
    prior_dict_file = os.path.join(prior_dict_path, f'prior_dict_nside{nside}.npy')
    prior_df.to_csv(prior_dict_file, index=False)

    # Plot the prior map
    plot_healpix_prior(prior_dict, nside, title="HEALPix Prior Map")