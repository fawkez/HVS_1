# imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.constants import kpc, au
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from numba import njit
import multiprocessing as mp


import random
import healpy as hp

from matplotlib.gridspec import GridSpec

# Now you can import from the 'scripts' package
from implied_d_vr import *  # Or import any other module
from selections import *
from scipy.stats import norm

from tqdm import tqdm


def implied_calculations_single_2(ra, dec, pmra, pmdec, pmra_error, pmdec_error):
    # Convert positions to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Convert proper motions to radians per second
    masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
    pmra_rad_s = pmra * masyr_to_radsec
    pmdec_rad_s = pmdec * masyr_to_radsec
    epmra_rad_s = pmra_error * masyr_to_radsec
    epmdec_rad_s = pmdec_error* masyr_to_radsec

    # Compute R0 and V0 in SI units (meters and meters per second)
    #print('Computing R0 and V0...')
    R0_SI, V0_SI = compute_R0_V0_SI()

    # Run optimized function
    #print('Computing distances and velocities...')
    plx_opt, eplx_opt, VGCR_opt, VR_opt = getdist_vectorized(
        ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, epmra_rad_s, epmdec_rad_s, R0_SI, V0_SI
    )
    #print('Distances and velocities computed successfully!')
    # Post-process the results
    plx_mas, eplx_mas, VGCR_kms, VR_kms = post_process_results(plx_opt, eplx_opt, VGCR_opt, VR_opt)

    return plx_mas, eplx_mas, VGCR_kms, VR_kms


def process_star(random_star):
    columns = ['SOURCE_ID', 'l', 'b', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
       'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
       'astrometric_params_solved', 'astrometric_excess_noise',
       'astrometric_excess_noise_sig', 'ruwe', 'phot_g_mean_mag',
       'phot_bp_mean_mag', 'phot_rp_mean_mag']
    # make randomstar a df
    random_star = pd.DataFrame([random_star], columns=columns)
    #print('Processing star...')
   # print(random_star)
    #print('Star:', random_star['ra'], random_star['dec'])
    """
    Process a single star from the Gaia DR3 data to generate the implied distance distribution.
    
    Parameters:
    -----------
    random_star : astropy.table.Row
        A single star from the Gaia DR3 data table.
    """
    # Define the path to save the plots
    plots_path = '/Users/mncavieres/Documents/2024-2/HVS/Plots/distance_distribution'
# get the distribution of distances by sampling from the errors

    plx_m_distribution = []
    pmra_distribution = []
    pmdec_distribution = []
    # define gaussian distribution for pmra and pmdec based on the errors
    for i in range(1000):
        pmra = np.random.normal(random_star['pmra'], random_star['pmra_error'])
        pmdec = np.random.normal(random_star['pmdec'], random_star['pmdec_error'])

        # calculate the distance
        plx_mas, eplx_mas, VGCR_kms, VR_kms = implied_calculations_single(random_star['ra'].values, random_star['dec'].values, pmra, pmdec, random_star['pmra_error'].values, random_star['pmdec_error'].values)

        plx_m_distribution.append(plx_mas)
        pmra_distribution.append(pmra)
        pmdec_distribution.append(pmdec)

    # get the central value and the error
    plx_mas, eplx_mas, VGCR_kms, VR_kms = implied_calculations_single(random_star['ra'].values, random_star['dec'].values, random_star['pmra'].values, random_star['pmdec'].values, random_star['pmra_error'].values, random_star['pmdec_error'].values)

    #plot
        # plot the distribution of the distance
    # Plot the distribution of the distance


    # Assuming plx_m_distribution, plx_mas, eplx_mas, ra, dec, random_star, and plots_path are defined

    # Create a figure with GridSpec to customize the layout
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, width_ratios=[2, 1])  # Main plot on the left, two smaller plots on the right

    # ======== Panel 1: Distance distribution ========
    ax1 = fig.add_subplot(gs[:, 0])  # Use the first column for the full height of panel 1

    # Calculate the 10th and 90th percentiles
    #lower_percentile = np.percentile(1 / np.array(plx_m_distribution), 0.001)
    #upper_percentile = np.percentile(1 / np.array(plx_m_distribution), 99.99)
    distance_error = (np.abs(1/(plx_mas.value + eplx_mas.value)) - np.abs(1/(plx_mas.value - eplx_mas.value))) / 2
    lower_percentile = 1/plx_mas.value[0] - 15*np.abs(distance_error[0])
    upper_percentile =  1/plx_mas.value[0] + 15*np.abs(distance_error[0])
    

    # Histogram: Compute the histogram data and normalize it to peak at 1
    hist_data, bins = np.histogram(1 / np.array(plx_m_distribution), bins='scott', range=(lower_percentile, upper_percentile))
    max_hist = np.max(hist_data)  # Find the peak value of the histogram

    # Normalize the histogram to peak at 1
    hist_data = hist_data / max_hist
    ax1.step(bins[:-1], hist_data, where='post', color='blue', label='Distance distribution')  # Step plot for histogram
    ax1.fill_between(bins[:-1], hist_data, step='post', color='lightblue', alpha=0.4)

    # Plot vertical lines for the central value and error range
    ax1.axvline(1/plx_mas.value, color='red', linestyle='--', label='Central value')
    ax1.axvline(1/(plx_mas.value + eplx_mas.value), color='black', linestyle='--', label=r'$+1\sigma$')
    ax1.axvline(1/(plx_mas.value - eplx_mas.value), color='black', linestyle='--', label=r'$-1\sigma$')

    # Compute sigma assuming 1/eplx_mas as the FWHM
    sigma = 1 / (2 * np.sqrt(2 * np.log(2)) * np.abs(eplx_mas.value))

    # Generate x values for the Gaussian distribution
    x = np.linspace(lower_percentile, upper_percentile, 1000)

    # Gaussian distribution: Compute the PDF and normalize it to peak at 1
    distance_error = (np.abs(1/(plx_mas.value + eplx_mas.value)) - np.abs(1/(plx_mas.value - eplx_mas.value))) / 2
    gaussian_data = norm.pdf(x, 1/plx_mas.value, distance_error)
    gaussian_data /= np.max(gaussian_data)  # Normalize Gaussian to peak at 1

    # Plot the normalized Gaussian distribution
    ax1.plot(x, gaussian_data, color='darkorange', linewidth=2, label='Gaussian distribution')

    # Labels, title, and legend for Panel 1
    ax1.set_xlabel('Distance [kpc]')
    ax1.set_ylabel('Normalized Frequency')
    ax1.set_title('Distance distribution (normalized)')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
   #ax1.set_xlim(1/plx_mas.value - 15*np.abs(distance_error), 1/plx_mas.value + 15*np.abs(distance_error))
    # set limit based on the percentile of the histogram
    #ax1.set_xlim(lower_percentile, upper_percentile)
    


    # ======== Panel 2: Sky Position (RA vs Dec) ========
    ax2 = fig.add_subplot(gs[0, 1])  # First smaller plot in the top-right
    ax2.scatter(random_star['ra'].values, random_star['dec'].values, color='blue', label='Sky Position')
    #print(random_star['ra'].values, random_star['dec'].values)
    ax2.set_xlabel('RA [deg]')
    ax2.set_ylabel('Dec [deg]')
    ax2.set_title('Sky Position')
    ax2.grid(True)
    #ax2.legend()

    # ======== Panel 3: Proper Motion Vector ========
    ax3 = fig.add_subplot(gs[1, 1])  # Second smaller plot in the bottom-right

    # Extract proper motion data from random_star table
    pmra = random_star['pmra'].values
    pmdec = random_star['pmdec'].values
    pmra_error = random_star['pmra_error'].values
    pmdec_error = random_star['pmdec_error'].values

    # Proper motion quiver plot and error bars
    ax3.quiver(0, 0, pmra, pmdec, angles='xy', scale_units='xy', scale=1, color='blue', label='Proper Motion')
    ax3.errorbar(pmra, pmdec, xerr=pmra_error, yerr=pmdec_error, fmt='o', color='black', label='Error bars', capsize=3)

    # Set limits and labels for the proper motion panel
    range_y = np.abs(pmdec) + 5 * pmdec_error
    range_x = np.abs(pmra) + 5 * pmra_error
    ax3.set_xlim(-range_x, range_x)
    ax3.set_ylim(-range_y, range_y)
    ax3.set_xlabel('pmra [mas/yr]')
    ax3.set_ylabel('pmdec [mas/yr]')
    ax3.set_title('Proper Motion Vector')
    ax3.grid(True)
    #ax3.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'{random_star['ra'].values}-{random_star['dec'].values}_distance.png'))
    plt.clf()


if __name__ == '__main__':
    """
    This script generates the implied distance distribution for the Gaia DR3 data.
    """
    path_plots = '/Users/mncavieres/Documents/2024-2/HVS/Plots/distance_distribution'

    # Load the Gaia DR3 data
    data = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/random_objects_homogeneous_sky.fits')
    # get 1000 random stars
    data = data[np.random.choice(data['ra'].shape[0], 100, replace=False)]

    data = data.to_pandas()
    num_cores = 8

    # Create a multiprocessing pool
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm for progress tracking and pool.imap for parallel execution
        # Using itertuples to pass the rows without the index
        for result in tqdm(pool.imap(process_star, data.itertuples(index=False, name=None)), total=len(data)):
            # Collect or process results if needed
            pass