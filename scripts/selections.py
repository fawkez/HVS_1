# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.constants import kpc, au
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from numba import njit

# load dust map sfd
from dustmaps.sfd import SFDQuery
sfd = SFDQuery()

# load healpy
import healpy as hp



def parallax_consistency(parallax, parallax_error, implied_parallax, implied_parallax_error, sigma= 2):
    """
    Check the consistency of the parallax values, if a star comes from the galactic center
    the observed and implied parallaxes should be consistent

    input:
        parallax: observed parallax in mas
        parallax_error: observed parallax error in mas
        implied_parallax: implied parallax in mas
        implied_parallax_error: implied parallax error in mas
        sigma: number of standard deviations to consider the parallax consistent
    output:
        bool: True if the parallax is consistent with the implied parallax

    """
    # check if the parallax is consistent with the implied parallax
    parallax_diff = np.abs(parallax - implied_parallax)
    parallax_diff_error = np.sqrt(parallax_error**2 + implied_parallax_error**2)
    return parallax_diff < sigma * parallax_diff_error



def is_fast(implied_radial_velocity, lower_limit = 800, upper_limit = 3500):
    """
    Check if the star is a fast star based on the implied radial velocity

    input:
        implied_radial_velocity: implied radial velocity in km/s

    output:
        bool: True if the star is fast

    """
    # check if the star is fast
    return np.logical_and(implied_radial_velocity > lower_limit, implied_radial_velocity < upper_limit)


def decent_astrometry(ruwe):
    """
    Check if the astrometry is decent based on the renormalized unit weight error

    input:
        ruwe: renormalized unit weight error

    output:
        bool: True if the astrometry is decent

    """
    return ruwe < 1.4


def distance_consistency(implied_distance, bailer_jones_distance, implied_distance_error, bailer_jones_distance_error, sigma = 2):
    """
    Check the consistency of the distance values, if a star comes from the galactic center
    the observed and implied distances should be consistent

    input:
        implied_distance: implied distance in kpc
        bailer_jones_distance: observed distance in kpc
        implied_distance_error: implied distance error in kpc
        bailer_jones_distance_error: observed distance error in kpc
        sigma: number of standard deviations to consider the distance consistent
    output:
        bool: True if the distance is consistent with the implied distance

    """
    # check if the distance is consistent with the implied distance
    distance_diff = np.abs(bailer_jones_distance - implied_distance)
    distance_diff_error = np.sqrt(bailer_jones_distance_error**2 + implied_distance_error**2)
    return distance_diff < sigma * distance_diff_error


def check_extinction(A_G, limit = 1.5):
    """
    Check if the star is affected by extinction

    input:
        ra: right ascension in degrees
        dec: declination in degrees
        limit: limit for the extinction in magnitudes

    output:
        bool: True if the star is affected by extinction

    """
    # check if the star is affected by extinction
    return A_G > limit


def is_main_sequence(bp_rp, M_g):
    """
    Check if the star is in the main sequence based on the color and absolute magnitude 
    computed from the implied distance

    input:
        bp_rp: color index
        M_g: absolute magnitude

    output:
        bool: True if the star is in the main sequence

    """
    # check if the star is in the main sequence
    return np.logical_and(-1.5 < M_g - 4.3 * bp_rp, M_g - 4.3 * bp_rp < 1.5)



def compute_absolute_magntiude(gmag, distance, extinction):
    """
    Compute the absolute magnitude of a star

    input:
        gmag: apparent magnitude
        distance: distance in kpc
        extinction: extinction in magnitudes

    output:
        float: absolute magnitude
    """
    return gmag - 5 * np.log10(distance * 1e3) + 5 - extinction



# gaia extintion correction
def extinction_correction(catalog):
    """
    Compute the extinction correction for the stars in the catalog

    input:
        catalog: Gaia catalog, must contain the columns ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag

    output:
        catalog: Gaia catalog with the extinction correction A_G
    """
    # define skycoord object
    coords = SkyCoord(ra=catalog['ra'], dec=catalog['dec'], unit=(u.degree, u.degree), frame='icrs')

    # retrive extinction values E(B - V) from sfd
    ebv = sfd(coords)

    # compute the extinction in V band
    A_V = 3.1 * ebv

    # transform to gaia G_bp, G_rp and G bands using PARSEC extinction coefficients for EDR3
    A_G = 0.83627 * A_V
    A_bp = 1.08337 * A_V
    A_rp = 0.63439 * A_V


    # add corrected photometry to the catalog
    catalog['A_G'] = A_G
    catalog['A_bp'] = A_bp
    catalog['A_rp'] = A_rp
    catalog['G_corr'] = catalog['phot_g_mean_mag'] - A_G
    catalog['bp_rp_corr'] = catalog['phot_bp_mean_mag'] - A_bp 
    catalog['bp_rp_corr'] = catalog['phot_rp_mean_mag'] - A_rp


    return catalog


def is_blue(bp_rp, limit = 0.5):
    """
    Check if the star is blue based on the color index

    input:
        bp_rp: color index
        limit: limit for the color index

    output:
        bool: True if the star is blue

    """
    return bp_rp < limit

def is_in_pixels(ra_deg, dec_deg):
    """
    Given arrays of RA and Dec in degrees, returns a boolean array indicating
    whether each source is within the specified HEALPix pixels at NSIDE=3 (ring scheme).

    Pixels: 0 to 19, 22 to 31, 35 to 42, 46 to 55, 58 to 62, 65, 66,
    69 to 73, 78, 82 to 85, 93 to 96, 101, 102, 103, and 107 in ring scheme.

    Parameters:
    - ra_deg: array-like, RA in degrees
    - dec_deg: array-like, Dec in degrees

    Returns:
    - in_pixels: boolean array, True if the source is in the specified pixels
    """
    nside = 3  # NSIDE corresponding to the HEALPix level with the given pixels

    # Expand the specified pixel ranges into a list
    pixels = np.concatenate([
        np.arange(0, 20),          # 0–19
        np.arange(22, 32),         # 22–31
        np.arange(35, 43),         # 35–42
        np.arange(46, 56),         # 46–55
        np.arange(58, 63),         # 58–62
        [65, 66],
        np.arange(69, 74),         # 69–73
        [78],
        np.arange(82, 86),         # 82–85
        np.arange(93, 97),         # 93–96
        [101, 102, 103],
        [107]
    ])
    pixels_set = set(pixels)  # Convert to a set for faster lookup

    # Convert RA and Dec to theta (colatitude) and phi (longitude) in radians
    theta = np.radians(90.0 - dec_deg)  # Colatitude in radians
    phi = np.radians(ra_deg)            # Longitude in radians

    # Compute the HEALPix pixel number for each coordinate (ring ordering)
    pix_nums = hp.ang2pix(nside, theta, phi, nest=False)

    # Check if each pixel number is in the specified list of pixels
    in_pixels = np.isin(pix_nums, list(pixels_set))

    return in_pixels

def outside_object(ra, dec, ra_obj, dec_obj, ang_dist_deg):
    """
    Check if the star has a distance greater than ang_dist_deg from the object
    """
    # define skycoord object
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    coords_obj = SkyCoord(ra=ra_obj, dec=dec_obj, unit=(u.degree, u.degree), frame='icrs')
    return coords.separation(coords_obj).degree > ang_dist_deg


if __name__ == "__main__":

    # path to the data, data must include implied distance and radial velocity
    path_data = '/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/200pc/bailer-jones_implied/implied_vr_d_bj.fits'
    output_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/filtered_stars/without_healpix_criteria'

    # load the data
    data = Table.read(path_data)

    # start time
    start = time.time()

    # add extinction correction
    print('Adding extinction correction')
    data = extinction_correction(data)

    # check if the star is blue
    print('Checking if the star is blue')
    data = data[is_blue(data['bp_rp_corr'])]
    print('Number of blue stars:', len(data))

    # check the consistency of the parallax
    print('Checking parallax consistency')
    parallax_consistency_bol = parallax_consistency(data['parallax'], data['parallax_error'], 
                                    data['implied_parallax'], data['implied_parallax_error'])
    # keep only consistent parallaxes
    data = data[parallax_consistency_bol]
    print('Number of consistent parallaxes:', len(data))

    # check if the star is fast
    print('Checking if the star is fast')
    data = data[is_fast(data['VR'])]
    print('Number of fast stars:', len(data))

    # check if the astrometry is decent
    print('Checking if the astrometry is decent')
    # keep only decent astrometry
    data = data[decent_astrometry(data['ruwe'])]
    print('Number of stars with decent astrometry:', len(data))

    # define implied distance and distance error
    data['implied_distance'] = 1/data['implied_parallax']
    data['implied_distance_error'] = data['implied_parallax_error']/data['implied_parallax']**2

    # check the consistency of the distance, note that if querying from Gaia database this must change as column names are different
    print('Checking distance consistency')
    data['distance_consistency'] = distance_consistency(data['implied_distance'], data['rgeo'], data['implied_distance_error'], data['B_rgeo_xa'] - data['b_rgeo_x'])
    # keep only consistent distances
    data = data[data['distance_consistency']]
    print('Number of consistent distances:', len(data))

    # check if the star is affected by extinction
    print('Checking if the star is heavily affected by extinction') 
    # keep only stars not affected by extinction
    data = data[check_extinction(data['A_G'])]
    print('Number of stars not affected by extinction:', len(data))

    # compute the absolute magnitude
    print('Computing absolute magnitude')
    data['Gmag'] = compute_absolute_magntiude(data['phot_g_mean_mag'], data['implied_distance'], data['A_G'])

    # check if the star is in the main sequence
    print('Checking if the star is in the main sequence')
    # keep only stars in the main sequence
    data = data[is_main_sequence(data['bp_rp_corr'], data['Gmag'])]
    print('Number of stars in the main sequence:', len(data))

    # check if the star is in the specified pixels
    print('Checking if the star is in the specified pixels')
    # keep only stars in the specified pixels
    data = data[is_in_pixels(data['ra'], data['dec'])]
    print('Number of stars in the specified pixels:', len(data))

    # check if the star is far from the LMC
    print('Checking if the star is far from the LMC and SMC')
    # Coordinates from https://www.aanda.org/articles/aa/abs/2003/47/aa3772/aa3772.html
    LMC_coordinates = [80.8942, -69.7561]
    SMC_coordinates = [13.1583, -72.8003]
    outside_LMC = outside_object(data['ra'], data['dec'], LMC_coordinates[0], LMC_coordinates[1], 8)
    outside_SMC = outside_object(data['ra'], data['dec'], SMC_coordinates[0], SMC_coordinates[1], 3)

    # keep only stars far from the LMC and SMC
    data = data[np.logical_and(outside_LMC, outside_SMC)]
    print('Number of stars far from the LMC and SMC:', len(data))

    #  save the data
    print('Saving the data to:', output_path + '/filtered_stars_noHP.fits')
    data.write(os.path.join(output_path, 'filtered_stars_noHP.fits'), overwrite=True)
    
    # print time it took to run the script
    print('Time it took to run the script:', time.time()-start, 'seconds')
    print('Done!')


