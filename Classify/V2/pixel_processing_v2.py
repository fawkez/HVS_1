
# Multithreading libraries
import numpy as np
from multiprocessing import Pool

# good stuff
import healpy as hp
from tqdm import tqdm
import os
from astropy.table import Table
import sys

# Custom script imports, this needs to be changed for it to work on ALICE
#sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')  # Add scripts folder to path

#from scripts.implied_d_vr import *  # Import specific functions or classes as needed
#from scripts.selections import *
#from scripts.CMD_selection import *
#from scripts.catalog_preparation.prepare_gaia import *
#from scripts.misc.fft_kde import WeightedFFTKDE
#from classifier import BayesianKDEClassifier
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS/Classify')
#from scripts.implied_d_vr import *  # Import specific functions or classes as needed
from V2.selections import *
from V2.iterative_correction import implied_calculations
from V2.prepare_gaia_iterative import prepare_gaia_iterative
from V2.prepare_gaia_iterative import prepare_speedystar
from V2.classifier_CMD import HistogramClassifier2D

import time
import pandas as pd


def process_path(path, healpix_pixel):
    global speedycatalog
    global processed_path
    global output_path

    """
    This should perform all the operations we need for a given Gaia catalog to classify.
    """

    # load the data
    gaia_catalog = Table.read(path)

    # load the prior map
    prior = pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Classify/V2/Prior_map.csv') # this is hardcoded,. not good

    # check that it is not empty and skip if it is
    if len(gaia_catalog) == 0:
        print('Empty array')
        return 

    # make sure that this column exists
    gaia_catalog['bp_rp'] = gaia_catalog['phot_bp_mean_mag'] - gaia_catalog['phot_rp_mean_mag']

    print('Processing HEALPix pixel', healpix_pixel)
    print('Catalog has', len(gaia_catalog), 'sources')
    # prepare the gaia catalog by computing implied quantities, correcting extinction, etc.
    gaia_catalog = prepare_gaia_iterative(gaia_catalog) # now gaia_catalog should have all the implied magnitudes and extinction corrections necessary for classificaiton

    # fourth cut, limit the color and magntiude range to a sensible space that also allows the KDE to fit correcty
    # particularly the extinction correctio is only valid within the bp-rp range -0.5 to 2.5, we will extend a bit because of the assumption that stars are beyond the extinction layer
    # gaia_catalog = gaia_catalog.loc[(gaia_catalog['bp_rp_corr'] > -1.2) &
    #                                  (gaia_catalog['bp_rp_corr'] < 2.5) &
    #                                    (gaia_catalog['implied_M_g_corr'] > -8) &
    #                                      (gaia_catalog['implied_M_g_corr'] < 9)].copy()

    print('Catalog has', len(gaia_catalog), 'sources after preparing')
    # initialize classifier
    x_hvs = speedycatalog['bp_rp_corr']
    y_hvs = speedycatalog['implied_M_g_corr']
    x_bg = gaia_catalog['bp_rp_corr']
    y_bg = gaia_catalog['implied_M_g_corr']

    # Build the classifier
    classifier = HistogramClassifier2D(
        x_hvs, y_hvs, x_bg, y_bg,
        bins_x=100, bins_y=100,
        x_range=(-1.2, 2.5),  # for example
        y_range=(-8, 10),
    )

    # compute the HEALPix pixel of the current catalog
    # healpix_level = 4
    # nside = 2**healpix_level
    # npix = hp.nside2npix(nside)
    # theta = np.radians(90.0 - np.mean(gaia_catalog['dec']))  # Colatitude
    # phi = np.radians(np.mean(gaia_catalog['ra']))           # Longitude
    # pix = hp.ang2pix(nside, theta, phi, lonlat=False, nest=True)

    # get the prior for this pixel
    prior_pixel = prior.loc[prior.pixel_nest == healpix_pixel]['prior'].values[0]
    # if len(prior_pixel) == 0:
    #     print('No prior for this pixel')
    #     prior_pixel = 0

    print('Prior for this pixel is', prior_pixel)
    # classify
    p_hvs, p_bg, p_data = classifier.classify(gaia_catalog['bp_rp_corr'],
                                               gaia_catalog['implied_M_g_corr'], 
                                               gaia_catalog['bp_rp_error'],
                                               gaia_catalog['implied_M_g_corr_error'])

    # add the classification to the catalog
    #gaia_catalog['HVS'] = classification
    gaia_catalog['p_hvs'] = p_hvs
    gaia_catalog['p_background'] = p_bg
    gaia_catalog['p_data'] = p_data
    gaia_catalog['p_hvs_over_bg'] = (p_hvs / p_bg )*prior_pixel
    gaia_catalog['healpix_pixel'] = healpix_pixel

    # save
    save_with_prob_path = os.path.join(processed_path, f'{healpix_pixel}.fits')
    # check that the directory exists
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    print('Saving catalog to ', save_with_prob_path)
    Table.from_pandas(gaia_catalog).write(save_with_prob_path, overwrite = True)

    # select sources that pass the classification
     # first cut, remove stars with inconsistent parallax
    gaia_catalog = gaia_catalog.loc[parallax_consistency(gaia_catalog.parallax, 
                                                                 gaia_catalog.parallax_error, 
                                                                 gaia_catalog.implied_parallax,
                                                                 gaia_catalog.implied_parallax_error)]
    
    # second cut, remove stars that are too slow to be a hypervelocity star
    gaia_catalog = gaia_catalog.loc[is_fast(gaia_catalog.VGCR, lower_limit=300)]

    # third cut, limit extinction to a sensible range, but still big
    gaia_catalog = gaia_catalog.loc[gaia_catalog['A_G'] < 3]

    hvs_candidates = gaia_catalog.loc[gaia_catalog.p_hvs > 0.5]
    if len(hvs_candidates > 0):
        # save
        candidate_path = os.path.join(output_path,f'candidates_{healpix_pixel}.fits')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'A total of {len(hvs_candidates)} were found in HEALPix pixel {healpix_pixel}')
        print('Saved to', candidate_path)
        Table.from_pandas(hvs_candidates).write(candidate_path, overwrite = True)
    else:
        print('No candidates found')


if __name__ == '__main__':
    # Define paths for local run
    output_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/Classification_test_v2/output" # This is the path in which we will save those that pass with 90% confidence
    gaia_catalogs_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/gaia_by_healpix" # This is the path within ALICE for the Gaia data
    processed_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/Classification_test_v2/processed_catalog" # Here we will basically put a copy of the Gaia catalog but with the added columns in case we want to change the confidence level or whatever
    simulation_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/stock_long.fits" # This should be the simulation that we will use to train the classifier

    # # Define paths for ALICE run
    # main_data_path = '/home/cavierescarreramc/data1'
    # output_path = os.path.join(main_data_path, 'candidates_90')
    # gaia_catalogs_path = os.path.join(main_data_path, 'gaia_dr3_photometric_uncertainties')
    # process_path = os.path.join(main_data_path,'gaia_dr3_processed' )
    # simulation_path = '/home/cavierescarreramc/data1/simulated_catalogs/prepared/stock_long_ready.fits'


    # Define the healpix pixel level, which defines the NSIDE parameter and the number of pixels, this has to stay the same because this is how the data was donwloaded
    healpix_level = 4
    nside = 2**healpix_level
    npix = hp.nside2npix(nside) 

    # load the simulation
    speedycatalog = Table.read(simulation_path)
    print('Simulation loaded')
    #speedycatalog = speedycatalog.to_pandas()
    # check that bp_rp_corr and implied_M_g_corr columns are in the simulation
    if 'bp_rp_corr' not in speedycatalog.colnames: # this deals with the case in which the input catalog is not ready
        print('Simulation not prepared, running preparation')
        speedycatalog = prepare_speedystar(speedycatalog) # this will turn it into a pandas dataframe

    else: # if it has already been processed we only need to turn it into a dataframe
        speedycatalog = speedycatalog.to_pandas()

    # save the catalog to not have to do it again
    #Table.from_pandas(speedycatalog).write('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/stock_long_ready.fits')
        
    # Loop over the pixels
    for healpix_pixel in tqdm(np.arange(0, npix+1)):
        #print(f"Processing HEALPix pixel {healpix_pixel}")

        catalog_path = os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits")

        # Check if the data is already downloaded
        if not os.path.exists(catalog_path):
            #print(f"File does not exist, skipping")
            # skip this iteration
            continue
        
        # process
        process_path(catalog_path, healpix_pixel)
