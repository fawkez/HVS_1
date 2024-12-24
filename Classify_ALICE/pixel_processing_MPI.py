# Multithreading libraries
import numpy as np
from mpi4py import MPI
import healpy as hp
from tqdm import tqdm
import os
from astropy.table import Table
import sys

# Custom script imports
from implied_d_vr import *
from selections import *
from prepare_gaia import *
from fft_kde import WeightedFFTKDE
from classifier import BayesianKDEClassifier

def process_path(path):
    global speedycatalog
    global processed_path
    global output_path
    """
    This should perform all the operations we need for a given Gaia catalog to classify.
    """

    # load the data
    gaia_catalog = Table.read(path)

    # check that it is not empty and skip if it is
    if len(gaia_catalog) == 0:
        print('Empty array')
        return 

    # make sure that this column exists
    gaia_catalog['bp_rp'] = gaia_catalog['phot_bp_mean_mag'] - gaia_catalog['phot_rp_mean_mag']
    gaia_catalog['bp_rp_error'] = np.log(gaia_catalog['phot_bp_mean_flux_error']**2 + gaia_catalog['phot_rp_mean_flux_error']**2)

    # prepare the gaia catalog by computing implied quantities, correcting extinction, etc.
    gaia_catalog = prepare_gaia(gaia_catalog)

    # first cut, remove stars with inconsistent parallax
    gaia_catalog = gaia_catalog.loc[parallax_consistency(
        gaia_catalog.parallax, 
        gaia_catalog.parallax_error, 
        gaia_catalog.implied_parallax,
        gaia_catalog.implied_parallax_error
    )]
    
    # second cut, remove stars that are too slow to be hypervelocity stars
    gaia_catalog = gaia_catalog.loc[is_fast(gaia_catalog.VGCR, lower_limit=300)]

    # third cut, limit extinction to a sensible range
    gaia_catalog = gaia_catalog.loc[gaia_catalog['A_G'] < 3]

    # fourth cut, limit the color and magnitude range
    gaia_catalog = gaia_catalog.loc[
        (gaia_catalog['bp_rp_corr'] > -1.2) &
        (gaia_catalog['bp_rp_corr'] < 2.5) &
        (gaia_catalog['implied_M_g_corr'] > -8) &
        (gaia_catalog['implied_M_g_corr'] < 9)
    ].copy()

    # initialize classifier
    classifier = BayesianKDEClassifier(
        speedy_catalog=speedycatalog,
        gaia_catalog=gaia_catalog,
        threshold=0.9,
        bandwidth=0.1,
        imf=None
    )

    # pre-compute the KDE grid
    classifier.compute_kde_grid(x_range=(-1, 2), y_range=(-7, 9), resolution=100)

    # initialize classification columns
    classification = np.zeros(len(gaia_catalog))
    p_class_given_data = np.zeros(len(gaia_catalog))
    p_not_class_given_data = np.zeros(len(gaia_catalog))
    p_data = np.zeros(len(gaia_catalog))

    # reset index
    gaia_catalog = gaia_catalog.reset_index(drop=True)

    # classify each source
    for i, star in tqdm(gaia_catalog.iterrows(), desc='Classifying sources', total=len(gaia_catalog)):
        try:
            classification[i], p_class_given_data[i], p_not_class_given_data[i], p_data[i] = classifier.classify_with_error_convolution(
                star['bp_rp_corr'],
                star['implied_M_g_corr'],
                x_err=0,
                y_err=0
            )
        except ValueError:
            continue

    # add the classification to the catalog
    gaia_catalog['HVS'] = classification
    gaia_catalog['p_hvs'] = p_class_given_data
    gaia_catalog['p_background'] = p_not_class_given_data
    gaia_catalog['p_data'] = p_data

    # save catalog
    save_with_prob_path = os.path.join(processed_path, f'{classifier.healpix_pixel}.fits')
    print('Saving catalog to ', save_with_prob_path)
    Table.from_pandas(gaia_catalog).write(save_with_prob_path, overwrite=True)

    # save candidates
    hvs_candidates = gaia_catalog.loc[gaia_catalog.p_hvs > 0.9]
    if len(hvs_candidates) > 0:
        candidate_path = os.path.join(output_path, f'candidates_{classifier.healpix_pixel}.fits')
        print(f'A total of {len(hvs_candidates)} were found in HEALPix pixel {classifier.healpix_pixel}')
        print('Saved to', candidate_path)
        Table.from_pandas(hvs_candidates).write(candidate_path, overwrite=True)
    else:
        print('No candidates found')

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define paths for ALICE run
    main_data_path = '/home/cavierescarreramc/data1'
    output_path = os.path.join(main_data_path, 'candidates_90')
    gaia_catalogs_path = os.path.join(main_data_path, 'gaia_dr3_photometric_uncertainties')
    processed_path = os.path.join(main_data_path, 'gaia_dr3_processed')
    simulation_path = '/home/cavierescarreramc/data1/simulated_catalogs/prepared/stock_long_ready.fits'

    # Define the healpix pixel level
    healpix_level = 4
    nside = 2**healpix_level
    npix = hp.nside2npix(nside)

    # load the simulation
    speedycatalog = Table.read(simulation_path)
    print('Simulation loaded')
    if 'bp_rp_corr' not in speedycatalog.colnames:
        print('Simulation not prepared, running preparation')
        speedycatalog = prepare_speedystar(speedycatalog)
    else:
        speedycatalog = speedycatalog.to_pandas()

    # Distribute HEALPix pixels across MPI ranks
    healpix_pixels = np.arange(0, npix + 1)
    for i, healpix_pixel in enumerate(healpix_pixels):
        if i % size == rank:
            catalog_path = os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits")
            if not os.path.exists(catalog_path):
                print(f"Rank {rank}: File {catalog_path} does not exist, skipping")
                continue
            process_path(catalog_path)

    comm.Barrier()  # Ensure all ranks finish
