from mpi4py import MPI
import numpy as np
import healpy as hp
from tqdm import tqdm
import os
from astropy.table import Table
import sys
import time
import pandas as pd

# Custom script imports
#sys.path.append('/Users/mncavieres/Documents/2024-2/HVS/Classify')
from selections import *
from iterative_correction import implied_calculations
from prepare_gaia_iterative import prepare_gaia_iterative, prepare_speedystar
from classifier_CMD import HistogramClassifier2D

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_path(path, healpix_pixel):
    global speedycatalog
    global processed_path
    global output_path
    global prior_path
    """
    Process one Gaia catalog file corresponding to a given HEALPix pixel.
    """
    # Load the Gaia catalog
    gaia_catalog = Table.read(path)

    # Load the prior map.
    # Here we assume the prior map is located in the same folder as the simulation;
    # adjust the path as needed.
    prior = pd.read_csv(prior_path)

    # Skip if the catalog is empty
    if len(gaia_catalog) == 0:
        # save the pixel number to a log file
        # Create the log file if it does not exist
        log_file_path = '/home/cavierescarreramc/HVS_1/Classify_ALICE_V2/empty_catalogs.log'
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write('Log of empty catalogs:\n')
        with open(log_file_path, 'a') as f:
            f.write(f'{healpix_pixel}\n')
        print(f'Rank {rank}: Empty catalog for pixel {healpix_pixel}')
        return 

    # Create the color column
    gaia_catalog['bp_rp'] = gaia_catalog['phot_bp_mean_mag'] - gaia_catalog['phot_rp_mean_mag']

    print(f'Rank {rank}: Processing HEALPix pixel {healpix_pixel}')
    print(f'Rank {rank}: Catalog has {len(gaia_catalog)} sources')

    # Prepare the catalog (e.g., compute implied magnitudes, correct for extinction, etc.)
    gaia_catalog = prepare_gaia_iterative(gaia_catalog)
    print(f'Rank {rank}: Catalog has {len(gaia_catalog)} sources after preparation')

    # Initialize classifier using simulation (speedycatalog) as the training data for HVS
    x_hvs = speedycatalog['bp_rp_corr']
    y_hvs = speedycatalog['implied_M_g_corr']
    x_bg = gaia_catalog['bp_rp_corr']
    y_bg = gaia_catalog['implied_M_g_corr']

    classifier = HistogramClassifier2D(
        x_hvs, y_hvs, x_bg, y_bg,
        bins_x=100, bins_y=100,
        x_range=(-1.2, 2.5),
        y_range=(-8, 10),
    )

    # Get the prior for the current pixel
    try:
        prior_pixel = prior.loc[prior.pixel_nest == healpix_pixel]['prior'].values[0]
    except IndexError:
        print(f'Rank {rank}: No prior found for pixel {healpix_pixel}')
        return

    print(f'Rank {rank}: Prior for pixel {healpix_pixel} is {prior_pixel}')

    # Classify the catalog sources
    p_hvs, p_bg, p_data = classifier.classify(
        gaia_catalog['bp_rp_corr'],
        gaia_catalog['implied_M_g_corr'], 
        gaia_catalog['bp_rp_error'],
        gaia_catalog['implied_M_g_corr_error']
    )

    # Append classification results to the catalog
    gaia_catalog['p_hvs'] = p_hvs
    gaia_catalog['p_background'] = p_bg
    gaia_catalog['p_data'] = p_data
    gaia_catalog['p_hvs_over_bg'] = (p_hvs / p_bg) * prior_pixel
    gaia_catalog['healpix_pixel'] = healpix_pixel

    # Save the processed catalog
    save_with_prob_path = os.path.join(processed_path, f'{healpix_pixel}.fits')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    print(f'Rank {rank}: Saving processed catalog to {save_with_prob_path}')
    Table.from_pandas(gaia_catalog).write(save_with_prob_path, overwrite=True)

    # Apply additional cuts and save candidate sources
    gaia_catalog = gaia_catalog.loc[parallax_consistency(
        gaia_catalog.parallax, 
        gaia_catalog.parallax_error, 
        gaia_catalog.implied_parallax,
        gaia_catalog.implied_parallax_error
    )]
    gaia_catalog = gaia_catalog.loc[is_fast(gaia_catalog.VGCR, lower_limit=300)]
    gaia_catalog = gaia_catalog.loc[gaia_catalog['A_G'] < 3]

    hvs_candidates = gaia_catalog.loc[gaia_catalog.p_hvs > 0.5]
    if len(hvs_candidates) > 0:
        candidate_path = os.path.join(output_path, f'candidates_{healpix_pixel}.fits')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'Rank {rank}: Found {len(hvs_candidates)} candidates in pixel {healpix_pixel}')
        print(f'Rank {rank}: Saving candidates to {candidate_path}')
        Table.from_pandas(hvs_candidates).write(candidate_path, overwrite=True)
    else:
        print(f'Rank {rank}: No candidates found for pixel {healpix_pixel}')

if __name__ == '__main__':
    # Define ALICE run paths
    main_data_path = '/home/cavierescarreramc/data1'
    output_path = os.path.join(main_data_path, 'candidates_50_v3')
    gaia_catalogs_path = os.path.join(main_data_path, 'gaia_dr3_photometric_uncertainties')
    processed_path = os.path.join(main_data_path, 'gaia_dr3_processed_v2')
    simulation_path = '/home/cavierescarreramc/data1/simulated_catalogs/prepared/stock_long_ready.fits'
    prior_path = '/home/cavierescarreramc/HVS_1/Classify_ALICE_V2/Prior_map.csv'

    # Define the HEALPix pixel level and compute the total number of pixels
    healpix_level = 4
    nside = 2 ** healpix_level
    npix = hp.nside2npix(nside)

    # load the list of pixels left to process
    pixels_left = pd.read_csv('/home/cavierescarreramc/HVS_1/Classify_ALICE_V2/pixels_left.csv')['0'].values

    # Rank 0 loads (and, if needed, prepares) the simulation catalog,
    # then broadcasts it to all MPI ranks.
    if rank == 0:
        speedycatalog = Table.read(simulation_path)
        print("Rank 0: Simulation loaded")
        if 'bp_rp_corr' not in speedycatalog.colnames:
            print("Rank 0: Simulation not prepared, running preparation")
            speedycatalog = prepare_speedystar(speedycatalog)
        else:
            speedycatalog = speedycatalog.to_pandas()
    else:
        speedycatalog = None
    speedycatalog = comm.bcast(speedycatalog, root=0)

    # Create the list of all pixels and split among ranks
    local_pixels = pixels_left[rank::size]
    print(f"Rank {rank} processing {len(local_pixels)} out of {len(pixels_left)} total pixels left.")

    # Process each assigned pixel
    for healpix_pixel in tqdm(local_pixels, desc=f"Rank {rank}"):
        catalog_path = os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits")
        if not os.path.exists(catalog_path):
            continue
        process_path(catalog_path, healpix_pixel)

    # Synchronize before exiting
    comm.Barrier()
    if rank == 0:
        print("All ranks have finished processing.")
