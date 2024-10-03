"""
Runs the HVS selection using Sill et al. (2024) alogrithm
"""

# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from selections import is_HVS
from implied_d_vr import implied_calculations
from download_gaia_by_healpix import query
from tqdm import tqdm
from astropy.table import vstack, Table



#
    # Define exit path

if __name__ == '__main__':
    output_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/replicated_candidates_by_healpix"
    gaia_catalogs_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/gaia_by_healpix"

    # Define the HEALPix pixel number and NSIDE parameter
    healpix_pixel = 100
    nside = 3

    # read gaia database credentials from file
    with open("/Users/mncavieres/Documents/2024-2/HVS/gaia_credentials.txt", "r") as f:
        username = f.readline().strip()
        password = f.readline().strip()

    # Define the healpix pixels to be queried
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

    HVS_counter = 0
    # Loop over the pixels
    for healpix_pixel in tqdm(pixels):
        print(f"Processing HEALPix pixel {healpix_pixel}")

        # Check if the data is already downloaded
        if os.path.exists(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits")):
            print(f"File exists, loading from file")
            data = Table.read(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"))
        else:
            # Query Gaia EDR3 for sources within the specified HEALPix pixel
            data = query(healpix_pixel, nside=nside, login= True, username= username, password = password)

            # save the data
            #data.write(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"), overwrite=True)

        # Calculate the implied radial velocity and parallax
        data = implied_calculations(data)

        print(f"Number of sources: {len(data)}")


        # run the HVS selection algorithm    
        data = is_HVS(data)

        # Update the counter
        HVS_counter += len(data)
        print(f"Number of HVS candidates: {HVS_counter}")

        # Save the results to a file
        if len(data) > 0:
            data.write(os.path.join(output_path, f"HVS_in_healpix_{healpix_pixel}.fits"), overwrite=True)

        # append the data to the final table
        final_data = data if healpix_pixel == pixels[0] else vstack([final_data, data])

    # Save the final table
    print('All done')
    print(f"Total number of HVS candidates: {HVS_counter}")
    print('Saving final table')
    final_data.write('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/HVS_candidates_DR3_SILL.fits', overwrite=True)
        