import os
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from selections import is_HVS
import time
from zero_point import zpt
from astropy.table import Table, join, vstack
import healpy as hp
from implied_d_vr import implied_calculations
from tqdm import tqdm
import astropy
from download_gaia_by_healpix import query_2

if __name__ == '__main__':
    # Define exit path
    gaia_catalogs_path = "/data1/cavierescarreramc/gaia_dr3"

    # Define the healpix pixel level, which defines the NSIDE parameter and the number of pixels
    healpix_level = 4
    nside = 2**healpix_level
    npix = hp.nside2npix(nside) 

    # read gaia database credentials from file
    with open("/home/cavierescarreramc/gaia_credentials.txt", "r") as f:
        username = f.readline().strip()
        password = f.readline().strip()
        
    # Loop over the pixels
    for healpix_pixel in tqdm(np.arange(0, npix+1)):
        print(f"Processing HEALPix pixel {healpix_pixel}")

        # Check if the data is already downloaded
        if os.path.exists(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits")):
            print(f"File exists, skipping")
            #data = Table.read(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"))
        else:
            # Query Gaia EDR3 for sources within the specified HEALPix pixel
            data = query_2(healpix_pixel, healpix_level=healpix_level, login= True, username= username, password = password)

            # save the data
            data.write(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"), overwrite=True)
