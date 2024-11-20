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
from download_gaia_by_healpix import query_photometric_uncertainties

if __name__ == '__main__':
    # Define exit path
    gaia_catalogs_path = "/data1/cavierescarreramc/gaia_dr3"

    # Define the new exit path
    gaia_catalogs_with_photometric_uncertainties_path = "/data1/cavierescarreramc/gaia_dr3_photometric_uncertainties"

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
        if os.path.exists(os.path.join(gaia_catalogs_with_photometric_uncertainties_path,
                             f"healpix_{healpix_pixel}.fits")):
            print(f"File exists, skipping")
            #data = Table.read(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"))


        else:
            # read the current catalog to add the photometric uncertainties+
            data_old = Table.read(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"))

            # check that the current catalog is not empty
            if len(data_old) == 0:
                print(f"File is empty, skipping")
                continue

            # Query Gaia EDR3 for sources within the specified HEALPix pixel
            data = query_photometric_uncertainties(healpix_pixel, healpix_level=healpix_level, login= True, username= username, password = password)
            
            print(data.columns)
            print(data_old.columns)

            # merge the tables on source_id
            data = join(data_old, data, keys='SOURCE_ID', join_type='left')

            # save the data
            data.write(os.path.join(gaia_catalogs_with_photometric_uncertainties_path, 
            f"healpix_{healpix_pixel}.fits"), overwrite=True)
        
            # delete variables to free memory
            del data
            del data_old    


        