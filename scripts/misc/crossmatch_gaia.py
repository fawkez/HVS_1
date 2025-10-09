import pandas as pd
import numpy as np
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import healpy as hp

def source_id_crossmatch(input_path, match_col_1 = 'SOURCE_ID', match_col_2 = 'source_id',
                          ra_col = 'ra', dec_col ='dec', gaia_path = 'path', match_sky = False):

    # Read the input data
    input_data = Table.read(input_path)

    # compute the healpix pixel for each source
    healpix_level = 4
    nside = 2**healpix_level
    npix = hp.nside2npix(nside) 

    # create a new column with the healpix pixel
    input_data['healpix_pixel'] = hp.ang2pix(nside, input_data[ra_col], input_data[dec_col], lonlat=True)

    # split the data into healpix pixels
    healpix_pixels = np.unique(input_data['healpix_pixel'])

    # for each healpix pixel, read the gaia data and crossmatch
    for pixel in healpix_pixels:
        # Read the gaia data
        gaia_file = os.path.join(gaia_path, f'gaia_healpix_{pixel}.fits')
        if not os.path.exists(gaia_file):
            print(f'File {gaia_file} does not exist')
            continue

        gaia_data = Table.read(gaia_file)

        # filter the input data for the current healpix pixel
        input_data_pixel = input_data[input_data['healpix_pixel'] == pixel]

        if match_sky:
            # crossmatch the data
            coords_input = SkyCoord(input_data_pixel[ra_col], input_data_pixel[dec_col], unit=(u.deg, u.deg))
            coords_gaia = SkyCoord(gaia_data['ra'], gaia_data['dec'], unit=(u.deg, u.deg))

            idx, d2d, d3d = coords_input.match_to_catalog_sky(coords_gaia)

            # add the match information to the input data
            input_data_pixel['match_idx'] = idx
            input_data_pixel['match_distance'] = d2d.to(u.arcsec).value

            # add the matched gaia data to the input data
            input_data_pixel['match_source_id'] = gaia_data['source_id'][idx]
            

        # save the matched data to a file
        output_file = os.path.join('output', f'matched_{pixel}.fits')
        input_data_pixel.write(output_file, overwrite=True)