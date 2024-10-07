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



def query(HEALPix_pixel, nside = 4, login = False, username = '', password = '', nested = True): #target_name must be the source id of the star to study
                        #returns a panda DataFrame with the required data to study cluster membership
                        #if the data is not present in the folder a gaia query is performed
        
        print('Starting Query')
        if nested:
            print('Processing nested healpix pixel:', HEALPix_pixel)

        if not nested: # note that this will not work with nside = 3
            print('Converting to nested')
            HEALPix_pixel = hp.ring2nest(nside, HEALPix_pixel)
            print('Nested healpix pixel:', HEALPix_pixel)
        

        start_time = time.time()

        source_id_range = {}

        for neighbour in [HEALPix_pixel] :
            source_id_range[neighbour] = neighbour*(2**35)*(4**(12- nside)), (neighbour+1)*(2**35)*(4**(12- nside))


        #login to gaia
        if login:
            Gaia.login(user=username, password=password) 
            #check if there are too many jobs and delete the extra to allow more
        
            jobs = [job for job in Gaia.list_async_jobs()]
            # To remove all the jobs at once:
            job_ids = [job.jobid for job in jobs]
            if len(job_ids)> 5:
                Gaia.remove_jobs(job_ids)
                print(f'Deleting {len(job_ids)} jobs')

        #change mirror for speed
        #Gaia = GaiaClass(gaia_tap_server='http://gaia.ari.uni-heidelberg.de/tap') #does not work now
        #query for gaia edr3 data
        job = Gaia.launch_job_async("SELECT source_id, l, b, ra, ra_error, dec, dec_error, parallax, parallax_error,"
            " pmra, pmra_error, pmdec, pmdec_error,"
            " astrometric_params_solved, astrometric_excess_noise, astrometric_excess_noise_sig,"
            " ruwe, pseudocolour, nu_eff_used_in_astrometry, pseudocolour, ecl_lat,"
            #-- Gaia photometry
            "phot_g_mean_mag, phot_g_mean_flux,"
            "phot_bp_mean_mag, phot_bp_mean_flux,"
            "phot_rp_mean_mag, phot_rp_mean_flux,"
            "phot_bp_rp_excess_factor"
            #-- From Gaia EDR3
            " FROM gaiadr3.gaia_source"
            #-- Select only valid points
            ' WHERE ((source_id >= ' + str(source_id_range[HEALPix_pixel][0]) + ' AND source_id < ' +  str(source_id_range[HEALPix_pixel][1]) + '))'
            ' AND ra IS NOT NULL '
            ' AND dec IS NOT NULL '
            ' AND parallax IS NOT NULL '
            ' AND ruwe < 1.4 '
            ' AND bp_rp < 2') # This might hunt me later, but if we impose that bp_rp < 2, we are removing stars that have bp_rp < 0.5, for the entirety of the allowed reddening range
        r = job.get_results()
        r['source_id'] = r['SOURCE_ID']


        #Correct parallax zero point bias

        zpt.load_tables()
        valid = r['astrometric_params_solved']>3
        zpvals = zpt.get_zpt(r['phot_g_mean_mag'][valid], r['nu_eff_used_in_astrometry'][valid], r['pseudocolour'][valid], r['ecl_lat'][valid], r['astrometric_params_solved'][valid])
        r['zpvals'] = np.nan
        r['zpvals'][valid] = zpvals
        r['parallax_corrected'] = r['parallax']-r['zpvals']

        #external query for geometrical distances
        job2 = Gaia.launch_job_async("SELECT source_id, r_med_geo, r_lo_geo, r_hi_geo, r_med_photogeo, r_lo_photogeo, r_hi_photogeo"
        #    #-- From Gaia EDR3
            " FROM external.gaiaedr3_distance"
        #    #-- Select only valid points
            ' WHERE ((source_id >= ' + str(source_id_range[HEALPix_pixel][0]) + ' AND source_id < ' +  str(source_id_range[HEALPix_pixel][1]) + '))')
        r_geo =  job2.get_results()

        print(f'Query done in {time.time() - start_time}')

        # merge tables by source_id given by the left astropy table
        merged = join(r, r_geo, keys='source_id', join_type='left')

        return merged



if __name__ == '__main__':
    # Define exit path
    output_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/replicated_candidates_by_healpix"
    gaia_catalogs_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/gaia_by_healpix"

    # Define the NSIDE parameter, for this Sill used 3 but I will use 4
    nside = 4

    # read gaia database credentials from file
    with open("/Users/mncavieres/Documents/2024-2/HVS/gaia_credentials.txt", "r") as f:
        username = f.readline().strip()
        password = f.readline().strip()
        
    # Loop over the pixels
    for healpix_pixel in tqdm(np.arange(0, 192)):
        print(f"Processing HEALPix pixel {healpix_pixel}")

        # Check if the data is already downloaded
        if os.path.exists(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits")):
            print(f"File exists, skipping")
            #data = Table.read(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"))
        else:
            # Query Gaia EDR3 for sources within the specified HEALPix pixel
            data = query(healpix_pixel, nside=nside, login= True, username= username, password = password)

            # save the data
            data.write(os.path.join(gaia_catalogs_path, f"healpix_{healpix_pixel}.fits"), overwrite=True)


    #     try:
    #         # Calculate the implied radial velocity and parallax
    #         data = implied_calculations(data)

    #         print(f"Number of sources: {len(data)}")

    #         # run the HVS selection algorithm    
    #         data = is_HVS(data)

    #         # Update the counter
    #         HVS_counter += len(data)
    #         print(f"Number of HVS candidates: {HVS_counter}")

    #         # Save the results to a file
    #         if len(data) > 0:
    #             data.write(os.path.join(output_path, f"HVS_in_healpix_{healpix_pixel}.fits"), overwrite=True)

    #         # append the data to the final table
    #         final_data = data if healpix_pixel == pixels[0] else vstack([final_data, data])

    #     except astropy.units.core.UnitConversionError:
    #         print(f"Unit conversion error in pixel {healpix_pixel}")
    #         continue

    # # Save the final table
    # print('All done')
    # print(f"Total number of HVS candidates: {HVS_counter}")
    # print('Saving final table')
    # final_data.write('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/HVS_candidates_DR3_SILL.fits', overwrite=True)
        