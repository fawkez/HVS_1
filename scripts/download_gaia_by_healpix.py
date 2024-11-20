import os
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from scripts.selections import is_HVS
import time
from zero_point import zpt
from astropy.table import Table, join, vstack
import healpy as hp
from scripts.implied_d_vr import implied_calculations
from tqdm import tqdm
import astropy



def query(HEALPix_pixel, nside = 4, login = False, username = '', password = '', nested = True):    
    """
    Retrieve Gaia DR3 data based on a specified HEALPix pixel using the nested HEALPix scheme.
    
    This function queries the Gaia DR3 catalog for astrometric, photometric, and external distance data,
    and returns the results as a pandas DataFrame. The query can be performed for a specific HEALPix pixel 
    using the nested scheme at different resolutions (nside), with optional login to the Gaia archive for 
    querying larger datasets.

    Parameters:
    -----------
    HEALPix_pixel : int
        The HEALPix pixel (in nested format) corresponding to the region of the sky for which data is queried.
        
    nside : int, default=4
        The resolution of the HEALPix grid, defining the number of sky subdivisions. Higher values increase
        resolution. The default nside is 4.
        
    login : bool, default=False
        A flag indicating whether to log in to the Gaia archive. If set to True, a valid username and password
        must be provided.
        
    username : str, default=''
        Username for logging into the Gaia archive, required if login is True.
        
    password : str, default=''
        Password for logging into the Gaia archive, required if login is True.
        
    nested : bool, default=True
        Specifies whether the HEALPix pixel index is in nested format. If set to False, the pixel index will
        be converted from ring format to nested format.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the queried Gaia DR3 data, including astrometric (proper motions, parallaxes), 
        photometric (magnitudes, fluxes), and optionally geometrical distance estimates from the external 
        distance catalog.

    """
            
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
        ' AND ruwe < 1.4 ') # This might hunt me later, but if we impose that bp_rp < 2, we are removing stars that have bp_rp < 0.5, for the entirety of the allowed reddening range
    r = job.get_results()
    r['source_id'] = r['SOURCE_ID']


    #Correct parallax zero point bias

    # zpt.load_tables()
    # valid = r['astrometric_params_solved']>3
    # zpvals = zpt.get_zpt(r['phot_g_mean_mag'][valid], r['nu_eff_used_in_astrometry'][valid], r['pseudocolour'][valid], r['ecl_lat'][valid], r['astrometric_params_solved'][valid])
    # r['zpvals'] = np.nan
    # r['zpvals'][valid] = zpvals
    # r['parallax_corrected'] = r['parallax']-r['zpvals']

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

def query_2(HEALPix_pixel, healpix_level = 5, login = False, username = '', password = '', nested = True):
    """
    Retrieve Gaia DR3 data based on a specified HEALPix pixel using the nested HEALPix scheme.
    
    This function queries the Gaia DR3 catalog for astrometric, photometric, and external distance data,
    and returns the results as a pandas DataFrame. The query can be performed for a specific HEALPix pixel 
    using the nested scheme at different resolutions (nside), with optional login to the Gaia archive for 
    querying larger datasets.

    Parameters:
    -----------
    HEALPix_pixel : int
        The HEALPix pixel (in nested format) corresponding to the region of the sky for which data is queried.
        
    nside : int, default=4
        The resolution of the HEALPix grid, defining the number of sky subdivisions. Higher values increase
        resolution. The default nside is 4.
        
    login : bool, default=False
        A flag indicating whether to log in to the Gaia archive. If set to True, a valid username and password
        must be provided.
        
    username : str, default=''
        Username for logging into the Gaia archive, required if login is True.
        
    password : str, default=''
        Password for logging into the Gaia archive, required if login is True.
        
    nested : bool, default=True
        Specifies whether the HEALPix pixel index is in nested format. If set to False, the pixel index will
        be converted from ring format to nested format.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the queried Gaia DR3 data, including astrometric (proper motions, parallaxes), 
        photometric (magnitudes, fluxes), and optionally geometrical distance estimates from the external 
        distance catalog.

    """
    if not nested: # note that this will not work with nside = 3
        HEALPix_pixel = hp.ring2nest(nside, HEALPix_pixel)
        print('Nested healpix pixel:', HEALPix_pixel)

    #login to gaia, this will be slower but it is necessary for large queries
    if login:
        Gaia.login(user=username, password=password)

        #check if there are too many jobs and delete the extra to allow more
        jobs = [job for job in Gaia.list_async_jobs()]

        # To remove all the jobs at once:
        job_ids = [job.jobid for job in jobs]
        if len(job_ids)> 5:
            Gaia.remove_jobs(job_ids)
            print(f'Deleting {len(job_ids)} jobs')
    Gaia.ROW_LIMIT = -1



    # Calculate source ID range for the HEALPix pixel at the given nside
    factor = (2**35) * (4**(12 - healpix_level))  # Scaling factor based on level
    source_id_min = int(HEALPix_pixel * factor)
    source_id_max = int((HEALPix_pixel + 1) * factor)

    job = Gaia.launch_job_async(f"""
    SELECT gs.source_id, gs.l, gs.b, gs.ra, gs.ra_error, gs.dec, gs.dec_error, 
           gs.parallax, gs.parallax_error, gs.pmra, gs.pmra_error, gs.pmdec, gs.pmdec_error,
           gs.astrometric_params_solved, gs.astrometric_excess_noise, gs.astrometric_excess_noise_sig, 
           gs.ruwe, gs.pseudocolour, gs.nu_eff_used_in_astrometry, gs.ecl_lat, 
           gs.phot_g_mean_mag, gs.phot_g_mean_flux, gs.phot_bp_mean_mag, gs.phot_bp_mean_flux, 
           gs.phot_rp_mean_mag, gs.phot_rp_mean_flux, gs.phot_bp_rp_excess_factor, 
           gd.r_med_geo, gd.r_lo_geo, gd.r_hi_geo, gd.r_med_photogeo, gd.r_lo_photogeo, gd.r_hi_photogeo
    FROM gaiadr3.gaia_source AS gs
    JOIN external.gaiaedr3_distance AS gd 
    USING (source_id)
    WHERE gs.source_id >= {source_id_min} 
      AND gs.source_id < {source_id_max}
      AND gs.ruwe < 1.4
    """)
    r = job.get_results()

    if len(r) == 3e6:
        # raise exception if the maximum number of sources is reached
        print("Warning: Maximum number of sources reached for HEALPix pixel", HEALPix_pixel)
        # stop the execution
        raise Exception("Maximum number of sources reached")
    return r


def query_photometric_uncertainties(HEALPix_pixel, healpix_level = 5, login = False, username = '', password = '', nested = True):
    """
    Retrieve Gaia DR3 photometric uncertainties based on a specified HEALPix pixel using the nested HEALPix scheme.
    
    This function queries the Gaia DR3 catalog for astrometric, photometric, and external distance data,
    and returns the results as a pandas DataFrame. The query can be performed for a specific HEALPix pixel 
    using the nested scheme at different resolutions (nside), with optional login to the Gaia archive for 
    querying larger datasets.

    Parameters:
    -----------
    HEALPix_pixel : int
        The HEALPix pixel (in nested format) corresponding to the region of the sky for which data is queried.
        
    nside : int, default=4
        The resolution of the HEALPix grid, defining the number of sky subdivisions. Higher values increase
        resolution. The default nside is 4.
        
    login : bool, default=False
        A flag indicating whether to log in to the Gaia archive. If set to True, a valid username and password
        must be provided.
        
    username : str, default=''
        Username for logging into the Gaia archive, required if login is True.
        
    password : str, default=''
        Password for logging into the Gaia archive, required if login is True.
        
    nested : bool, default=True
        Specifies whether the HEALPix pixel index is in nested format. If set to False, the pixel index will
        be converted from ring format to nested format.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the queried Gaia DR3 data, including astrometric (proper motions, parallaxes), 
        photometric (magnitudes, fluxes), and optionally geometrical distance estimates from the external 
        distance catalog.

    """
    if not nested: # note that this will not work with nside = 3
        HEALPix_pixel = hp.ring2nest(nside, HEALPix_pixel)
        print('Nested healpix pixel:', HEALPix_pixel)

    #login to gaia, this will be slower but it is necessary for large queries
    if login:
        Gaia.login(user=username, password=password)

        #check if there are too many jobs and delete the extra to allow more
        jobs = [job for job in Gaia.list_async_jobs()]

        # To remove all the jobs at once:
        job_ids = [job.jobid for job in jobs]
        if len(job_ids)> 5:
            Gaia.remove_jobs(job_ids)
            print(f'Deleting {len(job_ids)} jobs')
    Gaia.ROW_LIMIT = -1



    # Calculate source ID range for the HEALPix pixel at the given nside
    factor = (2**35) * (4**(12 - healpix_level))  # Scaling factor based on level
    source_id_min = int(HEALPix_pixel * factor)
    source_id_max = int((HEALPix_pixel + 1) * factor)

    job = Gaia.launch_job_async(f"""
    SELECT gs.source_id, gs.phot_g_mean_flux_error, gs.phot_bp_mean_flux_error, gs.phot_rp_mean_flux_error

    FROM gaiadr3.gaia_source AS gs
    WHERE gs.source_id >= {source_id_min} 
      AND gs.source_id < {source_id_max}
      AND gs.ruwe < 1.4
    """)
    r = job.get_results()


    return r



if __name__ == '__main__':
    # Define exit path
    output_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/replicated_candidates_by_healpix"
    gaia_catalogs_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/gaia_by_healpix"

    # Define the healpix pixel level, which defines the NSIDE parameter and the number of pixels
    healpix_level = 4
    nside = 2**healpix_level
    npix = hp.nside2npix(nside) 

    # read gaia database credentials from file
    with open("/Users/mncavieres/Documents/2024-2/HVS/gaia_credentials.txt", "r") as f:
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
        