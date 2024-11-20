"""
Use MIST isochrones to obtain Gaia photometry for stars sampled from the IMF

"""

# imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from astropy.table import Table
from astropy.table import hstack, vstack
import astropy.units as u

from isochrones.mist import MISTEvolutionTrackGrid, MISTIsochroneGrid, MIST_EvolutionTrack
track = MIST_EvolutionTrack()
iso_grid = MISTIsochroneGrid()

from isochrones.mist import MIST_Isochrone
from isochrones import get_ichrone
mist = get_ichrone('mist')

from isochrones import get_ichrone
import numpy as np
from astropy.table import Table


# Add the path to the 'scripts' folder directly
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')


# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *

def get_star_photometry_v2(feh_array, initial_mass_array, age_ejection_array, flight_time_array):
    """
    Retrieve Gaia photometry for a set of stars given their initial masses, metallicities (feh),
    and ages using the MIST isochrone model.

    Parameters:
    - feh_array: array-like, metallicities [Fe/H] of the stars.
    - initial_mass_array: array-like, initial masses of the stars in solar masses.
    - age_ejection_array: array-like, ejection ages of the stars in Gyr.
    - flight_time_array: array-like, flight times of the stars in Gyr.

    Returns:
    - pandas DataFrame with Gaia magnitudes 'G', 'BP', 'RP', and other stellar properties.
    """
    # Load the MIST isochrone model
    mist = get_ichrone('mist')

    results = []

    # Iterate over each star's parameters
    for feh, initial_mass, age_ejection, flight_time in zip(
            feh_array, initial_mass_array, age_ejection_array, flight_time_array):
        
        age = np.log10((age_ejection + flight_time)) + 9 # Convert age to log10(age) in yr

        try:
            # Generate the star's model
            star_model = mist.generate(mass=initial_mass, age=age, feh=feh, accurate=True)
            
            # Extract scalar values from the DataFrame
            G_mag = star_model['G_mag'].values[0]
            BP_mag = star_model['BP_mag'].values[0]
            RP_mag = star_model['RP_mag'].values[0]
            Teff = star_model['Teff'].values[0]
            logg = star_model['logg'].values[0]
            radius = star_model['radius'].values[0]
            logL = star_model['logL'].values[0]
            feh_value = star_model['feh'].values[0]
            eep = star_model['eep'].values[0]
            
            result = {
                'G': G_mag,
                'BP': BP_mag,
                'RP': RP_mag,
                'Teff': Teff,
                'logg': logg,
                'radius': radius,
                'logL': logL,
                'feh': feh_value,
                'eep': eep
            }
            print(f"Generated model for star with mass {initial_mass}, age {age}, feh {feh}")
        except (ValueError, RuntimeError, IndexError) as e:
            print(f"Error during model generation for star with mass {initial_mass}, "
                  f"age {age}, feh {feh}: {e}")
            # Assign NaN values if generation fails
            result = {
                'G': np.nan,
                'BP': np.nan,
                'RP': np.nan,
                'Teff': np.nan,
                'logg': np.nan,
                'radius': np.nan,
                'logL': np.nan,
                'feh': np.nan,
                'eep': np.nan
            }
        
        results.append(result)

    return Table(results)


def build_training_catalog(data_gaia_big, simulated_catalog, subsample=100000):
        # subsample the gaia catalog
    data_gaia_big = data_gaia_big.sample(subsample) 

    data_gaia_big = implied_calculations(data_gaia_big)

    # keep only positive implied parallaxes
    data_gaia_big = data_gaia_big.loc[data_gaia_big['implied_parallax'] > 0]

    # add extinction correction
    data_gaia_big = extinction_correction(data_gaia_big)

    # compute the implied absolute magnitude
    data_gaia_big['implied_M_g'] = data_gaia_big['phot_g_mean_mag'] - 5*np.log10(1000/data_gaia_big['implied_parallax']) + 5
    data_gaia_big['implied_M_g_corr'] = data_gaia_big['phot_g_mean_mag'] - 5*np.log10(1000/data_gaia_big['implied_parallax']) + 5

    # set all gaia stars as not hvs
    data_gaia_big['is_hvs'] = 0


    # now add the simulated cmd
    # keep in mind that the simulated photometry should be called implied_M_g_corr and bp_rp_corr
    simulated_catalog = Table()
    simulated_catalog['bp_rp_corr'] = pmage_catalog_with_photometry['BP'] - pmage_catalog_with_photometry['RP']
    simulated_catalog['implied_M_g_corr'] = pmage_catalog_with_photometry['G']
    simulated_catalog['phot_g_mean_mag'] = pmage_catalog_with_photometry['G']
    simulated_catalog['phot_bp_mean_mag'] = pmage_catalog_with_photometry['BP']
    simulated_catalog['phot_rp_mean_mag'] = pmage_catalog_with_photometry['RP']
    simulated_catalog['is_hvs'] = 1

    # add some scatter in bp_rp_corr and implied_M_g_corr
    #simulated_catalog['bp_rp_corr'] += np.random.normal(0, 0.003, len(simulated_catalog))
    #simulated_catalog['implied_M_g_corr'] += np.random.normal(0, 0.003, len(simulated_catalog))

    # create an astropy table with just the required columns from the gaia data
    data_gaia_training = Table.from_pandas(data_gaia_big[['bp_rp', 'implied_M_g_corr', 'bp_rp_corr', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'is_hvs']])

    # merge the pandas dataframes
    training_catalog = vstack([data_gaia_training, simulated_catalog])

    return training_catalog



if __name__ == '__main__':
    # usage
    pmage_catalog = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/SFH_sampling_catalogs/initial/sample_constant_SFH.fits')
    
    feh = 0.29 # in dex from https://www.aanda.org/articles/aa/pdf/2022/10/aa44411-22.pdf 
    pmage_catalog_with_photometry = hstack([pmage_catalog, 
                                    get_star_photometry_v2([feh]*len(pmage_catalog), 
                                    pmage_catalog['mass'], pmage_catalog['age'], 
                                    pmage_catalog['flight_time'])])
    pmage_catalog_with_photometry = pmage_catalog_with_photometry.to_pandas().dropna()

    # Save the output
    output_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/SFH_sampling_catalogs/MIST_photometry'
    pmage_catalog_with_photometry = Table.from_pandas(pmage_catalog_with_photometry)
    pmage_catalog_with_photometry.write(output_path + f'/sample_constant_SFH.fits', overwrite=True)

    # load gaia data
    data_gaia_big = pd.read_feather('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/200pc/raw_gaia_catalog/3M_sources_goodruwe.feather')
    data_gaia_big['bp_rp'] = data_gaia_big['phot_bp_mean_mag'] - data_gaia_big['phot_rp_mean_mag']
    data_gaia_big = data_gaia_big.dropna(subset=['bp_rp', 'phot_g_mean_mag'])

    # build training catalog
    output_path_training = '/Users/mncavieres/Documents/2024-2/HVS/Data/SFH_sampling_catalogs/training_catalog_CMD_classification'
    training_catalog = build_training_catalog(data_gaia_big, pmage_catalog_with_photometry)
    training_catalog.write( os.path.join(output_path_training, 'constant_SFH.fits'), overwrite=True)