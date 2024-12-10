"""
Since we are skipping some of the speedystar photometry, we need to recompute
apparent magnitudes for the stars in the speedystar catalog. This script will
interpolate the MIST isochrone model to get the Gaia G, BP, and RP magnitudes
for each star in the speedystar catalog, using the EEP, age, and metallicity,
combined with the distance from speedystar propagations and the extinction from
the dustmaps package. The output will be a new catalog with the Gaia magnitudes
appended.
"""

import numpy as np
import pandas as pd
from isochrones.mist import MIST_EvolutionTrack
from isochrones.mist import MIST_Isochrone
from tqdm import tqdm
from astropy.table import Table

from isochrones import get_ichrone
from isochrones import get_ichrone
from isochrones.mist import MIST_Isochrone
from isochrones.mist import MIST_EvolutionTrack




def get_star_photometry_from_eep(dataframe):
    """
    Retrieve Gaia photometry for a set of stars given their EEPs, ages, and metallicities
    using the MIST isochrone model.

    Parameters:
    - dataframe: pandas DataFrame, output of `generate_uniform_eep`, containing the following columns:
        - 'mass': Stellar mass
        - 'radius': Stellar radius
        - 'Teff': Effective temperature
        - 'age': Stellar age in log10(years)
        - 'logL': Logarithm of stellar luminosity
        - 'logg': Logarithm of surface gravity
        - 'feh': Metallicity [Fe/H]
        - 'eep': Equivalent evolutionary phase

    Returns:
    - pandas DataFrame with additional columns for Gaia photometry: 'G', 'BP', 'RP'.
    """
    # Load the MIST isochrone model
    #mist = get_ichrone('mist')
    mist_track = MIST_EvolutionTrack()
    #mist = MIST_Isochrone()


    # Prepare a list to store results
    results = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        try:
            # Extract the necessary parameters
            eep, age, feh, mass = row['stage'], row['tage'], row['met'], row['m']
            dist, Av = row['dist'], row['Av']

            # change distance from kpc to pc for mist_track
            dist = dist * 1000

            # Interpolate Gaia magnitudes
            mist_track_star = mist_track(mass, eep, feh, distance = dist, AV= Av)
            gmag, bpmag, rpmag = mist_track_star['G_mag'], mist_track_star['BP_mag'], mist_track_star['RP_mag']
            #teff, logg, feh, magnitudes = mist.interp_mag([eep, age, feh], ['G', 'BP', 'RP'])
            #teff, logg, feh, magnitudes = mist_track.interp_mag([mass, eep, feh], ['G', 'BP', 'RP'])
            # Add magnitudes to the result
            result = row.to_dict()
            # result['G'] = magnitudes[0]
            # result['BP'] = magnitudes[1]
            # result['RP'] = magnitudes[2]
            result['Gaia_G_M'] = gmag[0]
            result['Gaia_BP_M'] = bpmag[0]
            result['Gaia_RP_M'] = rpmag[0]

            results.append(result)
        except Exception as e:
            print(f"Error for row: {row}: {e}")
            # Add NaNs for failed interpolations
            result = row.to_dict()
            result.update({'G': np.nan, 'BP': np.nan, 'RP': np.nan})
            results.append(result)

    return pd.DataFrame(results)


if __name__ == '__main__':

    # load the speedystar catalog
    speedystar_catalog = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/from_MIST/test_eep_propagated_phot_1e5.fits').to_pandas()

    # Generate synthetic photometry for the population
    photometry = get_star_photometry_from_eep(speedystar_catalog)

    # Save the results
    Table.from_pandas(photometry).write('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/from_MIST/test_eep_propagated_phot_MIST_1e5.fits', overwrite=True)