from isochrones import get_ichrone
import numpy as np
# imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

# install and import isochrones
install_and_import('isochrones')
install_and_import('tqdm')

from astropy.table import Table
from isochrones.mist import MISTEvolutionTrackGrid, MISTIsochroneGrid, MIST_EvolutionTrack

from isochrones import get_ichrone
import numpy as np
from astropy.table import Table, hstack
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from isochrones import get_ichrone
import numpy as np
from astropy.table import Table

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


if __name__ == "__main__":
    # test on catalog
    initial_catalog = '/home/cavierescarreramc/data1/simulated_catalogs/initial'
    pmage_catalog = Table.read(os.path.join(initial_catalog, 'sample_NSC.fits')
    feh = [0]*len(pmage_catalog)
    initial_mass = pmage_catalog['mass']
    age_ejection = pmage_catalog['age']
    flight_time = pmage_catalog['flight_time']


    pmage_catalog_with_photometry = hstack([pmage_catalog, get_star_photometry(feh, initial_mass, age_ejection, flight_time)])


    # Save the output
    pmage_catalog_with_photometry.write('/home/cavierescarreramc/data1/simulated_catalogs/photometry/sample_NSC_with_photometry.fits', overwrite=True)
