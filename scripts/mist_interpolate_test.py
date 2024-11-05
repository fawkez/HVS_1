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


def generate_star_properties( feh, mass, age):
    """Generate star properties with error handling."""
    try:
        mist = get_ichrone('mist')
        star_model = mist.generate(mass=mass, age=age, feh=feh, accurate=True)
        return {
            'G': star_model.get('G_mag', np.nan),
            'BP': star_model.get('BP_mag', np.nan),
            'RP': star_model.get('RP_mag', np.nan),
            'Teff': star_model.get('Teff', np.nan),
            'logg': star_model.get('logg', np.nan),
            'radius': star_model.get('radius', np.nan),
            'logL': star_model.get('logL', np.nan),
            'feh': star_model.get('feh', np.nan),
            'eep': star_model.get('eep', np.nan)
        }
    
        print('Interpolation succesful')
    except (ValueError, RuntimeError):
        # Reset the mist model on failure and return NaNs
        mist = get_ichrone('mist')
        return {
            'G': np.nan, 'BP': np.nan, 'RP': np.nan, 'Teff': np.nan,
            'logg': np.nan, 'radius': np.nan, 'logL': np.nan,
            'feh': np.nan, 'eep': np.nan
        }

def get_star_photometry(feh, initial_mass, age_ejection, flight_time):
    """
    Retrieve Gaia photometry for a batch of stars given their initial masses,
    metallicities (feh), and ages using the MIST isochrone model.
    """
    # Initialize the MIST isochrone model once at the start

    ages = age_ejection + flight_time

    # Prepare to store all results
    results = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for feh_val, mass, age in zip(feh, initial_mass, ages):
            # Submit each star's property generation as a parallel task
            futures.append(executor.submit(generate_star_properties, feh_val, mass, age))
        
        # Gather results as tasks complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    
    # Convert list of results to an Astropy Table for easy handling
    return Table(rows=results)



def get_star_photometry(feh: np.array, initial_mass: np.array, age_ejection: np.array, flight_time: np.array):
    """
    Retrieve Gaia photometry for a star given its initial mass, metallicity (feh),
    and age using the MIST isochrone model.

    Parameters:
    - feh: float, the metallicity [Fe/H] of the star.
    - initial_mass: float, the initial mass of the star in solar masses.
    - age: float, the current age of the star in Gyr.

    Returns:
    - dict with Gaia magnitudes 'G', 'BP', and 'RP' if available.
    """


    age = age_ejection + flight_time
    mist = get_ichrone('mist')

    # Generate the star's model with `generate`, which returns a dictionary of properties
    # Documentation Reference: https://isochrones.readthedocs.io/en/latest/usage.html#model-parameters
    for feh, initial_mass, age in tqdm(zip(feh, initial_mass, age), total=len(feh)):
        try:
            #print(f"feh: {feh}, mass: {initial_mass}, age: {age}")
                # Load the MIST isochrone model
            star_model = mist.generate(mass=initial_mass, age=age, feh=feh, accurate=True)

                # Extract Gaia photometry (G, BP, RP) from the star model
            result = {
                'G': star_model.get('G_mag', np.nan),
                'BP': star_model.get('BP_mag', np.nan),
                'RP': star_model.get('RP_mag', np.nan),
                'Teff': star_model.get('Teff', np.nan),
                'logg': star_model.get('logg', np.nan),
                'radius': star_model.get('radius', np.nan),
                'logL': star_model.get('logL', np.nan),
                'feh': star_model.get('feh', np.nan),
                'eep': star_model.get('eep', np.nan)
            }

            
        except (ValueError, RuntimeError) as e:
            mist = get_ichrone('mist')
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



    return Table(result)



from multiprocessing import Pool, set_start_method

# Set start method for multiprocessing to "fork" (helpful on MacOS)
try:
    set_start_method('fork')
except RuntimeError:
    pass  # Ignore if the context is already set

def generate_star_properties(data):
    feh, mass, age = data
    """Generate star properties with error handling."""
    mist = get_ichrone('mist')  # Initialize in each process
    try:
        star_model = mist.generate(mass=mass, age=age, feh=feh, accurate=True)
        return {
            'G': star_model.get('G_mag', np.nan),
            'BP': star_model.get('BP_mag', np.nan),
            'RP': star_model.get('RP_mag', np.nan),
            'Teff': star_model.get('Teff', np.nan),
            'logg': star_model.get('logg', np.nan),
            'radius': star_model.get('radius', np.nan),
            'logL': star_model.get('logL', np.nan),
            'feh': star_model.get('feh', np.nan),
            'eep': star_model.get('eep', np.nan)
        }
        print('Interpolation succesful')
    except (ValueError, RuntimeError):
        # Return NaNs if there's an error
        return {
            'G': np.nan, 'BP': np.nan, 'RP': np.nan, 'Teff': np.nan,
            'logg': np.nan, 'radius': np.nan, 'logL': np.nan,
            'feh': np.nan, 'eep': np.nan
        }

def get_star_photometry(feh: np.array, initial_mass: np.array, age_ejection: np.array, flight_time: np.array):
    """
    Retrieve Gaia photometry for a batch of stars given their initial masses,
    metallicities (feh), and ages using the MIST isochrone model.
    """
    # Calculate the current ages of stars
    ages = age_ejection + flight_time

    # Prepare input for each star in a list of tuples
    input_data = list(zip(feh, initial_mass, ages))

    # Use multiprocessing Pool with 8 cores
    with Pool(processes=100) as pool:
        # Use tqdm to show progress with imap_unordered
        results = list(tqdm(pool.imap(generate_star_properties, input_data), total=len(input_data)))

    # Convert list of results to an Astropy Table for easy handling
    return Table(rows=results)

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
