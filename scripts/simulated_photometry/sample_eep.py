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





import pandas as pd

def generate_uniform_eep(n_sample, mass_range=(0.3, 10.0),
    metallicity=0.25, metallicity_range=None, eep_range=(202, 808)):
    """
    Generate synthetic photometry for n_sample stars using MIST evolutionary tracks,
    with mass, EEP, and metallicity as inputs.

    Parameters:
    - n_sample: int, number of stars to sample.
    - metallicity: float, fixed metallicity [Fe/H]. If provided, overrides metallicity_range.
    - metallicity_range: tuple, range of [Fe/H] values to sample from (min, max).
    - eep_range: tuple, range of EEP values to sample from (min, max). Defaults to full MIST EEP range.

    Returns:
    - pandas DataFrame with columns ['mass', 'age', 'G', 'BP', 'RP', 'Teff', 'logg',
      'radius', 'logL', 'feh', 'eep'].
    """
    # Validate inputs
    if metallicity is not None and metallicity_range is not None:
        raise ValueError("Specify either metallicity or metallicity_range, not both.")
    if metallicity_range is not None:
        if not isinstance(metallicity_range, tuple) or len(metallicity_range) != 2:
            raise ValueError("metallicity_range must be a tuple of (min, max).")

    # Load the MIST evolutionary track
    mist_track = MIST_EvolutionTrack()
    mist = MIST_Isochrone()

    # Sample stellar parameters
    initial_masses = np.random.uniform(mass_range[0], mass_range[1], n_sample)  # Masses in solar masses
    eeps = np.random.uniform(eep_range[0], eep_range[1], n_sample)  # EEPs
    # generate integer eeps in the range
    eeps = np.round(eeps).astype(int)

    if metallicity is not None:
        fehs = np.full(n_sample, metallicity)
    else:
        fehs = np.random.uniform(metallicity_range[0], metallicity_range[1], n_sample)

    # Collect results
    results = []

    for mass, eep, feh in tqdm(zip(initial_masses, eeps, fehs), total=n_sample):
        try:
            # Interpolate stellar properties based on mass, EEP, and Fe/H
            params = [mass, eep,  feh]
            values = mist_track.interp_value(params, ['mass', 'radius', 'Teff', 'age', 'logL', 'logg', 'initial_mass'])#, 'G', 'BP', 'RP'])
            values = values[0]
            result = {
                'mass': values[0],
                'radius': values[1],
                'Teff': values[2],
                'age': values[3],  # Age is interpolated from the track
                'logL': values[4],
                'logg': values[5],
                'initial_mass': values[6],
                # 'G': values[6],
                # 'BP': values[7],
                # 'RP': values[8],
                'feh': feh,
                'eep': eep
            }
        except Exception as e:
            print(f"Error for star with mass={mass}, eep={eep}, feh={feh}: {e}")
            # Fill missing data with NaNs
            result = {key: np.nan for key in ['mass', 'radius', 'Teff', 'age', 'logL', 'logg', 'feh', 'eep']}
            result.update({'mass': mass, 'eep': eep, 'feh': feh})
        results.append(result)

    return pd.DataFrame(results)



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
            eep, age, feh, mass = row['eep'], row['age'], row['feh'], row['mass']

            # Interpolate Gaia magnitudes
            mist_track_star = mist_track(mass, eep, feh)
            gmag, bpmag, rpmag = mist_track_star['G_mag'], mist_track_star['BP_mag'], mist_track_star['RP_mag']
            #teff, logg, feh, magnitudes = mist.interp_mag([eep, age, feh], ['G', 'BP', 'RP'])
            #teff, logg, feh, magnitudes = mist_track.interp_mag([mass, eep, feh], ['G', 'BP', 'RP'])
            # Add magnitudes to the result
            result = row.to_dict()
            # result['G'] = magnitudes[0]
            # result['BP'] = magnitudes[1]
            # result['RP'] = magnitudes[2]
            result['G'] = gmag[0]
            result['BP'] = bpmag[0]
            result['RP'] = rpmag[0]

            results.append(result)
        except Exception as e:
            print(f"Error for row: {row}: {e}")
            # Add NaNs for failed interpolations
            result = row.to_dict()
            result.update({'G': np.nan, 'BP': np.nan, 'RP': np.nan})
            results.append(result)

    return pd.DataFrame(results)



if __name__ == '__main__':


    # Generate population in uniform EEP for 10 stars with random parameters
    uniform_population = generate_uniform_eep(1000000)

    # # Generate synthetic photometry for the population
    photometry = get_star_photometry_from_eep(uniform_population)

    # save the table to a fits file
    photometry = Table.from_pandas(photometry)
    photometry.write('/Users/mncavieres/Documents/2024-2/HVS/Data/importance_sampling/uniform_eep.fits', overwrite=True)

