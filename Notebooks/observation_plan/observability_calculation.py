import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
from astroplan import Observer
from datetime import datetime
import concurrent.futures
from functools import partial
from tqdm import tqdm
from astropy.table import Table

def _compute_best_for_star(ra, dec, location, middle_night_data):
    """
    Compute the best (lowest) airmass and the corresponding night-month for a given star.
    
    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    location : EarthLocation
        Observer's location.
    middle_night_data : list of tuple
        List of tuples (middle_night, night_month) where middle_night is the computed
        midpoint between sunset and sunrise, and night_month is the month (from the sunset time).
    
    Returns
    -------
    tuple(bool, str)
        A tuple containing:
          - observable: True if the star's best airmass is below 1.5.
          - best_month: The month (as a string, e.g. 'October') corresponding to the night (sunset)
                        when the star reaches its lowest airmass, or an empty string if the star never rises.
    """
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    best_airmass = np.inf
    best_month = ""
    for mid, night_month in middle_night_data:
        altaz = coord.transform_to(AltAz(obstime=mid, location=location))
        # Only consider times when the source is above the horizon.
        if altaz.alt < 0*u.deg:
            continue
        airmass = altaz.secz
        if np.isfinite(airmass) and airmass < best_airmass:
            best_airmass = airmass
            best_month = night_month
    observable = best_airmass < 1.5
    return (observable, best_month)

def check_observability_midnight(astropy_table, start_date, end_date, location=None):
    """
    Check which sources in an Astropy table are observable with airmass < 1.5
    at the middle of the night (where "middle of the night" is defined as the
    midpoint between sunset and sunrise). For each source, return also the month
    (taken from the sunset time) on which it is best observed.

    This version uses multiprocessing (with a progress bar) so that each star is
    processed in parallel.

    Parameters
    ----------
    astropy_table : astropy.table.Table
        An Astropy Table containing at least the columns 'ra', 'dec', and 'name'.
    start_date : str
        Start date in 'dd-mm-yyyy' format (e.g. '01-10-2025').
    end_date : str
        End date in 'dd-mm-yyyy' format (e.g. '30-04-2026').
    location : astropy.coordinates.EarthLocation, optional
        Observer's location. If None, defaults to Mauna Kea, Hawaii.

    Returns
    -------
    astropy.table.Table
        The input table with two additional columns:
          - 'observable': True if the source reaches an airmass < 1.5 on at least one night.
          - 'best_month': The month (from the sunset time) when the source is best observed,
                          or an empty string if the source never rises above the horizon at any computed time.
    """
    # Default location: Mauna Kea, Hawaii.
    if location is None:
        location = EarthLocation(lat=19.8207*u.deg, lon=-155.4681*u.deg, height=4205*u.m)
        timezone = 'US/Hawaii'
    else:
        timezone = 'UTC'
    
    observer = Observer(location=location, timezone=timezone)
    
    # Parse the start and end dates using datetime.
    start_dt = datetime.strptime(start_date, '%d-%m-%Y')
    end_dt = datetime.strptime(end_date, '%d-%m-%Y')
    t_start = Time(start_dt)
    t_end = Time(end_dt)
    
    # Create a list of days in the date range.
    n_days = int((t_end - t_start).to(u.day).value) + 1
    days = t_start + np.arange(n_days)*u.day

    # For each day, compute the middle-of-night time (midpoint between sunset and sunrise)
    # and record the month of the sunset (i.e. the night’s "start" month).
    middle_night_data = []
    for day in days:
        try:
            sunset = observer.sun_set_time(day, which='next')
            sunrise = observer.sun_rise_time(sunset, which='next')
            middle_night = sunset + (sunrise - sunset) / 2
            # Use the sunset's month as the identifier for the night.
            night_month = sunset.datetime.strftime('%B')
            middle_night_data.append((middle_night, night_month))
        except Exception:
            # Skip days where sunset/sunrise cannot be computed.
            continue

    # Prepare lists of RA and Dec for each source.
    ra_list = [row['ra'] for row in astropy_table]
    dec_list = [row['dec'] for row in astropy_table]
    
    # Use multiprocessing to compute observability for each star.
    func = partial(_compute_best_for_star, location=location, middle_night_data=middle_night_data)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, ra_list, dec_list), total=len(ra_list), desc="Processing stars"))
    
    # Unpack the results and add them as new columns to the table.
    observable_list, best_month_list = zip(*results) if results else ([], [])
    
    # Ensure best_month_list is a homogeneous list of strings.
    best_month_list = [month if month is not None else "" for month in best_month_list]
    
    astropy_table['observable'] = observable_list
    astropy_table['best_month'] = best_month_list
    
    return astropy_table




# Example usage:
if __name__ == '__main__':
    # Create a dummy astropy Table with sample sources.
    table = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/candidates/high_likelihood_candidates.dat', format='ascii')
    table.rename_column('col1', 'name')
    table.rename_column('col2', 'ra')
    table.rename_column('col3', 'dec')
    
    # Define the date range.
    start_date = '01-10-2025'
    end_date = '28-04-2026'

    # Define the location to paranal for the current ESO call
    location_paranal = EarthLocation(lat=-24.627222222222*u.deg, lon=-70.404166666667*u.deg, height=2635*u.m)
    
    # Run the observability check.
    result_table = check_observability_midnight(table, start_date, end_date, location_paranal)
    
    # Print the resulting table.
    print(result_table)

    # save the table to a file
    result_table.write('/Users/mncavieres/Documents/2024-2/HVS/Data/candidates/observability_results.fits', overwrite=True)