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


# new code

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
from astroplan import Observer
from datetime import datetime
import concurrent.futures
from functools import partial
from tqdm import tqdm
from astropy.table import Table, Column
from collections import Counter, defaultdict

# -------------------------------------------------------------
# Helper: robust date parser for 'dd-mm-yyyy' or ISO 'yyyy-mm-dd'
# -------------------------------------------------------------
def _parse_date(dt_str: str) -> Time:
    for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
        try:
            return Time(datetime.strptime(dt_str, fmt))
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {dt_str!r}. Use 'dd-mm-yyyy' or 'yyyy-mm-dd'.")

# -------------------------------------------------------------
# Build astronomical nights within [start, end]
# Each item: dict with keys {'t_start', 't_end', 'times', 'month', 'cadence'}
# -------------------------------------------------------------

def _build_astronomical_nights(start: Time,
                               end: Time,
                               observer: Observer,
                               time_resolution: u.Quantity = 5*u.min):
    """Return list of astronomical nights between start and end (inclusive).

    Night is defined from evening astronomical twilight (Sun alt = -18 deg)
    to next morning astronomical twilight. Nights that cannot be computed
    (polar day/night) are skipped.
    """
    # Generate UTC midnights across the range to seed twilight searches
    n_days = int(np.floor(((end - start).to(u.day)).value)) + 1
    days = start + np.arange(n_days)*u.day

    nights = []
    for day in days:
        try:
            # Evening astronomical twilight after 'day'
            eve_tw = observer.twilight_evening_astronomical(day, which='next')
            # Morning astronomical twilight after the evening twilight
            morn_tw = observer.twilight_morning_astronomical(eve_tw, which='next')
        except Exception:
            # Skip dates where twilight cannot be computed
            continue

        # Guard against partial overlap with requested window
        if not (morn_tw < start or eve_tw > end):
            t0 = max(eve_tw, start)
            t1 = min(morn_tw, end)
            if t1 <= t0:
                continue
            # Build uniform time grid
            n_steps = int(np.ceil(((t1 - t0)/time_resolution).to(1).value)) + 1
            times = t0 + np.arange(n_steps)*time_resolution
            month = eve_tw.datetime.strftime('%B')  # month from the night's start (evening twilight)
            cadence_hours = time_resolution.to(u.hour).value
            nights.append({
                't_start': t0,
                't_end': t1,
                'times': times,
                'month': month,
                'cadence': cadence_hours,
            })
    return nights

# -------------------------------------------------------------
# Per-star, per-night metrics inside astronomical night only
# -------------------------------------------------------------

def _metrics_for_star_single_night(ra_deg: float,
                                   dec_deg: float,
                                   night: dict,
                                   location: EarthLocation,
                                   airmass_limit: float) -> dict:
    """Compute nightly metrics for a single star within a single night.

    Returns dict with keys:
      - 'min_airmass', 'median_airmass', 'hours_leq_limit', 'month'
      NaNs if the star never rises (no finite airmass values) within the night.
    """
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    altaz_frame = AltAz(obstime=night['times'], location=location)
    altaz = coord.transform_to(altaz_frame)

    # Valid when above horizon and finite airmass
    finite = np.isfinite(altaz.secz.value)
    above_horizon = altaz.alt > 0*u.deg
    ok = finite & above_horizon

    if not np.any(ok):
        return {
            'min_airmass': np.nan,
            'median_airmass': np.nan,
            'hours_leq_limit': 0.0,
            'month': night['month'],
        }

    airm = altaz.secz.value[ok]
    min_airm = np.nanmin(airm)
    med_airm = np.nanmedian(airm)

    # Hours with airmass <= limit (approximate via cadence)
    leq = airm <= airmass_limit
    hours_good = float(np.count_nonzero(leq) * night['cadence'])

    return {
        'min_airmass': float(min_airm),
        'median_airmass': float(med_airm),
        'hours_leq_limit': hours_good,
        'month': night['month'],
    }


def _compute_star_over_range(ra_deg: float,
                             dec_deg: float,
                             nights: list,
                             location: EarthLocation,
                             airmass_limit: float = 1.5) -> dict:
    """Compute per-night metrics across the range and pick the best night.

    Best night is defined as the night with the lowest *minimum* airmass.
    Returns dict with keys used to populate the output table for this star:
      - 'best_month'
      - 'best_night_min_airmass'
      - 'best_night_median_airmass'
      - 'best_night_hours_leq_{airmass_limit}'

    Also returns 'best_night_month_for_counting' to help aggregate a program-wide
    optimal month as the month with the most stars' best nights.
    """
    best = {
        'best_month': '',
        'best_night_min_airmass': np.nan,
        'best_night_median_airmass': np.nan,
        'best_night_hours_leq_limit': 0.0,
    }

    best_min = np.inf
    best_month = ''
    best_med = np.nan
    best_hours = 0.0

    for night in nights:
        m = _metrics_for_star_single_night(ra_deg, dec_deg, night, location, airmass_limit)
        min_airm = m['min_airmass']
        if np.isfinite(min_airm) and min_airm < best_min:
            best_min = min_airm
            best_month = m['month']
            best_med = m['median_airmass']
            best_hours = m['hours_leq_limit']

    if np.isfinite(best_min):
        best['best_month'] = best_month
        best['best_night_min_airmass'] = float(best_min)
        best['best_night_median_airmass'] = float(best_med)
        best['best_night_hours_leq_limit'] = float(best_hours)
    # else: remains NaNs/empty if never above horizon at astro night

    return best

# -------------------------------------------------------------
# Public API
# -------------------------------------------------------------

def plan_observations(
    astropy_table: Table,
    start_date: str,
    end_date: str,
    location: EarthLocation = None,
    timezone: str = 'UTC',
    time_resolution: u.Quantity = 5*u.min,
    airmass_limit: float = 1.5,
    n_processes: int | None = None,
):
    """
    Determine an *overall optimal month* for a program and add per-star nightly metrics.

    Parameters
    ----------
    astropy_table : astropy.table.Table
        Must contain columns 'ra' and 'dec' (degrees). Other columns are preserved.
    start_date, end_date : str
        Date strings in 'dd-mm-yyyy' or ISO 'yyyy-mm-dd'. Inclusive window.
    location : astropy.coordinates.EarthLocation, optional
        Observatory location. Defaults to Mauna Kea if None.
    timezone : str, optional
        Timezone string for the Observer. Affects month labeling of nights.
        Defaults to 'UTC'.
    time_resolution : Quantity, optional
        Sampling cadence within each astronomical night. Default 5 minutes.
    airmass_limit : float, optional
        Threshold used when counting "good" hours (airmass <= limit). Default 1.5.
    n_processes : int or None
        Number of worker processes. Default uses ProcessPoolExecutor's default.

    Returns
    -------
    overall_optimal_month : str
        The month (by name) in which the largest number of stars achieve their *best night*
        (lowest nightly minimum airmass) within the given date range.
    table_out : astropy.table.Table
        Copy of the input table with added columns:
          - 'best_month'
          - 'best_night_min_airmass'
          - 'best_night_median_airmass'
          - f"best_night_hours_leq_{airmass_limit:.1f}"
        Metrics are computed *only* during astronomical night (Sun altitude < -18°).
    """
    # Defaults
    if location is None:
        location = EarthLocation(lat=19.8207*u.deg, lon=-155.4681*u.deg, height=4205*u.m)  # Mauna Kea
        if timezone == 'UTC':
            timezone = 'US/Hawaii'

    observer = Observer(location=location, timezone=timezone)

    t_start = _parse_date(start_date)
    t_end = _parse_date(end_date)

    # Build all astronomical nights in the range
    nights = _build_astronomical_nights(t_start, t_end, observer, time_resolution=time_resolution)
    if len(nights) == 0:
        raise RuntimeError("No astronomical nights found in the requested range at this location.")

    # Prepare for parallel per-star computation
    ra_list = np.array(astropy_table['ra'], dtype=float)
    dec_list = np.array(astropy_table['dec'], dtype=float)

    worker = partial(_compute_star_over_range,
                     nights=nights,
                     location=location,
                     airmass_limit=airmass_limit)

    if len(ra_list) == 0:
        # Return early if table empty
        table_out = astropy_table.copy()
        hours_colname = f"best_night_hours_leq_{airmass_limit:.1f}"
        for name, data in [
            ('best_month', []),
            ('best_night_min_airmass', []),
            ('best_night_median_airmass', []),
            (hours_colname, []),
        ]:
            table_out[name] = data
        return '', table_out

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as ex:
        results = list(tqdm(ex.map(worker, ra_list, dec_list), total=len(ra_list), desc="Computing per-star metrics"))

    # Unpack and attach to table
    best_months = []
    best_min = []
    best_med = []
    best_hours = []

    for r in results:
        best_months.append(r['best_month'])
        best_min.append(r['best_night_min_airmass'])
        best_med.append(r['best_night_median_airmass'])
        best_hours.append(r['best_night_hours_leq_limit'])

    table_out = astropy_table.copy()
    hours_colname = f"best_night_hours_leq_{airmass_limit:.1f}"
    table_out['best_month'] = best_months
    table_out['best_night_min_airmass'] = best_min
    table_out['best_night_median_airmass'] = best_med
    table_out[hours_colname] = best_hours

    # Program-wide optimal month: month containing the *best night* for the most stars
    month_counter = Counter([m for m in best_months if isinstance(m, str) and len(m)])
    overall_optimal_month = month_counter.most_common(1)[0][0] if month_counter else ''

    return overall_optimal_month, table_out


# -------------------------------------------------------------
# Example usage (commented)
# -------------------------------------------------------------
# if __name__ == '__main__':
#     from astropy.table import Table
#     # Example: load your table with columns 'ra' (deg), 'dec' (deg)
#     # table = Table.read('your_sources.fits')
#     # location_paranal = EarthLocation(lat=-24.627222222222*u.deg,
#     #                                  lon=-70.404166666667*u.deg,
#     #                                  height=2635*u.m)
#     # opt_month, out = plan_observations(table,
#     #                                    start_date='01-11-2025',
#     #                                    end_date='30-11-2025',
#     #                                    location=location_paranal,
#     #                                    timezone='Chile/Continental',
#     #                                    time_resolution=5*u.min,
#     #                                    airmass_limit=1.5)
#     # print('Overall optimal month:', opt_month)
#     # out.write('observability_results.fits', overwrite=True)


# Example usage:
if __name__ == '__main__':
    # Create a dummy astropy Table with sample sources.
    #table = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/candidates/high_likelihood_candidates.dat', format='ascii')
    #table.rename_column('col1', 'name')
    #table.rename_column('col2', 'ra')
    #table.rename_column('col3', 'dec')
    table = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/candidates/november/candidates_HL_purple_disk_south_nov.fits')
    
    # Define the date range.
    start_date = '01-05-2026'
    end_date = '30-04-2027'

    # Define the location to paranal for the current ESO call
    #location_paranal = EarthLocation(lat=-24.627222222222*u.deg, lon=-70.404166666667*u.deg, height=2635*u.m)
    location_la_silla = EarthLocation.of_site('lasilla')

    # Run the observability check.
    #result_table = check_observability_midnight(table, start_date, end_date, location_la_silla)

    # Alternatively, use the more detailed planning function:
    overall_optimal_month, table_out = plan_observations(table,
                                                            start_date=start_date,
                                                            end_date=end_date,
                                                            location=location_la_silla,
                                                            airmass_limit=1.8,
                                                            time_resolution=10*u.min)
    print('Overall optimal month for the program:', overall_optimal_month)
    result_table = table_out
    
    # Print the resulting table.
    print(result_table)

    # save the table to a file
    result_table.write('/Users/mncavieres/Documents/2024-2/HVS/Data/p117/OBS_candidates_HL_blue.fits', overwrite=True)