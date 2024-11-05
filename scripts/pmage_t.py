import numpy as np
import astropy.units as u
from numba import njit
from astropy.table import Table


@njit
def loop(r_ages, lookback, flight_times, age_arr, CDF_IMF, mass_arr, SFR_arr):
    age_samples = np.empty(len(flight_times), dtype=np.float64)
    mass_samples = np.empty(len(flight_times), dtype=np.float64)

    for i, flight_time in enumerate(flight_times):
        idx = np.argmin(np.absolute(lookback-flight_time))
        age_samples[i] = np.interp(r_ages[i,0], SFR_arr[idx,:], age_arr)

        idx_mass = np.argmin(np.absolute(age_arr-age_samples[i]))
        mass_samples[i] = np.interp(r_ages[i,1], CDF_IMF[idx_mass], mass_arr)
    return mass_samples, age_samples


def sample_MF_PDMF2(n_samp, Mmin=0.1, kappa = 4.1, NSC = True):
        from isochrones.mist import MISTEvolutionTrackGrid
        from scipy.interpolate import interp1d
        from scipy.integrate import simpson
        from scipy.integrate import cumulative_simpson
        track_grid = MISTEvolutionTrackGrid()

        #Find the maximum age for each initial mass from MIST isochrones
        max_ages = []
        masses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for mass in masses:
            mass = round(mass, 2)
            rel = track_grid.df.xs((0.0, mass), level=(0, 1))
            records = rel.to_records(index=False)
            rel = np.array(records, dtype = records.dtype.descr)
            max_ages.append(np.max(rel['age']))
        max_ages = np.array(max_ages)
        #Make an interpolation to determine the maximum mass for a given age (log10(yr))
        interpfunc = interp1d(max_ages, masses, kind='cubic', bounds_error=False, fill_value=(100, np.nan)) #log of max age

        if NSC:

            # # PDMF NSC
            # table 3 from https://www.aanda.org/articles/aa/pdf/2020/09/aa36688-19.pdf column 2Z without young, massive stars (CWD)
            times = np.flip([13, 11, 9, 7, 6, 5, 4, 3, 2, 1, 0.8, 0.5, 0.25, 0.15, 0.1, 0.08, 0.03, 0]) #Gyr
            fracs = np.flip([0.748, 0, 0, 0, 0.084, 0, 0, 0.116, 0, 0.002, 0.003, 0.039, 0.001, 0, 0.002, 0.002, 0.002]) #fraction of total initially formed stellar mass
            rate = fracs/np.diff(times)
            rate = np.append(rate[0], rate)
            #Function that returns the star formation rate as a function of time [Msun/yr]
            SFR = interp1d(times, rate, kind='next', bounds_error=False, fill_value=0)
        else:

            # PDMF CWD
            burst_time = 10*10**(-3) # Lookback time for the star formation burst (Gyr)
            burst_duration = 1*10**(-3) # Duration of the star formation burst (Gyr)
            times = np.linspace(0, 20, 100)*10**(-3) # Gyr
            rate = np.exp(-0.5*(times-burst_time)**2/burst_duration**2)
            SFR = interp1d(times, rate, kind='linear', bounds_error=False, fill_value=0)

        # age_arr gives the grid over the age of the stars that are drawn, lookback over the flight time of the sampled stars
        resolution = 1000
        age_arr = np.append(0, np.logspace(-3, np.log10(max(times)), resolution))
        lookback = np.append(10**-6, np.logspace(-3, np.log10(max(times)), 1000))[:-1] #lookback time [Gyr], remove last point to avoid issue with interpolation
        mass_arr = np.logspace(np.log10(Mmin), 2, 100)

        # calculate the CDF of the truncated IMF (function of age) and the C(age) fraction of stars alive
        mass_IMF = mass_arr**(-kappa)
        CDF_IMF = np.zeros((len(age_arr), len(mass_arr)))
        Cage = []
        norm_fact = 1/(-kappa*(100**(-kappa+1) - Mmin**(-kappa+1)))
        for i, t in enumerate(np.log10(age_arr*10**9)):
            max_mass = interpfunc(t)
            mass_IMF_ = mass_IMF.copy()
            mass_IMF_[mass_arr>max_mass] = 0
            CDF_IMF[i, :] = cumulative_simpson(mass_IMF_, x=mass_arr, initial=0)
            CDF_IMF[i, :] /= CDF_IMF[i, -1]
            Cage.append(-kappa*(interpfunc(t)**(-kappa+1) - Mmin**(-kappa+1))*norm_fact)
        Cage = np.array(Cage)

        # CDF of the SFR for each lookback time
        SFR_arr = np.zeros((len(lookback), len(age_arr)))
        rates = np.zeros(len(lookback))
        for i, time in enumerate(lookback):
            SFR_arr[i,:] = np.cumsum(SFR(age_arr+time)*Cage*np.diff(np.append([0], age_arr)))
            SFR_arr[i,:] /= np.max(SFR_arr[i,:])
            rates[i] = simpson(SFR(age_arr+time)*Cage, x=age_arr)

        rate_CDF = cumulative_simpson(rates, x=lookback, initial=0)
        rate_CDF /= rate_CDF[-1]

        flight_times = np.interp(np.random.rand(n_samp), rate_CDF, lookback)

        # Finite difference to get the current ejection rate
        present_rate = rate_CDF[1]*n_samp/(lookback[1]-lookback[0])*1/u.Gyr
        print ('current rate = ', present_rate.to(u.yr**-1))

        # random sample for the age and mass with inverse transform sampling
        r_ages = np.random.rand(len(flight_times),2)
        mass_samples, age_samples = loop(r_ages, lookback, flight_times, age_arr, CDF_IMF, mass_arr, SFR_arr)
        return np.array(mass_samples).flatten()*u.Msun, np.array(age_samples).flatten()*u.Gyr, flight_times*u.Gyr, present_rate

if __name__ == '__main__':
    sample = sample_MF_PDMF2( 10000, Mmin=0.8, NSC= True)
    mass = sample[0]
    age = sample[1]
    flight_times = sample[2]
    rate = sample[3]
    output_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/SFH_sampling_catalogs/initial'
    
    # Save the output
    table_out = Table([mass, age, flight_times], names=['mass', 'age', 'flight_time'], units=[u.Msun, u.Gyr, u.Gyr])
    table_out.write(output_path + f'/sample_NSC.fits', overwrite=True)