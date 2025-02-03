import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

def log_uniform_cdf(P, P_min, P_max, P0):
    """
    CDF of a random variable whose log10(P/P0) is uniform
    between log10(P_min/P0) and log10(P_max/P0).
    
    Parameters
    ----------
    P : array or Quantity
        Values at which to evaluate the CDF.
    P_min : Quantity
        Minimum bound of P.
    P_max : Quantity
        Maximum bound of P.
    P0 : Quantity
        Reference scale for the log, e.g. 1*u.day.
    
    Returns
    -------
    cdf : ndarray
        Values of the CDF at the input P.
    """
    # Convert everything to dimensionless by dividing by P0
    a = np.log10((P_min / P0).value)  # log10(P_min/P0)
    b = np.log10((P_max / P0).value)  # log10(P_max/P0)

    # Convert input P to dimensionless for the CDF formula
    x = np.log10((P / P0).value)      # log10(P/P0)
    
    cdf = np.zeros_like(x)
    # Region P < P_min  -> CDF = 0
    cdf[x < a] = 0
    # Region P > P_max  -> CDF = 1
    cdf[x > b] = 1
    # Region P_min <= P <= P_max
    inrange = (x >= a) & (x <= b)
    cdf[inrange] = (x[inrange] - a) / (b - a)
    
    return cdf

def sample_log_uniform(P_min, P_max, P0, size=10000, rng=None):
    """
    Draw random samples from a distribution whose log10(P/P0) is
    uniform between log10(P_min/P0) and log10(P_max/P0).
    
    Parameters
    ----------
    P_min : Quantity
        Minimum bound of P.
    P_max : Quantity
        Maximum bound of P.
    P0 : Quantity
        Reference scale for the log, e.g. 1*u.day.
    size : int
        Number of samples to generate.
    rng : np.random.Generator, optional
        Random number generator instance (for reproducibility).
    
    Returns
    -------
    samples : Quantity array
        Random samples of size `size`.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ratio (dimensionless) between bounds
    ratio = (P_max / P_min).decompose().value  
    # Uniform draws in [0,1]
    u = rng.random(size)
    # Inverse-CDF for log-uniform:
    # P(u) = P_min * ( (P_max)/(P_min) )^u
    # (The P0 cancels out in the simplified formula, but was used
    #  conceptually in the log10(P/P0) distribution.)
    
    samples = P_min * ratio**u
    return samples

def main():
    # Example usage:
    P_min = 1.0 * u.day
    P_max = 10.0 * u.day
    P0    = 1.0 * u.s  # reference unit (e.g. 1 day)

    # 1) Plot the CDF:
    pvals = np.logspace(
        np.log10(P_min.value),
        np.log10(P_max.value),
        500
    ) * P_min.unit  # keep astropy units
    
    cdf_vals = log_uniform_cdf(pvals, P_min, P_max, P0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(pvals.value, cdf_vals, label="Log-Uniform CDF")
    plt.xscale("log")
    plt.xlabel(f"P [{P_min.unit}]")
    plt.ylabel("CDF")
    plt.title("CDF of Log-Uniform Distribution")
    plt.legend()
    plt.grid(True)

    # 2) Sample from the distribution and plot the histogram:
    samples = sample_log_uniform(P_min, P_max, P0, size=10000)
    
    plt.figure(figsize=(8, 6))
    # We'll do a log-binning for better visualization
    bins = np.logspace(np.log10(P_min.value), np.log10(P_max.value), 50)
    plt.hist(samples.value, bins=bins, histtype="step", density=True, label="Sampled P")
    
    plt.xscale("log")
    plt.xlabel(f"P [{P_min.unit}]")
    plt.ylabel("Probability Density")
    plt.title("Histogram of Samples (Log Scale)")
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
