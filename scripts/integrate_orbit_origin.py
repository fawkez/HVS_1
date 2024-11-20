import os
import numpy as np
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from galpy.orbit import Orbit
from galpy.potential import McMillan17
import matplotlib.pyplot as plt

def integrate_orbits(data, output_path, n_samples=100, integration_time=-1*u.Gyr, n_steps=1000):
    """
    Integrate orbits of stars back in time, save orbit data, and estimate posterior distribution
    for their origin positions.
    
    Parameters:
    - data: ndarray
      Structured array with columns ['RA', 'DEC', 'pmra', 'pmdec', 'd_helio', 'v_helio', 
                                     'RA_error', 'DEC_error', 'pmra_error', 'pmdec_error', 
                                     'd_helio_error', 'v_helio_error'].
    - output_path: str
      Path to save orbit files for each star.
    - n_samples: int
      Number of Monte Carlo samples for error propagation.
    - integration_time: Quantity
      Integration time into the past (negative for backward integration).
    - n_steps: int
      Number of time steps for orbit integration.
      
    Returns:
    - origins: list of arrays
      List containing posterior positions at origin for each star.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Initialize McMillan 2017 potential
    pot = McMillan17()

    origins = []
    time_grid = np.linspace(0, integration_time.value, n_steps) * integration_time.unit
    
    for idx, star in enumerate(data):
        # Extract data and errors
        ra, dec = star['RA'], star['DEC']
        pmra, pmdec = star['pmra'], star['pmdec']
        d_helio, v_helio = star['d_helio'], star['v_helio']
        ra_err, dec_err = star['RA_error'], star['DEC_error']
        pmra_err, pmdec_err = star['pmra_error'], star['pmdec_error']
        d_helio_err, v_helio_err = star['d_helio_error'], star['v_helio_error']
        
        # Generate Monte Carlo samples
        ra_samples = np.random.normal(ra, ra_err, n_samples)
        dec_samples = np.random.normal(dec, dec_err, n_samples)
        pmra_samples = np.random.normal(pmra, pmra_err, n_samples)
        pmdec_samples = np.random.normal(pmdec, pmdec_err, n_samples)
        d_samples = np.random.normal(d_helio, d_helio_err, n_samples)
        v_samples = np.random.normal(v_helio, v_helio_err, n_samples)
        
        star_origins = []
        for i in range(n_samples):
            # Transform to Galactocentric frame
            c = SkyCoord(ra=ra_samples[i]*u.deg, dec=dec_samples[i]*u.deg, 
                         distance=d_samples[i]*u.kpc, pm_ra_cosdec=pmra_samples[i]*u.mas/u.yr, 
                         pm_dec=pmdec_samples[i]*u.mas/u.yr, radial_velocity=v_samples[i]*u.km/u.s)
            galactic = c.transform_to(Galactocentric())
            
            # Initialize orbit
            vx = galactic.v_x.to_value(u.km/u.s)
            vy = galactic.v_y.to_value(u.km/u.s)
            vz = galactic.v_z.to_value(u.km/u.s)
            o = Orbit(vxvv=[galactic.x.to_value(u.kpc), galactic.y.to_value(u.kpc), galactic.z.to_value(u.kpc), 
                            vx, vy, vz], ro=8.2, vo=220., solarmotion='schoenrich')
            
            # Integrate orbit
            o.integrate(time_grid, pot)
            
            # Record origin position
            star_origins.append([o.x(time_grid[-1]), o.y(time_grid[-1]), o.z(time_grid[-1])])
            
            # Save orbit to file
            orbit_data = np.array([o.x(time_grid), o.y(time_grid), o.z(time_grid),
                                   o.vx(time_grid), o.vy(time_grid), o.vz(time_grid)]).T
            filename = os.path.join(output_path, f"star_{idx}_sample_{i}.txt")
            np.savetxt(filename, orbit_data, 
                       header="X [kpc]    Y [kpc]    Z [kpc]    VX [km/s]    VY [km/s]    VZ [km/s]")
        
        origins.append(np.array(star_origins))
    
    return origins

def plot_origins(origins):
    """
    Plot posterior distributions of star origins.
    
    Parameters:
    - origins: list of arrays
      List of posterior positions for each star.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    for origin in origins:
        x, y, z = origin[:, 0], origin[:, 1], origin[:, 2]
        ax.scatter(x, y, alpha=0.3, label='Posterior samples', s=10)
    
    ax.scatter(0, 0, color='red', label='Galactic Center', s=100)
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()


