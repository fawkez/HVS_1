import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import agama
import os
from tqdm import tqdm
from astropy.coordinates import SkyCoord, Galactocentric
from multiprocessing import Pool, cpu_count
from functools import partial

agama.setUnits(mass=1, length=1, velocity=1)  # kpc, km/s, 1e10 M_sun

from scipy.integrate import dblquad

def compute_bulge_probability(kde, radius=4):
    """
    Compute the probability of a star originating within a circular region (the Galactic bulge).
    
    Parameters:
    - kde: gaussian_kde
      The kernel density estimator.
    - radius: float
      Radius of the circular region around (0, 0) in kpc.
    
    Returns:
    - probability: float
      The integrated probability within the defined radius.
    """
    # Define integration bounds and the KDE function in Cartesian coordinates
    def kde_func(y, x):
        return kde([x, y])
    
    # Integrate over the circular region
    prob, _ = dblquad(
        kde_func, -radius, radius,
        lambda x: -np.sqrt(radius**2 - x**2),  # Lower bound for y
        lambda x: np.sqrt(radius**2 - x**2)   # Upper bound for y
    )
    return prob



def plot_sigma_contours(origins, xlim=(-60, 60), ylim=(-60, 60), bw_method='scott', show= True, save = False, star_name='star', save_path = 'Plots/brown_stars/brown_potential'):
    """
    Plot the 1σ and 2σ contours of closest approach points in the X-Y plane.

    Parameters:
    - origins: list of tuples
      List of positions (X, Y, Z) at the closest approach for each sample.
    - xlim: tuple
      Limits for the X-axis (default: (-60, 60)).
    - ylim: tuple
      Limits for the Y-axis (default: (-60, 60)).
    """
    # Extract X, Y positions from origins
    x = np.array([origin[0] for origin in origins])
    y = np.array([origin[1] for origin in origins])

    # Create a 2D KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)
    xi, yi = np.linspace(xlim[0], xlim[1], 300), np.linspace(ylim[0], ylim[1], 300)
    xi, yi = np.meshgrid(xi, yi)
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    # Normalize the KDE to get cumulative probabilities
    zi_flat = zi.flatten()
    sorted_zi = np.sort(zi_flat)
    cumsum_zi = np.cumsum(sorted_zi) / np.sum(sorted_zi)

    # Determine contour levels for 1σ (68.26%) and 2σ (95.44%)
    level_1sigma = sorted_zi[np.searchsorted(cumsum_zi, 1 - 0.6826)]
    level_2sigma = sorted_zi[np.searchsorted(cumsum_zi, 1- 0.9544)]

    # Find mode of the distribution
    mode_index = np.argmax(zi_flat)
    mode_x, mode_y = xi.flatten()[mode_index], yi.flatten()[mode_index]

    # Plot the KDE with contour levels
    plt.figure(figsize=(10, 10))
    #plt.contourf(xi, yi, zi, levels=100, cmap="Blues")
    #plt.colorbar(label="Density")
    plt.contour(xi, yi, zi, levels=[level_2sigma, level_1sigma, zi.max()], colors=['k', 'k', 'k'], 
                linewidths=[2, 2, 2], linestyles=['--', '-','-'], labels=['1σ', '2σ', '0'])
    plt.contourf(xi, yi, zi, levels=[level_2sigma, level_1sigma, zi.max()], colors=['cyan', 'magenta', 'blue'], alpha=0.5)
    
    # Add annotations
    plt.scatter(mode_x, mode_y, color='black', label='Mode of Distribution', s=100, marker='x')
    plt.scatter(0, 0, color='red', label='Galactic Center', s=100)  # Galactic Center

    # Add solar circles
    circle1 = plt.Circle((0, 0), 8.21, color='r', fill=False, lw=2, label='8.21 kpc')
    circle2 = plt.Circle((0, 0), 20, color='g', fill=False, linestyle='--', label='20 kpc')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)

    # Customize plot
    plt.xlabel('X [kpc]')
    plt.ylabel('Y [kpc]')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('1σ and 2σ Contours of Closest Approach Points in X-Y Plane')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.show()
    if save:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, star_name+'.png'))

    if show:
        plt.show()



def plot_sigma_contours_m(origins, xlim=(-60, 60), ylim=(-60, 60), bw_method='scott', save = False, star_name='star', save_path = 'Plots/brown_stars/brown_potential'):
    """
    Plot the 1σ and 2σ contours of closest approach points in the X-Y plane.

    Parameters:
    - origins: list of tuples
      List of positions (X, Y, Z) at the closest approach for each sample.
    - xlim: tuple
      Limits for the X-axis (default: (-60, 60)).
    - ylim: tuple
      Limits for the Y-axis (default: (-60, 60)).
    """
    # Extract X, Y positions from origins
    x = np.array([origin[0] for origin in origins])
    y = np.array([origin[1] for origin in origins])

    # Create a 2D KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)
    xi, yi = np.linspace(xlim[0], xlim[1], 300), np.linspace(ylim[0], ylim[1], 300)
    xi, yi = np.meshgrid(xi, yi)
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    # Normalize the KDE to get cumulative probabilities
    zi_flat = zi.flatten()
    sorted_zi = np.sort(zi_flat)
    cumsum_zi = np.cumsum(sorted_zi) / np.sum(sorted_zi)

    # Determine contour levels for 1σ (68.26%) and 2σ (95.44%)
    level_1sigma = sorted_zi[np.searchsorted(cumsum_zi, 1 - 0.6826)]
    level_2sigma = sorted_zi[np.searchsorted(cumsum_zi, 1- 0.9544)]

    # Find mode of the distribution
    mode_index = np.argmax(zi_flat)
    mode_x, mode_y = xi.flatten()[mode_index], yi.flatten()[mode_index]

    # Get the probability of the Galactic Center origin considering a 4 kpc radius of the bulge
    prob_00 = compute_bulge_probability(kde, radius=4)

    # Plot the KDE with contour levels
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, zi, levels=100, cmap="Blues")
    plt.colorbar(label="Density")
    plt.contour(xi, yi, zi, levels=[level_2sigma, level_1sigma, zi.max()], colors=['red', 'darkorange', 'k'], 
                linewidths=[2, 2, 2], linestyles=['--', '-','-'], labels=['1σ', '2σ', '0'])
    #plt.contourf(xi, yi, zi, levels=[level_2sigma, level_1sigma, zi.max()], colors=['cyan', 'magenta', 'blue'], alpha=0.5)
    
    # Add annotations
    plt.scatter(mode_x, mode_y, color='black', label='Mode of Distribution', s=100, marker='x')
    plt.scatter(0, 0, color='red', label='Galactic Center', s=100)  # Galactic Center

    # Add solar circles
    circle1 = plt.Circle((0, 0), 8.21, color='r', fill=False, lw=2, label='8.21 kpc')
    circle2 = plt.Circle((0, 0), 20, color='g', fill=False, linestyle='--', label='20 kpc')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)

    # Customize plot
    plt.xlabel('X [kpc]')
    plt.ylabel('Y [kpc]')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(f'1σ and 2σ Contours of Closest Approach Points in X-Y Plane \n Probability of GC origin {prob_00:.2e}')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.show()
    if save:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, star_name+'.png'))


def plot_closest_origins(origins, xlim=(-60, 60), ylim=(-60, 60)):
    """
    Plot the closest approach points in the X-Y plane.
    
    Parameters:
    - origins: list of tuples
      List of positions (X, Y, Z) at the closest approach for each sample.
    """
    # Extract X, Y positions
    x = np.array([origin[0] for origin in origins])
    y = np.array([origin[1] for origin in origins])
    
    # Create a 2D KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    #xi, yi = np.linspace(x.min(), x.max(), 300), np.linspace(y.min(), y.max(), 300)
    xi, yi = np.linspace(xlim[0], xlim[1], 300), np.linspace(ylim[0], ylim[1], 300)
    xi, yi = np.meshgrid(xi, yi)
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    # Plot the KDE as a contour plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, zi, levels=30, cmap="Blues")
    plt.colorbar(label="Density")
    plt.scatter(0, 0, color='red', label='Galactic Center', s=100)  # Mark the Galactic Center

    # solar circle
    circle1 = plt.Circle((0, 0), 8.21, color='r', fill=False, lw=2, label='8.21 kpc')
    # add circle with a 15 kpc radius
    circle2 = plt.Circle((0, 0), 20, color='g', fill=False, linestyle='--', label='15 kpc')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)
    plt.xlabel('X [kpc]')
    plt.ylabel('Y [kpc]')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Closest Approach Points in X-Y Plane')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    plt.show()



def sample_positive_distances(d_helio, d_helio_err, n_samples):
    """
    Sample distances from a Gaussian distribution, ensuring all values are positive.
    
    Parameters:
        d_helio (float): Mean distance (e.g., heliocentric distance) in the distribution.
        d_helio_err (float): Standard deviation of the distribution.
        n_samples (int): Number of samples to generate.
    
    Returns:
        numpy.ndarray: Array of sampled distances, all positive.
    """
    samples = []
    while len(samples) < n_samples:
        # Generate samples
        new_samples = np.random.normal(d_helio, d_helio_err, n_samples - len(samples))
        # Keep only positive values
        positive_samples = new_samples[new_samples > 0]
        # Append valid samples to the list
        samples.extend(positive_samples)
    
    return np.array(samples)


def integrate_orbits_to_closest_plane_crossing_agama(
    star, ra_col='ra', dec_col='dec', pmra_col='pmra',
    pmdec_col='pmdec', d_col='d_helio', v_col='v_helio',
    pmra_error_col='pmra_error', pmdec_error_col='pmdec_error',
    d_col_error='d_helio_error', v_col_error='v_helio_error',
    output_path=None, n_samples=100, integration_time=-0.5,
    potential = None 
):
    """
    Integrate orbits of a star back in time using AGAMA and find the closest approach to X, Y, Z = 0.

    Parameters:
    - star: pandas.DataFrame
      Single-row DataFrame with columns ['RAx', 'DECx', 'pmra', 'pmdec', 'd_helio', 'v_helio', 
                                         'pmra_error', 'pmdec_error', 'd_helio_error', 'v_helio_error'].
    - output_path: str
      Path to save orbit files (optional).
    - n_samples: int
      Number of Monte Carlo samples for error propagation.
    - integration_time: float
      Integration time into the past (negative for backward integration) in Myr.
    - n_steps: int
      Number of time steps for orbit integration.
      
    Returns:
    - closest_origins: list
      List of positions (X, Y, Z) at the closest approach for each sample.
    """


    #potential = agama.Potential(file='Agama/McMillan2011.pot')  # Adjust file path if needed
    # just in case
    agama.setUnits(mass=1, length=1, velocity=1)  # kpc, km/s, 1e10 M_sun
    
    # Define AGAMA potential (McMillan17 equivalent)
    if potential is None:
      potential_file = 'Agama/McMillan2011.pot'
      potential = agama.Potential(potential_file)

    from astropy import units as u

    ### galpy uses these units, so instruct agama to do the same
    #agama.setUnits( mass=1., length=1, velocity=1)  # Msun, kpc, km/s


    
    # Extract data and errors
    ra = star[ra_col].iloc[0]
    dec = star[dec_col].iloc[0]
    pmra = star[pmra_col].iloc[0]
    pmdec = star[pmdec_col].iloc[0]
    d_helio = star[d_col].iloc[0]
    v_helio = star[v_col].iloc[0]
    pmra_err = star[pmra_error_col].iloc[0]
    pmdec_err = star[pmdec_error_col].iloc[0]
    d_helio_err = star[d_col_error].iloc[0]
    v_helio_err = star[v_col_error].iloc[0]

    
    # Generate Monte Carlo samples
    ra_samples = np.full(n_samples, ra)
    dec_samples = np.full(n_samples, dec)
    pmra_samples = np.random.normal(pmra, pmra_err, n_samples)
    pmdec_samples = np.random.normal(pmdec, pmdec_err, n_samples)
    v_samples = np.random.normal(v_helio, v_helio_err, n_samples)
    # sample only positive distances
    #d_samples = np.random.normal(d_helio, d_helio_err, n_samples)
    d_samples = sample_positive_distances(d_helio, d_helio_err, n_samples)
    
    
    # Initialize output
    closest_origins = []

    # Prepare output directory if saving results
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # Integrate orbits
    for i in tqdm(range(n_samples)):
        # Transform coordinates to Galactocentric frame
        c = SkyCoord(
            ra=ra_samples[i] * u.deg, dec=dec_samples[i] * u.deg, 
            distance=d_samples[i] * u.kpc, pm_ra_cosdec=pmra_samples[i] * u.mas / u.yr, 
            pm_dec=pmdec_samples[i] * u.mas / u.yr, radial_velocity=v_samples[i] * u.km / u.s
        )
        galactic = c.transform_to(Galactocentric())
        x, y, z = galactic.x.to_value(u.kpc), galactic.y.to_value(u.kpc), galactic.z.to_value(u.kpc)
        vx, vy, vz = galactic.v_x.to_value(u.km / u.s), galactic.v_y.to_value(u.km / u.s), galactic.v_z.to_value(u.km / u.s)

        # Prepare initial conditions for AGAMA
        ic = np.array([x, y, z, vx, vy, vz])  # Initial conditions
        
        # Integrate orbit
        orbit = agama.orbit(potential=potential, ic=ic, time=integration_time, dtype = object)#, trajsize=n_steps)

        # extract time steps and trajectory
        ts=np.array([t for t in orbit])
        trajectory=orbit(orbit)[:,:3]
        rs=np.linalg.norm(trajectory,axis=1)
        Rs=np.linalg.norm(trajectory[:,:2],axis=1)
        vels=orbit(orbit)[:,3:]


        # Split trajectory into positions and velocities
        x_vals, y_vals, z_vals = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        positions = [x_vals, y_vals, z_vals]

        # Find the closest approach to the Galactic plane (Z=0)
        distances = np.abs(z_vals)  # Z values
        min_idx = np.argmin(distances)
        periapse = (x_vals[min_idx], y_vals[min_idx], z_vals[min_idx])
        closest_origins.append(periapse)

        # Save orbit data to file (optional)
        if output_path:
            filename = os.path.join(output_path, f"star_sample_{i}.txt")
            np.savetxt(filename, orbit, header="X [kpc]    Y [kpc]    Z [kpc]    VX [km/s]    VY [km/s]    VZ [km/s]")
    
    return closest_origins


def integrate_orbits_to_plane_crossing_agama(
    star, ra_col='ra', dec_col='dec', pmra_col='pmra',
    pmdec_col='pmdec', d_col='d_helio', v_col='v_helio',
    pmra_error_col='pmra_error', pmdec_error_col='pmdec_error',
    d_col_error='d_helio_error', v_col_error='v_helio_error',
    output_path=None, n_samples=100, integration_time=-0.5,
    potential=None
):
    """
    Integrate orbits of a star back in time using AGAMA and find the first crossing of the Galactic plane (Z=0).

    Parameters:
    - star: pandas.DataFrame
      Single-row DataFrame with columns ['RAx', 'DECx', 'pmra', 'pmdec', 'd_helio', 'v_helio', 
                                         'pmra_error', 'pmdec_error', 'd_helio_error', 'v_helio_error'].
    - output_path: str
      Path to save orbit files (optional).
    - n_samples: int
      Number of Monte Carlo samples for error propagation.
    - integration_time: float
      Integration time into the past (negative for backward integration) in Myr.
      
    Returns:
    - plane_crossings: list
      List of positions (X, Y, Z) at the first crossing of the Galactic plane for each sample.
    """

    import numpy as np
    from astropy.coordinates import SkyCoord, Galactocentric
    from astropy import units as u
    import os
    from tqdm import tqdm
    import agama

    # Initialize AGAMA units and potential
    agama.setUnits(mass=1, length=1, velocity=1)  # kpc, km/s, 1e10 M_sun
    if potential is None:
        potential = agama.Potential(file='Agama/McMillan2011.pot')

    # Extract data and errors
    ra = star[ra_col].iloc[0]
    dec = star[dec_col].iloc[0]
    pmra = star[pmra_col].iloc[0]
    pmdec = star[pmdec_col].iloc[0]
    d_helio = star[d_col].iloc[0]
    v_helio = star[v_col].iloc[0]
    pmra_err = star[pmra_error_col].iloc[0]
    pmdec_err = star[pmdec_error_col].iloc[0]
    d_helio_err = star[d_col_error].iloc[0]
    v_helio_err = star[v_col_error].iloc[0]

    # Generate Monte Carlo samples
    ra_samples = np.full(n_samples, ra)
    dec_samples = np.full(n_samples, dec)
    pmra_samples = np.random.normal(pmra, pmra_err, n_samples)
    pmdec_samples = np.random.normal(pmdec, pmdec_err, n_samples)
    d_samples = np.random.normal(d_helio, d_helio_err, n_samples)
    v_samples = np.random.normal(v_helio, v_helio_err, n_samples)
    
    # Initialize output
    plane_crossings = []

    # Prepare output directory if saving results
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # Integrate orbits
    for i in tqdm(range(n_samples)):
        # Transform coordinates to Galactocentric frame
        c = SkyCoord(
            ra=ra_samples[i] * u.deg, dec=dec_samples[i] * u.deg, 
            distance=d_samples[i] * u.kpc, pm_ra_cosdec=pmra_samples[i] * u.mas / u.yr, 
            pm_dec=pmdec_samples[i] * u.mas / u.yr, radial_velocity=v_samples[i] * u.km / u.s
        )
        galactic = c.transform_to(Galactocentric())
        x, y, z = galactic.x.to_value(u.kpc), galactic.y.to_value(u.kpc), galactic.z.to_value(u.kpc)
        vx, vy, vz = galactic.v_x.to_value(u.km / u.s), galactic.v_y.to_value(u.km / u.s), galactic.v_z.to_value(u.km / u.s)

        # Prepare initial conditions for AGAMA
        ic = np.array([x, y, z, vx, vy, vz])  # Initial conditions
        
        # Integrate orbit
        orbit = agama.orbit(potential=potential, ic=ic, time=integration_time)
        trajectory = orbit(orbit)[:, :3]
        z_vals = trajectory[:, 2]

        # Find the first crossing of the Galactic plane (Z=0)
        sign_changes = np.where(np.diff(np.sign(z_vals)))[0]
        if len(sign_changes) > 0:
            first_crossing_idx = sign_changes[0] + 1  # Index after sign change
            first_crossing = trajectory[first_crossing_idx]
            plane_crossings.append(tuple(first_crossing))
        else:
            plane_crossings.append(None)  # No crossing detected

        # Save orbit data to file (optional)
        if output_path:
            filename = os.path.join(output_path, f"star_sample_{i}.txt")
            np.savetxt(filename, orbit, header="X [kpc]    Y [kpc]    Z [kpc]    VX [km/s]    VY [km/s]    VZ [km/s]")
    
    return plane_crossings


from multiprocessing import Pool, cpu_count
from functools import partial
from astropy.coordinates import SkyCoord, Galactocentric
import numpy as np
import os
from tqdm import tqdm
import agama

def worker_initialize(potential_file):
    """
    Initialize the potential in each worker process.
    """
    global worker_potential
    worker_potential = agama.Potential(potential_file)


def integrate_orbit_single_sample(sample_data, integration_time):
    """
    Integrate orbit for a single Monte Carlo sample and find the closest approach to Z=0.
    """
    global worker_potential
    import astropy.units as u
    
    ra, dec, d, pmra, pmdec, v_helio = sample_data
    c = SkyCoord(
        ra=ra * u.deg, dec=dec * u.deg, 
        distance=d * u.kpc, pm_ra_cosdec=pmra * u.mas / u.yr, 
        pm_dec=pmdec * u.mas / u.yr, radial_velocity=v_helio * u.km / u.s
    )
    galactic = c.transform_to(Galactocentric())
    x, y, z = galactic.x.to_value(u.kpc), galactic.y.to_value(u.kpc), galactic.z.to_value(u.kpc)
    vx, vy, vz = galactic.v_x.to_value(u.km / u.s), galactic.v_y.to_value(u.km / u.s), galactic.v_z.to_value(u.km / u.s)

    # Prepare initial conditions for AGAMA
    ic = np.array([x, y, z, vx, vy, vz])

    # Integrate orbit
    orbit = agama.orbit(potential=worker_potential, ic=ic, time=integration_time, dtype=object)

    # Extract trajectory and find the closest approach to Z=0
    trajectory = orbit(orbit)[:, :3]
    z_vals = trajectory[:, 2]
    distances = np.abs(z_vals)
    min_idx = np.argmin(distances)
    closest_origin = trajectory[min_idx]

    # Find the first crossing of the Galactic plane (Z=0)
    sign_changes = np.where(np.diff(np.sign(z_vals)))[0]
    if len(sign_changes) > 0:
        first_crossing_idx = sign_changes[0] + 1  # Index after sign change
        first_crossing = trajectory[first_crossing_idx]
    else: 
        # Since there is no crossing, do not return any value
         # No crossing detected
        #first_crossing = None  # No crossing detected
        first_crossing = (np.nan, np.nan, np.nan)
    return tuple(first_crossing)
    #return tuple(closest_origin)


def integrate_orbits_to_plane_crossing_agama_parallel(
    star, ra_col='ra', dec_col='dec', pmra_col='pmra',
    pmdec_col='pmdec', d_col='d_helio', v_col='v_helio',
    pmra_error_col='pmra_error', pmdec_error_col='pmdec_error',
    d_col_error='d_helio_error', v_col_error='v_helio_error',
    output_path=None, n_samples=100, integration_time=-0.5,
    potential_file='Agama/McMillan2011.pot'
):
    """
    Parallelized integration of orbits back in time to find the closest approach to Z=0.
    """
    # Extract data and errors
    ra = star[ra_col].iloc[0]
    dec = star[dec_col].iloc[0]
    pmra = star[pmra_col].iloc[0]
    pmdec = star[pmdec_col].iloc[0]
    d_helio = star[d_col].iloc[0]
    v_helio = star[v_col].iloc[0]
    pmra_err = star[pmra_error_col].iloc[0]
    pmdec_err = star[pmdec_error_col].iloc[0]
    d_helio_err = star[d_col_error].iloc[0]
    v_helio_err = star[v_col_error].iloc[0]

    # Generate Monte Carlo samples
    ra_samples = np.full(n_samples, ra)
    dec_samples = np.full(n_samples, dec)
    pmra_samples = np.random.normal(pmra, pmra_err, n_samples)
    pmdec_samples = np.random.normal(pmdec, pmdec_err, n_samples)
    #d_samples = np.random.normal(d_helio, d_helio_err, n_samples)
    d_samples = sample_positive_distances(d_helio, d_helio_err, n_samples) # sample the distances gaussian but truncated to be only positive
    
    v_samples = np.random.normal(v_helio, v_helio_err, n_samples)

    # Create sample data tuples
    sample_data = list(zip(ra_samples, dec_samples, d_samples, pmra_samples, pmdec_samples, v_samples))

    # Parallel computation
    with Pool(cpu_count(), initializer=worker_initialize, initargs=(potential_file,)) as pool:
        closest_origins = list(tqdm(pool.imap(
            partial(integrate_orbit_single_sample, integration_time=integration_time),
            sample_data), total=n_samples))
    
    # remove tuples containing nans (no crossing) and return the closest origins
    closest_origins = [origin for origin in closest_origins if not np.isnan(origin).any()]

    return closest_origins


if __name__ == '__main__':
    potential = agama.Potential(file='Agama/brown2015.pot')  # Adjust file path if needed

    # Load the star data
    brown_data = pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/brown_stars_gaia.csv')

    # compute orbits for all brown stars using Gaia DR3
    plane_crossings = []
    for i in range(len(brown_data)):
        star = brown_data.iloc[[i]]
        plane_crossings.append(integrate_orbits_to_plane_crossing_agama(star, n_samples=10000,
                                                            integration_time=-0.3,  
                                                            pmra_col='mu_ra',
                                                            pmdec_col='mu_dec',
                                                            pmra_error_col='mu_ra_error',
                                                            pmdec_error_col='mu_dec_error', 
                                                            potential_file=potential))