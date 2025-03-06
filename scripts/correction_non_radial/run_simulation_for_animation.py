import gala.potential as gp
import gala.integrate as gi
import gala.dynamics as gd
import astropy.units as auni
import astropy.coordinates as acoo
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Set up the potential and unit conversions.
pot = gp.MilkyWayPotential2022()
kms = auni.km / auni.s
acoo.galactocentric_frame_defaults.set('latest')

def doit(vej, angle, t):
    # Calculate cos and sin for the full 360° range.
    cosang = np.cos(angle)
    sinang = np.sin(angle)
    # Start at 10 pc (0.01 kpc) in the XZ plane.
    startpos = np.array([cosang, 0, sinang]) * 0.01 * auni.kpc
    vel = np.array([cosang, 0, sinang]) * vej
    w0 = gd.PhaseSpacePosition(startpos, vel=vel * kms)
    
    nsteps = 1000
    timestep = t * auni.Myr / nsteps
    orbit = gp.Hamiltonian(pot).integrate_orbit(
        w0,
        dt=timestep,
        n_steps=nsteps,
        Integrator=gi.DOPRI853Integrator,
    )
    # Return only the x and z coordinates (ignoring y) as arrays.
    return orbit.x.to_value(auni.kpc), orbit.z.to_value(auni.kpc)

def doall(N=10, seed=3):
    rng = np.random.default_rng(seed)
    vej = 10**rng.uniform(2.7, 3, size=N)
    # Choose angles uniformly between 0 and 2π for full 360° coverage.
    angles = rng.uniform(0, np.pi, size=N)
    times = 300
    x_list, z_list = [], []
    for curv, angle in tqdm(zip(vej, angles), total=N):
        x, z = doit(curv, angle, times)
        x_list.append(x)
        z_list.append(z)
    return x_list, z_list

print('Starting simulation')
# Run the simulation for 1000 orbits.
#x_coords, z_coords = doall(300, seed=3)
x_coords, z_coords = doit(vej = 700, angle= 0.8*np.pi/4, t=300)
print('Simulation Done')

# Plot the first orbit.
plt.plot(x_coords, z_coords)
plt.xlabel("X [kpc]")
plt.ylabel("Z [kpc]")
plt.title("Ejected star orbit")
plt.show()

# Create a dictionary with columns "x0", "z0", "x1", "z1", etc.
#data = {}
#for i, (x, z) in enumerate(zip(x_coords, z_coords)):
#    data[f'x{i}'] = x
#    data[f'z{i}'] = z

# Build a DataFrame.
# The index corresponds to the time steps and the columns are labeled with the orbit and coordinate.
#df = pd.DataFrame(data)
df = pd.DataFrame({'x0': x_coords, 'z0': z_coords})
print(df.head())

# Save the DataFrame to a CSV file.
df.to_csv("/Users/mncavieres/Documents/2024-2/HVS/Data/orbits/gala_ejections/orbits_animation.csv", index=True)
