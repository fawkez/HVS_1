import gala.potential as gp
import gala.integrate as gi
import gala.dynamics as gd
import astropy.units as auni
import astropy.coordinates as acoo
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import sys
from matplotlib.widgets import Slider
import os

pot = gp.MilkyWayPotential2022()
kms = auni.km / auni.s

acoo.galactocentric_frame_defaults.set('latest')

def doit(vej, cosang, t):
    sinang = np.sqrt(1 - cosang**2)
    # start from 10 pc radius
    startpos = np.array([cosang, 0, sinang]) * 0.01 * auni.kpc
    vel = np.array([cosang, 0, sinang]) * vej
    w0 = gd.PhaseSpacePosition(startpos, vel=vel * kms)
    
    # First integration to find the apoapsis
    nsteps = 10000
    timestep = t * auni.Myr / nsteps
    orbit = gp.Hamiltonian(pot).integrate_orbit(
        w0,
        dt=timestep,
        n_steps=nsteps,
        Integrator=gi.DOPRI853Integrator,
    )
    
    # Find the apoapsis point where vx changes sign
    vx = orbit.v_x.to_value(kms)
    apoapsis_indices = np.where(np.diff(np.sign(vx)))[0]
    
    if len(apoapsis_indices) == 0:
        # If no apoapsis is found, return the initial orbit
        R = (orbit.x**2 + orbit.z**2 + orbit.y**2)**.5
        z = orbit.z.to_value(auni.kpc)
        VR = orbit.v_x * R / orbit.x
        Vz = orbit.v_z - orbit.v_x * orbit.z / orbit.x
        return R.to_value(auni.kpc), z, VR.to_value(kms), Vz.to_value(kms)
    
    apoapsis_index = apoapsis_indices[0]
    apoapsis_time = orbit.t[apoapsis_index]
    
    # Second integration up to apoapsis time with 10000 points
    nsteps_apoapsis = 10000
    timestep_apoapsis = apoapsis_time / nsteps_apoapsis
    orbit_apoapsis = gp.Hamiltonian(pot).integrate_orbit(
        w0,
        dt=timestep_apoapsis,
        n_steps=nsteps_apoapsis,
        Integrator=gi.DOPRI853Integrator,
    )
    
    R = (orbit_apoapsis.x**2 + orbit_apoapsis.z**2 + orbit_apoapsis.y**2)**.5
    z = orbit_apoapsis.z.to_value(auni.kpc)
    VR = orbit_apoapsis.v_x * R / orbit_apoapsis.x
    Vz = orbit_apoapsis.v_z - orbit_apoapsis.v_x * orbit_apoapsis.z / orbit_apoapsis.x
    return R.to_value(auni.kpc), z, VR.to_value(kms), Vz.to_value(kms)

def doall(N=10000, seed=3):
    rng = np.random.default_rng(seed)
    vej = 10**rng.uniform(2.8, 3.5, size=N)
    cosa = rng.uniform(0, 1, size=N)
    times = 200
    r1, r2, r3, r4 = [], [], [], []
    for curv, curc in tqdm(zip(vej, cosa), total=N):
        R, z, VR, Vz = doit(curv, curc, times)
        r1.append(R)
        r2.append(z)
        r3.append(VR)
        r4.append(Vz)

    return [np.array(_) for _ in [r1, r2, r3, r4]]

print('Starting')
# Do the simulation
R, z, VR, Vz = doall(10000, 3)

print('Simulation Done')

# Plot it
xbins = 100  # bins in z
ybins = 100  # bins in log10(VR)
zbins = 100  # bins in R

zf = z.flatten()
VRf = VR.flatten()
Vzf = Vz.flatten()
Rf = R.flatten()

# save the data
#data= pd.DataFrame({'R':Rf, 'z':zf, 'VR':VRf, 'Vz':Vzf})
#data.to_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/orbits/gala_ejections/gala_ejections_10000_4.csv')

data_interpolator = np.array([np.log10(Rf), zf/Rf, np.log10(VRf)])

stat, bin_edges, binnumber = binned_statistic_dd(
    data_interpolator.T, Vzf*VRf/Rf,
    statistic='mean',
    bins=[xbins, ybins, zbins])

xedges = bin_edges[0]
yedges = bin_edges[1]
zedges = bin_edges[2]


# construct a regular grid interpolator that will give Vz/R*VR for any R, z
import numpy as np
from scipy.interpolate import RegularGridInterpolator

xcenters = 0.5*(xedges[:-1] + xedges[1:])
ycenters = 0.5*(yedges[:-1] + yedges[1:])
zcenters = 0.5*(zedges[:-1] + zedges[1:])

interp_func = RegularGridInterpolator(
    (xcenters, ycenters, zcenters),  # The 3D grid coordinates
    stat,                            # The mean ratio table on that grid
    method='linear',                 # or 'nearest', 'cubic', etc.
    bounds_error=False,              # If False, points outside will not raise an error...
    fill_value=0                     # ...and will return 0. (You can choose NaN or None, etc.)
)

# save the interpolator
# Save the interpolator to a file
with open('/Users/mncavieres/Documents/2024-2/HVS/Data/vz_interpolator/vz_rf_vr_sergey_v4_return0.pkl', 'wb') as f:
    pickle.dump(interp_func, f)


# plt.figure(figsize=(8, 6))
# plt.pcolormesh(xedges, yedges, stat.T[0], cmap='jet', shading='auto')

# cb = plt.colorbar()
# cb.set_label(r'$V_z/R*V_R$ ')

# plt.xlabel(r'$\log_{10}(R)$')
# plt.ylabel(r'$z/R$')
# plt.tight_layout()
# plt.savefig('/Users/mncavieres/Documents/2024-2/HVS/Plots/interpolator/vi2.png')
# plt.show()

# Choose an initial value for the slider (e.g. the middle value)
init_log10_VRf = zcenters[len(zcenters) // 2]
# Find the index of the closest center to this initial value
init_index = np.argmin(np.abs(zcenters - init_log10_VRf))

# Create the figure and the initial pcolormesh plot.
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # leave room at the bottom for the slider

# Plot the initial statistic slice:
# Note: stat is indexed as [x, y, z]. Since pcolormesh expects the grid defined by the bin edges,
# we plot using xedges and yedges. We take the transpose to match the axis orientation.
pc = ax.pcolormesh(xedges, yedges, stat[:, :, init_index].T, shading='auto')
cbar = fig.colorbar(pc, ax=ax)
ax.set_xlabel('log10(Rf)')
ax.set_ylabel('zf/Rf')
ax.set_title(f'log10(VRf) = {zcenters[init_index]:.2f}')

# Create a slider axis and the Slider itself.
# The slider will allow values between the minimum and maximum of zcenters.
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'log10(VRf)', zcenters[0], zcenters[-1], valinit=init_log10_VRf)

# Update function that gets called whenever the slider value changes.
def update(val):
    current_val = slider.val
    # Find the index corresponding to the current slider value
    current_index = np.argmin(np.abs(zcenters - current_val))
    # Update the pcolormesh data.
    # pcolormesh stores its data as a flattened array, so we flatten the transposed slice.
    pc.set_array(stat[:, :, current_index].T.ravel())
    # Update the title to show the current log10(VRf) value.
    ax.set_title(f'log10(VRf) = {zcenters[current_index]:.2f}')
    fig.canvas.draw_idle()

# Connect the slider to the update function.
slider.on_changed(update)

plt.show()