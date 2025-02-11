import gala.potential as gp
import gala.integrate as gi
import gala.dynamics as gd
import astropy.units as auni
import astropy.coordinates as acoo
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import sys
import os

pot = gp.MilkyWayPotential2022()
kms = auni.km / auni.s

acoo.galactocentric_frame_defaults.set('latest')


# def doit(vej, cosang, t):
#     #cosang = np.cos(ang)
#     #sinang = np.sin(ang)
#     sinang = np.sqrt(1 - cosang**2)
#     startpos = np.array([cosang, 0, sinang]) * auni.kpc
#     vel = np.array([cosang, 0, sinang]) * vej
#     w0 = gd.PhaseSpacePosition(startpos, vel=vel * kms)
#     nsteps = 30000
#     timestep = t * auni.Myr / nsteps
#     orbit = gp.Hamiltonian(pot).integrate_orbit(
#         w0,
#         dt=timestep,
#         n_steps=nsteps,
#         Integrator=gi.DOPRI853Integrator,
#         # Integrator_kwargs = dict(atol=1e-15,rtol=1e-15)
#     )
#     R = (orbit.x**2 + orbit.z**2 + orbit.y**2)**.5
#     z = orbit.z.to_value(auni.kpc)
#     VR = orbit.v_x * R / orbit.x # this is some decomposition of the velocity in a weird way
#     Vz = orbit.v_z - orbit.v_x * orbit.z / orbit.x
#     return R.to_value(auni.kpc), z, VR.to_value(kms), Vz.to_value(kms)
def doit(vej, cosang, t):
    sinang = np.sqrt(1 - cosang**2)
    # start from 10 pc radius
    startpos = np.array([cosang, 0, sinang]) * 0.01 * auni.kpc
    vel = np.array([cosang, 0, sinang]) * vej
    w0 = gd.PhaseSpacePosition(startpos, vel=vel * kms)
    nsteps = 10000
    timestep = t * auni.Myr / nsteps
    orbit = gp.Hamiltonian(pot).integrate_orbit(
        w0,
        dt=timestep,
        n_steps=nsteps,
        Integrator=gi.DOPRI853Integrator,
        # Integrator_kwargs = dict(atol=1e-15,rtol=1e-15)
    )
    R = (orbit.x**2 + orbit.z**2 + orbit.y**2)**.5
    z = orbit.z.to_value(auni.kpc)
    VR = orbit.v_x * R / orbit.x
    Vz = orbit.v_z - orbit.v_x * orbit.z / orbit.x
    return R.to_value(auni.kpc), z, VR.to_value(kms), Vz.to_value(kms)

def doall(N=10000, seed=3):
    rng = np.random.default_rng(seed)
    vej = 10**rng.uniform(2.6, 3.5, size=N)
    cosa = rng.uniform(0, 1, size=N)
    times = 100
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
R, z, VR, Vz = doall(20000, 3)

print('Simulation Done')

# Plot it
xbins = 100  # bins in z
ybins = 100  # bins in log10(VR)

zf = z.flatten()
VRf = VR.flatten()
Vzf = Vz.flatten()
Rf = R.flatten()

# save the data
#data= pd.DataFrame({'R':Rf, 'z':zf, 'VR':VRf, 'Vz':Vzf})
#data.to_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/orbits/gala_ejections/gala_ejections_10000_3.csv')

stat, xedges, yedges, binnum = binned_statistic_2d(
    y= zf/Rf, # to avoid log(0) we add a small number
    x=np.log10(Rf),
    values=Vzf/Rf*VRf,
    statistic='mean',
    bins=[xbins, ybins]
)

# construct a regular grid interpolator that will give Vz/R*VR for any R, z
import numpy as np
from scipy.interpolate import RegularGridInterpolator

xcenters = 0.5*(xedges[:-1] + xedges[1:])
ycenters = 0.5*(yedges[:-1] + yedges[1:])

interp_func = RegularGridInterpolator(
    (xcenters, ycenters),   # The 2D grid coordinates
    stat,                   # The mean ratio table on that grid
    method='linear',        # or 'nearest', 'cubic', etc.
    bounds_error=False,     # If False, points outside will not raise an error...
    fill_value=np.nan       # ...and will return NaN. (You can choose 0 or None, etc.)
)

# save the interpolator
# Save the interpolator to a file
with open('/Users/mncavieres/Documents/2024-2/HVS/Data/vz_interpolator/vz_rf_vr_sergey.pkl', 'wb') as f:
    pickle.dump(interp_func, f)


plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, stat.T, cmap='jet', shading='auto')

cb = plt.colorbar()
cb.set_label(r'$V_z/R*V_R$ ')

plt.xlabel(r'$\log_{10}(R)$')
plt.ylabel(r'$z/R$')
plt.tight_layout()
plt.show()