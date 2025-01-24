from iterative_correction import compute_R0xez
from iterative_correction import compute_R0_V0_SI
from iterative_correction import compute_ez 
import numpy as np
import astropy.units as u
import os
import sys
# Add the path to the 'scripts' folder directly
from tqdm import tqdm
import time

# Add the path to the 'scripts' folder directly
# This needs to be changed to the folder in which I will have the scripts in ALICE
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS') 

from scripts.catalog_preparation.prepare_gaia import prepare_speedystar

import astropy.coordinates as coord

galcen_coord = coord.Galactocentric(
    x=0 * u.kpc,
    y=0 * u.kpc,
    z=0 * u.kpc,
    v_x=0 * u.km / u.s,
    v_y=0 * u.km / u.s,
    v_z=0 * u.km / u.s
)
# Transform to ICRS frame
icrs_coord = galcen_coord.transform_to(coord.ICRS())
# Get the position and velocity in ICRS frame (in SI units)
R0 = -icrs_coord.cartesian.xyz.to(u.kpc).value  # position in meters
V0 = -icrs_coord.velocity.d_xyz.to(u.km / u.s).value  # velocity in m/s


R0, V0 = compute_R0_V0_SI()
R0_kpc = R0*u.m.to(u.kpc)
mod_R0_kpc = np.sqrt(R0_kpc[0]**2 + R0_kpc[1]**2 + R0_kpc[2]**2)
print('Astropy solar positions is',mod_R0_kpc)
#R0xez = compute_R0xez(R0)

#ez = compute_ez()

#print(V0)

mod_V0 = np.sqrt(V0[0]**2 + V0[1]**2 + V0[2]**2)
UVW_V0_speedystar = [-11.1, 12.24, 7.25] # this is UVW solar peculiar motion in km/s, from Schonrich 2010, I need this in ICRS to compare fairly

LSR_V0_speedystar = coord.LSR(0*u.deg, 0*u.deg, 0*u.kpc, 0*u.mas/u.yr, 0*u.mas/u.yr ,0*u.km/u.s , v_bary = UVW_V0_speedystar*u.km/u.s)
ICRS_V0_speedystar = LSR_V0_speedystar.transform_to(coord.ICRS())
# Transform to ICRS frame
V0_speedystar = ICRS_V0_speedystar.velocity.d_xyz.to(u.km / u.s).value  # velocity in km/s

print('V0 from speedystar',V0_speedystar)
print('V0 in astropy', V0)

mod_V0_speedystar = np.sqrt(V0_speedystar[0]**2 + V0_speedystar[1]**2 + V0_speedystar[2]**2)

print(mod_V0, mod_V0_speedystar)
print(mod_V0_speedystar/mod_V0)