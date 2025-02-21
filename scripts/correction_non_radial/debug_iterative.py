import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the path to the 'scripts' folder directly
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')

# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *
from scripts.correction_non_radial.iterative_correction import compute_R0xez
from scripts.correction_non_radial.iterative_correction import compute_ez
from scripts.correction_non_radial.iterative_correction import compute_R0_V0_SI
from scripts.correction_non_radial.iterative_correction import getdist_corrected
from scripts.correction_non_radial.iterative_correction import iterative_correction
#from scripts.catalog_preparation.prepare_gaia import prepare_speedystar
# load some data to test
from astropy.table import Table


# lets test with some speedystar data
data = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/stock_long.fits')

data = data[data['v0']> 1500]

from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactocentric
import astropy.units as u





coords = Galactocentric(x = data['x']*u.kpc, y = data['y']*u.kpc, z = data['z']*u.kpc,
                    v_x = data['vx']*u.km/u.s, v_y = data['vy']*u.km/u.s, v_z = data['vz']*u.km/u.s)

icrs_coord = coords.transform_to(coord.ICRS())

data['ra_clean'] = icrs_coord.ra.degree
data['dec_clean'] = icrs_coord.dec.degree
data['pmra_clean'] = icrs_coord.pm_ra_cosdec
data['pmdec_clean'] = icrs_coord.pm_dec
data['radial_velocity'] = icrs_coord.radial_velocity

ra_rad = icrs_coord.ra.rad
dec_rad = icrs_coord.dec.rad
pmra_rad_s = icrs_coord.pm_ra_cosdec.to(u.rad/u.s).value
pmdec_rad_s = icrs_coord.pm_dec.to(u.rad/u.s).value
vz = data['vz']
l = data['l']
b = data['b']

VGCR, VR, Darr, D_for_it = iterative_correction(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                       0, 0)