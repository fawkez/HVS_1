"""
Sanity check for the  implementation of selection,
If applied to the 600 candidates from the paper
every candidate should pass the selection
"""

# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.constants import kpc, au
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from numba import njit

from implied_d_vr import implied_calculations
from selections import is_HVS


# Load the data
data = pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/Sill Candidates/sills_candidates.csv')

# Convert the data to an astropy table
data = Table.from_pandas(data)

# Calculate the implied radial velocity and parallax
#data = implied_calculations(data)

# run the HVS selection algorithm
data = is_HVS(data)

data.write('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/sill_candidates_through_myselection.fits', overwrite=True)