
#Import what you need
import numpy as np
import os
#os.chdir('/mnt/c/Users/frase/')
from speedystar import starsample
from speedystar.eject import Hills
from speedystar.eject import HillsFromCatalog
from speedystar.utils.mwpotential import MWPotential
import astropy.units as u
from galpy import potential
import mwdust
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.table import Table
#Print a lot of documentation
#help(starsample)



# import os
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.constants import kpc, au
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from numba import njit

import random
import healpy as hp

from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize


# # Add the path to the 'scripts' folder directly
#sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')
# #sys.path.append('/Users/mncavieres/Documents/2024-2/HVS/speedystar_v2')

# # import eep sampler
from scripts.simulated_photometry import sample_eep

# # set directory to speedystar

# os.chdir('/Users/mncavieres/Documents/2024-2/HVS/speedystar_v2')
