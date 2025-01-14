#Import what you need
import numpy as np
import os
#os.chdir('/mnt/c/Users/frase/')
from speedystar import starsample
from speedystar.eject import Hills
from speedystar.utils.mwpotential import MWPotential
import astropy.units as u
from galpy import potential
import mwdust
from tqdm import tqdm
import matplotlib.pyplot as plt
#Print a lot of documentation
#help(starsample)


photometric_catalogs_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/stock/phot_no_extinction_prop'
propagated_catalogs_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/speedystar_catalogs/stock/propagated'
for catalog in tqdm(os.listdir(propagated_catalogs_path)):
    # Only compute actual catalogs FITS files
    if not catalog.endswith('.fits'):
        print('Skipping', catalog)
        continue

    # Load propagated sample
    mysample = starsample(os.path.join(propagated_catalogs_path, catalog))

    # Select only HVSs that are fast
    #fast = mysample.GCv > 300
    #mysample = mysample[fast]
    
    #Assign the dust map. Will be downloaded if it doesn't already exist in the working directory or where you've
    #   specified above
    mysample.dust = mwdust.Combined15()

    #Get mock apparent magnitudes . By default magnitudes are computed in the Johnson-Cousins V and I bands 
    #   and the Gaia G, G_RP, G_BP and G_RVS bands.
    #   By default this also returns Gaia astrometric and radial velocity errors assuming Gaia DR4 precision
    #Set Av to 0 for all stars
    # set self.Av to 0 for all stars to avoid magnitudes being affected by extinction
    mysample.Av = np.zeros(len(mysample.m))

    mysample.photometry(extiction= True)

    print(mysample.Av)
    #Save the sample with mock photometry
    mysample.save(os.path.join(photometric_catalogs_path, catalog))