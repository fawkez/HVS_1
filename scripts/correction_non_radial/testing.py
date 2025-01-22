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

R0, V0 = compute_R0_V0_SI()

R0xez = compute_R0xez(R0)

ez = compute_ez()

# load some data to test
from astropy.table import Table


# lets test with some speedystar data
data = Table.read('Data/speedystar_catalogs/stock/phot_no_extinction_prop/cat_ejection_0.fits')#.to_pandas()


# just select 10 points
#data = data[:10]

# we need the implied distance and velocity to get the correction term (0th order solution)
data_ready = prepare_speedystar(data)


# select only things with positive implied distances
data_ready = data_ready[data_ready['implied_distance'] > 0]
#print(len(data_ready))
#print(data_ready.columns)

# compute the correction
from iterative_correction import iterative_correction

# Convert positions to radians
ra_rad = np.deg2rad(data_ready['ra'].values)
dec_rad = np.deg2rad(data_ready['dec'].values)

# Convert proper motions to radians per second
masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
pmra_rad_s = data_ready['pmra'].values * masyr_to_radsec
pmdec_rad_s = data_ready['pmdec'].values * masyr_to_radsec
epmra_rad_s = data_ready['pmra_error'].values * masyr_to_radsec
epmdec_rad_s = data_ready['pmdec_error'].values * masyr_to_radsec

start = time.time()

VGCR, VR, Darr, D_for_it = iterative_correction(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                       epmra_rad_s, epmdec_rad_s)

end = time.time()
print('Iterative correction completed in:', end - start)
print('It took:', (end - start)/len(data_ready), 'for each source')
#print('VGCR:', VGCR)
#print('VR:', VR)
#print('D:', Darr)
#print('D for each iteration:', D_for_it)

# save results
data_ready['VGCR_corrected'] = VGCR
data_ready['VR_corrected'] = VR
data_ready['D_corrected'] = Darr

# iteratively save the distances for each iteration
#print(np.array(D_for_it[2, :]).shape)
#print(D_for_it[0])
#print(len(D_for_it[0]))
for i, dist in enumerate(D_for_it):
    #print(dist[0])
    print(dist[0])
    data_ready[f'D_corrected_{i}'] = dist[0]


# save the data
data_ready.to_csv('Data/speedystar_catalogs/stock_long_corrected.csv')