from scipy.stats import binned_statistic_2d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

# Load the data
print('Loading data')
data = pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/orbits/gala_ejections/gala_ejections_10000_3.csv')
#'R':Rf, 'z':zf, 'VR':VRf, 'Vz':Vzf
Rf = data['R'].values
zf = data['z'].values
VRf = data['VR'].values
Vzf = data['Vz'].values

xbins = 100  # bins in z/R
ybins = 100  # bins in log10(VR)

print('Computing 2d statistic')
print('Range of log10(R):', np.min(np.log10(Rf)), np.max(np.log10(Rf))) 
print('Range of z/R:', np.min(zf/Rf), np.max(zf/Rf))

stat, xedges, yedges, binnum = binned_statistic_2d(
    y= zf/Rf, # to avoid log(0) we add a small number
    x=np.log10(Rf),
    values=(Vzf/Rf)*VRf,
    statistic='mean',
    bins=[xbins, ybins]
)

# construct a regular grid interpolator that will give Vz/R*VR for any R, z
import numpy as np
from scipy.interpolate import RegularGridInterpolator

xcenters = 0.5*(xedges[:-1] + xedges[1:])
ycenters = 0.5*(yedges[:-1] + yedges[1:])

print('Constructing interpolator')
interp_func = RegularGridInterpolator(
    (xcenters, ycenters),   # The 2D grid coordinates
    stat,                   # The mean ratio table on that grid
    method='linear',        # or 'nearest', 'cubic', etc.
    bounds_error=False,     # If False, points outside will not raise an error...
    fill_value=None    # ...and will return NaN. (You can choose 0 or None, etc.)
)
print('Interpolator constructed')
# save the interpolator
# Save the interpolator to a file
with open('/Users/mncavieres/Documents/2024-2/HVS/Data/vz_interpolator/vz_rf_vr_sergey_extrapolate.pkl', 'wb') as f:
    pickle.dump(interp_func, f)


print('Interpolator saved')

# Plot the 2D statistic 
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, stat.T, cmap='jet', shading='auto')

cb = plt.colorbar()
cb.set_label(r'$V_z/R*V_R$ ')

plt.xlabel(r'$\log_{10}(R)$')
plt.ylabel(r'$z/R$')
plt.tight_layout()
plt.savefig('/Users/mncavieres/Documents/2024-2/HVS/Plots/interpolator/vz_rf_log10R_sergey.pdf', dpi = 300)
plt.show()