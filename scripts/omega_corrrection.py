# imports
#import agama
import numpy as np
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u
from tqdm import tqdm
import os
import sys
import pandas as pd
from astropy.table import Table
from matplotlib import pyplot as plt
import pickle
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
import astropy.units as u
import astropy.coordinates as coord
import os
import sys

from astropy.coordinates import SkyCoord, Galactocentric, ICRS
import astropy.units as u
import astropy.coordinates as coord
import os
import pickle

# Add the path to the 'scripts' folder directly
sys.path.append('/app/data/scripts')
from scripts.implied_d_vr import compute_R0_V0_SI, implied_calculations # Import specific functions or classes as needed
from scripts.selections import *
from scripts.catalog_preparation.prepare_gaia import prepare_speedystar

# set current directory to /app/data so we can work with relative paths
os.chdir('/app/data/')

# Add the path to the 'scripts' folder directly
sys.path.append('/app/data/')

#from scripts import orbit_integration_agama as oia

#import agama
#agama.setUnits(mass=1, length=1, velocity=1)

from matplotlib.ticker import MaxNLocator
# set up the plotting
# set font size
plt.rcParams.update({'font.size': 18})
# set the figure size
plt.rcParams.update({'figure.figsize': (10, 7)})
# set the font to latex
plt.rcParams.update({'text.usetex': True})

from logging import warn
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

class OrbitInterpolator2D:
    def __init__(self, R, z, Vtot, Omega):
        """
        Initialize the interpolator with precomputed orbit data.

        Parameters:
        - R, z, VR, Vz, t: 2D arrays of shape (n_orbits, n_points_per_orbit).
        """
        # Flatten the data
        self.R = R.flatten()
        self.z = z.flatten()
        # self.V = V.flatten()
        # self.Vz = Vz.flatten()
        self.Vtot =   Vtot.flatten() #np.sqrt(self.VR**2 + self.Vz**2) # I will try to use the total velocity instead of splitting in RZ components

        # Omega should be a flattened array of just the Y component
        self.Omega = Omega 
        #assert np.all(self.Omega >= 0), "All Omega values should be positive!" # seems that Omega can be negative ups


        # Create a KDTree for efficient nearest-neighbor search
        self.tree = cKDTree(np.column_stack((self.R, self.z, self.Vtot)))

        # Create an interpolator for Vz
        self.interpolator = LinearNDInterpolator(
            np.column_stack((self.R, self.z, self.Vtot)),
            self.Omega
        )
    
    def query(self, R, z, Vtot):
        """
        Query the interpolator to find Vz for a given R, z, and total velocity.

        Parameters:
        - R, z, Vtot: Input parameters to match.

        Returns:
        - Interpolated Vz value.
        """
        # Try to interpolate Vz
        Omega_interp = self.interpolator(R, z, Vtot)
        if np.isnan(Omega_interp):  # Fall back to nearest neighbor
                    # Find the nearest neighbor if interpolation fails
            dist, idx = self.tree.query([R, z, Vtot])
            Omega_nearest = self.Omega[idx]
            #warn(f"Interpolation failed at R={R:.2f}, z={z:.2f}, V={Vtot:.2f}! Using nearest neighbor instead.")
            return Omega_nearest
        return Omega_interp

    def load(path):
        """
        Load the interpolator from a file.

        Parameters:
        path (string): Path to the file to load.

        Returns:
        OrbitInterpolator2D object.
        """
        import pickle
        # Load the interpolator that is needed
        with open(path, "rb") as f:
            interpolator = pickle.load(f)
        return interpolator
    
    def save(self, path):
        """
        Save the interpolator to a file.

        Parameters:
        path (string): Path to save the file.
        """
        import pickle
        # Save the interpolator that is needed
        with open(path, "wb") as f:
            pickle.dump(self, f)


def interpolate_pre_comp(R_gc, R, z, Vtot, path_interpolators='Data/omega_interpolator_3d'):
    """
    Interpolates the Omega values for the given R, z, and Vtot values
    using a precomputed interpolator. The file name convention is:
        interpolator_{r_min}-{r_max}kpc.pkl
    where r_min and r_max are in kpc (integers or floats).

    Parameters:
        R_gc (float): Galactocentric radius in kpc (the value to match in the file range).
        R (float): Cylindrical radius in kpc.
        z (float): Height above the Galactic plane in kpc.
        Vtot (float): Total velocity in km/s.
        path_interpolators (str): Path to the folder containing the precomputed interpolators.

    Returns:
        array: Interpolated Omega values.
    """

    start = time.time()
    # List all files in the folder
    file_names = os.listdir(path_interpolators)

    # Variable to store the correct file name once found
    correct_file = None

    # Loop over all files and parse r_min, r_max
    for file_name in file_names:
        # Look for a name like "interpolator_5-10kpc.pkl"
        if file_name.startswith("interpolator_") and file_name.endswith(".pkl"):
            # Example: file_name = "interpolator_5-10kpc.pkl"
            # Split on underscore, take the second chunk e.g. "5-10kpc.pkl"
            range_part = file_name.split("_", 1)[1]  # "5-10kpc.pkl"

            # Remove the trailing "kpc.pkl", leaving something like "5-10"
            range_part = range_part.replace("kpc.pkl", "")  # "5-10"

            # Split on hyphen => ["5", "10"]
            r_min_str, r_max_str = range_part.split("-")

            # Convert to float (or int, if you are sure they are integers)
            r_min = float(r_min_str)
            r_max = float(r_max_str)

            # Check if R_gc is in [r_min, r_max)
            if r_min <= R_gc < r_max:
                correct_file = file_name
                break

    if correct_file is None:
        raise ValueError(
            f"No interpolator found matching R_gc={R_gc:.2f} kpc in directory {path_interpolators}!"
        )

    # print time taken to find the correct file
    print(f"Time taken to find correct file: {time.time() - start:.2f} seconds")
    # Load the chosen interpolator
    interpolator_path = os.path.join(path_interpolators, correct_file)
    #interpolator = OrbitInterpolator2D()
    with open(interpolator_path, "rb") as f:
        interpolator = pickle.load(f)
        #pickle.dump(interpolator, f)

    # Print time to load the interpolator
    print(f"Time taken to load interpolator: {time.time() - start:.2f} seconds")

    start = time.time()
    # Query the interpolator
    Omega = interpolator.query(R, z, Vtot)
    
    # print time taken to query the interpolator
    print(f"Time taken to query interpolator: {time.time() - start:.2f} seconds")
    return Omega



R0_SI, V0_SI = compute_R0_V0_SI()

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

def z_rotation_2d(theta):
    """
    Returns the 3x3 rotation matrix for a rotation around the z-axis by angle `theta`.
    Note this can handle array `theta` using broadcasting if carefully shaped.
    """
    # If theta is an array of shape (N,), we want an array of shape (N,3,3).
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # We'll build a stack of rotation matrices
    R = np.zeros((len(theta), 3, 3))
    R[:,0,0] = cos_t
    R[:,0,1] = -sin_t
    R[:,1,0] = sin_t
    R[:,1,1] = cos_t
    R[:,2,2] = 1.0
    return R

R0_SI, V0_SI = compute_R0_V0_SI()
def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

def z_rotation_2d(theta):
    """
    Returns the 3x3 rotation matrix for a rotation around the z-axis by angle `theta`.
    Note this can handle array `theta` using broadcasting if carefully shaped.
    """
    # If theta is an array of shape (N,), we want an array of shape (N,3,3).
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # We'll build a stack of rotation matrices
    if isinstance(theta, float):
        R = np.zeros((1, 3, 3))
    else:
        R = np.zeros((len(theta), 3, 3))
    R[:,0,0] = cos_t
    R[:,0,1] = -sin_t
    R[:,1,0] = sin_t
    R[:,1,1] = cos_t
    R[:,2,2] = 1.0
    return R

def compute_distance_correction(ra, dec, omega, R0, pmra, pmdec, D_i):
    """
    Compute distance correction using the sources ra, dec coordinates to define the n vector, pmra, pmdec for the mu vector.

    The Omega vector needs to be defined from the amplitude omega that is passed, which corresponds to the Y
    component of the Omega vector if the trajectory is in the XZ plane. Otherwise it needs to be rotated to align with 
    the plane of the orbit. 

    For consistency all vectors should be in SI units and in the ICRS frame.

    input:
        ra: right ascension in degrees (float or array)
        dec: declination in degrees 
        omega: amplitude of the Omega vector in SI units
        R0: position vector pointing from the Galactic Center to the Sun in ICRS frame
        pmra: proper motion in right ascension in mas/yr
        pmdec: proper motion in declination in mas/yr
        D_i: initial distance in pc
    """

        # Convert positions to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Convert proper motions to radians per second
    masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
    pmra_rad_s = pmra * masyr_to_radsec
    pmdec_rad_s = pmdec * masyr_to_radsec

    # Initialize the arrays
    if isinstance(ra, float):
        N = 1
        plx = np.empty(1)
        eplx = np.empty(1)
        VGCR = np.empty(1)
        VR = np.empty(1)
    else:
        N = len(ra_rad)
        plx = np.empty(N)
        eplx = np.empty(N)
        VGCR = np.empty(N)
        VR = np.empty(N)

    R02 = np.sum(R0**2)
    #V0R0 = np.dot(V0, R0)


    ra_r = ra_rad
    dec_r = dec_rad

    cos_ra = np.cos(ra_r)
    sin_ra = np.sin(ra_r)
    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)

    # Compute the unit vector n
    n0 = cos_ra * cos_dec
    n1 = sin_ra * cos_dec
    n2 = sin_dec

    # pmra basis vector
    e10 = -sin_ra
    e11 = cos_ra
    e12 = 0.0

    # pmdec basis vector
    e20 = -cos_ra * sin_dec
    e21 = -sin_ra * sin_dec
    e22 = cos_dec
    
    # proper motions in rad/s (already converted)
    mu0 = pmra_rad_s * e10 + pmdec_rad_s * e20
    mu1 = pmra_rad_s * e11 + pmdec_rad_s * e21
    mu2 = pmra_rad_s * e12 + pmdec_rad_s * e22

    # Cross product of R0 and n
    R0n0 = n1 * R0[2] - n2 * R0[1]
    R0n1 = - (n0 * R0[2] - n2 * R0[0])
    R0n2 = n0 * R0[1] - n1 * R0[0]

    # Dot product of R0xn and mu
    R0xn_mu = mu0 * R0n0 + mu1 * R0n1 + mu2 * R0n2


    # Define skycoord to obtain the angle of the orbit
    coord_object = SkyCoord(np.array(ra)*u.deg, np.array(dec)*u.deg, np.array(D_i)*u.pc, frame='icrs')
    x, y, z = coord_object.transform_to(Galactocentric).cartesian.xyz.to(u.kpc).value
    #print(x, y, z)
    # Compute the galactocentric longitude angle
    l = np.arctan2(y, x) # this should be an array now

    # Define omega vector with prior knowledge that if the orbit is in the xz plane the amplitude is the Y component
    #Omega = np.array([0, omega, 0]) # not sure if this will make it work with arrays
    omega_array = np.ones(N)*omega  # If needed, or handle if omega is already array
    Omega_gal = np.zeros((N,3))
    Omega_gal[:,0] = 0.0
    Omega_gal[:,1] = omega_array     # "Y" component
    Omega_gal[:,2] = 0.0

    # Rotate Omega by the galactocentric longitude angle of the source, this implies a rotation around the Z axis
    # to align it with the plane of the orbit
    #Omega_rotated = z_rotation(Omega, l)
    # Get the rotation matrices around the Z axis for an angle l
    Rot = z_rotation_2d(l)  # shape (N,3,3)

    # Rotate the Omega vector by the rotation matrices
    Omega_gal_rot = np.einsum("nij,nj->ni", Rot, Omega_gal)

    # transform the Omega vector from Galactocentric to ICRS frame
    Omega_rotated = SkyCoord(x=Omega_gal_rot[:,0]*u.kpc, y=Omega_gal_rot[:,1]*u.kpc,
                              z=Omega_gal_rot[:,2]*u.kpc,
                                frame= Galactocentric).transform_to(ICRS).cartesian.xyz.value
    
    # turn to unit vector (it might have been better to do everything in unit vectors but this is the way it is)
    Omega_rotated = Omega_rotated / np.linalg.norm(Omega_rotated)
    
    # dot product with n, and scaling by omega, this way I know that the output is in the same units as the omega input, which could be in pc to make this easy 
    Omega_dot_n = (n0*Omega_rotated[0] + n1*Omega_rotated[1] + n2*Omega_rotated[2])*omega

    Omega_dot_mu = (mu0*Omega_rotated[0] + mu1*Omega_rotated[1] + mu2*Omega_rotated[2])

    

    # Compute the extra term
    return Omega_dot_n / R0xn_mu, Omega_dot_mu / R0xn_mu


def interpolate_pre_comp(R_gc, R, z, Vtot, path_interpolators='Data/omega_interpolator_3d'):
    """
    Interpolates the Omega values for the given R, z, and Vtot values
    using a precomputed interpolator. The file name convention is:
        interpolator_{r_min}-{r_max}kpc.pkl
    where r_min and r_max are in kpc (integers or floats).

    Parameters:
        R_gc (float): Galactocentric radius in kpc (the value to match in the file range).
        R (float): Cylindrical radius in kpc.
        z (float): Height above the Galactic plane in kpc.
        Vtot (float): Total velocity in km/s.
        path_interpolators (str): Path to the folder containing the precomputed interpolators.

    Returns:
        array: Interpolated Omega values.
    """

    # List all files in the folder
    file_names = os.listdir(path_interpolators)

    # Variable to store the correct file name once found
    correct_file = None

    # Loop over all files and parse r_min, r_max
    for file_name in file_names:
        # Look for a name like "interpolator_5-10kpc.pkl"
        if file_name.startswith("interpolator_") and file_name.endswith(".pkl"):
            # Example: file_name = "interpolator_5-10kpc.pkl"
            # Split on underscore, take the second chunk e.g. "5-10kpc.pkl"
            range_part = file_name.split("_", 1)[1]  # "5-10kpc.pkl"

            # Remove the trailing "kpc.pkl", leaving something like "5-10"
            range_part = range_part.replace("kpc.pkl", "")  # "5-10"

            # Split on hyphen => ["5", "10"]
            r_min_str, r_max_str = range_part.split("-")

            # Convert to float (or int, if you are sure they are integers)
            r_min = float(r_min_str)
            r_max = float(r_max_str)

            # Check if R_gc is in [r_min, r_max)
            if r_min <= R_gc < r_max:
                correct_file = file_name
                break

    if correct_file is None:
        raise ValueError(
            f"No interpolator found matching R_gc={R_gc:.2f} kpc in directory {path_interpolators}!"
        )

    # Load the chosen interpolator
    interpolator_path = os.path.join(path_interpolators, correct_file)
    #interpolator = OrbitInterpolator2D()
    with open(interpolator_path, "rb") as f:
        interpolator = pickle.load(f)
        #pickle.dump(interpolator, f)

    # Query the interpolator
    Omega = interpolator.query(R, z, Vtot)
    return Omega
