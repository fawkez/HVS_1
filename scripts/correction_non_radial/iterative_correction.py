# imports 
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy.table import Table
import os
import astropy
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import pickle

# set galactocentric frame to latest
astropy.coordinates.galactocentric_frame_defaults.set('latest')



def getdist_vectorized2(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                       epmra_rad_s, epmdec_rad_s, R0, V0):
    """
    Compute several derived quantities from Gaia-like observables, assuming a radial trajectory:
      - plx_i : implied parallax [1/m]
      - eplx_i: parallax error    [1/m]
      - VGCR_i: velocity in the Galactocentric reference frame [m/s]
      - VR_i  : radial velocity   [m/s]
      - D_i   : distance          [m]
      - eD_i  : distance error    [m]

    Parameters
    ----------
    ra_rad       : float or ndarray
        Right ascension in radians.
    dec_rad      : float or ndarray
        Declination in radians.
    pmra_rad_s   : float or ndarray
        Proper motion in RA (radians/sec).
    pmdec_rad_s  : float or ndarray
        Proper motion in Dec (radians/sec).
    epmra_rad_s  : float or ndarray
        Uncertainty in pmra_rad_s (radians/sec).
    epmdec_rad_s : float or ndarray
        Uncertainty in pmdec_rad_s (radians/sec).
    R0           : 3-element array
        Position of the Sun / observer in Galactocentric coords (m).
    V0           : 3-element array
        Velocity of the Sun / observer in Galactocentric coords (m/s).

    Returns
    -------
    plx_i  : ndarray
        Implied parallax (1/m).
    eplx_i : ndarray
        Error on the parallax (1/m).
    VGCR_i : ndarray
        Velocity in the Galactocentric reference frame (m/s).
    VR_i   : ndarray
        Radial velocity (m/s).
    D_i    : ndarray
        Distance (m).
    eD_i   : ndarray
        Distance uncertainty (m).

    Notes
    -----
    1) D_i is computed as 1 / plx_i, which is also equivalent to:
         D_i = - (dot_V0_R0n / dot_mu_R0n).
       We then propagate errors directly from pmra, pmdec to get eD_i.

    2) R0 and V0 are treated here as exact (no uncertainty). If you need
       to account for their errors, you must propagate them separately.

    3) This linear approximation is valid only if the fractional errors
       in pmra, pmdec remain small. If Gaia parallax is near zero or
       negative, or you have large fractional errors, you may need a
       more careful (Bayesian) approach.
    """
    # Convert single floats to arrays
    if isinstance(ra_rad, np.float64):
        ra_rad = np.array([ra_rad])
    N = len(ra_rad)
    
    plx = np.empty(N)
    eplx = np.empty(N)
    VGCR = np.empty(N)
    VR = np.empty(N)
    Darr = np.empty(N)
    eDarr = np.empty(N)

    R02 = np.sum(R0**2)
    V0R0 = np.dot(V0, R0)

    ra_r = ra_rad
    dec_r = dec_rad

    cos_ra = np.cos(ra_r)
    sin_ra = np.sin(ra_r)
    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)

    n0 = cos_ra * cos_dec
    n1 = sin_ra * cos_dec
    n2 = sin_dec

    # Cross product of R0 and n
    R0n0 = n1 * R0[2] - n2 * R0[1]
    R0n1 = -(n0 * R0[2] - n2 * R0[0])
    R0n2 = n0 * R0[1] - n1 * R0[0]

    # pmra basis vector
    e10 = -sin_ra
    e11 =  cos_ra
    e12 =  0.0

    # pmdec basis vector
    e20 = -cos_ra * sin_dec
    e21 = -sin_ra * sin_dec
    e22 =  cos_dec

    # Build velocity vector from pmra, pmdec
    mu0 = pmra_rad_s * e10 + pmdec_rad_s * e20
    mu1 = pmra_rad_s * e11 + pmdec_rad_s * e21
    mu2 = pmra_rad_s * e12 + pmdec_rad_s * e22

    # Dot products
    dot_V0_R0n = V0[0]*R0n0 + V0[1]*R0n1 + V0[2]*R0n2  # A
    dot_mu_R0n = mu0*R0n0 + mu1*R0n1 + mu2*R0n2       # B

    # Implied parallax
    plx_i = - dot_mu_R0n / dot_V0_R0n  # 1/m

    # Derivative components for eplx
    dot_e1_R0n = e10*R0n0 + e11*R0n1 + e12*R0n2
    dot_e2_R0n = e20*R0n0 + e21*R0n1 + e22*R0n2

    eplx_i = np.sqrt(
        (dot_e1_R0n * epmra_rad_s)**2 + 
        (dot_e2_R0n * epmdec_rad_s)**2
    ) / dot_V0_R0n  # 1/m

    # Distance in meters
    # Equivalent to 1.0 / plx_i, but let's be explicit
    D = - dot_V0_R0n / dot_mu_R0n

    # For error propagation, define:
    #   D = -(A / B), where A = dot_V0_R0n, B = dot_mu_R0n.
    #   B = alpha*pmra + beta*pmdec, with:
    alpha = dot_e1_R0n  # partial B wrt pmra
    beta  = dot_e2_R0n  # partial B wrt pmdec

    # partial D / partial pmra = A * alpha / B^2
    # partial D / partial pmdec = A * beta / B^2
    # => (sigma_D)^2 = (A^2 / B^4)[alpha^2 (sigma_pmra)^2 + beta^2 (sigma_pmdec)^2]
    A = dot_V0_R0n
    B = dot_mu_R0n
    sig_pmra  = epmra_rad_s
    sig_pmdec = epmdec_rad_s

    eD = np.abs(A / (B**2)) * np.sqrt(alpha**2 * sig_pmra**2 + 
                                      beta**2  * sig_pmdec**2)

    # Next, compute VR_i and VGCR_i as in your original code
    nV0  = V0[0]*n0 + V0[1]*n1 + V0[2]*n2
    nR0  = R0[0]*n0 + R0[1]*n1 + R0[2]*n2
    mun  = mu0*n0 + mu1*n1 + mu2*n2
    muR0 = mu0*R0[0] + mu1*R0[1] + mu2*R0[2]

    denom = R02 - nR0**2
    VR_i = -1.0 / denom * (
        (nV0 * R02 - V0R0 * nR0)
        + D * (nV0 * nR0 - V0R0 + mun*R02 - muR0*nR0)
        + D**2 * (mun * nR0 - muR0)
    )  # [m/s]

    # Compute VGCR
    numerator = V0R0 + D*muR0 + VR_i*nR0 + D*nV0 + D**2*mun + D*VR_i
    R = np.sqrt(R02 + D**2 + 2.0 * D * nR0)
    VGCR_i = numerator / R  # [m/s]

    # Store results
    plx[:]   = plx_i
    eplx[:]  = eplx_i
    VGCR[:]  = VGCR_i
    VR[:]    = VR_i
    Darr[:]  = D
    eDarr[:] = eD

    return plx, eplx, VGCR, VR, Darr, eDarr


# Define the vectors that are constant for the Milky Way
def compute_R0_V0_SI():
    """
    Define the position and velocity of the Galactic Center in Galactocentric frame
    Here, we're using the position of the Sun in the Galactocentric frame,
    which by definition is at (x, y, z) = (0, 0, 0) in Galactocentric coordinates.
    We then transform this to the ICRS frame to get the position and velocity vectors.

    input:
        None
    output:
        R0: position vector pointing from the Galactic Center to the Sun in ICRS frame
        V0: velocity vector of the Sun in ICRS frame
    """
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
    X0 = icrs_coord.cartesian.xyz.to(u.m).value  # position in meters
    V0 = icrs_coord.velocity.d_xyz.to(u.m / u.s).value  # velocity in m/s

    # Get the vector pointing from the Galactic Center to the Sun in ICRS frame
    R0 = -X0  # Shape (3,)
    V0 = -V0  # Shape (3,)
    return R0, V0


def dot_2D(a, b):
    """Handles dot product between vectors of shape (3, N) and (3,)."""
    if a.ndim == 2:  # If `a` has shape (3, N)
        return np.sum(a * b[:, np.newaxis], axis=0)
    return np.dot(a, b)  # If both are (3,)

def cross_2D(a, b):
    """
    Computes the cross product between:
    - `a`: A 2D array of shape (3, N) or a 1D array of shape (3,).
    - `b`: A 1D array of shape (3,).
    
    Returns:
    - A 2D array of shape (3, N) if `a` is 2D.
    - A 1D array of shape (3,) if both `a` and `b` are 1D.
    """
    if a.ndim == 2:  # If `a` is (3, N)
        return np.cross(a.T, b).T  # Compute cross product column-wise
    return np.cross(a, b)  # If both are (3,)


def compute_extra_term(R0, mu, n, Vz, ez):
    """
    Computes the correction term for the implied distance by considering a velocity component in the galactic Z direction

    Parameters:
        R0 (array): Vector from the galactic center to the sun in the ICRS coordinate system and SI units
        mu (array): Proper motion of the star in the ICRS coordinate system and SI units
        n (array): Normal vector from the sun to the source in the ICRS coordinate system (unit vector, from RA, DEC)
        Vz (float): Velocity component in the galactic Z direction in SI units
        ez (array): Unit vector in the galactic Z direction (orthogonal to the plane of the disk) in the ICRS coordinate system
    
    Returns:
        float: The correction term for the implied distance
    """
    r0xez = cross_2D(R0, ez)
    n_r0xez = dot_2D(n, r0xez)
    r0xmu = cross_2D(R0, mu)
    n_r0xmu = dot_2D(n, r0xmu)

    return Vz*(n_r0xez / n_r0xmu )

def compute_R0xez(R0):
    """
    Computes the cross product of the vector from the galactic center to the sun and the unit vector in the galactic Z direction

    Parameters:
        R0 (array): Vector from the galactic center to the sun in the ICRS coordinate system and SI units

    Returns:
        array: The cross product R0 x ez in ICRS coordinate system and SI units
    """
    # define ez
    ez = compute_ez()
    return np.cross(R0, ez)

def compute_ez():
    """
    Computes the cross product of the vector from the galactic center to the sun and the unit vector in the galactic Z direction

    Parameters:
        R0 (array): Vector from the galactic center to the sun in the ICRS coordinate system and SI units

    Returns:
        array: The cross product R0 x ez in ICRS coordinate system and SI units
    """
    # define ez
    ez = SkyCoord(0, 0, 1, unit='m', representation_type='cartesian', frame = 'galactocentric' ).transform_to('icrs').cartesian.xyz.value

    # make extra sure that is a unit vector
    ez = ez / np.linalg.norm(ez)

    return ez



def compute_correction_distance(Vz, n, R0xez, denominator):
    """
    Computes the correction term for the implied distance by considering a velocity component in the galactic Z direction

    Parameters:
        Vz (float): Velocity component in the galactic Z direction in SI units
        n (array): Normal vector from the sun to the source in the ICRS coordinate system (unit vector, from RA, DEC)
        R0xez (array): Cross product of the vector from the galactic center to the sun and the unit vector in the galactic Z direction in ICRS coordinate system and SI units
        numerator (float): Numerator of the correction term for the implied distance

    Returns:
        float: The correction term for the implied distance
    """
    #print(n.shape, R0xez.shape)
    return Vz * dot_2D(n, R0xez) / (denominator)


def compute_correction_velocity(Vz, mu, n, R0xez, ez, D_I):
    """
    Computes the correction term for the radial velocity by considering a 
    velocity component in the galactic Z direction.
    
    According to the derivation, the correction is:
    
        ΔV_r = [ Vz * (mu·(R0 × ez)) + D_I * Vz * (mu·(n × ez)) ]
               / [ mu·(R0 × n) ]
    
    Parameters:
        Vz (float): Velocity component in the galactic Z direction (SI units).
        mu (array): Proper motion vector in ICRS, shape (3, N) [SI units].
        n (array): Unit vector from the Sun to the source in ICRS, shape (3, N).
        R0xez (array): Cross product of R0 and ez, shape (3,).
        ez (array): Unit vector in the galactic Z direction in ICRS, shape (3,).
        D_I (array): Implied distance (from the strictly radial solution), shape (N,).
    
    Returns:
        array: Correction term for the radial velocity, shape (N,).
    """
    global R0
    # --- Corrected First Term ---
    # According to the derivation the first term must be: Vz * (mu · (R0 x ez))
    R0xez_dot_mu = np.einsum('i,ij->j', R0xez, mu)  # shape (N,)
    
    # --- Second Term ---
    # Compute cross product of n and ez, then dot with mu:
    mu_cross_n_ez = np.cross(n.T, ez).T  # shape (3, N)
    mu_dot_cross_n_ez = np.einsum('ij,ij->j', mu, mu_cross_n_ez)  # shape (N,)
    
    # Numerator: Vz*(mu·(R0 x ez)) + D_I * Vz*(mu·(n x ez))
    numerator = Vz * R0xez_dot_mu + D_I * Vz * mu_dot_cross_n_ez

    # --- Denominator ---
    # Compute mu · (R0 × n).  Note that -np.cross(n, R0) equals (R0 × n).
    nxR = -np.cross(n.T, R0).T  # shape (3, N)
    mu_dot_nxR = np.einsum('ij,ij->j', mu, nxR)  # shape (N,)
    
    # Final correction:
    correction_velocity = numerator / mu_dot_nxR
    return correction_velocity


# define the global parameters that do not need to change, namely R0, V0, ez and R0xez
R0, V0 = compute_R0_V0_SI()
R0xez = compute_R0xez(R0)
ez = compute_ez()

def getdist_corrected(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, Vz, l, b):
    
    # give access to the global variables
    global R0, V0, R0xez, ez
    """
    Compute several derived quantities from Gaia-like observables, assuming a radial trajectory:
      - plx_i : implied parallax [1/m]
      - eplx_i: parallax error    [1/m]
      - VGCR_i: velocity in the Galactocentric reference frame [m/s]
      - VR_i  : radial velocity   [m/s]
      - D_i   : distance          [m]
      - eD_i  : distance error    [m]

    Parameters
    ----------
    ra_rad       : float or ndarray
        Right ascension in radians.
    dec_rad      : float or ndarray
        Declination in radians.
    pmra_rad_s   : float or ndarray
        Proper motion in RA (radians/sec).
    pmdec_rad_s  : float or ndarray
        Proper motion in Dec (radians/sec).
    epmra_rad_s  : float or ndarray
        Uncertainty in pmra_rad_s (radians/sec).
    epmdec_rad_s : float or ndarray
        Uncertainty in pmdec_rad_s (radians/sec).
    R0           : 3-element array
        Position of the Sun / observer in Galactocentric coords (m).
    V0           : 3-element array
        Velocity of the Sun / observer in Galactocentric coords (m/s).

    Returns
    -------
    plx_i  : ndarray
        Implied parallax (1/m).
    eplx_i : ndarray
        Error on the parallax (1/m).
    VGCR_i : ndarray
        Velocity in the Galactocentric reference frame (m/s).
    VR_i   : ndarray
        Radial velocity (m/s).
    D_i    : ndarray
        Distance (m).
    eD_i   : ndarray
        Distance uncertainty (m).

    Notes
    -----
    1) D_i is computed as 1 / plx_i, which is also equivalent to:
         D_i = - (dot_V0_R0n / dot_mu_R0n).
       We then propagate errors directly from pmra, pmdec to get eD_i.

    2) R0 and V0 are treated here as exact (no uncertainty). If you need
       to account for their errors, you must propagate them separately.

    3) This linear approximation is valid only if the fractional errors
       in pmra, pmdec remain small. If Gaia parallax is near zero or
       negative, or you have large fractional errors, you may need a
       more careful (Bayesian) approach.
    """
    # Convert single floats to arrays
    if isinstance(ra_rad, np.float64):
        ra_rad = np.array([ra_rad])
    N = len(ra_rad)
    

    VGCR = np.empty(N)
    VR = np.empty(N)
    Darr = np.empty(N)

    R02 = np.sum(R0**2)
    V0R0 = np.dot(V0, R0)

    ra_r = ra_rad
    dec_r = dec_rad

    cos_ra = np.cos(ra_r)
    sin_ra = np.sin(ra_r)
    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)

    n0 = cos_ra * cos_dec
    n1 = sin_ra * cos_dec
    n2 = sin_dec

    # Cross product of R0 and n
    R0n0 = n1 * R0[2] - n2 * R0[1]
    R0n1 = -(n0 * R0[2] - n2 * R0[0])
    R0n2 = n0 * R0[1] - n1 * R0[0]

    # pmra basis vector
    e10 = -sin_ra
    e11 =  cos_ra
    e12 =  0.0

    # pmdec basis vector
    e20 = -cos_ra * sin_dec
    e21 = -sin_ra * sin_dec
    e22 =  cos_dec

    # Build velocity vector from pmra, pmdec
    mu0 = pmra_rad_s * e10 + pmdec_rad_s * e20
    mu1 = pmra_rad_s * e11 + pmdec_rad_s * e21
    mu2 = pmra_rad_s * e12 + pmdec_rad_s * e22

    # Dot products
    dot_V0_R0n = V0[0]*R0n0 + V0[1]*R0n1 + V0[2]*R0n2  # A
    dot_mu_R0n = mu0*R0n0 + mu1*R0n1 + mu2*R0n2       # B

    # Distance in meters
    # Equivalent to 1.0 / plx_i, but let's be explicit
    D = - dot_V0_R0n / dot_mu_R0n

    # assemble the n vector as it is needed for the correction term
    n = np.array([n0, n1, n2])

    # compute correction to distance, we added a minus because the denominator is inverted to the equations we have in the overleaf
    # if b > 0: # This is required since the Vz in the interpolator is only negative, so we need to reflect the sign of the correction term if b > 0
    #     D += compute_correction_distance(Vz, n, R0xez, dot_mu_R0n)
    # else:
    #     D -= compute_correction_distance(Vz, n, R0xez, dot_mu_R0n)

    # removed this since it is better to invert Vz as a function of b before calling this function

    D += compute_correction_distance(Vz, n, R0xez, dot_mu_R0n)

    # Next, compute VR_i and VGCR_i as in your original code
    nV0  = V0[0]*n0 + V0[1]*n1 + V0[2]*n2
    nR0  = R0[0]*n0 + R0[1]*n1 + R0[2]*n2
    mun  = mu0*n0 + mu1*n1 + mu2*n2
    muR0 = mu0*R0[0] + mu1*R0[1] + mu2*R0[2]

    denom = R02 - nR0**2
    VR_i = -1.0 / denom * (
        (nV0 * R02 - V0R0 * nR0)
        + D * (nV0 * nR0 - V0R0 + mun*R02 - muR0*nR0)
        + D**2 * (mun * nR0 - muR0)
    )  # [m/s]

    # Set up proper motion vector for the correction term
    mu_v = np.array([mu0, mu1, mu2])
    # Compute correction term for radial velocity
    VR_i += compute_correction_velocity(Vz, mu_v, n, R0xez, ez, D)

    # Compute VGCR
    numerator = V0R0 + D*muR0 + VR_i*nR0 + D*nV0 + D**2*mun + D*VR_i
    R = np.sqrt(R02 + D**2 + 2.0 * D * nR0)
    VGCR_i = numerator / R  # [m/s]

    VGCR[:]  = VGCR_i
    VR[:]    = VR_i
    Darr[:]  = D


    return VGCR, VR, Darr

# load the interpolator 
with open('/Users/mncavieres/Documents/2024-2/HVS/Data/vz_interpolator/vz_rf_vr_sergey.pkl', 'rb') as f:
#with open('/Users/mncavieres/Documents/2024-2/HVS/Data/vz_interpolator/vz_rf_vr_sergey_extrapolate.pkl', 'rb') as f:
    interpolator_vz = pickle.load(f)

def do_iteration(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, D_i, Vr):
    
    D_i = np.maximum(D_i, 0) # this is here to prevent negative distance errors

    distances = Distance(D_i, unit=u.m, allow_negative=True)
    # make skycoord object
    skycoord = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad, pm_ra_cosdec=pmra_rad_s*u.rad/u.s,
                         pm_dec=pmdec_rad_s*u.rad/u.s, distance=distances, 
                         radial_velocity=Vr*u.m/u.s)

    # get galactocentric coordinates in kpc,
    x, y, z = skycoord.transform_to('galactocentric').cartesian.xyz.to(u.kpc).value
    vx, vy, _ = skycoord.transform_to('galactocentric').velocity.d_xyz.to(u.km/u.s).value

    # get galactic coordinates
    l, b = skycoord.galactic.l.value, skycoord.galactic.b.value

    # get the galactocentric distance
    R_gc = (x**2 + y**2 + z**2)**0.5
    
    # interpolate to get vz/r*vr
    logR = np.log10(R_gc)
    ratios = z/R_gc
    points = np.array([logR, ratios]).T  # Combine logR and z/R_gc into a single array

    vz_rvr = interpolator_vz(points)
    #vz_rvr = interpolator_vz(logR, z/R_gc)

    V_r_sergey = vx * R_gc/x #VR = orbit.v_x * R / orbit.x, this corresponds to a non-orthogonal decomposition of the velocity in radial and vz components
    vz = vz_rvr * V_r_sergey/R_gc
    vz = vz * 1000 # convert to m/s

    # compute the correction term ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, Vz
    VGCR, VR, D_i = getdist_corrected(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, vz, l)
    
    return VGCR, VR, D_i

def iterative_correction(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                       epmra_rad_s, epmdec_rad_s):
    
    D_for_it = []
    global R0, V0, R0xez, ez
    # compute initial distance and velocity
    plx, eplx, VGCR, VR, Darr, eDarr = getdist_vectorized2(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                       epmra_rad_s, epmdec_rad_s, R0, V0)

    
    # Do 10 iterations of the correction
    for i in tqdm(range(10)): 
        
        VGCR, VR, Darr = do_iteration(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,  Darr, VR)
        
        D_for_it.append(Darr)
        
    return VGCR, VR, Darr, D_for_it
        

    