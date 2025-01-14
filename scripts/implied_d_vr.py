import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from numba import njit
from astropy.table import Table
import os
import astropy
import pandas as pd

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

@njit(parallel=True)
def getdist_numba(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                  epmra_rad_s, epmdec_rad_s, R0, V0):
    N = len(ra_rad)
    plx = np.empty(N)
    eplx = np.empty(N)
    VGCR = np.empty(N)
    VR = np.empty(N)

    R02 = np.sum(R0**2)
    V0R0 = np.dot(V0, R0)

    for i in range(N):
        ra_r = ra_rad[i]
        dec_r = dec_rad[i]

        cos_ra = np.cos(ra_r)
        sin_ra = np.sin(ra_r)
        cos_dec = np.cos(dec_r)
        sin_dec = np.sin(dec_r)

        n0 = cos_ra * cos_dec
        n1 = sin_ra * cos_dec
        n2 = sin_dec

        # Cross product of R0 and n
        R0n0 = n1 * R0[2] - n2 * R0[1]
        R0n1 = - (n0 * R0[2] - n2 * R0[0])
        R0n2 = n0 * R0[1] - n1 * R0[0]

        # pmra basis vector
        e10 = -sin_ra
        e11 = cos_ra
        e12 = 0.0

        # pmdec basis vector
        e20 = -cos_ra * sin_dec
        e21 = -sin_ra * sin_dec
        e22 = cos_dec

        # proper motions in rad/s (already converted)
        mu0 = pmra_rad_s[i] * e10 + pmdec_rad_s[i] * e20
        mu1 = pmra_rad_s[i] * e11 + pmdec_rad_s[i] * e21
        mu2 = pmra_rad_s[i] * e12 + pmdec_rad_s[i] * e22

        # DOT products
        dot_V0_R0n = V0[0] * R0n0 + V0[1] * R0n1 + V0[2] * R0n2
        dot_mu_R0n = mu0 * R0n0 + mu1 * R0n1 + mu2 * R0n2

        plx_i = - dot_mu_R0n / dot_V0_R0n  # Units: 1/m

        # Compute eplx
        dot_e1_R0n = e10 * R0n0 + e11 * R0n1 + e12 * R0n2
        dot_e2_R0n = e20 * R0n0 + e21 * R0n1 + e22 * R0n2

        eplx_i = np.sqrt(
            (dot_e1_R0n * epmra_rad_s[i])**2 + (dot_e2_R0n * epmdec_rad_s[i])**2
        ) / dot_V0_R0n  # Units: 1/m

        # Compute other DOT products
        nV0 = V0[0] * n0 + V0[1] * n1 + V0[2] * n2
        nR0 = R0[0] * n0 + R0[1] * n1 + R0[2] * n2
        mun = mu0 * n0 + mu1 * n1 + mu2 * n2
        muR0 = mu0 * R0[0] + mu1 * R0[1] + mu2 * R0[2]

        D = 1.0 / plx_i  # Units: m

        denom = R02 - nR0**2
        VR_i = -1 / denom * (
            (nV0 * R02 - V0R0 * nR0) + D *
            (nV0 * nR0 - V0R0 + mun * R02 - muR0 * nR0) + D**2 *
            (mun * nR0 - muR0)
        )  # Units: m/s

        # Compute VGCR
        numerator = V0R0 + D * muR0 + VR_i * nR0 + D * nV0 + D**2 * mun + D * VR_i
        R = np.sqrt(R02 + D**2 + 2 * D * nR0)
        VGCR_i = numerator / R  # Units: m/s

        plx[i] = plx_i  # Units: 1/m
        eplx[i] = eplx_i  # Units: 1/m
        VGCR[i] = VGCR_i  # Units: m/s
        VR[i] = VR_i  # Units: m/s

    return plx, eplx, VGCR, VR




@njit(parallel=True)
def getdist_numba(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                  epmra_rad_s, epmdec_rad_s, R0, V0):
    N = len(ra_rad)
    plx = np.empty(N)
    eplx = np.empty(N)
    VGCR = np.empty(N)
    VR = np.empty(N)

    R02 = np.sum(R0**2)
    V0R0 = np.dot(V0, R0)

    for i in range(N):
        ra_r = ra_rad[i]
        dec_r = dec_rad[i]

        cos_ra = np.cos(ra_r)
        sin_ra = np.sin(ra_r)
        cos_dec = np.cos(dec_r)
        sin_dec = np.sin(dec_r)

        n0 = cos_ra * cos_dec
        n1 = sin_ra * cos_dec
        n2 = sin_dec

        # Cross product of R0 and n
        R0n0 = n1 * R0[2] - n2 * R0[1]
        R0n1 = - (n0 * R0[2] - n2 * R0[0])
        R0n2 = n0 * R0[1] - n1 * R0[0]

        # pmra basis vector
        e10 = -sin_ra
        e11 = cos_ra
        e12 = 0.0

        # pmdec basis vector
        e20 = -cos_ra * sin_dec
        e21 = -sin_ra * sin_dec
        e22 = cos_dec

        # proper motions in rad/s (already converted)
        mu0 = pmra_rad_s[i] * e10 + pmdec_rad_s[i] * e20
        mu1 = pmra_rad_s[i] * e11 + pmdec_rad_s[i] * e21
        mu2 = pmra_rad_s[i] * e12 + pmdec_rad_s[i] * e22

        # DOT products
        dot_V0_R0n = V0[0] * R0n0 + V0[1] * R0n1 + V0[2] * R0n2
        dot_mu_R0n = mu0 * R0n0 + mu1 * R0n1 + mu2 * R0n2

        plx_i = - dot_mu_R0n / dot_V0_R0n  # Units: 1/m

        # Compute eplx
        dot_e1_R0n = e10 * R0n0 + e11 * R0n1 + e12 * R0n2
        dot_e2_R0n = e20 * R0n0 + e21 * R0n1 + e22 * R0n2

        eplx_i = np.sqrt(
            (dot_e1_R0n * epmra_rad_s[i])**2 + (dot_e2_R0n * epmdec_rad_s[i])**2
        ) / dot_V0_R0n  # Units: 1/m

        # Compute other DOT products
        nV0 = V0[0] * n0 + V0[1] * n1 + V0[2] * n2
        nR0 = R0[0] * n0 + R0[1] * n1 + R0[2] * n2
        mun = mu0 * n0 + mu1 * n1 + mu2 * n2
        muR0 = mu0 * R0[0] + mu1 * R0[1] + mu2 * R0[2]

        D = 1.0 / plx_i  # Units: m

        denom = R02 - nR0**2
        VR_i = -1 / denom * (
            (nV0 * R02 - V0R0 * nR0) + D *
            (nV0 * nR0 - V0R0 + mun * R02 - muR0 * nR0) + D**2 *
            (mun * nR0 - muR0)
        )  # Units: m/s

        # Compute VGCR
        numerator = V0R0 + D * muR0 + VR_i * nR0 + D * nV0 + D**2 * mun + D * VR_i
        R = np.sqrt(R02 + D**2 + 2 * D * nR0)
        VGCR_i = numerator / R  # Units: m/s

        plx[i] = plx_i  # Units: 1/m
        eplx[i] = eplx_i  # Units: 1/m
        VGCR[i] = VGCR_i  # Units: m/s
        VR[i] = VR_i  # Units: m/s

    return plx, eplx, VGCR, VR

def getdist_vectorized(ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s,
                  epmra_rad_s, epmdec_rad_s, R0, V0):
    # check if ra_rad is a numpy.float64
    if isinstance(ra_rad, np.float64):
        ra_rad = np.array([ra_rad])
    N = len(ra_rad)
    plx = np.empty(N)
    eplx = np.empty(N)
    VGCR = np.empty(N)
    VR = np.empty(N)

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
    R0n1 = - (n0 * R0[2] - n2 * R0[0])
    R0n2 = n0 * R0[1] - n1 * R0[0]

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

    # DOT products
    dot_V0_R0n = V0[0] * R0n0 + V0[1] * R0n1 + V0[2] * R0n2
    dot_mu_R0n = mu0 * R0n0 + mu1 * R0n1 + mu2 * R0n2

    plx_i = - dot_mu_R0n / dot_V0_R0n  # Units: 1/m

    # Compute eplx
    dot_e1_R0n = e10 * R0n0 + e11 * R0n1 + e12 * R0n2
    dot_e2_R0n = e20 * R0n0 + e21 * R0n1 + e22 * R0n2

    eplx_i = np.sqrt(
        (dot_e1_R0n * epmra_rad_s)**2 + (dot_e2_R0n * epmdec_rad_s)**2
    ) / dot_V0_R0n  # Units: 1/m

    # Compute other DOT products
    nV0 = V0[0] * n0 + V0[1] * n1 + V0[2] * n2
    nR0 = R0[0] * n0 + R0[1] * n1 + R0[2] * n2
    mun = mu0 * n0 + mu1 * n1 + mu2 * n2
    muR0 = mu0 * R0[0] + mu1 * R0[1] + mu2 * R0[2]

    D = 1.0 / plx_i  # Units: m

    denom = R02 - nR0**2
    VR_i = -1 / denom * (
        (nV0 * R02 - V0R0 * nR0) + D *
        (nV0 * nR0 - V0R0 + mun * R02 - muR0 * nR0) + D**2 *
        (mun * nR0 - muR0)
    )  # Units: m/s

    # Compute VGCR
    numerator = V0R0 + D * muR0 + VR_i * nR0 + D * nV0 + D**2 * mun + D * VR_i
    R = np.sqrt(R02 + D**2 + 2 * D * nR0)
    VGCR_i = numerator / R  # Units: m/s

    return plx_i, eplx_i, VGCR_i, VR_i



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


def post_process_results(plx, eplx, VGCR, VR, D, eD):
    """
    Post-process the results from the optimized function

    input:
        plx: parallax in 1/m
        eplx: parallax error in 1/m
        VGCR: VGCR in m/s
        VR: VR in m/s

    output:
        plx_mas: parallax in mas
        eplx_mas: parallax error in mas
        VGCR_kms: VGCR in km/s
        VR_kms: VR in km/s
    """
    # Remove units from plx, eplx, VGCR, and VR
    if isinstance(plx, u.Quantity):
        plx = plx.value
    if isinstance(eplx, u.Quantity):
        eplx = eplx.value

    # check if plx and eplx are 'Series' objects
    if isinstance(plx, pd.Series):
        plx = plx.values
    if isinstance(eplx, pd.Series):
        eplx = eplx.values


    # If this stops working try adding .value after plx and eplx
    try:
        plx_mas = (plx/u.meter).to(1/u.pc, equivalencies=u.parallax())*1e3
        eplx_mas = (eplx/u.meter).to(1/u.pc, equivalencies=u.parallax())*1e3
    
    except:
        print('Unit conversion error in plx and eplx')
        plx_mas = (plx.value / u.meter).to(1/u.pc, equivalencies=u.parallax())*1e3
        eplx_mas = (eplx.value / u.meter).to(1/u.pc, equivalencies=u.parallax())*1e3
    

    # VGCR and VR are in m/s, convert to km/s
    VGCR_kms = VGCR / 1e3
    VR_kms = VR / 1e3

    # D is in meters, convert to pc
    D_pc = D*u.m.to(u.pc)
    eD_pc = eD*u.m.to(u.pc)


    return plx_mas, eplx_mas, VGCR_kms, VR_kms, D_pc, eD_pc

def test_functions():
    # Create test data
    ra_test_deg = np.array([10.684, 56.75])  # degrees
    dec_test_deg = np.array([41.269, 24.116])  # degrees
    pmra_test = np.array([5.25, -3.12])  # mas/yr
    pmdec_test = np.array([-2.16, 4.58])  # mas/yr
    epmra_test = np.array([0.5, 0.3])  # mas/yr
    epmdec_test = np.array([0.4, 0.2])  # mas/yr

    # Convert positions to radians
    ra_rad = np.deg2rad(ra_test_deg)
    dec_rad = np.deg2rad(dec_test_deg)

    # Convert proper motions to radians per second
    masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
    pmra_rad_s = pmra_test * masyr_to_radsec
    pmdec_rad_s = pmdec_test * masyr_to_radsec
    epmra_rad_s = epmra_test * masyr_to_radsec
    epmdec_rad_s = epmdec_test * masyr_to_radsec

    # Compute R0 and V0 in SI units (meters and meters per second)
    R0_SI, V0_SI = compute_R0_V0_SI()

    # Run optimized function
    plx_opt, eplx_opt, VGCR_opt, VR_opt = getdist_numba(
        ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, epmra_rad_s, epmdec_rad_s, R0_SI, V0_SI
    )

    # Convert plx and eplx from 1/m to mas
    
    plx_mas = (plx_opt * u.m**-1).to(1/u.pc, equivalencies=u.parallax())*1e3
    eplx_mas = (eplx_opt * u.m**-1).to(1/u.pc, equivalencies=u.parallax())*1e3

    # VGCR and VR are in m/s, convert to km/s
    VGCR_kms = VGCR_opt / 1e3
    VR_kms = VR_opt / 1e3

    # Print the outputs
    print("Parallax (mas):", plx_mas.value)
    print("Parallax error (mas):", eplx_mas.value)
    print("VGCR (km/s):", VGCR_kms)
    print("VR (km/s):", VR_kms)

    # # compare time it takes to run the optimized function
    # plx, eplx, vgcr,  vr = getdist(ra_test_deg, dec_test_deg, pmra_test, pmdec_test, epmra_test, epmdec_test)

    # print("Previous Implementation:")
    # print('plx:', plx)
    # print('eplx:', eplx)
    # print('vgcr:', vgcr)
    # print('vr:', vr)

def implied_calculations(data):
    # Convert positions to radians
    ra_rad = np.deg2rad(data['ra'])
    dec_rad = np.deg2rad(data['dec'])

    # Convert proper motions to radians per second
    masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
    pmra_rad_s = data['pmra'] * masyr_to_radsec
    pmdec_rad_s = data['pmdec'] * masyr_to_radsec
    epmra_rad_s = data['pmra_error'] * masyr_to_radsec
    epmdec_rad_s = data['pmdec_error'] * masyr_to_radsec

    # Compute R0 and V0 in SI units (meters and meters per second)
    #print('Computing R0 and V0...')
    R0_SI, V0_SI = compute_R0_V0_SI()

    # Run optimized function
    #print('Computing distances and velocities...')
    # plx_opt, eplx_opt, VGCR_opt, VR_opt = getdist_vectorized(
    #     ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, epmra_rad_s, epmdec_rad_s, R0_SI, V0_SI
    # )

    plx_opt, eplx_opt, VGCR_opt, VR_opt, D_i, D_i_error = getdist_vectorized2(
    ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, epmra_rad_s, epmdec_rad_s, R0_SI, V0_SI
    )
    #print('Distances and velocities computed successfully!')
    # Post-process the results
    plx_mas, eplx_mas, VGCR_kms, VR_kms, D_pc, eD_pc = post_process_results(plx_opt, eplx_opt, 
                                                                            VGCR_opt, VR_opt, 
                                                                            D_i, D_i_error)

    # Save the results to a new table
    data['implied_parallax'] = plx_mas #mas
    data['implied_parallax_error'] = eplx_mas #mas
    data['VGCR'] = VGCR_kms
    data['VR'] = VR_kms
    data['implied_distance'] = D_pc
    data['implied_distance_error'] = eD_pc

    return data

def implied_calculations_single(ra, dec, pmra, pmdec, pmra_error, pmdec_error):
    # Convert positions to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Convert proper motions to radians per second
    masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
    pmra_rad_s = pmra * masyr_to_radsec
    pmdec_rad_s = pmdec * masyr_to_radsec
    epmra_rad_s = pmra_error * masyr_to_radsec
    epmdec_rad_s = pmdec_error* masyr_to_radsec

    # Compute R0 and V0 in SI units (meters and meters per second)
    #print('Computing R0 and V0...')
    R0_SI, V0_SI = compute_R0_V0_SI()

    # Run optimized function
    #print('Computing distances and velocities...')
    plx_opt, eplx_opt, VGCR_opt, VR_opt = getdist_vectorized(
        ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, epmra_rad_s, epmdec_rad_s, R0_SI, V0_SI
    )
    #print('Distances and velocities computed successfully!')
    # Post-process the results
    plx_mas, eplx_mas, VGCR_kms, VR_kms = post_process_results(plx_opt, eplx_opt, VGCR_opt, VR_opt)

    return plx_mas, eplx_mas, VGCR_kms, VR_kms


if __name__ == "__main__":
    #test_functions()

    # set paths
    path_data = '/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/200pc/raw_gaia_catalog/nearby_200pc-result_smaller.fits' # Update with the path to the data file (e.g., Gaia data)
    path_output = '/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/200pc/implied_D_v_r' # Update with the path to the output file

    # Read the data
    print('Reading data...')
    data = Table.read(path_data)
    print('Data read successfully!')

    # Convert positions to radians
    ra_rad = np.deg2rad(data['ra'])
    dec_rad = np.deg2rad(data['dec'])

    # Convert proper motions to radians per second
    masyr_to_radsec = (1 * u.mas / u.yr).to(u.rad / u.s).value
    pmra_rad_s = data['pmra'] * masyr_to_radsec
    pmdec_rad_s = data['pmdec'] * masyr_to_radsec
    epmra_rad_s = data['pmra_error'] * masyr_to_radsec
    epmdec_rad_s = data['pmdec_error'] * masyr_to_radsec

    # Compute R0 and V0 in SI units (meters and meters per second)
    print('Computing R0 and V0...')
    R0_SI, V0_SI = compute_R0_V0_SI()

    # Run optimized function
    print('Computing distances and velocities...')
    plx_opt, eplx_opt, VGCR_opt, VR_opt = getdist_vectorized(
        ra_rad, dec_rad, pmra_rad_s, pmdec_rad_s, epmra_rad_s, epmdec_rad_s, R0_SI, V0_SI
    )
    print('Distances and velocities computed successfully!')
    # Post-process the results
    plx_mas, eplx_mas, VGCR_kms, VR_kms = post_process_results(plx_opt, eplx_opt, VGCR_opt, VR_opt)

    # Save the results to a new table
    data['implied_parallax'] = plx_mas
    data['implied_parallax_error'] = eplx_mas
    data['VGCR'] = VGCR_kms
    data['VR'] = VR_kms

    # Write the output to a new file
    print('Writing output...')
    data.write(os.path.join(path_output, 'implied_distance_v_r.fits'), format='fits', overwrite=True)
    print('Output written successfully!')
    