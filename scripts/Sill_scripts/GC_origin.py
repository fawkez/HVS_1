import os
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table

class constant:
    masyr = u.mas / u.year
    radyr = u.rad / u.year
    kms = u.km / u.s
    mult = (1 * masyr).to_value(radyr) / u.year
    inv_kpc = 1 / u.kpc

def getdist(ra, dec, pmra, pmdec, epmra=0, epmdec=0):
    """ Get distance assuming GC origin
    Return
    1) plx (mas)
    2) plx uncertainty (mas),
    3) implied galactocentric radial velocity (km/s)
    4) Implied radial velocity (km/s)
    """
    # GC position in rectangular ICRS system
    X0 = coord.Galactocentric(x=0 * u.kpc,
                             y=0 * u.kpc,
                             z=0 * u.kpc,
                             v_x=0 * constant.kms,
                             v_y=0 * constant.kms,
                             v_z=0 * constant.kms).transform_to(
                                 coord.ICRS()).represent_as(
                                     coord.CartesianRepresentation)
    DX0 = X0.differentials['s']

    R0 = -u.Quantity([X0.x, X0.y, X0.z])
    # vector pointing from GC to the Sun in the ICRS rectangular system

    V0 = -u.Quantity([DX0.d_x, DX0.d_y, DX0.d_z])
    # Sun's velocity vector in the ICRS rectangular system

    # Convert sky position to units of rad
    ra_r = np.deg2rad(ra)
    dec_r = np.deg2rad(dec)

    # vector pointing to the star
    n = [
        np.cos(ra_r) * np.cos(dec_r),
        np.sin(ra_r) * np.cos(dec_r),
        np.sin(dec_r)
    ]

    # cross product of R0 and n
    R0n = u.Quantity([
        n[1] * R0[2] - n[2] * R0[1], -(n[0] * R0[2] - n[2] * R0[0]),
        n[0] * R0[1] - n[1] * R0[0]
    ])

    # pmra basis vector
    e1 = [-np.sin(ra_r), np.cos(ra_r), 0]
    # pmdec basis vector
    e2 = [
        -np.cos(ra_r) * np.sin(dec_r), -np.sin(ra_r) * np.sin(dec_r),
        np.cos(dec_r)
    ]

    pmra_u = pmra * constant.mult
    pmdec_u = pmdec * constant.mult
    epmra_u = epmra * constant.mult
    epmdec_u = epmdec * constant.mult
    # proper motions and errors in year^-1 (ie rad/year)
    # I ignore correlations in errors

    mu = [
        pmra_u * e1[0] + pmdec_u * e2[0],
        pmra_u * e1[1] + pmdec_u * e2[1],
        pmra_u * e1[2] + pmdec_u * e2[2]
    ]

    # proper motion vector in rectangular ICRS system

    def DOT(x, y):
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]

    # here we use the fact that det(ABC) = A . (B x C)
    # and that plx = -det(mu R0 n)/det(V0 R0 n)
    dot_V0_R0n = DOT(V0, R0n)
    plx = -DOT(mu, R0n) / dot_V0_R0n
    eplx = np.sqrt(
        DOT(e1, R0n)**2 * epmra_u**2 + DOT(e2, R0n)**2 * epmdec_u**2) / \
                                                                    dot_V0_R0n

    R02 = DOT(R0, R0)
    V0R0 = DOT(V0, R0)
    nV0 = DOT(V0, n)
    nR0 = DOT(R0, n)
    mun = DOT(mu, n)
    muR0 = DOT(mu, R0)

    D = 1. / plx
    # This is just the result of some algebra of first
    # cross producting the R x V
    # and then dot product the result with (n x R0)
    VR = -1 / (R02 - nR0**2) * (
        (nV0 * R02 - V0R0 * nR0) + D *
        (nV0 * nR0 - V0R0 + mun * R02 - muR0 * nR0) + D**2 *
        (mun * nR0 - muR0))

    # dot product of (V . R)/||R||
    # This is galactocentric radial velocity (total velocity with the sign)
    VGCR = (V0R0 + D * muR0 + VR * nR0 + D * nV0 + D**2 * mun +
            D * VR) / np.sqrt(R02 + D**2 + 2 * D * nR0)
    # Vtot = np.sqrt((V0[0] + VR * n[0] + mu[0] * D)**2 +
    #               (V0[1] + VR * n[1] + mu[1] * D)**2 +
    #               (V0[2] + VR * n[2] + mu[2] * D)**2)
    plx = plx.to_value(constant.inv_kpc)
    eplx = eplx.to_value(constant.inv_kpc)
    return plx, eplx, VGCR.to_value(constant.kms), VR.to_value(constant.kms)

def main():
    #Path to the file with the raw data downloaded from Gaia (in fits table)
    mypath = '/path/to/file'
    data = fits.getdata(mypath)

    plx, eplx, VGCR, VR = getdist(data['ra'],
                                    data['dec'],
                                    data['pmra'],
                                    data['pmdec'],
                                    data['pmra_error'],
                                    data['pmdec_error'])

    #Save the results to new table
    t = Table([data['source_id'], data['ra'], data['dec'],
            data['phot_g_mean_mag'], data['phot_bp_mean_mag'],
            data['phot_rp_mean_mag'], data['parallax'],
            data['parallax_error'], plx, eplx,  VGCR, VR],
            names=('source_id', 'ra', 'dec', 'phot_g_mean_mag',
            'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax',
            'parallax_error', 'implied_parallax', 'implied_parallax_error',
            'VGCR', 'VR'))
    t.write('file_name.fits', format='fits', overwrite=True)

if __name__ == '__main__':
    main()
