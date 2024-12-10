__author__ = 'Fraser Evans'
__version__ = '2.0'

__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

import time

#try:
import os
from astropy import units as u
import numpy as np
    #import scanninglaw.asf as astrospreadfunc
#except ImportError:
#    raise ImportError(__ImportError__)

from .eject import EjectionModel
from .eject import HillsfromCatalog
#from .utils.mwpotential import PotDiff

class starsample:

    from .dynamics import propagate, backprop, get_vesc, _R, likelihood
    from .observation import photometry, zero_point, get_Punbound, get_P_velocity_greater, evolve, get_Gaia_errors, get_e_beta, photometry_brutus
    from .saveload import save, _load, _loadExt, saveall, loadall
    from .subsample import subsample
    from .config import fetch_dust, config_dust, config_astrosf, config_rvssf,                          set_ast_sf, set_Gaia_release
    '''
    HVS sample class. Main features:

    - Generate a sample of HVS with a specified ejection model
    - Propagate the ejection sample in the Galaxy
    - Obtain mock photometry for the sample in a variety of bands
    - Obtain mock astrometric and radial velocity errors
    - Subsample the population according to various criteria
    - Save/Load catalogues FITS files

   # Common attributes
    ---------
    self.size : int
        Size of the sample
    self.name : str
        Catalog name, 'Unknown'  by default
    self.ejmodel_name : str
        String of the ejection model used to generate the sample.
       'Unknown' by default
    self.dt : Quantity
        Timestep used for orbit integration, 0.01 Myr by default
    self.T_MW : Quantity
        Milky Way maximum lifetime

    self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0
        Initial configuration at ejection in galactocentric 
        spherical coordinates
    self.tage, self.tflight : Quantity
        Age and flight time of the stars
    self.m
        Stellar masses of the sample
    self.a, self.P, self.q : Quantity/Quantity/Float
        Orbital separation, orbital period and mass ratio of HVS
        progenitor binary
    self.met : Float
        Metallicity xi = log10(Z/Z_sun)
    self.stage, self.stagebefore : integer
        Evolutionary stage of the star _today_ and at the moment of ejection
        see for the meanings of each stage:
        https://ui.adsabs.harvard.edu/abs/2000MNRAS.315..543H/abstract      
    self.Rad, self.T_eff, self.Lum : Quantity
        Radius, temperature and luminosity

    self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos : Quantity
        ICRS positions and velocities. 
        Proper motion in ra direction is declination-corrected.
    self.par : Quantity
        Parallax
    self.e_ra, self.e_dec, self.e_par : Quantity
        Position and parallax uncertainties
    self.e_pmra, self.e_pmdec, self.e_vlos : Quantity
        Proper motion and line-of-sight velocity uncertainties

    self.x, self.y, self.z, self.vx, self.vy, self.vz : Quantity
        Positions and velocities in Galactocentric Cartesian coordinates
    self.GCdist, self.GCv : Quantity
        Galactocentric total distance and total velocity
    self.e_GCv : Quantity
        Standard deviation of the star's total velocity, given
        observational errors
    self.thetaf, self.phif : Float
        Polar (+z towards xy) and azimuthal (+x towards +y) coordinates in 
        Galactocentric spherical coordinates

    self.Pub : Float
        Probability, given the star's observational error, that it will be
        it is fast enough to escape the Galaxy

    self.Vesc : Quantity
        Escape velocity to infinity from the star's current position
    

    self.GRVS, self.G, self.RP, ... : ndarray
        Apparent magnitude in many photometric bands

    Methods
    -------
    __init__():
        Initializes the class: loads catalog if one is provided, 
        otherwise creates one based on a given ejection model
    _eject(): 
        Initializes a sample at t=0 ejected from Sgr A*
    backprop():
        Propagates a sample backwards in time for a given max integration time
    propagate():
        Propagates a sample forewards in time for a given max integrating time
    get_vesc():
        Determines the escape velocity from the Galaxy for each mock HVS 
    photometry():
        Calculates the apparent magnitudes in several different bands.
        Can also calculate astrometric and radial velocity uncertainties
    get_P_unbound():
        Samples over observational errors,
        determines unbound probabilities depending on errors
    get_P_velocity_greater():
        Samples over observational errors, determines probabilities that mock
        HVSs are faster than a specified velocity cutoff
    subsample():
        For a propagated sample, returns a subsample - either specific indices,
        a random selection of stars or ones that meet given criteria
    save():
        Saves the sample in a FITS file
    _load():
        Load existing HVSsample, either ejected or propagated
    loadExt():
        If a sample was NOT created here, reads it in as an HVSsample
    likelihood():
        Checks the likelihood of the sample for a given potential
    zero_point():
        Estimates the Gaia parallax zero point for each star
    '''

    dt   = 0.01*u.Myr
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015

    #@init
    def __init__(self, inputdata=None, name=None, isExternal=False,**kwargs):

        '''
        Parameters
        ----------
        inputdata : EjectionModel or str
            Instance of an ejection model or string to the catalog path
        name : str
            Name of the catalog
        isExternal : Bool
            Flag if the loaded catalog was externally generated, 
            i.e. not by this package
        '''

        if(inputdata is None):
            raise ValueError('Initialize the class by either providing an \
                                ejection model or an input HVS catalog.')

        #Name catalog
        if(name is None):
            self.name = 'HVS catalog '+str(time.time())
        else:
            self.name = name

        #When Gaia errors are calculated, default Gaia data release is assumed
        #to be Gaia DR4. This can be changed with 
        #speedystar.config.set_Gaia_release()

        self.Gaia_release = 'DR4'

        #By default, astrometric errors are NOT calculated using the actual
        #Gaia astrometric spread function, but rather with PyGaia based on 
        #pre-launch predicted Gaia performance. Setting use_ast_sf to True with 
        #speedystar.config.set_ast_sf() calculates astrometric errors using 
        #the Gaiaunlimited package, which is computationally more expensive 
        #but is more accurate, particularly for bright sources.

        self.use_ast_sf = False

        #If inputdata is ejection model, create new ejection sample
        if isinstance(inputdata, EjectionModel):
            self._eject(inputdata, **kwargs)

        # If inputdata is a filename and isExternal=True, 
        # loads existing sample of star observations    
        if(isinstance(inputdata, str) and (isExternal)):
            self._loadExt(inputdata)

        #If inputdata is a filename and isExternal=False, 
        # loads existing already-propagated sample of stars
        if (isinstance(inputdata, str) and (not isExternal)):
            self._load(inputdata,**kwargs)

    def __getitem__(self,item):
        #print(item)
        self.subsample(np.where(item)[0])
        return self

    #@eject
    def _eject(self, ejmodel, **kwargs):

        '''
        Initializes the sample as an ejection sample
        '''

        self.ejmodel_name = ejmodel._name

        self.propagated = False

        # See ejmodel.sampler() for descriptions of returned attributes
        #self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
        #self.m, self.tage, self.tflight, self.a, self.P, self.q, self.mem, \
        #self.met, self.stage, self.stagebefore, self.Rad, self.T_eff, \
        #self.Lum, self.size = ejmodel.sampler(**kwargs)
        
        ejargs = ejmodel.sampler(**kwargs)

        for key in list(ejargs.keys()):
            setattr(self,key,ejargs[key])
