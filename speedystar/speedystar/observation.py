PackageError = "One or more required external packages have not been installed. Please see requirements.txt"

PunboundAstrometryError = 'Computing unbound probabilities requires full equatorial positions and velocities (ra, dec, parallax/distance, pmra, pmdec, vlos). Please make sure your sample includes these attributes.'

asfWarning = 'Gaia G band apparent magnitude must be known to compute Gaia astrometric spread function. Calculating...'

PunboundUncertaintyWarning = 'Computing unbound probabilities requires uncertainties on positions and velocities in the equatorial frame. Calculating...'

ErrorWarning = 'Warning: One or more requested error component not yet implemented. Only available error options are e_ra, e_dec, e_par, e_pmra, e_pmdec, e_vlos. See speedystar.observation.get_Gaia_errors() docstring'

releaseError = 'Error: invalid Gaia data release. Options are DR2, EDR3, DR3, DR4, DR5. See speedystar.observation.get_Gaia_errors() docstring'

radecError = 'Error: right ascension and declination of sources must be known. Please ensure "ra" and "dec" are attributes of your sample.'

photoattrError = 'All of the following attributes are required as astropy quantities to compute mock photometry for the sample: distance "dist", mass "m", metallicity "met", effective temperate "T_eff", stellar radius "Rad", luminosity "Lum". Please ensure all these attributes exist.'

propagatedWarning = 'Warning: sample appears not to have been propagated. Proceeding with mock photometry regardless'

SolarMotionWarning = 'Warning: UVW Solar motion not provided. Defaulting to [-11.1, 12.24, 7.25]*u.km/u.s (Schonrich+2010)'

roWarning = 'Warning: ro not provided. Defaulting to value in galpy config file.'

voWarning = 'Warning: vo not provided. Defaulting to value in galpy config file'

zoWarning = 'Warning: zo not provided. Defaulting to 20.8 pc (Bovy & Bennett 2019)'

metWarning = 'WARNING: speedystar mock magnitudes are unreliable for stars with metallicities outside the range [-0.3, +0.3]. Some stars outside this range are present in the sample. Proceed with caution.'

vcutError = 'Error: Invalid velocity cut in get_P_velocity_greater(). Supplied cut must be either float (assumed to be km/s) or an astropy velocity quantity'

astroErrorError = 'Error: sample needs an attribute "e_vlos" (line-of-sight velocity error) AND EITHER an attribute "cov" (astrometric covariance matrix) or all of the attributes "e_par", "e_pmra", "e_pmdec" to call this function. See speedystar.observation.get_Gaia_errors for more information'

astroErrorWarning = 'Warning: Some stars without valid errors are included in the sample, most likely because they are too dim. They are being removed'

vlosError = 'Error: Effective temperature attribute is required to compute radial velocity error. Please see speedystar.utils.evolve.evolve for more information.'
 
_Gaia_releases = ['DR2', 'EDR3', 'DR3', 'DR4', 'DR5']

_Gaia_errors = ['e_ra', 'e_dec', 'e_par', 'e_pmra', 'e_pmdec', 'e_vlos']

from astropy import units as u
import numpy as np
from tqdm import tqdm
#from galpydev.galpy.util.conversion import get_physical
from galpy.util.conversion import get_physical

def get_Gaia_errors(self, errors = ['e_par', 'e_pmra', 
                                    'e_pmdec', 'e_vlos']):

    global dr3sf, dr3astsf, dr3rvssf, dr3rvssfvar
    global dr2sf, dr2astsf, dr2rvssf, dr2rvssfvar, dr2rvssf2

    from .utils.selectionfunctions.source import Source as rSource
    from .utils.MIST_photometry import get_e_vlos
    #from .utils.selectionfunctions import cog_v as CoGV

    #from gaiaunlimited.selectionfunctions import DR2SelectionFunction, DR3RVSSelectionFunction, DR3SelectionFunctionTCG
    #from gaiaunlimited.utils import get_healpix_centers
    #from gaiaunlimited.selectionfunctions import EDR3RVSSelectionFunction

    '''
    Calculates mock Gaia astrometric and radial velocity errors depending on
    the user-specified data release. 

    Parameters
    ----------
    errors: List of strings
        The Gaia errors to calculate. 
        - Options include:
            - e_ra -- Error in right ascension (uas)
            - e_dec -- Error in declination (uas)
            - e_par -- parallax error (mas)
            - e_pmra, e_pmdec -- predicted proper motion in the 
                    ra (cosdec corrected) and dec directions (mas/yr)
            - e_vlos -- Radial velocity error (km/s)
            - NOTE: errors are computed agnostic of whether or not Gaia 
                could actually detect a given source. Recall that the 
                faint-end magnitude limit of the Gaia astrometric catalogue 
                is G ~ 21 and G_RVS ~ 16.2 for the radial velocity catalogue.

    '''

    #Check to make sure supplied data release is valid option
    #if data_release not in _Gaia_releases:
    #    raise ValueError(releaseError)

    #Check to make sure the requested errors are valid options
    if len([i for i in errors if i not in _Gaia_errors]) > 0:
        print(ErrorWarning)

    errors = [err for err in errors if err in _Gaia_errors]

    #Check to make sure 

    _position_scale = {'DR5': 0.7, 'DR4': 1.0, 'DR3': 1.335, 
                                'EDR3': 1.335, 'DR2': 1.7}
    _propermotion_scale = {'DR5': 0.7*0.5, 'DR4': 1.0, 'DR3': 1.335*1.78, 
                                'EDR3': 1.335*1.78, 'DR2': 4.5}
    _vrad_scale = {'DR5': 0.707, 'DR4': 1.0, 'DR3': 1.33 , 
                                'EDR3': 1.65, 'DR2': 1.65}

    if self.size>0:
        if self.use_ast_sf:
            from scanninglaw.source import Source as aSource
            import scanninglaw.asf as astrospreadfunc
            if 'dr2astsf' not in globals():
                #Load in DR2 astrometric spread function
                dr2astsf = astrospreadfunc.asf() 

            if 'dr3astsf' not in globals():
                #Load in DR3 astrometric spread function
                dr3astsf = astrospreadfunc.asf(version='dr3_nominal') 

            if not ( hasattr(self,'Gaia_G') ):
                print(asfWarning)
                self.photometry(bands=['Gaia_G'],errors=errors)

            if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                raise ValueError(asfError)

            _which_ast_sf = {'DR2': dr2astsf, 'EDR3': dr3astsf, 
                            'DR3':dr3astsf, 'DR4': dr3astsf, 'DR5': dr3astsf}

            #Position and brightness of each star
            asource = aSource(self.ra,self.dec,frame='icrs',
                                photometry={'gaia_g':self.Gaia_G})

            #Calling the astrometric spread function. 
            # Gives the 5x5 covariance matrix, whose diagonal
            # elements are the variances of ra/dec/par/pmra/pmdec
            #self.cov = dr2astsf(asource)
            self.cov = _which_ast_sf[self.Gaia_release](asource)

            if self.Gaia_release == 'DR4':
                self.cov[0,0,:] /= _position_scale['DR3']
                self.cov[1,1,:] /= _position_scale['DR3']
                self.cov[2,2,:] /= _position_scale['DR3']
                self.cov[3,3,:] /= _propermotion_scale['DR3']
                self.cov[4,4,:] /= _propermotion_scale['DR3']
            elif self.Gaia_release == 'DR5':
                self.cov[0,0,:] /= ( _position_scale['DR3']
                                    / _position_scale['DR5'])
                self.cov[1,1,:] /= ( _position_scale['DR3']
                                    / _position_scale['DR5'])
                self.cov[2,2,:] /= ( _position_scale['DR3']
                                    / _position_scale['DR5'])
                self.cov[3,3,:] /= ( _propermotion_scale['DR3']
                                    / _propermotion_scale['DR5'])
                self.cov[4,4,:] /= ( _propermotion_scale['DR3'] 
                                    / _propermotion_scale['DR5'] )

            #print(self.cov[:,:,0])
            #assign astrometric error
            self.e_ra = np.sqrt(self.cov[0,0])*u.uas
            #print(self.e_ra[0])
            self.e_dec = np.sqrt(self.cov[1,1])*u.uas
            self.e_par   = np.sqrt(self.cov[2,2])*u.mas
            self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
            self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

            if hasattr(self,'e_vlos'):
                self.e_vlos *= _vrad_scale[self.Gaia_release]
            elif 'e_vlos' in errors:
                if not hasattr(self,'T_eff'):
                    raise ValueError(vlosError)
                if hasattr(self, 'Bessell_V'):
                    self.e_vlos = get_e_vlos(self.Bessell_V,self.T_eff.value)
                else:
                    self.photometry(bands=['Bessell_V'], errors=['e_vlos'])
        else:

            _which_error = {'e_ra': _position_scale, 'e_dec': _position_scale, 
                        'e_par': _position_scale, 
                        'e_pmra': _propermotion_scale, 
                        'e_pmdec': _propermotion_scale, 
                        'e_vlos': _vrad_scale}
 
            #for err in errors:
            #    print(err)
            #    print(hasattr(self,err))

            if all(hasattr(self,err) for err in errors):       
                for err in errors:
                    setattr(self, err, getattr(self,err) 
                                    * _which_error[err][self.Gaia_release])
            else:
                self.photometry(errors=[err for err in errors 
                                    if err not in ['e_ra', 'e_dec']]) 

#@photometry
def photometry(self, bands=['Bessell_V', 'Bessell_I', 
                   'Gaia_GRVS', 'Gaia_G', 'Gaia_BP', 'Gaia_RP'],
                   errors = ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos'], data_release='DR4'):

    '''
    Computes mock apparent magnitudes in the Gaia bands (and also others).
    Also calculates mock DR4 astrometric errors using pygaia. 
    These may or may not be overwritten later (see subsample()).

    Parameters
    ----------
    dustmap : DustMap
        Dustmap object to be used
    bands: List of strings
        The photometric bands in which apparent magnitudes are calculated. 
        Names are more or less self-explanatory. Options for now include:
        - Bessell_U, Bessell_B, Bessell_V, Bessell_R, Bessell_I 
          Johnson-Cousins UBVRI filters (Bessell 1990)
        - Gaia_G, Gaia_BP, Gaia_RP, Gaia_GRVS bands
            - NOTE: Only EDR3 bands are currently implemented in MIST. DR3 
              bands are available from Gaia and this code will be updated 
              when DR3 bands are implemented in MIST.
            - NOTE as well: This subroutine calculates G_RVS magnitudes not 
              using the G_RVS transmission curve directly but by a power-law 
              fit using the Bessell_V, Bessell_I and Gaia_G filters 
              (Jordi et al. 2010). Transmission curve was not available prior 
              to Gaia DR3 and is not yet implemented in MIST.
        - VISTA Z, Y, J, H, K_s filters 
        - DECam u, g, r, i, z, Y filters 
        - LSST u, g, r, i, z, y filters
    errors: List of strings
        The Gaia errors to calculate. 
        Fairly inexpensive if you are already calculating Bessell_I, 
        Bessell_V, Gaia_G.
        - Options include:
            - e_par -- DR4 predicted parallax error (mas)
            - e_pmra, e_pmdec -- DR4 predicted proper motion in the 
                    ra (cosdec corrected) and dec directions (mas/yr)
            - e_vlos -- DR4 predicted radial velocity error (km/s)
            - NOTE: errors are computed agnostic of whether or not Gaia 
                could actually detect a given source. Recall that the 
                faint-end magnitude limit of the Gaia astrometric catalogue 
                is G ~ 21 and G_RVS ~ 16.2 for the radial velocity catalogue.
            - NOTE: These error calculations are computationally inexpensive 
                but not the most accurate, particularly for bright sources. 
                Get_Gaia_errors() is slow but more robustly simulates the 
                Gaia astrometric performance 

    '''

    from galpy.util.coords import radec_to_lb
    #from .utils.dustmap import DustMap
    from .utils.MIST_photometry import get_Mags
    import mwdust

    if not hasattr(self,'dust'):
        print('Warning: A dust map is not provided. Attempting to default '\
                'to Combined15')
        try:
            self.dust = mwdust.Combined15()
        except ValueError:
            raise ValueError('Default dust map could not be loaded. See'\
                    'myexample.py ' \
                    'or https://github.com/jobovy/mwdust for more information.' \
                    ' Call speedystar.config.fetch_dust() to download dust ' \
                    'map(s) and set DUST_DIR environment variable.')

    if (not hasattr(self,'propagated') or not self.propagated):
        print(propagatedWarning)

    #Needs galactic lat/lon to get G_RVS 
    # converts to it if only equatorial are available
    if(hasattr(self,'ll')):
            l = self.ll
            b = self.bb
    elif(hasattr(self,'ra')):
        data = radec_to_lb(self.ra.to('deg').value, 
                                self.dec.to('deg').value, degree=True)
        l, b = data[:, 0], data[:, 1]
    else:
        raise ValueError('RA/Dec or Galactic lat/lon are required to perform'\
                                'mock photometry. Please check your sample.')           
    if not hasattr(self,'Av'):
        self.Av = None

    if not all(hasattr(self,attr) 
            for attr in ['dist', 'm', 'met', 'T_eff', 'Rad', 'Lum']):
        raise ValueError(photoattrError)

    if any(self.met>0.3) or any(self.met<-0.3):
        print(metWarning)

    if(self.size==0):
        self.Av, self.e_par, self.e_pmra, self.e_pmdec, self.e_vlos \
            = ([] for i in range(5))
        for band in bands:
            setattr(self,band,[])
    else:

        # Calculates visual extinction, apparent magnitudes in 
        # appropriate bands, and errors 
        self.Av, Mags, errs = get_Mags(self.Av, self.dist.to('kpc').value, l,
                                       b, self.m.to('Msun').value, self.met,
                                       self.T_eff.value, self.Rad.value, 
                                       self.Lum.value, self.dust, bands, errors)

        #Sets attributes
        for band in bands:
            setattr(self, band, Mags[band]*u.dimensionless_unscaled)
        for err in errors:
            setattr(self, err, errs[err]) 
        if(hasattr(self,'e_par')):         
            self.e_par = self.e_par * u.mas
        if(hasattr(self,'e_pmra')):         
            self.e_pmra = self.e_pmra * u.mas / u.yr
        if(hasattr(self,'e_pmdec')):         
            self.e_pmdec = self.e_pmdec * u.mas /u.yr
        if(hasattr(self,'e_vlos')):         
            self.e_vlos = self.e_vlos * u.km/u.s

        if len([i for i in errors if i not in _Gaia_errors]) > 0:
            print(ErrorWarning)

        if len([i for i in errors if i in _Gaia_errors]) > 0:
            self.get_Gaia_errors(errors)

#@photometry
def photometry_brutus(self, bands=['Bessell_V', 'Bessell_I', 
                   'Gaia_GRVS', 'Gaia_G', 'Gaia_BP', 'Gaia_RP'],
                   errors = ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos'], data_release='DR4'):

    '''
    Computes mock apparent magnitudes in the Gaia bands (and also others).
    Also calculates mock DR4 astrometric errors using pygaia. 
    These may or may not be overwritten later (see subsample()).

    Parameters
    ----------
    dustmap : DustMap
        Dustmap object to be used
    bands: List of strings
        The photometric bands in which apparent magnitudes are calculated. 
        Names are more or less self-explanatory. Options for now include:
        - Bessell_U, Bessell_B, Bessell_V, Bessell_R, Bessell_I 
          Johnson-Cousins UBVRI filters (Bessell 1990)
        - Gaia_G, Gaia_BP, Gaia_RP, Gaia_GRVS bands
            - NOTE: Only EDR3 bands are currently implemented in MIST. DR3 
              bands are available from Gaia and this code will be updated 
              when DR3 bands are implemented in MIST.
            - NOTE as well: This subroutine calculates G_RVS magnitudes not 
              using the G_RVS transmission curve directly but by a power-law 
              fit using the Bessell_V, Bessell_I and Gaia_G filters 
              (Jordi et al. 2010). Transmission curve was not available prior 
              to Gaia DR3 and is not yet implemented in MIST.
        - VISTA Z, Y, J, H, K_s filters 
        - DECam u, g, r, i, z, Y filters 
        - LSST u, g, r, i, z, y filters
    errors: List of strings
        The Gaia errors to calculate. 
        Fairly inexpensive if you are already calculating Bessell_I, 
        Bessell_V, Gaia_G.
        - Options include:
            - e_par -- DR4 predicted parallax error (mas)
            - e_pmra, e_pmdec -- DR4 predicted proper motion in the 
                    ra (cosdec corrected) and dec directions (mas/yr)
            - e_vlos -- DR4 predicted radial velocity error (km/s)
            - NOTE: errors are computed agnostic of whether or not Gaia 
                could actually detect a given source. Recall that the 
                faint-end magnitude limit of the Gaia astrometric catalogue 
                is G ~ 21 and G_RVS ~ 16.2 for the radial velocity catalogue.
            - NOTE: These error calculations are computationally inexpensive 
                but not the most accurate, particularly for bright sources. 
                Get_Gaia_errors() is slow but more robustly simulates the 
                Gaia astrometric performance 

    '''

    from galpy.util.coords import radec_to_lb
    from brutus2.brutus.seds import SEDmaker
    from brutus2.brutus import filters
    #from .utils.dustmap import DustMap
    from .utils.MIST_photometry import get_Mags
    import mwdust

    if not hasattr(self,'dust'):
        print('Warning: A dust map is not provided. Attempting to default '\
                'to Combined15')
        try:
            self.dust = mwdust.Combined15()
        except ValueError:
            raise ValueError('Default dust map could not be loaded. See'\
                    'myexample.py ' \
                    'or https://github.com/jobovy/mwdust for more information.' \
                    ' Call speedystar.config.fetch_dust() to download dust ' \
                    'map(s) and set DUST_DIR environment variable.')

    if (not hasattr(self,'propagated') or not self.propagated):
        print(propagatedWarning)

    #Needs galactic lat/lon to get G_RVS 
    # converts to it if only equatorial are available
    if(hasattr(self,'ll')):
            l = self.ll
            b = self.bb
    elif(hasattr(self,'ra')):
        data = radec_to_lb(self.ra.to('deg').value, 
                                self.dec.to('deg').value, degree=True)
        l, b = data[:, 0], data[:, 1]
    else:
        raise ValueError('RA/Dec or Galactic lat/lon are required to perform'\
                                'mock photometry. Please check your sample.')           
    if not hasattr(self,'Av'):
        self.Av = None

    if not all(hasattr(self,attr) 
            for attr in ['dist', 'm', 'met', 'T_eff', 'Rad', 'Lum']):
        raise ValueError(photoattrError)

    if any(self.met>0.3) or any(self.met<-0.3):
        print(metWarning)

    if(self.size==0):
        self.Av, self.e_par, self.e_pmra, self.e_pmdec, self.e_vlos \
            = ([] for i in range(5))
        for band in bands:
            setattr(self,band,[])
    else:

        # Calculates visual extinction, apparent magnitudes in 
        # appropriate bands, and errors 

        filts = np.array(filters.FILTERS)
        brutussed = SEDmaker(mistfile='/data1/Cats/brutus_cats/MIST_1.2_EEPtrk.h5',nnfile='/data1/Cats/brutus_cats/nn_c3k.h5')

        self.Av = self.dust(l, b, self.dist.to(u.kpc).value) * 2.682

        mags = np.zeros( (self.size, len(filters.FILTERS)) )

        for i in tqdm(range(self.size)):
    
            eep = brutussed.get_eep(loga = np.log10(self.tage[i].to('yr').value), eep = 300., mini = self.m[i].value, feh = self.met[i])

            mags[i,] = brutussed.get_sed(mini = self.m[i].value, eep=eep, feh=self.met[i], afe=0., av=self.Av[i], dist=self.dist[i].to('pc').value)[0]

            #self.Av, Mags, errs = get_Mags(self.Av, self.dist.to('kpc').value, l,
            #                               b, self.m.to('Msun').value, self.met,
            #                               self.T_eff.value, self.Rad.value, 
            #                               self.Lum.value, self.dust, bands, errors)

        #newG = np.append(newG,mags[0][0])
        #newRP = np.append(newRP,mags[0][2])
        #newV = np.append(newV,mags[0][np.where(filts=='Bessell_V')[0]])
        #newI = np.append(newI,mags[0][np.where(filts=='Bessell_I')[0]])
        #newg = np.append(newg,mags[0][np.where(filts=='DECam_g')[0]])
        #newr = np.append(newr,mags[0][np.where(filts=='DECam_r')[0]])

        #Sets attributes
        for band in bands:
            #setattr(self, band, Mags[band]*u.dimensionless_unscaled)
            setattr(self, band, mags[:,np.where(filts==band)[0]]*u.dimensionless_unscaled)
        for err in errors:
            setattr(self, err, errs[err]) 
        if(hasattr(self,'e_par')):         
            self.e_par = self.e_par * u.mas
        if(hasattr(self,'e_pmra')):         
            self.e_pmra = self.e_pmra * u.mas / u.yr
        if(hasattr(self,'e_pmdec')):         
            self.e_pmdec = self.e_pmdec * u.mas /u.yr
        if(hasattr(self,'e_vlos')):         
            self.e_vlos = self.e_vlos * u.km/u.s

        if len([i for i in errors if i not in _Gaia_errors]) > 0:
            print(ErrorWarning)

        if len([i for i in errors if i in _Gaia_errors]) > 0:
            self.get_Gaia_errors(errors)



def zero_point(self):

    '''
        Calculate the predicted Gaia zero point offset for each mock HVS.
        NOT playtested or validated, proceed with caution

    '''
    import astropy.coordinates as coord
    from astropy.coordinates import SkyCoord
    from zero_point import zpt
    
    if not(hasattr(self,'Gaia_G') and hasattr(self,'Gaia_BP') \
                and hasattr(self,'Gaia_RP')):
        self.photometry(bands=['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

    #Get the astrometric nu_eff (Lindegren et al. 2021 Eq. 3 (A&A, 649, A2))
    nu_eff = (1.76 - (1.61/np.pi)*np.arctan(0.531*(self.Gaia_BP \
                - self.Gaia_RP)))

    zpt.load_tables()

    #Get the ecliptic latitude of each source
    c = SkyCoord(ra=self.ra, dec=self.dec, frame='icrs')
    b = c.transform_to('barycentricmeanecliptic').lat.value

    #Estimate the parallax zero point offset
    self.zp = zpt.get_zpt(self.Gaia_G, nu_eff, -99*np.ones(len(b)), 
                            b, 31*np.ones(len(b)))

#@get_Punbound
def get_Punbound(self, potential, numsamp = int(5e1), par_cut_flag=True, 
                    par_cut_val = 0.2,solarmotion = None, 
                    zo=None, vo=None, ro=None, t = 0.*u.Myr):

    '''
    Sampling over provided observations w/ errors, returns probability 
    that star is unbound in the provided Galactic potential.

    Parameters:
    ---------------

    potential : galpy potential instance
        The assumed Galactic potential. MUST be either defined with physical
        units or `physicalized' with .turn_physical_on()

    numsamp : integer
        Number of times observations should be sampled to 
        compute unbound probabilities

    par_cut_flag : Boolean
        If True, computes only unbound probabilities for sources with 
        (relative) parallax uncertaintainties less than par_cut_val. 
        Recommended to keep as true -- unbound probabilities are not 
        particularly meaningful for sources with large distance 
        uncertainties and the computation of these probabilities can take a
        long time for populations for whom this cut is not performed.

    par_cut_val : real
        The if par_cut_flag is True, the relative parallax error cut 
        to impose. Default is 20% and it is recommended to keep it here. 
        20% is where estimating distances by inverting parallaxes starts 
        to become problematic -- see Bailer-Jones 2015 
        (https://ui.adsabs.harvard.edu/abs/2015PASP..127..994B)

    solarmotion : length-3 list of floats
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped. Default is Schonrich+2010

    zo : Float or astropy distance quantity
             Offset of the Sun above or below the Galactic plane.
             Default is 20.8 pc (Bennett+Bovy 2019)
             If float, units are assumed to be kpc

    ro : Float (astropy length quantity)
        Distance to the Galactic Centre in kpc. Default is None, in which 
        case ro is set to the ro assumed by the provided potential

    vo : Float (astropy velocity quantity)
        Circular velocity at the Solar position in km/s. Default is None, 
        in which case vo is set to the ro assumed by the provided potential

    t : Float (astropy time quantity)
        Time at which the potential is evaluated.
        Only relevant if potential is time-dependent
    '''
        
    import astropy.coordinates as coord
    from astropy.table import Table
    #from galpydev.galpy.util.coords import radec_to_lb, pmrapmdec_to_pmllpmbb
    #from galpydev.galpy.potential import evaluatePotentials
    #from galpydev.galpy.orbit import Orbit
    #from galpydev.galpy.potential.mwpotentials import MWPotential2014
    from galpy.potential import evaluatePotentials
    from galpy.orbit import Orbit
    from galpy.potential.mwpotentials import MWPotential2014

    from .utils.dustmap import DustMap

    #Check to make sure astrometry exists
    if not ( hasattr(self,'ra') and hasattr(self,'dec') \
            and hasattr(self,'pmra') and hasattr(self,'pmdec') \
            and hasattr(self,'dist') and hasattr(self,'vlos')):
        raise ValueError(PunboundAstrometryError)

    #Check to see if errors exist. Compute them if not
    if not hasattr(self,'cov'):
        if not ( hasattr(self,'e_par') and hasattr(self,'e_pmra') and \
                hasattr(self,'e_pmdec') and hasattr(self,'e_vlos') ):
            print(PunboundUncertaintyWarning)
            self.photometry(self.dust)

    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])

    if(self.size==0):
        self.Pub = np.zeros(self.size)
        return

    #Solar position and velocity in McMillan17
    #vo               = 233.1 
    #ro               = 8.21*u.kpc
    #Usun, Vsun, Wsun = 8.6, 13.9, 7.1

    #Initialize...
    self.Pub = np.empty(self.size)
    self.e_GCv = np.empty(self.size)*u.km/u.s

    #vSun = [-Usun, Vsun, Wsun] * u.km / u.s # (U, V, W)
        
    #v_sun = coord.CartesianDifferential(vSun + [0, vo, 0]*u.km/u.s)

    #GCCS = coord.Galactocentric(galcen_distance=ro, z_sun=0*u.kpc, 
    #                                galcen_v_sun=v_sun)

    #if potential is None:
        #if self.potential is None:
        #raise ValueError(PotentialError)
        #potential = self.potential

    if solarmotion is None:
        if self.solarmotion is None:
            print(SolarMotionWarning)
            solarmotion = [-11.1, 12.24, 7.25]*u.km/u.s
        else:
            solarmotion = self.solarmotion

    if vo is None:
        vo = get_physical(potential)['vo']*u.km/u.s
    if ro is None:
        ro = get_physical(potential)['ro']*u.kpc

    if zo is None:
            if self.zo is None:
                print(zoWarning)
                zo = 0.0208*u.kpc
            else:
                zo = self.zo

    for i in tqdm(range(self.size),desc='Calculating unbound probability...'):

        #Don't even calculate if star is very fast
        if(self.v0[i]>1500*u.km/u.s):
            self.Pub[i] = 1.
            self.e_GCv[i] = 0.
            
        else:
            #Sample a radial velocity
            vlos = np.random.normal(self.vlos[i].value, 
                                    self.e_vlos[i].value,numsamp)*u.km/u.s

            #Get the 'true' astrometry
            means = [self.ra[i].to('mas').value,self.dec[i].to('mas').value, 
                    self.par[i].value,self.pmra[i].to(u.mas/u.yr).value, 
                    self.pmdec[i].to(u.mas/u.yr).value
                    ]

            if hasattr(self, 'cov'):

                # Sample astrometry n times based on covariance matrix
                ratmp, dectmp, partmp, pmra, pmdec = \
                    np.random.multivariate_normal(means,self.cov[:,:,i],
                                                        numsamp).T

                ra = ratmp*u.mas.to(u.deg)*u.deg
                dec = dectmp*u.mas.to(u.deg)*u.deg
                dist = u.kpc/np.abs(partmp)
                pmra, pmdec = pmra*u.mas/u.yr, pmdec*u.mas/u.yr

            else:
                # Sample astrometry based only on errors.
                # Errors are assumed to be uncorrelated
                ra = self.ra[i]*np.ones(numsamp)
                dec = self.dec[i]*np.ones(numsamp)

                dist = u.kpc / abs(np.random.normal(self.par[i].value, 
                                self.e_par[i].to(u.mas).value,numsamp))

                pmra = np.random.normal(self.pmra[i].to(u.mas/u.yr).value, 
                       self.e_pmra[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr
                pmdec=np.random.normal(self.pmdec[i].to(u.mas/u.yr).value, 
                      self.e_pmdec[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr

            #Get galactocrentric position and velocity 
            #data   = radec_to_lb(ra.value, dec.value, degree=True)
            #ll, bb = data[:, 0], data[:, 1]

            #data       = pmrapmdec_to_pmllpmbb(pmra,pmdec, ra.value, 
            #                                        dec.value, degree=True)
            #pmll, pmbb = data[:, 0], data[:, 1]

            #galactic_coords = coord.Galactic(l=ll*u.deg, b=bb*u.deg, 
            #                        distance=dist, pm_l_cosb=pmll*u.mas/u.yr, 
            #                        pm_b=pmbb*u.mas/u.yr, radial_velocity=vlos)

            #galactocentric_coords = galactic_coords.transform_to(GCCS)

            #GCv = np.sqrt(galactocentric_coords.v_x**2. 
            #                + galactocentric_coords.v_y**2. 
            #                + galactocentric_coords.v_z**2.).to(u.km/u.s)

            #R = np.sqrt(galactocentric_coords.x**2 
            #                + galactocentric_coords.y**2).to(u.kpc)

            #z = galactocentric_coords.z.to(u.kpc)

            o = Orbit([ra, dec, dist, pmra, pmdec, vlos],radec=True, ro=ro, 
                    solarmotion=solarmotion, vo=vo, zo=zo)

            GCv2 = np.sqrt(o.vx(quantity=True)**2 + o.vy(quantity=True)**2 
                    + o.vz(quantity=True)**2)

            R2 = np.sqrt(o.x(quantity=True)**2 + o.y(quantity=True)**2)

            z2 = o.z(quantity=True)

            phi = np.arctan2(o.y(quantity=True),o.x(quantity=True))

            #For each sampled entry, get escape velocity
            Vesc = np.zeros(numsamp)*u.km/u.s
            for j in range(numsamp):
                Vesc[j] = np.sqrt(2*(- evaluatePotentials(potential, 
                        R2[j],z2[j],phi=phi[j], t = t, quantity=True)))
                #Vesc[j] = np.sqrt(2*(- evaluatePotentials(MWPotential2014,R[j],z[j])))*u.km/u.s
                #Vesc[j] = np.sqrt(2*(evaluatePotentials(MWPotential2014,1e10*R[j],z[j]) - evaluatePotentials(MWPotential2014,R[j],z[j])))*220*u.km/u.s

            #Calculate fraction of iterations above escape velocity
            inds = (GCv2 > Vesc)
            self.Pub[i] = len(GCv2[inds])/len(Vesc)

            #Calculate spread of sampled galactocentric velocity
            self.e_GCv[i] = np.std(GCv2)

#@get_Punbound
def get_P_velocity_greater(self, vcut, numsamp = int(5e1), 
                            par_cut_flag=True, par_cut_val = 0.2, 
                            solarmotion = None, zo=None, vo=None, ro=None):

    '''
    Sampling over provided observations w/ errors, returns probability 
    that star is observed with a total velocity above a certain threshold.

    Parameters:
    ---------------

    vcut : float
        Galactocentric velocity (in km/s) that is used for the cut

    numsamp : integer
        Number of times observations should be sampled to 
        compute unbound probabilities

    par_cut_flag : Boolean
        If True, computes only unbound probabilities for sources with 
        (relative) parallax uncertaintainties less than par_cut_val. 
        Recommended to keep as true -- unbound probabilities are not 
        particularly meaningful for sources with large distance 
        uncertainties and the computation of these probabilities can take a
        long time for populations for whom this cut is not performed.

    par_cut_val : real
        The if par_cut_flag is True, the relative parallax error cut 
        to impose. Default is 20% and it is recommended to keep it here. 
        20% is where estimating distances by inverting parallaxes starts 
        to become problematic -- see Bailer-Jones 2015 
        (https://ui.adsabs.harvard.edu/abs/2015PASP..127..994B)

    solarmotion : None or length-3 list of floats or astropy quantity (km/s)
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped.
            Defaults to self.solarmotion if it exists. If it does not, defaults
            to Schonrich+2010

    zo : None or Float or astropy distance quantity
             Offset of the Sun above or below the Galactic plane.
             Default is self.zo if it exists.
             If self.zo does not exists, defaults to 20.8 pc (Bennett+Bovy 2019)
             If float, units are assumed to be kpc

    ro : None or Float (astropy length quantity)
        Distance to the Galactic Centre in kpc. 
        Defaults to self.ro, if it exists
        If self.ro does not exist, reads it from .galpyrc config file

    vo : None or Float (astropy velocity quentity)
        Circular velocity at the Solar position in km/s. 
        Defaults to self.vo, if it exists.
        If self.vo does not exist, reads it from the .galpyrc config file

    '''

    import astropy.coordinates as coord
    from astropy.table import Table
    #from galpydev.galpy.util.coords import radec_to_lb, pmrapmdec_to_pmllpmbb

    #from galpydev.galpy.potential import evaluatePotentials
    #from galpydev.galpy.orbit import Orbit
    #from galpydev.galpy.potential.mwpotentials import MWPotential2014
    from galpy.potential import evaluatePotentials
    from galpy.orbit import Orbit
    from galpy.potential.mwpotentials import MWPotential2014

    from .utils.dustmap import DustMap

    #Check to make sure astrometry exists
    if not ( hasattr(self,'ra') and hasattr(self,'dec') \
            and hasattr(self,'pmra') and hasattr(self,'pmdec') \
            and hasattr(self,'dist') and hasattr(self,'vlos')):
        raise ValueError(PunboundAstrometryError)

    #Check to see if errors exist. Compute them if not
    if not hasattr(self,'cov'):
        if not ( hasattr(self,'e_par') and hasattr(self,'e_pmra') and \
                hasattr(self,'e_pmdec') and hasattr(self,'e_vlos') ):
            print(PunboundUncertaintyWarning)
            self.photometry(self.dust)

    if(self.size==0):
        self.p_GCvcut = np.zeros(self.size)
        self.e_GCv = np.zeros(self.size)
        return

    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])

    #Solar position and velocity in McMillan17
    #vo               = 233.1 
    #ro               = 8.21*u.kpc
    #ro               = 8.122*u.kpc
    #Usun, Vsun, Wsun = 8.6, 13.9, 7.1
    #Usun, Vsun, Wsun = 12.9, 12.5, 7.78

    #Initialize...
    self.p_GCvcut = np.empty(self.size)
    self.GCv_lb = np.empty(self.size)*u.km/u.s
    self.GCv_ub = np.empty(self.size)*u.km/u.s

    #Assign vo, ro, zo, solarmotion
    if vo is None:
            if self.vo is None:
                print(voWarning)
                vo = get_physical(MWPotential2014)['vo']*u.km/u.s
            else:
                vo = self.vo

    if ro is None:
            if self.ro is None:
                print(roWarning)
                ro = get_physical(MWPotential2014)['ro']*u.kpc
            else:
                ro = self.ro
    if zo is None:
            if self.zo is None:
                print(zoWarning)
                zo = 0.0208*u.kpc
            else:
                zo = self.zo

    if solarmotion is None:
        if self.solarmotion is None:
            print(SolarMotionWarning)
            solarmotion = [-11.1, 12.24, 7.25]*u.km/u.s
        else:
            solarmotion = self.solarmotion

    if isinstance(vcut,float):
            vcut = vcut*u.km/u.s
    if u.get_physical_type('speed') != u.get_physical_type(vcut):
        raise ValueError(vcutError)

    if not (hasattr(self,attr) for attr in ['e_vlos', 'cov']) \
            or not (hasattr(self,attr) \
                for attr in ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos']):
        raise ValueError(astroErrorError)

    if any(np.isnan(self.e_par)):
        print(astroErrorWarning)
        self.subsample(np.where(np.isreal(self.e_par))[0])

    #if any(self.met>0.3) or any(self.met<-0.3):

    for i in tqdm(range(self.size),
                desc='Calculating high-v prob...'):

            #Sample a radial velocity
            vlos = np.random.normal(self.vlos[i].value, 
                                    self.e_vlos[i].value,numsamp)*u.km/u.s

            #Get the 'true' astrometry
            means = [self.ra[i].to('mas').value,self.dec[i].to('mas').value, 
                    self.par[i].value,self.pmra[i].to(u.mas/u.yr).value, 
                    self.pmdec[i].to(u.mas/u.yr).value
                    ]

            if hasattr(self, 'cov'):

                # Sample astromerry n times based on covariance matrix
                ratmp, dectmp, partmp, pmra, pmdec = \
                    np.random.multivariate_normal(means,self.cov[:,:,i],
                                                        numsamp).T

                ra = ratmp*u.mas.to(u.deg)*u.deg
                dec = dectmp*u.mas.to(u.deg)*u.deg
                dist = u.kpc/np.abs(partmp)
                pmra, pmdec = pmra*u.mas/u.yr, pmdec*u.mas/u.yr

            else:
                # Sample astrometry based only on errors.
                # Errors are assumed to be uncorrelated
                ra = self.ra[i]*np.ones(numsamp)
                dec = self.dec[i]*np.ones(numsamp)

                dist = u.kpc / abs(np.random.normal(self.par[i].value, 
                                self.e_par[i].to(u.mas).value,numsamp))

                pmra = np.random.normal(self.pmra[i].to(u.mas/u.yr).value, 
                       self.e_pmra[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr
                pmdec = np.random.normal(self.pmdec[i].to(u.mas/u.yr).value, 
                      self.e_pmdec[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr

            #Get galactocrentric position and velocity 
            #data   = radec_to_lb(ra.value, dec.value, degree=True)
            #ll, bb = data[:, 0], data[:, 1]

            #data       = pmrapmdec_to_pmllpmbb(pmra,pmdec, ra.value, 
            #                                        dec.value, degree=True)
            #pmll, pmbb = data[:, 0], data[:, 1]

            #galactic_coords = coord.Galactic(l=ll*u.deg, b=bb*u.deg, 
            #                        distance=dist, pm_l_cosb=pmll*u.mas/u.yr, 
            #                        pm_b=pmbb*u.mas/u.yr, radial_velocity=vlos)

            #galactocentric_coords = galactic_coords.transform_to(GCCS)

            #GCv = np.sqrt(galactocentric_coords.v_x**2. 
            #                + galactocentric_coords.v_y**2. 
            #                + galactocentric_coords.v_z**2.).to(u.km/u.s)

            o = Orbit([ra, dec, dist, pmra, pmdec, vlos],radec=True, ro=ro, 
                    solarmotion=solarmotion, vo=vo, zo=zo)

            GCv2 = np.sqrt(o.vx(quantity=True)**2 + o.vy(quantity=True)**2 
                            + o.vz(quantity=True)**2)
            #Calculate fraction of iterations above escape velocity
            inds = (GCv2 > vcut)
            self.p_GCvcut[i] = len(GCv2[inds])/len(GCv2)

            #Calculate spread of sampled galactocentric velocity
            self.GCv_lb[i], self.GCv_ub[i] = np.quantile(GCv2, [0.16,0.84])

def get_e_beta(self, par_cut_flag=True, par_cut_val = 0.2, numsamp = int(7.5e2), 
               solarmotion = None, zo=None, vo=None, ro=None):

    '''
    Samples over observations and errors to return azimuthal angle between HVS's velocity in Galactocentric frame and position in Galactocentric frame

    Parameters:
    ---------------
    
    par_cut_flag: bool
        If True, samples over HVS with relative parallax uncertainties less 
        than par_cut_value

    numsamp : integer
        Number of times observations are sampled  
        
    err_GC, err_vo, err_vlos, err_par, err_pmra_pmmdec: boolean 
        Defaults are False. If true, error in that measurement is considered when 
        samping
    
    ro, vo: integer 
        values of ro and vo used in propagation. ro in kpc vo in km/s
        
    Returns:
    --------------
    self.beta: array-like
        A numsamp x self.size where each column is an HVS and each row is the
        beta calculated for that sample.
    '''
    import astropy.coordinates as coord
    from astropy.table import Table
    #from galpy.util.coords import radec_to_lb, pmrapmdec_to_pmllpmbb
    #from galpy.potential import evaluatePotentials
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from galpy.orbit import Orbit

    #lt.rcParams.update({'font.size': 15})
    #from .utils.dustmap import DustMap

    #Assign vo, ro, zo, solarmotion
    if vo is None:
            if self.vo is None:
                print(voWarning)
                vo = get_physical(MWPotential2014)['vo']*u.km/u.s
            else:
                vo = self.vo

    if ro is None:
            if self.ro is None:
                print(roWarning)
                ro = get_physical(MWPotential2014)['ro']*u.kpc
            else:
                ro = self.ro
    if zo is None:
            if self.zo is None:
                print(zoWarning)
                zo = 0.0208*u.kpc
            else:
                zo = self.zo

    if solarmotion is None:
        if self.solarmotion is None:
            print(SolarMotionWarning)
            solarmotion = [-11.1, 12.24, 7.25]*u.km/u.s
        else:
            solarmotion = self.solarmotion
    
    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])
    
    if not (hasattr(self,attr) for attr in ['e_vlos', 'cov']) \
            or not (hasattr(self,attr) \
                for attr in ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos']):
        raise ValueError(astroErrorError)

    #if any(self.e_par is None):
    #    print(astroErrorWarning)
    #    self.subsample(np.where(np.isreal(self.e_par))[0])

    if any(np.isnan(self.e_par)):
        print(astroErrorWarning)
        self.subsample(np.where(np.isreal(self.e_par))[0])

    #solar position 
    #Usun, Vsun, Wsun = 11.1, 12.24, 7.25
    
    #solar cords
    #vSun = [-Usun, Vsun, Wsun] * u.km / u.s # (U, V, W)
        
    #v_sun = coord.CartesianDifferential(vSun + [0, vo, 0]*u.km/u.s)

    #print('Computing e_beta...')
    
    #For each sampled entry, get beta 
    #array to hold beta samples
    self.beta = np.zeros((numsamp,self.size))
    
    #for i in range(self.size):
    for i in tqdm(range(self.size),desc='Calculating deflection...'):

        #Sample a radial velocity
        vlos = np.random.normal(self.vlos[i].value, 
                                    self.e_vlos[i].value,numsamp)*u.km/u.s

        #Get the 'true' astrometry
        means = [self.ra[i].to('mas').value,self.dec[i].to('mas').value, 
                    self.par[i].value,self.pmra[i].to(u.mas/u.yr).value, 
                    self.pmdec[i].to(u.mas/u.yr).value
                    ]

        if hasattr(self, 'cov'):

            # Sample astromerry n times based on covariance matrix
            ratmp, dectmp, partmp, pmra, pmdec = \
                np.random.multivariate_normal(means,self.cov[:,:,i], numsamp).T

            ra = ratmp*u.mas.to(u.deg)*u.deg
            dec = dectmp*u.mas.to(u.deg)*u.deg
            dist = u.kpc/np.abs(partmp)
            pmra, pmdec = pmra*u.mas/u.yr, pmdec*u.mas/u.yr

        else:

            # Sample astrometry based only on errors.
            # Errors are assumed to be uncorrelated
            ra = self.ra[i]*np.ones(numsamp)
            dec = self.dec[i]*np.ones(numsamp)

            dist = u.kpc / abs(np.random.normal(self.par[i].value, 
                                self.e_par[i].to(u.mas).value,numsamp))

            pmra = np.random.normal(self.pmra[i].to(u.mas/u.yr).value, 
                       self.e_pmra[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr
            pmdec=np.random.normal(self.pmdec[i].to(u.mas/u.yr).value, 
                      self.e_pmdec[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr         
        #string to do an array of orbits 
        o = Orbit([ra, dec, dist, pmra, pmdec, vlos], radec=True, \
                        solarmotion = solarmotion, ro=ro, vo=vo)
            
        for j in range(numsamp): 
                        
            #total velicity and position
            vtot = np.sqrt(o.vx(quantity=True)[j]**2+o.vy(quantity=True)[j]**2)
            postot = np.sqrt(o.x(quantity=True)[j]**2+o.y(quantity=True)[j]**2)
            
            #now make the vectors 
            r = [o.x(quantity=True)[j].value, o.y(quantity=True)[j].value]
            v = [o.vx(quantity=True)[j].value, o.vy(quantity=True)[j].value]
            dot = np.dot(r,v)
            angle = dot/(vtot*postot)
            
            self.beta[j][i] = np.arccos(angle.value) 

def evolve(self,Zsun=0.02):
        #Evolve a star of a certain mass and metallicity until a certain age 
        #using the SSE module within AMUSE
        from amuse.units import units
        from amuse.community.sse.interface import SSE
        #from amuse.test.amusetest import get_path_to_results
        from amuse import datamodel

        self.T_eff = np.ones(self.size)*u.K
        self.Rad = np.ones(self.size)*u.Rsun
        self.Lum = np.ones(self.size)*u.Lsun
        self.stage = np.ones(self.size)

        for z in (np.unique(self.met)):  

            #indices with ith metallicity
            idx = np.where(self.met==z)[0]

            #Initialize
            stellar_evolution = SSE()
            #Adjust metallicity for new Zsun assumption
            stellar_evolution.parameters.metallicity = Zsun*10**(z)
            star      = datamodel.Particles(len(self.m[idx].value))
            star.mass = self.m[idx].value | units.MSun

            age = self.tage[idx].to('Myr').value | units.Myr

            #Evolve the star
            star = stellar_evolution.particles.add_particles(star)
            stellar_evolution.commit_particles()
            stellar_evolution.evolve_model(end_time = age)

            stellar_evolution.stop()

            #Extract HVS effective temperature, radius, 
            #luminosity, evolutionary stage
            self.T_eff[idx] = star.temperature.as_astropy_quantity().to('K')#.value
            self.Rad[idx] = star.radius.as_astropy_quantity().to('Rsun')#.value 
            self.Lum[idx] = star.luminosity.as_astropy_quantity().to('Lsun')#.value 
            self.stage[idx] = star.stellar_type.as_astropy_quantity()

        #self.Rad = R*u.Rsun
        #self.Lum = Lum*u.Lsun
        #self.T_eff = T_eff*u.K
        #self.stage = stage

