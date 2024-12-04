PackageError = "One or more required external packages have not been installed. Please see requirements.txt"

try:
    from astropy import units as u
    import astropy
    from astropy.table import Table
    import numpy as np
except ImportError:
    raise ImportError(PackageError)

#@save
def save(self, path):
        '''
        Saves the sample in a FITS file to be grabbed later.
        ALL attributes which are arrays of length self.size are saved.
        See docstring of zippystar.starsample for list of common attributes
        Some metavariables saved as well.

        Parameters
        ----------
        path : str
            Path to the output fits file
        '''

        import warnings
        warnings.filterwarnings('ignore',module='astropy.io')

        #if(self.size==0):
        #    print('-----------WARNING--------')
        #    print('No stars exist in sample. Saving to file anyways.')

        datalist = []

        namelist = ['r0','phi0','phiv0','theta0','thetav0','v0', 'm', 'tage', 
                    'tflight','a','P','q','mem','met','stage','stagebefore',
                    'ra', 'dec', 'pmra', 'pmdec', 'dist', 'par', 'vlos', 'Av',
                    'Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R', 
                    'Bessell_I', 'Gaia_GRVS', 'Gaia_G','Gaia_BP','Gaia_RP', 
                    'VISTA_Z', 'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_K', 
                    'DECam_u', 'DECam_g', 'DECam_r', 'DECam_i', 'DECam_z', 
                    'DECam_Y', 'LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 
                    'LSST_y', 'T_eff','Lum','Rad','e_par', 'e_pmra', 'e_pmdec', 
                    'e_vlos', 'obsprob','obsvar','run','Pub','potind','GCdist', 
                    'GCv', 'GCv_lb', 'GCv_ub', 'p_GCvcut', 'Vesc', 'x', 'y', 
                    'z', 'vx', 'vy', 'vz', 'n', 'k', 'kappa'
                    ]

        namelist = []

        #Some metavariables are saved
        #meta_var = {'name' : self.name, 'ejmodel' : self.ejmodel_name, \
        #                'dt' : self.dt.to('Myr').value}
        meta_var = {}

        #print(vars(self).keys())
        #Every attribute which is a numpy/astropy array of length
        #self.size is saved to file
        #Yes, this is ugly
        #Ugliness is required to deal with edge cases
        for name in vars(self).keys():
            if name != 'cov':
                if isinstance(vars(self)[name],astropy.units.quantity.Quantity):
                    if isinstance(vars(self)[name].value,np.ndarray):
                        if len(vars(self)[name])==self.size:
                            datalist.append(getattr(self,name))
                            namelist.append(name)
                        else:
                            #print(name)
                            meta_var[name] = getattr(self, name).value
                            #print(meta_var[name])
                    else:
                        meta_var[name] = getattr(self, name).value
                elif isinstance(vars(self)[name],np.ndarray):
                    if len(vars(self)[name])==self.size:
                        datalist.append(getattr(self,name))
                        namelist.append(name)
                    else:
                            #print(name)
                            meta_var[name] = getattr(self, name)
                            #print(meta_var[name])
                else:
                    #print(name)
                    meta_var[name] = getattr(self, name)
                    #print(meta_var[name])
        data_table = Table(data=datalist, names=namelist, meta=meta_var)
        data_table.write(path, overwrite=True)

#@load
def _load(self, path):

        '''
            Loads a HVS sample from a fits table.
            Creates a starsample object with attributes corresponding
            to each column in the fits file.

        Parameters
        --------------
    
        path: string
            File path to be read from
        '''

        #List of common coloumns to be read in and their associated units.
        #Read columns do NOT need to be explicitly specified here, all columns
        #of the fits file are read. This is simply useful to assign units to
        #common quantities

        default_units = {'r0': u.pc, 'phi0': u.rad, 'theta0':u.rad, 
                        'v0':u.km/u.s, 'vc':u.km/u.s, 'phiv0':u.rad, 
                        'thetav0':u.rad, 'm':u.solMass, 'tage':u.Myr,
                        'tflight':u.Myr, 'a':u.Rsun,'P':u.day,'q':None,
                        'mem':None,'met':None, 'stage':None,'stagebefore':None,
                        'ra':u.deg,'dec':u.deg, 'pmra':u.mas/u.yr, 
                        'pmdec':u.mas/u.yr,'dist':u.kpc, 'vlos':u.km/u.s,
                        'Av':None, 'Bessell_U':None, 'Bessell_B':None, 
                        'Bessell_V':None, 'Bessell_R':None, 'Bessell_I':None, 
                        'Gaia_GRVS':None, 'Gaia_G':None, 'Gaia_BP':None, 
                        'Gaia_RP':None, 'VISTA_Z':None, 'VISTA_Y':None, 
                        'VISTA_J':None, 'VISTA_H':None, 'VISTA_K':None, 
                        'DECam_u':None, 'DECam_g':None,
                        'DECam_r':None, 'DECam_i':None, 'DECam_z':None, 
                        'DECam_Y':None, 'LSST_u':None, 'LSST_g':None, 
                        'LSST_r':None, 'LSST_i':None, 'LSST_z':None, 
                        'LSST_y':None, 'BP_RP':None,'T_eff':u.K,'Lum':u.Lsun,
                        'Rad':u.Rsun,'par':u.mas,'e_par':1e-6*u.arcsec,
                        'e_pmra':u.mas/u.yr,'e_pmdec':u.mas/u.yr,
                        'e_vlos':u.km/u.s,'obsprob':None,'obsvar':None, 
                        'run':None, 'k':None, 'n':None,
                        'potind':None,'Pub':None,'GCdist':u.kpc,'GCv':u.km/u.s, 
                        'e_GCv':u.km/u.s,'GCv_lb': u.km/u.s, 'GCv_ub': u.km/u.s, 
                        'Vesc':u.km/u.s, 'theta':u.rad,
                        'phi':u.rad, 'x':u.kpc, 'y':u.kpc, 'z':u.kpc,
                        'vx':u.km/u.s,'vy':u.km/u.s, 'vz':u.km/u.s}

        meta_units = {'name': None, 'gaia_release': None, 'use_ast_sf': None, 
                    'ejmodel_name':None, 'size': None, 'zsun': None, 'Met': None,
                     'current_a': u.pc, 'tlb': u.Myr, 'cut_factor': None, 
                    'mc': u.Msun, 'm_bh': u.Msun, 'a0': u.pc,
                    'rho': u.Msun / u.pc**3, 'sigma': u.km/u.s, 
                    'tflightmax': u.Myr, 'saveorbit': None, 'orbitpath': None, 
                    'vmin': u.km/u.s, 'solarmotion': None, 'dt': u.Myr, 
                    'm_range': u.Msun, 'v_range':u.km/u.s, 'eta': u.yr**-1, 
                    'gamma': None, 'alpha': None, 'zo':u.kpc, 'ro': u.kpc,
                    'vo': u.km/u.s, 'potential': None}

        namelist = list(default_units.keys())
        meta_names = list(meta_units.keys())

        #Read in data
        data_table = Table.read(path)

        #get metadata
        data_table.meta =  {k.lower(): v for k, v in data_table.meta.items()}

        #Assign metadata as attributes
        for key in data_table.meta.keys():
            if key in meta_names:
                if meta_units[key] is not None:
                    #print(key)
                    #print(meta_units[key])
                    #print(data_table.meta[key])
                    setattr(self, key, data_table.meta[key]*meta_units[key])
                else:
                    setattr(self,key, data_table.meta[key])
            else:    
                setattr(self,key, data_table.meta[key])

        self.size = len(data_table)

        #Assign columns as attributes
        for colname in data_table.colnames:
                if colname in namelist:
                    if data_table[colname].unit is None \
                                        and default_units[colname] is not None:
                        setattr(self, colname, data_table[colname].quantity \
                                    * default_units[colname])
                    else:
                        setattr(self, colname, data_table[colname].quantity)
                else:
                    setattr(self, colname, data_table[colname])

#@loadExt
#Probably out of date, ignore
def _loadExt(self, path, ejmodel='Contigiani2018',dt=0.01*u.Myr):

        try:
            from astropy.coordinates import SkyCoord
        except ImportError:
            raise ImportError(PackageError)
        '''
            Loads an external HVS sample from external source (e.g., from literature)

        Parameters
        ----------
            path: str
                Path to catalog
            ejmodel = str
                Suspected ejection model generating the sample. Not sure if this would do anything right now if only the likelihood method is being used
            dt = float
                Timestep to be used for the back-propagation

            See self.likelihood() for other parameters

        '''

        from astropy.table import Table

        #namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', 'dec', 'pmra',
        #            'pmdec', 'dist', 'vlos', 'GRVS', 'V', 'G', 'e_par', 'e_pmra', 'e_pmdec', 'GCdist', 'GCv']

        data_table = Table.read(path)

        #data_table['pmra_Gaia'][1] = -0.175
        #data_table['pmdec_Gaia'][1] = -0.719
        #data_table['err_pmra_Gaia'][1] = 0.316
        #data_table['err_pmdec_Gaia'][1] = 0.287

        #Manually set variables that would normally be in metadata
        self.ejmodel_name = ejmodel
        self.dt = dt
        #self.cattype = 2
        self.size = len(data_table)

        setattr(self,'m',data_table['M']*u.solMass)
        setattr(self,'pmra',data_table['pmra_Gaia']*u.mas/u.yr)
        setattr(self,'pmdec',data_table['pmdec_Gaia']*u.mas/u.yr)
        setattr(self,'vlos',data_table['vrad']*u.km/u.second)
        setattr(self,'dist',data_table['d']*u.kpc)
        setattr(self,'tage',data_table['tage']*u.Myr)
        setattr(self,'ID',data_table['ID'])
        setattr(self,'e_pmra',data_table['err_pmra_Gaia']*u.mas/u.yr)
        setattr(self,'e_pmdec',data_table['err_pmdec_Gaia']*u.mas/u.yr)
        setattr(self,'e_dist',data_table['d_errhi']*u.kpc)
        setattr(self,'e_vlos',data_table['vrad_errhi']*u.km/u.second)
        setattr(self,'ra',data_table['ra']*u.degree)
        setattr(self,'dec',data_table['dec']*u.degree)


        #self.pmra[1] = -0.175*u.mas/u.yr
        #setattr(self,'pmra[1]',-0.175*u.mas/u.yr)
        #setattr(self,'pmdec[1]',-0.719*u.mas/u.yr)
        #setattr(self,'err_pmra[1]',0.316*u.mas/u.yr)
        #setattr(self,'err_pmdec[1]',0.287*u.mas/u.yr)

        #Read in ra, dec in hhmmss.ss/DDmmss.ss, convert to degrees
        #ratmp = data_table['RA']
        #dectmp = data_table['Dec']
        #c = SkyCoord(ratmp,dectmp)
        #setattr(self,'ra',c.ra.value*u.degree)
        #setattr(self,'dec',c.dec.value*u.degree)

        #l = c.galactic.l.value*np.pi/180.0
        #b = c.galactic.b.value*np.pi/180.0

        #p1 = (np.cos(b)**2)*(np.cos(l)**2) + (np.cos(b)**2)*(np.sin(l)**2) + np.sin(b)**2
        #p2 = -16.0*np.cos(b)*np.cos(l)
        #p3 = 64 - data_table['RGC']**2

        #dist = np.zeros(self.size)

        #for i in range(self.size):
        #    dist[i] = max(np.roots([p1[i],p2[i],p3[i]]))

        #DATA
        #i=0
        #for colname in data_table.colnames:
        #    try:
        #        i = namelist.index(colname)
        #        setattr(self, colname, data_table[colname].quantity)
        #        i+=1
        #    except ValueError:
        #        print('Column not recognized: ' + str(colname))
        #        i+=1
        #        continue
def saveall(self, path):
    '''
    Saves the sample in a FITS file to be retrieved later.
    ALL attributes which are arrays of length `self.size` are saved, 
    including input catalog columns and other dynamically created attributes.

    Parameters
    ----------
    path : str
        Path to the output FITS file.
    '''
    import warnings
    warnings.filterwarnings('ignore', module='astropy.io')

    # If no stars exist in the sample, save an empty file with metadata
    if self.size == 0:
        print('-----------WARNING--------')
        print('No stars exist in the sample. Saving an empty file with metadata.')

    datalist = []
    namelist = []

    # Some metadata variables to be saved
    meta_var = {}

    # Iterate over all attributes in the object
    for name in vars(self).keys():
        if name != 'cov':  # Exclude 'cov' explicitly
            attr = getattr(self, name)
            # Check if the attribute is an astropy Quantity or numpy array
            if isinstance(attr, astropy.units.quantity.Quantity):
                if isinstance(attr.value, np.ndarray) and len(attr) == self.size:
                    datalist.append(attr)
                    namelist.append(name)
                else:
                    meta_var[name] = attr.value
            elif isinstance(attr, np.ndarray):
                if len(attr) == self.size:
                    datalist.append(attr)
                    namelist.append(name)
                else:
                    meta_var[name] = attr
            elif isinstance(attr, (int, float, str, list)):
                meta_var[name] = attr

    # Create an astropy table to store the data and metadata
    data_table = Table(data=datalist, names=namelist, meta=meta_var)

    # Write the table to a FITS file
    data_table.write(path, overwrite=True)


def loadall(self, path):
    '''
    Loads a HVS sample from a FITS table.
    Creates a `starsample` object with attributes corresponding
    to each column in the FITS file.

    Parameters
    ----------
    path: string
        File path to be read from.
    '''
    import warnings
    warnings.filterwarnings('ignore', module='astropy.io')

    # List of common columns and their associated units.
    # Columns do NOT need to be explicitly specified here;
    # all columns in the FITS file will be read.
    default_units = {
        'r0': u.pc, 'phi0': u.rad, 'theta0': u.rad, 'v0': u.km / u.s,
        'phiv0': u.rad, 'thetav0': u.rad, 'm': u.Msun, 'tage': u.Myr,
        'tflight': u.Myr, 'sep': u.Rsun, 'P': u.day, 'q': None, 'mem': None,
        'met': None, 'stage': None, 'stagebefore': None, 'Rad': u.Rsun,
        'T_eff': u.K, 'Lum': u.Lsun
    }

    meta_units = {
        'name': None, 'ejmodel_name': None, 'size': None, 'zsun': None,
        'Met': None, 'tflightmax': u.Myr, 'm_bh': u.Msun, 'v_range': u.km / u.s,
        'm_range': u.Msun, 'rate': u.yr**-1, 'alpha': None, 'gamma': None,
        'kappa': None
    }

    # Read in data
    data_table = Table.read(path)

    # Assign metadata as attributes
    data_table.meta = {k.lower(): v for k, v in data_table.meta.items()}
    for key, value in data_table.meta.items():
        if key in meta_units:
            if meta_units[key] is not None:
                setattr(self, key, value * meta_units[key])
            else:
                setattr(self, key, value)
        else:
            setattr(self, key, value)

    # Determine the size of the sample
    self.size = len(data_table)

    # Assign columns as attributes
    for colname in data_table.colnames:
        # Assign known columns with units
        if colname in default_units:
            if data_table[colname].unit is None and default_units[colname] is not None:
                setattr(self, colname, data_table[colname].quantity * default_units[colname])
            else:
                setattr(self, colname, data_table[colname].quantity)
        else:
            # Dynamically handle any additional columns
            setattr(self, colname, data_table[colname])
