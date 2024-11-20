# __init__.py

# Import key modules or functions from within the package

from .implied_d_vr import *  # Import everything from implied_d_vr.py
from .selections import *    # Import everything from selections.py
#from .download_gaia_alice import *  # Import everything from download_gaia_alice.py
from .download_gaia_by_healpix import *  # Import everything from download_gaia_by_healpix.py
from .kde_classifier import *
# You can also define the __all__ variable to specify what gets imported when using * 
# when importing from this package. For example:

__all__ = ['implied_d_vr', 'selections', 'download_gaia_by_healpix', 'kde_classifier']
