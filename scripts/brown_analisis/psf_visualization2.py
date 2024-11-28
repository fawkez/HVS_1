import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.table import Table
from mpl_toolkits.mplot3d import Axes3D


def onclick(event, data, ax_3d, fig_3d, log_scale):
    if event.inaxes is not None:  # Check if the click is within the axes
        x, y = int(event.xdata), int(event.ydata)
        
        # Define a small neighborhood around the click
        window_size = 20  # Size of the area to display in 3D (10x10 pixels)
        x_min = max(0, x - window_size // 2)
        x_max = min(data.shape[1], x + window_size // 2)
        y_min = max(0, y - window_size // 2)
        y_max = min(data.shape[0], y + window_size // 2)
        
        # Extract the region around the clicked point
        region = data[y_min:y_max, x_min:x_max]
        
        # Apply log scaling if enabled
        if log_scale:
            region = np.log10(np.maximum(region, 1))  # Prevent log(0) by setting a floor
        
        # Generate grid for 3D surface
        Y, X = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max))
        
        # Clear the previous 3D plot
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, region, cmap='viridis', edgecolor='none')
        ax_3d.set_title(f"3D Surface Plot at ({x}, {y}) {'(Log Scale)' if log_scale else ''}")
        ax_3d.set_xlabel('X Coordinate')
        ax_3d.set_ylabel('Y Coordinate')
        ax_3d.set_zlabel('Counts (log10)' if log_scale else 'Counts')
        
        # Redraw the 3D plot figure
        fig_3d.canvas.draw()

def overlay_catalog(ax, wcs, catalog_path):
    """Overlay catalog points on the image."""
    try:
        catalog = Table.read(catalog_path)  # Read the catalog as an astropy Table
    except Exception as e:
        print(f"Error reading catalog: {e}")
        return
    
    # Check if RA and DEC columns exist
    if 'RA' in catalog.colnames and 'DEC' in catalog.colnames:
        ra = catalog['RA']
        dec = catalog['DEC']
        
        # Convert RA, DEC to pixel coordinates
        pixel_coords = np.array(wcs.all_world2pix(np.column_stack((ra, dec)), 0))  # Ensure 2D input and output
        x_pixels = pixel_coords[:, 0]
        y_pixels = pixel_coords[:, 1]
        
        # Plot the catalog points
        ax.scatter(x_pixels, y_pixels, s=10, color='red', label='Catalog Points', alpha=0.7)
        ax.legend(loc='upper right')
    else:
        print("Catalog does not contain 'RA' and 'DEC' columns. Skipping overlay.")


def interactive_plot(fits_file_path, catalog_path=None, log_scale=False):
    # Open the FITS file
    with fits.open(fits_file_path) as hdul:
        header = hdul[1].header  # Extract WCS info from the header
        data = hdul[1].data  # Extract image data
    
    # Create WCS object
    wcs = WCS(header)
    
    # Ensure the data is numeric
    if not np.issubdtype(data.dtype, np.number):
        data = np.array(data, dtype=float)
    
    # Replace NaNs or invalid values with zeros
    data = np.nan_to_num(data)

    # Apply Zscale for intensity normalization
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    # Create the initial 2D plot
    fig, ax = plt.subplots(subplot_kw={'projection': wcs})
    im = ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title("Click a point to generate 3D plot")
    fig.colorbar(im, ax=ax, orientation='vertical', label='Counts')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    # Overlay catalog points if provided
    if catalog_path is not None:
        overlay_catalog(ax, wcs, catalog_path)
    
    # Create a second figure for the 3D plot
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Connect the click event to the onclick function
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, data, ax_3d, fig_3d, log_scale))

    # Show the plots
    plt.show()


# Replace with the path to your catalog FITS file
catalog_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/hvs10_hst.fits"


# Replace with the path to your FITS file
fits_file_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/HST/MAST_2024-11-27T12_42_50.627Z/HST/ibsp05010_drz.fits"


# Set `log_scale=True` for logarithmic Z-axis in the 3D plot
interactive_plot(fits_file_path, catalog_path=catalog_path, log_scale=False)
