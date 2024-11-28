import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from mpl_toolkits.mplot3d import Axes3D


def onclick(event, data, ax_3d, fig_3d):
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
        
        # Generate grid for 3D surface
        Y, X = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max))
        
        # Clear the previous 3D plot
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, region, cmap='viridis', edgecolor='none')
        ax_3d.set_title(f"3D Surface Plot at ({x}, {y})")
        ax_3d.set_xlabel('X Coordinate')
        ax_3d.set_ylabel('Y Coordinate')
        ax_3d.set_zlabel('Counts')
        
        # Redraw the 3D plot figure
        fig_3d.canvas.draw()


def interactive_plot(fits_file_path):
    # Open the FITS file
    with fits.open(fits_file_path) as hdul:
        data = hdul[1].data
    
    # Ensure the data is numeric
    if not np.issubdtype(data.dtype, np.number):
        data = np.array(data, dtype=float)
    
    # Replace NaNs or invalid values with zeros
    data = np.nan_to_num(data)

    # Apply Zscale for intensity normalization
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    # Create the initial 2D plot
    fig, ax = plt.subplots()
    im = ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title("Click a point to generate 3D plot")
    fig.colorbar(im, ax=ax, orientation='vertical', label='Counts')
    
    # Create a second figure for the 3D plot
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Connect the click event to the onclick function
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, data, ax_3d, fig_3d))

    # Show the plots
    plt.show()




# Replace with the path to your FITS file
fits_file_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/HST/MAST_2024-11-27T12_42_50.627Z/HST/ibsp05010_drz.fits"
interactive_plot(fits_file_path)
