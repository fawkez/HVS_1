import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.table import Table
from astropy.modeling import models, fitting
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def interactive_plot(fits_file_path, catalog_path=None, psf_stddev=2, log_scale=False):
    # Open the FITS file and extract data and WCS information
    with fits.open(fits_file_path) as hdul:
        data = hdul[1].data.astype(float)
        header = hdul[1].header
        wcs = WCS(header)

    # Replace NaN values with zero
    data = np.nan_to_num(data)

    # Apply ZScale for image normalization
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    # Create the main figure with WCS projection
    fig_main, ax_main = plt.subplots(subplot_kw={'projection': wcs})
    im = ax_main.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax_main.set_xlabel('RA')
    ax_main.set_ylabel('DEC')
    ax_main.set_title('FITS Image - Click to Analyze')

    # Overlay catalog if provided
    if catalog_path:
        overlay_catalog(ax_main, catalog_path, wcs)

    # Create figures for the interactive plots
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    fig_profiles, (ax_row, ax_col) = plt.subplots(1, 2, figsize=(12, 5))

    # Connect the click event
    fig_main.canvas.mpl_connect('button_press_event', lambda event: onclick(event, data, ax_3d, fig_3d, ax_row, ax_col, fig_profiles, psf_stddev))
    plt.tight_layout()
    plt.show()

def overlay_catalog(ax, catalog_path, wcs):
    """Overlay catalog points on the image."""
    try:
        catalog = Table.read(catalog_path)
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return

    if 'RA' in catalog.colnames and 'DEC' in catalog.colnames:
        ra = catalog['RA']
        dec = catalog['DEC']
        x, y = wcs.world_to_pixel_values(ra, dec)
        ax.scatter(x, y, s=20, color='red', alpha=0.7, label='Catalog')
        ax.legend(loc='upper right')
    else:
        print("Catalog does not contain 'RA' and 'DEC' columns. Skipping overlay.")

def onclick(event, data, ax_3d, fig_3d, ax_row, ax_col, fig_profiles, psf_stddev):
    if event.inaxes is None:
        return

    xdata = event.xdata
    ydata = event.ydata

    if xdata is None or ydata is None:
        return

    x = int(xdata)
    y = int(ydata)

    window_size = 20  # Size of the region to analyze
    x_min = max(0, x - window_size // 2)
    x_max = min(data.shape[1], x + window_size // 2)
    y_min = max(0, y - window_size // 2)
    y_max = min(data.shape[0], y + window_size // 2)

    region = data[y_min:y_max, x_min:x_max]

    # Coordinates for plotting
    x_vals = np.arange(x_min, x_max)
    y_vals = np.arange(y_min, y_max)

    if event.button == 1:  # Left-click: 3D plot and profiles
        # Update 3D plot
        X, Y = np.meshgrid(x_vals, y_vals)
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, region, cmap='viridis')
        ax_3d.set_title(f"3D Surface Plot at ({x}, {y})")
        ax_3d.set_xlabel('X Pixel')
        ax_3d.set_ylabel('Y Pixel')
        ax_3d.set_zlabel('Counts')
        fig_3d.canvas.draw_idle()

        # Update row profile
        ax_row.clear()
        row_data = data[y, x_min:x_max]
        ax_row.plot(x_vals, row_data, label='Data')
        ax_row.set_title(f"Row Profile at Y={y}")
        ax_row.set_xlabel('X Pixel')
        ax_row.set_ylabel('Counts')

        # Update column profile
        ax_col.clear()
        col_data = data[y_min:y_max, x]
        ax_col.plot(y_vals, col_data, label='Data')
        ax_col.set_title(f"Column Profile at X={x}")
        ax_col.set_xlabel('Y Pixel')
        ax_col.set_ylabel('Counts')

        fig_profiles.canvas.draw_idle()

    elif event.button == 3:  # Right-click: Component analysis
        # Fit 2D Gaussian models with 1, 2, and 3 components
        best_model, best_n = fit_gaussian_2d(region, psf_stddev)

        # Generate model data for plotting
        y_grid, x_grid = np.mgrid[0:region.shape[0], 0:region.shape[1]]
        model_data = best_model(x_grid, y_grid)

        # Plot the best fit
        fig_fit, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(region, origin='lower', cmap='viridis')
        axes[0].set_title('Data')
        axes[1].imshow(model_data, origin='lower', cmap='viridis')
        axes[1].set_title(f'Best Fit ({best_n} Component{"s" if best_n > 1 else ""})')
        residual = region - model_data
        axes[2].imshow(residual, origin='lower', cmap='viridis')
        axes[2].set_title('Residual')
        plt.show()

def fit_gaussian_2d(data, psf_stddev):
    """Fit 2D Gaussian models with 1, 2, and 3 components and select the best one."""
    y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    models_list = []
    aic_list = []

    for n in [1, 2, 3]:
        # Create an initial guess for the model
        model_init = None
        for i in range(n):
            amp = data.max()
            x_mean = data.shape[1] / 2
            y_mean = data.shape[0] / 2
            gauss = models.Gaussian2D(amplitude=amp, x_mean=x_mean, y_mean=y_mean,
                                      x_stddev=psf_stddev, y_stddev=psf_stddev)
            if model_init is None:
                model_init = gauss
            else:
                model_init += gauss

        # Fit the model to the data
        fitter = fitting.LevMarLSQFitter()
        with np.errstate(all='ignore'):
            fitted_model = fitter(model_init, x, y, data)

        # Calculate AIC
        residual = data - fitted_model(x, y)
        ss_res = np.sum(residual**2)
        k = fitted_model.parameters.size
        aic = 2 * k + data.size * np.log(ss_res / data.size)
        models_list.append(fitted_model)
        aic_list.append(aic)

    # Select the model with the lowest AIC
    best_index = np.argmin(aic_list)
    best_model = models_list[best_index]
    best_n = best_index + 1  # Because n starts from 1
    return best_model, best_n


# Replace with the path to your catalog FITS file
catalog_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/hvs10_hst.fits"


# Replace with the path to your FITS file
fits_file_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/HST/MAST_2024-11-27T12_42_50.627Z/HST/ibsp05010_drz.fits"


# Set `log_scale=True` for logarithmic Z-axis in the 3D plot
interactive_plot(fits_file_path, catalog_path=catalog_path)
