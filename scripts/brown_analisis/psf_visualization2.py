import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.table import Table
from photutils.psf import PSFPhotometry, GriddedPSFModel, stdpsf_reader
from photutils.background import MMMBackground
from photutils.psf.groupers import SourceGrouper
from astropy.modeling.fitting import LevMarLSQFitter

def interactive_plot(fits_file_path, catalog_path=None, psf_model=None, log_scale=False):
    """
    Main function to visualize the FITS image and perform interactive PSF analysis.

    Parameters:
        fits_file_path (str): Path to the FITS image file.
        catalog_path (str): Path to the catalog file (optional).
        psf_model (GriddedPSFModel): PSF model object.
        log_scale (bool): Whether to display the image in log scale.
    """
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
    if log_scale:
        im = ax_main.imshow(data, origin='lower', cmap='gray', norm=LogNorm(), vmin=vmin, vmax=vmax)
    else:
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

    # Create figure for PSF fit and residuals
    fig_psf, axes_psf = plt.subplots(1, 3, figsize=(15, 5))
    axes_psf[0].set_title('Data')
    axes_psf[1].set_title('Best Fit')
    axes_psf[2].set_title('Residual')

    # Check if PSF model is provided
    if psf_model is None:
        raise ValueError("PSF model must be provided.")

    # Connect the click event
    fig_main.canvas.mpl_connect('button_press_event', lambda event: onclick(
        event, data, ax_3d, fig_3d, ax_row, ax_col, fig_profiles, psf_model, fig_psf, axes_psf))

    plt.show()

def overlay_catalog(ax, catalog_path, wcs):
    """
    Overlay catalog points on the image.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        catalog_path (str): Path to the catalog file.
        wcs (astropy.wcs.WCS): WCS object for coordinate transformation.
    """
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

def onclick(event, data, ax_3d, fig_3d, ax_row, ax_col, fig_profiles, psf_model, fig_psf, axes_psf):
    """
    Handle mouse click events for interactive analysis.

    Parameters:
        event: Matplotlib event object.
        data (ndarray): Image data array.
        ax_3d, ax_row, ax_col: Matplotlib axes for plotting.
        fig_3d, fig_profiles, fig_psf: Matplotlib figures.
        psf_model (GriddedPSFModel): PSF model object.
        axes_psf: Axes for PSF fit and residual plots.
    """
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

    elif event.button == 3:  # Right-click: PSF fitting using provided PSF
        # Perform PSF fitting
        best_model, residual, best_n_components = fit_psf_photometry(region, psf_model)

        # Update the PSF fit and residuals plots
        for ax in axes_psf:
            ax.clear()

        axes_psf[0].imshow(region, origin='lower', cmap='viridis')
        axes_psf[0].set_title('Data')

        axes_psf[1].imshow(best_model, origin='lower', cmap='viridis')
        axes_psf[1].set_title(f'Best Fit ({best_n_components} Component{"s" if best_n_components > 1 else ""})')

        axes_psf[2].imshow(residual, origin='lower', cmap='viridis')
        axes_psf[2].set_title('Residual')

        fig_psf.canvas.draw_idle()

def fit_psf_photometry(region, psf_model):
    """
    Fit the provided PSF to the data region using astropy.modeling.
    Tries fitting with 1, 2, and 3 components and selects the best fit.

    Parameters:
        region (ndarray): The data region to fit.
        psf_model (GriddedPSFModel): The PSF model to use for fitting.

    Returns:
        best_model_image (ndarray): The best-fit model image.
        best_residual (ndarray): The residual between data and model.
        best_n_components (int): The number of components in the best fit.
    """
    from astropy.modeling import models, fitting

    y_size, x_size = region.shape

    # Estimate background
    bkg_estimator = MMMBackground()
    bkg = bkg_estimator(region)

    # Subtract background
    region_sub = region - bkg

    models_list = []
    gof_list = []
    n_components_list = [1, 2, 3]

    for n_components in n_components_list:
        # Create initial positions and amplitudes
        x_cen = x_size / 2
        y_cen = y_size / 2

        # Initialize compound model
        compound_model = None

        for i in range(n_components):
            # Offset positions for initial guess
            offset = (i - n_components // 2) * 2  # Adjust offset as needed
            psf_i = psf_model.copy()

            psf_i.x_0 = x_cen + offset
            psf_i.y_0 = y_cen + offset
            psf_i.flux = region_sub.max() / n_components

            # Combine models
            if compound_model is None:
                compound_model = psf_i
            else:
                compound_model += psf_i

        # Fit the model to the data
        y, x = np.mgrid[:y_size, :x_size]
        fitter = LevMarLSQFitter()
        fitted_model = fitter(compound_model, x, y, region_sub)

        # Generate model image
        model_image = fitted_model(x, y)

        # Add background back
        model_image += bkg

        # Compute residual
        residual = region - model_image

        # Compute goodness-of-fit metric (sum of squared residuals)
        ss_res = np.sum(residual ** 2)

        models_list.append((n_components, model_image))
        gof_list.append(ss_res)

    # Select the model with the lowest goodness-of-fit metric
    best_index = np.argmin(gof_list)
    best_n_components, best_model_image = models_list[best_index]

    # Compute residual for the best model
    best_residual = region - best_model_image

    return best_model_image, best_residual, best_n_components


# Load PSF model from your PSF FITS file using stdpsf_reader
psf_fits_file = "/Users/mncavieres/Documents/2024-2/HVS/Data/HST/PSFSTD_WFC3UV_F606W.fits"

# Read the PSF model using stdpsf_reader
psf_model = stdpsf_reader(psf_fits_file, detector_id=1)

# Replace with the path to your catalog FITS file
catalog_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/hvs10_hst.fits"

# Replace with the path to your FITS file
fits_file_path = "/Users/mncavieres/Documents/2024-2/HVS/Data/Brown_targets/HVS10/HST/MAST_2024-11-27T12_42_50.627Z/HST/ibsp05010_drz.fits"

# Run the interactive plot
interactive_plot(fits_file_path, catalog_path, psf_model=psf_model)
