import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.path import Path

# Extract necessary columns

def compute_CMD_hvs_ratio(data_gaia_big, x_col= 'bp_rp_corr', y_col= 'implied_M_g_corr', is_hvs_col= 'is_hvs', n_bins_x= 100, n_bins_y= 100):
    """
    Compute the ratio of high-velocity stars (HVS) to total stars in a 2D histogram of the color-magnitude diagram (CMD).

    inputs:
        data_gaia_big: pandas DataFrame containing the Gaia DR3 data
        x_col: column name for the x-axis (bp_rp_corr)
        y_col: column name for the y-axis (implied_M_g_corr)
        is_hvs_col: column name for the high-velocity star flag (is_hvs)
        n_bins_x: number of bins for the x-axis
        n_bins_y: number of bins for the y-axis

    returns:
        ratio_hist: 2D numpy array of the ratio histogram
        x_bins: 1D numpy array of bin edges for the x-axis
        y_bins: 1D numpy array of bin edges for the y-axis
    """

    # Filter data to avoid outliers and wasting bins on non useful data       
    data_gaia_big_for_hist = data_gaia_big.loc[(data_gaia_big[y_col] < 15) 
                                            & (data_gaia_big[y_col] > -8)
                                            & (data_gaia_big[x_col] < 2.5) 
                                            & (data_gaia_big[x_col] > -1)]
    
    x = data_gaia_big_for_hist[x_col].values
    y = data_gaia_big_for_hist[y_col].values
    is_hvs = data_gaia_big_for_hist[is_hvs_col].values

    # Define bin edges for X (bp_rp_corr) and Y (implied_M_g_corr)
    num_bins_x = n_bins_x  # Adjust the number of bins as needed
    num_bins_y = n_bins_y  # Adjust the number of bins as needed

    x_bins = np.linspace(np.min(x), np.max(x), num_bins_x + 1)
    y_bins = np.linspace(np.min(y), np.max(y), num_bins_y + 1)

    # 2D histogram of total objects
    total_hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # 2D histogram of high-velocity stars (HVS)
    hvs_hist, _, _ = np.histogram2d(x[is_hvs == 1], y[is_hvs == 1], bins=[x_bins, y_bins])

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_hist = np.divide(hvs_hist, total_hist)
        ratio_hist[total_hist == 0] = np.nan  # Set bins with no total objects to NaN

    return ratio_hist, x_bins, y_bins



def smooth_histogram(ratio_hist, x_bins, y_bins, sigma= 1):
    """
    Apply a gaussian kernel based smoothing to the ratio histogram.

    inputs:
        ratio_hist: 2D numpy array of the ratio histogram
        x_bins: 1D numpy array of bin edges for the x-axis
        y_bins: 1D numpy array of bin edges for the y-axis
        sigma: Standard deviation for the gaussian kernel

    returns:
        Z_smooth: 2D numpy array of the smoothed ratio histogram
        X_grid: 2D numpy array of the x-axis bin centers
        Y_grid: 2D numpy array of the y-axis bin centers
    """

    # Parameters for Gaussian smoothing
    # convert sigma from mag to pixels assuming 0.035 mag per pixel
    sigma = sigma  # Standard deviation for Gaussian kernel in pixels; adjust for more or less smoothing

    # Apply Gaussian smoothing to the histogram
    ratio_hist_filled = np.nan_to_num(ratio_hist, nan=0.0)
    Z_smooth = gaussian_filter(ratio_hist_filled.T, sigma=sigma)

    # Create meshgrid for the histogram bin centers
    X_centers = (x_bins[:-1] + x_bins[1:]) / 2
    Y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    X_grid, Y_grid = np.meshgrid(X_centers, Y_centers)

    return Z_smooth, X_grid, Y_grid


def oversample_histogram(ratio_hist, x_bins, y_bins):
    from scipy.interpolate import RegularGridInterpolator

    # Original grid coordinates (bin centers)
    X_centers = (x_bins[:-1] + x_bins[1:]) / 2
    Y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    # Define the new higher-resolution grid
    oversample_factor = 4  # Adjust this factor to control oversampling
    num_x = len(X_centers) * oversample_factor
    num_y = len(Y_centers) * oversample_factor

    X_new = np.linspace(X_centers.min(), X_centers.max(), num_x)
    Y_new = np.linspace(Y_centers.min(), Y_centers.max(), num_y)
    X_grid_new, Y_grid_new = np.meshgrid(X_new, Y_new)

    # Interpolate the smoothed data onto the new grid
    interpolating_function = RegularGridInterpolator(
        (Y_centers, X_centers), ratio_hist, bounds_error=False, fill_value=0
    )
    points = np.array([Y_grid_new.ravel(), X_grid_new.ravel()]).T
    Z_smooth_oversampled = interpolating_function(points)
    Z_smooth_oversampled = Z_smooth_oversampled.reshape(X_grid_new.shape)

    return Z_smooth_oversampled, X_grid_new, Y_grid_new, X_centers, Y_centers

def make_contours(Z_smooth_oversampled, X_grid_new, Y_grid_new, levels = 10):
    """
    Create contour lines for the smoothed and oversampled ratio histogram.

    inputs:
        Z_smooth_oversampled: 2D numpy array of the smoothed and oversampled ratio histogram
        X_grid_new: 2D numpy array of the x-axis bin centers for the oversampled grid
        Y_grid_new: 2D numpy array of the y-axis bin centers for the oversampled grid
        levels: number of contour levels to plot

    returns:
        contour_lines: matplotlib contour object
    """

    # Define contour levels
    # Define levels for the contour
    contour_levels = np.linspace(0, 1, levels)

    # Generate the contour plot from the oversampled, smoothed data
    contour_lines = plt.contour(
        X_grid_new, Y_grid_new, Z_smooth_oversampled,
        levels=contour_levels, cmap='coolwarm', vmin=0, vmax=1
    )
    # do not show the plot
    
    return contour_lines, contour_levels


def filter_points_within_contour(contour_lines, df, x_col = 'bp_rp_corr', y_col= 'implied_M_g_corr', level=None):
    """
    Filters a DataFrame to include only points within a specified contour level.

    Parameters:
        contour_lines: QuadContourSet object from plt.contour.
        df: pandas DataFrame containing the data points.
        x_col: Name of the column in df representing x-values.
        y_col: Name of the column in df representing y-values.
        level: The contour level to use for filtering. Defaults to the maximum level.

    Returns:
        filtered_df: DataFrame containing only points within the specified contour.
    """
    # Determine the contour level to use
    if level is None:
        level = contour_lines.levels[-2]#np.max(contour_lines.levels)
    
    # Find the index of the specified level
    try:
        level_idx = np.where(contour_lines.levels == level)[0][0]
    except IndexError:
        raise ValueError(f"Level {level} is not in the contour levels.")
    
    
    # Get the collections of LineCollections at the specified level
    collection = contour_lines.collections[level_idx]
  
    # Get all the paths (contours) at this level
    paths = []
    for path in collection.get_paths():
        # For each sub-path in the path (in case of discontinuous contours)
        vertices = path.vertices
        codes = path.codes
        if codes is None:
            # Simple path
            paths.append(Path(vertices))
        else:
            # Compound path
            paths.append(Path(vertices, codes))
    
    # Prepare the points to test
    points = df[[x_col, y_col]].values
    
    # Initialize a mask of False
    mask = np.zeros(len(df), dtype=bool)
    
    # Check each path and update the mask
    for path in paths:
        mask |= path.contains_points(points)
    
    # Filter the DataFrame
    filtered_df = df[mask].copy()
    
    return filtered_df


