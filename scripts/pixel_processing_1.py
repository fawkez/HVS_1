"""
This python script will perform the following tasks:
1. Load the data for 1 lvl 4 HEALPix pixel
2. Compute the implied quantities
3. Keep only positive implied parallaxes
4. Apply the parallax consistency check
5. Add extinction correction
6. Compute the implied absolute magnitude
7. Fit KDE to background stars
8. Compute the KDE probability for each star
9. Save the results
10. Save stars with KDE probability > 0.5 to a separate file
"""
# fit a 2D KDE to the CMD
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
from astropy.table import Table
from tqdm import tqdm

# Add the path to the 'scripts' folder directly
# This needs to be changed to the folder in which I will have the scripts in ALICE
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS') 


# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *
from scripts.CMD_selection import *

   
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator



class BayesianKDEClassifier:

    def __init__(self, X_class, Y_class, X_not_class, Y_not_class, threshold=0.5, bandwidth=1.0):

        self.threshold = threshold
        self.base_bandwidth = bandwidth

        # Split data by class
        self.X_class = X_class
        self.Y_class = Y_class
        self.X_not_class = X_not_class
        self.Y_not_class = Y_not_class

        # Calculate prior probabilities
        self.p_class = 1e-5 # This should be replaced with the HVS fraction in Gaia
        self.p_not_class = 1 - self.p_class

        # Fit 2D KDEs for each class with base bandwidth
        XY_class = np.vstack([self.X_class, self.Y_class]).T
        XY_not_class = np.vstack([self.X_not_class,self.Y_not_class]).T

        self.kde_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_class)
        self.kde_not_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_not_class)


        # Number of samples in each class
        self.n_class = len(self.X_class)
        self.n_not_class = len(self.X_not_class)


    def compute_kde_grid(self, x_range=(-1, 2.5), y_range=(-7, 15), resolution=100):
        """
        Compute a grid of KDE evaluations for each class.
        """
        X_range = np.linspace(x_range[0], x_range[1], resolution)
        Y_range = np.linspace(y_range[0], y_range[1], resolution)
        X_mesh, Y_mesh = np.meshgrid(X_range, Y_range)
        XY_mesh = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T

        log_p_class_grid = self.kde_class.score_samples(XY_mesh).reshape(resolution, resolution)
        log_p_not_class_grid = self.kde_not_class.score_samples(XY_mesh).reshape(resolution, resolution)

        self.p_class_grid = np.exp(log_p_class_grid)
        self.p_not_class_grid = np.exp(log_p_not_class_grid)
        self.X_range, self.Y_range = X_range, Y_range

    def classify_with_error_convolution(self, x, y, x_err, y_err):
        """
        Classify points using convolution with Gaussian error and grid interpolation.
        """
        # Create interpolators for the KDE grids
        interpolator_class = RegularGridInterpolator((self.X_range, self.Y_range), self.p_class_grid)
        interpolator_not_class = RegularGridInterpolator((self.X_range, self.Y_range), self.p_not_class_grid)

        # Convolve KDE grids with Gaussian filters based on errors
        convolved_class_grid = gaussian_filter(self.p_class_grid, sigma=[x_err, y_err])
        convolved_not_class_grid = gaussian_filter(self.p_not_class_grid, sigma=[x_err, y_err])

        # Interpolators for convolved KDE grids
        convolved_interpolator_class = RegularGridInterpolator((self.X_range, self.Y_range), convolved_class_grid)
        convolved_interpolator_not_class = RegularGridInterpolator((self.X_range, self.Y_range), convolved_not_class_grid)

        # Interpolate convolved KDEs at the (x, y) point
        data_points = np.vstack([x, y]).T
        p_data_given_class = convolved_interpolator_class(data_points)
        p_data_given_not_class = convolved_interpolator_not_class(data_points)

        # Calculate total probability and posterior probabilities
        p_data = p_data_given_class * self.p_class + p_data_given_not_class * self.p_not_class
        p_class_given_data = (p_data_given_class * self.p_class) / p_data
        p_not_class_given_data = (p_data_given_not_class * self.p_not_class) / p_data

        # Classification based on threshold
        classification = p_class_given_data >= self.threshold

        return classification, p_class_given_data, p_not_class_given_data, p_data



    def classify_integral(self, x, y, x_err, y_err):
        """
        Classify points by performing the 2D integral of the measurement Gaussian
        with the KDE for each class.

        Parameters:
        - x (array-like): Input data for feature X.
        - y (array-like): Input data for feature Y.
        - x_err (float or array-like): Standard deviation of errors in X.
        - y_err (float or array-like): Standard deviation of errors in Y.

        Returns:
        - classification (bool array): True if classified as part of the class, False otherwise.
        - p_class_given_data (array): Probability of being in the class.
        - p_not_class_given_data (array): Probability of not being in the class.
        - p_data (array): Total probability of the data.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        x_err = np.atleast_1d(x_err)
        y_err = np.atleast_1d(y_err)

        # Ensure arrays are of the same length
        if len(x_err) == 1:
            x_err = np.full_like(x, x_err[0])
        if len(y_err) == 1:
            y_err = np.full_like(y, y_err[0])

        # Bandwidth squared
        h2 = self.base_bandwidth ** 2

        # For each data point
        p_data_given_class = []
        p_data_given_not_class = []

        for xi, yi, xei, yei in zip(x, y, x_err, y_err):
            # Variances for convolution
            var_x_class = xei ** 2 + h2
            var_y_class = yei ** 2 + h2

            # Compute for class
            delta_x_class = xi - self.X_class
            delta_y_class = yi - self.Y_class
            exponent_class = -0.5 * ((delta_x_class ** 2) / var_x_class + (delta_y_class ** 2) / var_y_class)
            norm_class = (2 * np.pi * np.sqrt(var_x_class * var_y_class)) * self.n_class
            sum_weights_class = np.sum(np.exp(exponent_class))
            p_d_given_class = sum_weights_class / norm_class
            p_data_given_class.append(p_d_given_class)

            # Compute for not class
            var_x_not_class = xei ** 2 + h2
            var_y_not_class = yei ** 2 + h2

            delta_x_not_class = xi - self.X_not_class
            delta_y_not_class = yi - self.Y_not_class
            exponent_not_class = -0.5 * ((delta_x_not_class ** 2) / var_x_not_class + (delta_y_not_class ** 2) / var_y_not_class)
            norm_not_class = (2 * np.pi * np.sqrt(var_x_not_class * var_y_not_class)) * self.n_not_class
            sum_weights_not_class = np.sum(np.exp(exponent_not_class))
            p_d_given_not_class = sum_weights_not_class / norm_not_class
            p_data_given_not_class.append(p_d_given_not_class)

        p_data_given_class = np.array(p_data_given_class)
        p_data_given_not_class = np.array(p_data_given_not_class)

        # Total probability of data
        p_data = p_data_given_class * self.p_class + p_data_given_not_class * self.p_not_class

        # Posterior probabilities P(class|data) and P(not class|data)
        p_class_given_data = (p_data_given_class * self.p_class) / p_data
        p_not_class_given_data = (p_data_given_not_class * self.p_not_class) / p_data

        # Classification based on threshold
        classification = p_class_given_data >= self.threshold

        return classification, p_class_given_data, p_not_class_given_data, p_data



    def debug_kde_fit(self, x_range=(-1, 2.5), y_range=(15, -7), resolution=100):
        """
        Debug KDE by plotting it alongside the original data to compare fit.

        Parameters:
        - x_range (tuple): Range of x values for the plot (default is (-1, 2.5)).
        - y_range (tuple): Range of y values for the plot (default is (15, -7)).
        - resolution (int): Number of points along each axis (default is 100).
        """

        # calculate log-likelihoods with the KDEs for a range of data points to plot contours
        X_range = np.linspace(x_range[0], x_range[1], resolution)
        Y_range = np.linspace(y_range[0], y_range[1], resolution)
        X_mesh, Y_mesh = np.meshgrid(X_range, Y_range)
        XY_mesh = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T

        log_p_data_given_hvs = self.kde_class.score_samples(XY_mesh)
        log_p_data_given_not_hvs = self.kde_not_class.score_samples(XY_mesh)

        # plot the KDE and the original data for each class
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Plot the "class" data
        ax[0].scatter(self.X[self.C], self.Y[self.C], s=1, c='blue', alpha=0.5)
        cbar_1 = ax[0].contourf(X_mesh, Y_mesh, np.exp(log_p_data_given_hvs).reshape(resolution, resolution),
                                 cmap='Blues', levels = 20, alpha = 0.6)
        ax[0].set_title('HVS KDE')
        ax[0].set_xlim(x_range)
        ax[0].set_ylim(y_range)
        ax[0].set_xlabel("$G_{bp} - G_{rp}$")
        ax[0].set_ylabel("$G_{I}$")

        # Plot the "not class" data
        ax[1].scatter(self.X[~self.C], self.Y[~self.C], s=1, c='red', alpha=0.5)
        cbar_2 = ax[1].contourf(X_mesh, Y_mesh, np.exp(log_p_data_given_not_hvs).reshape(resolution, resolution),
                                 cmap='Reds', levels = 30, alpha = 0.6)
        ax[1].set_title('Non-HVS KDE')
        ax[1].set_xlim(x_range)
        ax[1].set_ylim(y_range)
        ax[1].set_xlabel("$G_{bp} - G_{rp}$")
        ax[1].set_ylabel("$G_{I}$")

        # add colorbars
        fig.colorbar(cbar_1, ax=ax[0], label = '$P(data|hvs)$')
        fig.colorbar(cbar_2, ax=ax[1], label = '$P(data|non-hvs)$')

        plt.tight_layout()
        plt.show()

def process_pixel(data):
    """
    Process a single HEALPix pixel.

    Parameters:
    - data (astropy Table): Data for the HEALPix pixel.

    Returns:
    - results (astropy Table): Results of the processing.
    - high_prob (astropy Table): Stars with high probability of being HVS.
    """

    # Compute implied quantities
    print('Computing implied quantities...')
    data = implied_calculations(data)

    # Keep only positive implied parallaxes
    data = data.loc[data['implied_parallax'] > 0]

    # Apply the parallax consistency check
    print('Applying parallax consistency check...')
    data['parallax_consistency'] = parallax_consistency(data)

    # Compute the implied absolute magnitude
    data['implied_M_g'] = data['phot_g_mean_mag'] - 5*np.log10(1000/data['implied_parallax']) + 5

    # Compute the implied absolute magnitude with using the two bounds of the parallax error
    data['implied_M_g_lo'] = np.abs(data['phot_g_mean_mag'] - 5*np.log10(1000/(data['implied_parallax'] + data['implied_parallax_error'])) + 5 - data['implied_M_g'])
    data['implied_M_g_hi'] = np.abs(data['phot_g_mean_mag'] - 5*np.log10(1000/(data['implied_parallax'] - data['implied_parallax_error'])) + 5 - data['implied_M_g'])
    data['implied_M_g_error'] = np.mean(data['implied_M_g_lo'], data['implied_M_g_hi'])

    # Add extinction correction
    print('Adding extinction correction...')
    data = extinction_correction(data)

    # # Add color error from bp and rp flux errors
    # data['bp_rp_flux_error'] = np.sqrt(data['phot_bp_mean_flux_error']**2 + data['phot_rp_mean_flux_error']**2)
    # data['bp_rp_error'] = np.abs(2.5*np.log10(data['bp_rp_flux_error']))
    # Compute magnitude errors for BP and RP bands
    const = 2.5 / np.log(10)
    data['phot_bp_error'] = const * (data['phot_bp_mean_flux_error'] / data['phot_bp_mean_flux'])
    data['phot_rp_error'] = const * (data['phot_rp_mean_flux_error'] / data['phot_rp_mean_flux'])

    # Combine errors to compute color uncertainty
    data['bp_rp_error'] = np.sqrt(data['phot_bp_error']**2 + data['phot_rp_error']**2)

    # Load the simulated catalog for training the KDE
    simulated_catalog = pd.read_csv('/data1/cavierescarrera/simulated_catalogs/top_heavy_k17_with_implied.csv')

    # define the KDE classifier object
    X_not_class = data['bp_rp_corr']
    Y_not_class = data['implied_M_g_corr']

    X_class = simulated_catalog['bp_rp_corr']
    Y_class = simulated_catalog['implied_M_g_corr']

    print('Fitting KDE...')
    classifier = BayesianKDEClassifier(X_class, Y_class,
                 X_not_class, Y_not_class, threshold=0.5,
                  bandwidth=0.1)
    
    # Compute the KDE grid
    print('Computing KDE grid...')
    classifier.compute_kde_grid()

    # compute probabilities 
    print('Computing HVS probability')
    classification, p_class_given_data, p_not_class_given_data, p_data = classifier.classify_convolution(data['bp_rp_corr'], 
                                                                    data['implied_M_g_corr'],
                                                                    data['bp_rp_error'] ,
                                                                    data['implied_M_g_error']) 
    
    # Save the results
    data['p_hvs'] = p_class_given_data  
    data['p_not_hvs'] = p_not_class_given_data
    data['p_data'] = p_data
    data['is_hvs'] = classification

    return data


if __name__ == '__main__':

    gaia_data_path = '/data1/cavierescarrera/gaia_dr3'
    output_path = '/data1/cavierescarrera/probability_catalogs/with_probability'
    output_path_high_prob = '/data1/cavierescarrera/probability_catalogs/high_prob'
    
    # Load the data 
    for file in os.listdir(gaia_data_path):
        # Only load fits files
        if file.endswith('.fits'):
            data = Table.read(os.path.join(gaia_data_path, file))
            break
        print('Loaded data for pixel', file)
        
        # Process the pixel
        results = process_pixel(data)

        # Save the results
        results.write(os.path.join(output_path, file), overwrite=True)

        # Save stars with high probability of being HVS
        high_prob = results[results['p_hvs'] > 0.9]
        high_prob.write(os.path.join(output_path, 'high_prob', file), overwrite=True)

        print('Processed pixel', file)