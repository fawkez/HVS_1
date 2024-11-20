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
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')


# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *
from scripts.CMD_selection import *



# class BayesianKDEClassifier:

#     def __init__(self, X, Y, C, threshold=0.5, bandwidth=1.0):
#         self.X = np.array(X)
#         self.Y = np.array(Y)
#         self.C = np.array(C)
#         self.threshold = threshold
#         self.base_bandwidth = bandwidth

#         # Split data by class
#         X_class = self.X[self.C]
#         Y_class = self.Y[self.C]
#         X_not_class = self.X[~self.C]
#         Y_not_class = self.Y[~self.C]

#         # Calculate prior probabilities
#         self.p_class = np.mean(self.C)
#         self.p_not_class = 1 - self.p_class

#         # Fit 2D KDEs for each class with base bandwidth
#         XY_class = np.vstack([X_class, Y_class]).T
#         XY_not_class = np.vstack([X_not_class, Y_not_class]).T

#         self.kde_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_class)
#         self.kde_not_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_not_class)
    
    
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

class BayesianKDEClassifier:

    def __init__(self, X, Y, C, threshold=0.5, bandwidth=1.0):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.C = np.array(C)
        self.threshold = threshold
        self.base_bandwidth = bandwidth

        # Split data by class
        X_class = self.X[self.C]
        Y_class = self.Y[self.C]
        X_not_class = self.X[~self.C]
        Y_not_class = self.Y[~self.C]

        # Calculate prior probabilities
        self.p_class = np.mean(self.C) # This should be replaced with the HVS fraction in Gaia
        self.p_not_class = 1 - self.p_class

        # Fit 2D KDEs for each class with base bandwidth
        XY_class = np.vstack([X_class, Y_class]).T
        XY_not_class = np.vstack([X_not_class, Y_not_class]).T

        self.kde_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_class)
        self.kde_not_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_not_class)

        # for grid interpolation
        self.X_class = self.X[self.C]
        self.Y_class = self.Y[self.C]
        self.X_not_class = self.X[~self.C]
        self.Y_not_class = self.Y[~self.C]

        # Number of samples in each class
        self.n_class = len(self.X_class)
        self.n_not_class = len(self.X_not_class)

    def classify(self, x, y, x_err=0, y_err=0):
            """
            Classify points based on the KDE-based Bayesian model with analytical error integration.

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
            # Adjust the bandwidth based on measurement error (assuming isotropic errors)
            effective_bandwidth = np.sqrt(self.base_bandwidth**2 + x_err**2 + y_err**2)
            #print(effective_bandwidth)
            # Create temporary KDEs with adjusted bandwidths for each case
            kde_class = KernelDensity(kernel='gaussian', bandwidth=effective_bandwidth)
            kde_class.fit(np.vstack([self.X[self.C], self.Y[self.C]]).T)
            
            kde_not_class = KernelDensity(kernel='gaussian', bandwidth=effective_bandwidth)
            kde_not_class.fit(np.vstack([self.X[~self.C], self.Y[~self.C]]).T)
            
            # Prepare input data
            data = np.vstack([x, y]).T
            
            # Calculate log-likelihoods with adjusted KDEs
            log_p_data_given_class = kde_class.score_samples(data)
            log_p_data_given_not_class = kde_not_class.score_samples(data)
            
            # Convert log-likelihoods to likelihoods
            p_data_given_class = np.exp(log_p_data_given_class)
            p_data_given_not_class = np.exp(log_p_data_given_not_class)
            
            # Total probability of data
            p_data = p_data_given_class * self.p_class + p_data_given_not_class * self.p_not_class
            
            # Posterior probabilities P(class|data) and P(not class|data)
            p_class_given_data = (p_data_given_class * self.p_class) / p_data
            p_not_class_given_data = (p_data_given_not_class * self.p_not_class) / p_data
            
            # Classification based on threshold
            classification = p_class_given_data >= self.threshold
            
            return classification, p_class_given_data, p_not_class_given_data, p_data


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

    def save(self, filename):
        """
        Save the KDE object to a file.

        Parameters:
        - filename (str): Name of the file to save the classifier.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
    def load(self, filename):
        """
        Load the KDE object from a pickle dump.

        Parameters:
        - filename (str): Name of the file to load the classifier.
        """
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self
        



def flux_to_mag(flux):
    return -2.5*np.log10(flux)

if __name__ == '__main__':

    # load training data

    data_gaia_big= pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/CMD_training_catalogs/kappa_1.7_speedystar.csv') 

    # select the features
    X = data_gaia_big['bp_rp_corr'].values
    Y = data_gaia_big['implied_M_g_corr'].values
    C = np.bool(data_gaia_big['is_hvs'])

    # train the classifier
    bayesian_kde = BayesianKDEClassifier(X, Y, C, bandwidth=0.1)

    # compute the KDE grid for further interpolation
    bayesian_kde.compute_kde_grid()

    # plot the KDE fit
    #bayesian_kde.debug_kde_fit()


    # classify a S5 HVS1
    # get data for the star
    s5_hvs1 = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/s5hvs1-result.fits')

    s5_hvs1 = extinction_correction(s5_hvs1)

    s5_hvs1 = implied_calculations(s5_hvs1)

    s5_hvs1['bp_rp_corr_error'] = np.sqrt(flux_to_mag(s5_hvs1['phot_rp_mean_flux_error'])**2
         + flux_to_mag(s5_hvs1['phot_bp_mean_flux_error'])**2)

    s5_hvs1['implied_M_g_corr'] = compute_absolute_magntiude(s5_hvs1['G_corr'], 1000/s5_hvs1['implied_parallax'],[0])
    s5_hvs1['implied_M_g_corr_error'] = compute_absolute_magntiude(s5_hvs1['G_corr'], 1000/(s5_hvs1['implied_parallax']
                                                                     + s5_hvs1['implied_parallax_error']), [0])

    # classify the star
    start = time.time()
    classification, p_class_given_data, p_not_class_given_data, p_data = bayesian_kde.classify(s5_hvs1['bp_rp_corr'],
                                                 s5_hvs1['implied_M_g_corr'], 0.3, s5_hvs1['implied_M_g_corr_error'].value[0] )
    
    print('For S5 HVS1:')
    print('Re-fit the KDE using a new bandwidth given by the measurement errors')
    print('Classification time:', time.time() - start)
    print('Classification:', classification[0])
    print('P(class|data):', p_class_given_data[0])
    print('P(not class|data):', p_not_class_given_data[0])
    print('P(data):', p_data[0])
    print('\n')
        

    print('Testing with integral method')
    
    start = time.time()
    # Assuming bayesian_kde is an instance of BayesianKDEClassifier
    classification, p_class_given_data, p_not_class_given_data, p_data = bayesian_kde.classify_integral(
        s5_hvs1['bp_rp_corr'],
        s5_hvs1['implied_M_g_corr'],
        x_err= s5_hvs1['bp_rp_corr_error'],  # Assuming you have this error
        y_err=s5_hvs1['implied_M_g_corr_error'].value[0]
    )

    print('For S5 HVS1 using classify_integral:')
    print('Classification:', classification[0])
    print('P(class|data):', p_class_given_data[0])
    print('P(not class|data):', p_not_class_given_data[0])
    print('P(data):', p_data[0])
    print('Classification time:', time.time() - start)
    print('\n')

    print('Testing with convolution and interpolation method')

    # Using classify_with_error_convolution
    start = time.time()
    classification_conv, p_class_conv, p_not_class_conv, p_data_conv = bayesian_kde.classify_with_error_convolution(
        s5_hvs1['bp_rp_corr'],
        s5_hvs1['implied_M_g_corr'],
        x_err=s5_hvs1['bp_rp_corr_error'],
        y_err=s5_hvs1['implied_M_g_corr_error'].value[0]
    )
    # save the results
    s5_hvs1['classification'] = classification_conv
    s5_hvs1['p_hvs'] = p_class_conv
    s5_hvs1['p_background'] = p_not_class_conv
    s5_hvs1['p_data'] = p_data_conv
    s5_hvs1.write('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/s5hvs1-result_with_probabilities.fits', overwrite=True)



    print('Classification time convolution:', time.time() - start)
    print('Classification:', classification_conv[0])
    print('P(class|data):', p_class_conv[0])
    print('P(not class|data):', p_not_class_conv[0])
    print('P(data):', p_data_conv[0])
    print('\n')

    # Compare results
    print('Difference in P(class|data):', p_class_given_data[0] - p_class_conv[0])

    # now for the training set
    print('Classifying the training set')

    # # obtain the posterior probability distribution for the training set
    data_gaia_big['implied_M_g_corr_error'] = compute_absolute_magntiude(data_gaia_big['G_corr'], 1000/(data_gaia_big['implied_parallax']
                                                                      + data_gaia_big['implied_parallax_error']), [0])
    bayesian_kde.compute_kde_grid(x_range=(-1, 2.5), y_range=(-7, 15), resolution=100)

    # Prepare arrays
    p_class_given_data_train = np.zeros(len(data_gaia_big))
    p_not_class_given_data_train = np.zeros(len(data_gaia_big))
    classification_array = np.zeros(len(data_gaia_big), dtype=bool)

    # Loop over data
    for i in tqdm(range(len(data_gaia_big))):
        x = data_gaia_big['bp_rp_corr'].values[i]
        y = data_gaia_big['implied_M_g_corr'].values[i]
        x_err = data_gaia_big['bp_rp_corr_error'].values[i] if 'bp_rp_corr_error' in data_gaia_big else 0.01  # Adjust as needed
        y_err = np.abs(data_gaia_big['implied_M_g_corr_error'].values[i])

        # Ensure x and y are within the grid ranges
        if (-1 <= x <= 2.5) and (-7 <= y <= 15):
            classif, p_class, p_not_class, _ = bayesian_kde.classify_with_error_convolution([x], [y], x_err, y_err)
            classification_array[i] = classif[0]
            p_class_given_data_train[i] = p_class[0]
            p_not_class_given_data_train[i] = p_not_class[0]
        else:
            classification_array[i] = False
            p_class_given_data_train[i] = np.nan
            p_not_class_given_data_train[i] = np.nan

    # Add results to DataFrame
    data_gaia_big['p_hvs'] = p_class_given_data_train
    data_gaia_big['p_background'] = p_not_class_given_data_train
    data_gaia_big['classification'] = classification_array

    # Save results
    data_gaia_big.to_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/CMD_training_catalogs/kappa_1.7_speedystar_with_probabilities.csv', index=False)
