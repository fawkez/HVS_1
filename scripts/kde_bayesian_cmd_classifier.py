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

# Add the path to the 'scripts' folder directly
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')


# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *
from scripts.CMD_selection import *



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
        self.p_class = np.mean(self.C)
        self.p_not_class = 1 - self.p_class

        # Fit 2D KDEs for each class with base bandwidth
        XY_class = np.vstack([X_class, Y_class]).T
        XY_not_class = np.vstack([X_not_class, Y_not_class]).T

        self.kde_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_class)
        self.kde_not_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(XY_not_class)
    
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

if __name__ == '__main__':

    # load training data

    data_gaia_big= pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/CMD_training_catalogs/kappa_1.7_speedystar.csv') 

    # select the features
    X = data_gaia_big['bp_rp_corr'].values
    Y = data_gaia_big['implied_M_g_corr'].values
    C = np.bool(data_gaia_big['is_hvs'])

    # train the classifier
    bayesian_kde = BayesianKDEClassifier(X, Y, C, bandwidth=0.1)

    # classify a S5 HVS1
    # get data for the star
    s5_hvs1 = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/s5hvs1-result.fits')

    s5_hvs1 = extinction_correction(s5_hvs1)

    s5_hvs1 = implied_calculations(s5_hvs1)

    s5_hvs1['implied_M_g_corr'] = compute_absolute_magntiude(s5_hvs1['G_corr'], 1000/s5_hvs1['implied_parallax'],[0])
    s5_hvs1['implied_M_g_corr_error'] = compute_absolute_magntiude(s5_hvs1['G_corr'], 1000/(s5_hvs1['implied_parallax']
                                                                     + s5_hvs1['implied_parallax_error']), [0])

    # classify the star
    start = time.time()
    classification, p_class_given_data, p_not_class_given_data, p_data = bayesian_kde.classify(s5_hvs1['bp_rp_corr'],
                                                 s5_hvs1['implied_M_g_corr'], 2e-3, s5_hvs1['implied_M_g_corr_error'].value[0] )
    
    print('For S5 HVS1:')
    print('Classification time:', time.time() - start)
    print('Classification:', classification[0])
    print('P(class|data):', p_class_given_data[0])
    print('P(not class|data):', p_not_class_given_data[0])
    print('P(data):', p_data[0])
        