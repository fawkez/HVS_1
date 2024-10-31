# imports
# imports 

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.constants import kpc, au
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from numba import njit

import random
import healpy as hp

from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize


# Add the path to the 'scripts' folder directly
sys.path.append('/Users/mncavieres/Documents/2024-2/HVS')


# Now you can import from the 'scripts' package
from scripts.implied_d_vr import *  # Or import any other module
from scripts.selections import *
from scripts.CMD_selection import *
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# set up the plotting
# set font size
plt.rcParams.update({'font.size': 18})
# set the figure size
plt.rcParams.update({'figure.figsize': (10, 7)})
# set the font to latex
plt.rcParams.update({'text.usetex': True})


# set the path to save plots
plots_path = '/Users/mncavieres/Documents/2024-2/HVS/Plots/new_cmd'


class BayesianGMMClassifier:
    def __init__(self, X, Y, C, threshold=0.5, n_components=3):
        """
        Initialize a Bayesian classifier based on Gaussian Mixture Models (GMMs).

        Parameters:
        - X (array-like): Input data for feature X.
        - Y (array-like): Input data for feature Y.
        - C (array-like): Boolean feature indicating if a point is in the True class.
        - threshold (float): Probability threshold for classification (default is 0.5).
        - n_components (int): Number of components for the Gaussian Mixture Model.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.C = np.array(C)
        self.threshold = threshold
        self.n_components = n_components

        # Split data by class
        X_class = np.column_stack((self.X[self.C], self.Y[self.C]))
        X_not_class = np.column_stack((self.X[~self.C], self.Y[~self.C]))

        # Calculate prior probabilities
        self.p_class = np.mean(self.C)
        self.p_not_class = 1 - self.p_class

        # Fit GMM for each class
        self.gmm_class = GaussianMixture(n_components=self.n_components).fit(X_class)
        self.gmm_not_class = GaussianMixture(n_components=self.n_components).fit(X_not_class)
        
    def classify(self, x, y):
        """
        Classify points based on the GMM-based Bayesian model.

        Parameters:
        - x (array-like): Input data for feature X.
        - y (array-like): Input data for feature Y.

        Returns:
        - classification (bool array): True if classified as part of the class, False otherwise.
        - p_class_given_data (array): Probability of being in the class.
        - p_not_class_given_data (array): Probability of not being in the class.
        - p_data (array): Total probability of the data.
        """
        xy = np.column_stack((x, y))

        # Calculate log-probabilities
        log_p_data_given_class = self.gmm_class.score_samples(xy)
        log_p_data_given_not_class = self.gmm_not_class.score_samples(xy)
        
        # Convert log-probabilities to probabilities
        p_data_given_class = np.exp(log_p_data_given_class)
        p_data_given_not_class = np.exp(log_p_data_given_not_class)
        
        # Total probability of data
        p_data = p_data_given_class * self.p_class + p_data_given_not_class * self.p_not_class
        
        # Posterior probabilities
        p_class_given_data = (p_data_given_class * self.p_class) / p_data
        p_not_class_given_data = (p_data_given_not_class * self.p_not_class) / p_data
        
        # Classification based on threshold
        classification = p_class_given_data >= self.threshold
        
        return classification, p_class_given_data, p_not_class_given_data, p_data
    
    def classify_errors(self, x, y, x_err=0, y_err=0):
        """
        Classify points based on the GMM-based Bayesian model with measurement errors.

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
        # Prepare input data
        xy = np.column_stack((x, y))
        x_var = x_err ** 2
        y_var = y_err ** 2

        # Vectorized computation of adjusted covariances for "class" GMM
        adjusted_cov_class = self.gmm_class.covariances_ + np.array([[x_var, 0], [0, y_var]])
        adjusted_probs_class = np.sum(
            [
                weight * multivariate_normal(mean=mean, cov=adjusted_cov).pdf(xy)
                for mean, adjusted_cov, weight in zip(self.gmm_class.means_, adjusted_cov_class, self.gmm_class.weights_)
            ],
            axis=0
        )

        # Vectorized computation of adjusted covariances for "not class" GMM
        adjusted_cov_not_class = self.gmm_not_class.covariances_ + np.array([[x_var, 0], [0, y_var]])
        adjusted_probs_not_class = np.sum(
            [
                weight * multivariate_normal(mean=mean, cov=adjusted_cov).pdf(xy)
                for mean, adjusted_cov, weight in zip(self.gmm_not_class.means_, adjusted_cov_not_class, self.gmm_not_class.weights_)
            ],
            axis=0
        )

        # Total probability of data
        p_data = adjusted_probs_class * self.p_class + adjusted_probs_not_class * self.p_not_class

        # Posterior probabilities
        p_class_given_data = (adjusted_probs_class * self.p_class) / p_data
        p_not_class_given_data = (adjusted_probs_not_class * self.p_not_class) / p_data

        # Classification based on threshold
        classification = p_class_given_data >= self.threshold

        return classification, p_class_given_data, p_not_class_given_data, p_data
    

    
    def debug_gmm_fit(self, x_range=(-1, 2.5), y_range=(15, -7), resolution=100):
        """
        Debug GMM by plotting it alongside the original data to compare fit.

        Parameters:
        - x_range (tuple): Range of x values for the plot (default is (-1, 2.5)).
        - y_range (tuple): Range of y values for the plot (default is (15, -7)).
        - resolution (int): Number of points along each axis (default is 100).
        """
        # Create a meshgrid for plotting GMM densities
        x_i = np.linspace(x_range[0], x_range[1], resolution)
        y_i = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_i, y_i)
        xy_flat = np.column_stack([X.ravel(), Y.ravel()])

        # Calculate GMM density values for both classes
        prob_class = np.exp(self.gmm_class.score_samples(xy_flat)).reshape(X.shape)
        prob_not_class = np.exp(self.gmm_not_class.score_samples(xy_flat)).reshape(X.shape)

        # Plot the GMM and original data for each class
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot for the "class" data
        ax[0].scatter(self.X[self.C], self.Y[self.C], color="blue", alpha=0.5, s = 0.1, label="Class Data")
        cbar_1 = ax[0].contourf(X, Y, prob_class, levels=20, cmap="Blues", alpha=0.6)
        ax[0].set_title("GMM Fit for Class")
        ax[0].set_xlabel("$G_{bp} - G_{rp}$")
        ax[0].set_ylabel("$G_{I}$")
        ax[0].set_xlim(x_range)
        ax[0].set_ylim(y_range)
        fig.colorbar(cbar_1, ax=ax[0], label='$P(x)$')
        
        # Plot for the "not class" data
        ax[1].scatter(self.X[~self.C], self.Y[~self.C], color="red", alpha=0.5, s = 0.1, label="Not Class Data")
        cbar_2 = ax[1].contourf(X, Y, prob_not_class, levels=20, cmap="Reds", alpha=0.6)
        ax[1].set_title("GMM Fit for Not Class")
        ax[1].set_xlabel("$G_{bp} - G_{rp}$")
        ax[1].set_xlim(x_range)
        ax[1].set_ylim(y_range)
        fig.colorbar(cbar_2, ax=ax[1], label='$P(X)$')

        plt.tight_layout()
        plt.show()



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

        # Fit KDEs for each feature in each class with base bandwidth
        self.kde_x_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(X_class.reshape(-1, 1))
        self.kde_y_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(Y_class.reshape(-1, 1))
        self.kde_x_not_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(X_not_class.reshape(-1, 1))
        self.kde_y_not_class = KernelDensity(kernel='gaussian', bandwidth=self.base_bandwidth).fit(Y_not_class.reshape(-1, 1))
        
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
        # Adjust the bandwidths based on measurement error
        effective_bandwidth_x_class = np.sqrt(self.base_bandwidth**2 + x_err**2)
        effective_bandwidth_y_class = np.sqrt(self.base_bandwidth**2 + y_err**2)
        effective_bandwidth_x_not_class = np.sqrt(self.base_bandwidth**2 + x_err**2)
        effective_bandwidth_y_not_class = np.sqrt(self.base_bandwidth**2 + y_err**2)
        
        # Create temporary KDEs with adjusted bandwidths for each case
        kde_x_class = KernelDensity(kernel='gaussian', bandwidth=effective_bandwidth_x_class)
        kde_x_class.fit(self.X[self.C].reshape(-1, 1))
        kde_y_class = KernelDensity(kernel='gaussian', bandwidth=effective_bandwidth_y_class)
        kde_y_class.fit(self.Y[self.C].reshape(-1, 1))
        
        kde_x_not_class = KernelDensity(kernel='gaussian', bandwidth=effective_bandwidth_x_not_class)
        kde_x_not_class.fit(self.X[~self.C].reshape(-1, 1))
        kde_y_not_class = KernelDensity(kernel='gaussian', bandwidth=effective_bandwidth_y_not_class)
        kde_y_not_class.fit(self.Y[~self.C].reshape(-1, 1))

        # Calculate log-likelihoods with adjusted KDEs
        log_p_x_given_class = kde_x_class.score_samples(np.array(x).reshape(-1, 1))
        log_p_y_given_class = kde_y_class.score_samples(np.array(y).reshape(-1, 1))
        log_p_x_given_not_class = kde_x_not_class.score_samples(np.array(x).reshape(-1, 1))
        log_p_y_given_not_class = kde_y_not_class.score_samples(np.array(y).reshape(-1, 1))
        
        # Convert log-likelihoods to likelihoods
        p_x_given_class = np.exp(log_p_x_given_class)
        p_y_given_class = np.exp(log_p_y_given_class)
        p_x_given_not_class = np.exp(log_p_x_given_not_class)
        p_y_given_not_class = np.exp(log_p_y_given_not_class)
        
        # Joint probabilities
        p_data_given_class = p_x_given_class * p_y_given_class
        p_data_given_not_class = p_x_given_not_class * p_y_given_not_class
        
        # Total probability of data
        p_data = p_data_given_class * self.p_class + p_data_given_not_class * self.p_not_class
        
        # Posterior probabilities P(class|data) and P(not class|data)
        p_class_given_data = (p_data_given_class * self.p_class) / p_data
        p_not_class_given_data = (p_data_given_not_class * self.p_not_class) / p_data
        
        # Classification based on threshold
        classification = p_class_given_data >= self.threshold
        
        return classification, p_class_given_data, p_not_class_given_data, p_data
    
    def plot_kde(self, x_range=(-1, 2.5), y_range=(15, -7), resolution=100):
        """
        Plot the 2D KDE distributions for both classes.
        
        Parameters:
        - x_range (tuple): Range of x values for the plot (default is (-1, 2.5)).
        - y_range (tuple): Range of y values for the plot (default is (15, -7)).
        - resolution (int): Number of points along each axis (default is 100).
        """
        # Create a meshgrid
        x_i = np.linspace(x_range[0], x_range[1], resolution)
        y_i = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_i, y_i)
        X_flat, Y_flat = X.ravel(), Y.ravel()

        # Calculate KDE values for both classes
        log_prob_class_x = self.kde_x_class.score_samples(X_flat.reshape(-1, 1))
        log_prob_class_y = self.kde_y_class.score_samples(Y_flat.reshape(-1, 1))
        log_prob_not_class_x = self.kde_x_not_class.score_samples(X_flat.reshape(-1, 1))
        log_prob_not_class_y = self.kde_y_not_class.score_samples(Y_flat.reshape(-1, 1))

        # Calculate probabilities and reshape for plotting
        prob_class = np.exp(log_prob_class_x) * np.exp(log_prob_class_y)
        prob_not_class = np.exp(log_prob_not_class_x) * np.exp(log_prob_not_class_y)
        prob_class = prob_class.reshape(X.shape)
        prob_not_class = prob_not_class.reshape(X.shape)
        
        # Plot the KDEs
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        cbar_1 = ax[0].contourf(X, Y, prob_class, levels=20, cmap="Blues")
        ax[0].scatter(self.X[self.C], self.Y[self.C], color='yellow', alpha=0.1, s = 0.1, label='HVS')
        ax[0].scatter(self.X[~self.C], self.Y[~self.C], color='green', alpha=0.1, s = 0.1, label='Background')
        ax[0].set_title("KDE for Class")
        ax[0].set_xlabel("$G_{bp} - G_{rp}$")
        ax[0].set_ylabel("$G_{I}$")
        ax[0].set_xlim(x_range)
        ax[0].set_ylim(y_range)
        fig.colorbar(cbar_1, ax=ax[0], label='Probability')
        
        cbar_2 = ax[1].contourf(X, Y, prob_not_class, levels=20, cmap="Reds")
        ax[1].scatter(self.X[self.C], self.Y[self.C], color='yellow', alpha=0.1, s = 0.1, label='HVS')
        ax[1].scatter(self.X[~self.C], self.Y[~self.C], color='green', alpha=0.1, s = 0.1, label='Background')
        ax[1].set_title("KDE for Not Class")
        ax[1].set_xlabel("$G_{bp} - G_{rp}$")
        ax[1].set_xlim(x_range)
        ax[1].set_ylim(y_range)
        fig.colorbar(cbar_2, ax=ax[1], label='Probability')
        plt.legend()

        plt.savefig(os.path.join(plots_path, 'gmm_classifier_fit.png'))
        
        plt.show()
        
    
if __name__ == '__main__':
    # Example usage of BayesianGMMClassifier
    data_gaia_big= pd.read_feather('/Users/mncavieres/Documents/2024-2/HVS/Data/Gaia_tests/CMD_selection_testing_catalog/large_catalog.feather')
    X = data_gaia_big['bp_rp_corr'].values
    Y = data_gaia_big['implied_M_g_corr'].values
    C = np.bool(data_gaia_big['is_hvs'])
    gmm_classifier = BayesianGMMClassifier(X, Y, C, threshold=0.9, n_components=5)
    gmm_classifier.debug_gmm_fit()
    
