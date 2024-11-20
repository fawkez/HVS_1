from mpi4py import MPI
import numpy as np
import pandas as pd
import sys
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
        self.X_class = self.X[self.C]
        self.Y_class = self.Y[self.C]
        self.X_not_class = self.X[~self.C]
        self.Y_not_class = self.Y[~self.C]

        # Calculate prior probabilities
        self.p_class = np.mean(self.C)
        self.p_not_class = 1 - self.p_class

        # Store class data as arrays for efficient computation
        self.data_class = np.vstack([self.X_class, self.Y_class]).T
        self.data_not_class = np.vstack([self.X_not_class, self.Y_not_class]).T

    def classify(self, x, y, x_err=0, y_err=0):
        """
        Classify points based on the KDE-based Bayesian model with measurement uncertainties.

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

        N_eval = len(x)

        # Effective bandwidths for each sample
        h_x = np.sqrt(self.base_bandwidth**2 + x_err**2)
        h_y = np.sqrt(self.base_bandwidth**2 + y_err**2)

        # Precompute constants
        inv_hx = 1 / h_x
        inv_hy = 1 / h_y
        inv_2pi_hx_hy = 1 / (2 * np.pi * h_x * h_y)

        # Evaluate densities for the class
        p_data_given_class = self._evaluate_density(
            x, y, h_x, h_y, inv_hx, inv_hy, inv_2pi_hx_hy, self.data_class)

        # Evaluate densities for the not_class
        p_data_given_not_class = self._evaluate_density(
            x, y, h_x, h_y, inv_hx, inv_hy, inv_2pi_hx_hy, self.data_not_class)

        # Total probability of data
        p_data = p_data_given_class * self.p_class + p_data_given_not_class * self.p_not_class

        # Posterior probabilities P(class|data) and P(not class|data)
        p_class_given_data = (p_data_given_class * self.p_class) / p_data
        p_not_class_given_data = (p_data_given_not_class * self.p_not_class) / p_data

        # Classification based on threshold
        classification = p_class_given_data >= self.threshold

        return classification, p_class_given_data, p_not_class_given_data, p_data

    def _evaluate_density(self, x, y, h_x, h_y, inv_hx, inv_hy, inv_2pi_hx_hy, data):
        """
        Evaluate the density at points (x, y) given the data and bandwidths.

        Parameters:
        - x, y: Coordinates where density is evaluated.
        - h_x, h_y: Bandwidths in x and y directions.
        - inv_hx, inv_hy: Inverses of bandwidths.
        - inv_2pi_hx_hy: Precomputed constant.
        - data: Data points of the class.

        Returns:
        - density: Evaluated density at (x, y).
        """
        # Number of evaluation points and data points
        N_eval = len(x)
        N_data = data.shape[0]

        # Expand dimensions for broadcasting
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        h_x = h_x[:, np.newaxis]
        h_y = h_y[:, np.newaxis]
        inv_hx = inv_hx[:, np.newaxis]
        inv_hy = inv_hy[:, np.newaxis]
        inv_2pi_hx_hy = inv_2pi_hx_hy[:, np.newaxis]

        # Compute differences
        dx = x - data[:, 0][np.newaxis, :]
        dy = y - data[:, 1][np.newaxis, :]

        # Compute exponent
        exponent = -0.5 * ((dx * inv_hx) ** 2 + (dy * inv_hy) ** 2)

        # Compute kernel values
        K = inv_2pi_hx_hy * np.exp(exponent)

        # Sum over data points
        density = np.sum(K, axis=1) / N_data

        return density

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load training data on root process
    if rank == 0:
        data_gaia_big = pd.read_csv('/Users/mncavieres/Documents/2024-2/HVS/Data/CMD_training_catalogs/kappa_1.7_speedystar.csv')

        # Select the features
        X = data_gaia_big['bp_rp_corr'].values
        Y = data_gaia_big['implied_M_g_corr'].values
        C = data_gaia_big['is_hvs'].astype(bool).values

        # Broadcast data to all processes
        X = comm.bcast(X, root=0)
        Y = comm.bcast(Y, root=0)
        C = comm.bcast(C, root=0)
    else:
        X = None
        Y = None
        C = None
        X = comm.bcast(X, root=0)
        Y = comm.bcast(Y, root=0)
        C = comm.bcast(C, root=0)

    # Each process initializes the classifier
    bayesian_kde = BayesianKDEClassifier(X, Y, C, bandwidth=0.1)

    # Load evaluation data (sources to classify)
    if rank == 0:
        # Load the sources you want to classify (e.g., billions of sources)
        # For demonstration, let's create synthetic data
        num_sources = 10**9  # One billion sources
        x_eval = np.random.uniform(-1, 2.5, num_sources)
        y_eval = np.random.uniform(-7, 15, num_sources)
        x_err_eval = np.full(num_sources, 2e-3)
        y_err_eval = np.full(num_sources, 0.1)

        # Split evaluation data among processes
        eval_indices = np.array_split(np.arange(num_sources), size)
    else:
        x_eval = None
        y_eval = None
        x_err_eval = None
        y_err_eval = None
        eval_indices = None

    # Scatter evaluation indices to all processes
    eval_indices = comm.scatter(eval_indices, root=0)

    # Each process gets its subset of data to classify
    x_eval_local = x_eval[eval_indices]
    y_eval_local = y_eval[eval_indices]
    x_err_eval_local = x_err_eval[eval_indices]
    y_err_eval_local = y_err_eval[eval_indices]

    # Classify the local data
    start_time = time.time()
    classification_local, p_class_given_data_local, p_not_class_given_data_local, p_data_local = bayesian_kde.classify(
        x_eval_local, y_eval_local, x_err_eval_local, y_err_eval_local)
    end_time = time.time()

    print(f'Process {rank} classification time:', end_time - start_time)

    # Gather results from all processes
    classification = comm.gather(classification_local, root=0)
    p_class_given_data = comm.gather(p_class_given_data_local, root=0)
    p_not_class_given_data = comm.gather(p_not_class_given_data_local, root=0)
    p_data = comm.gather(p_data_local, root=0)

    # On root process, concatenate results
    if rank == 0:
        classification = np.concatenate(classification)
        p_class_given_data = np.concatenate(p_class_given_data)
        p_not_class_given_data = np.concatenate(p_not_class_given_data)
        p_data = np.concatenate(p_data)

        print('Total classification completed.')
