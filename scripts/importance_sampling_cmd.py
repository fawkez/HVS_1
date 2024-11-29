# plot_kde_alpha.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KernelDensity

# Set up plotting parameters
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': (10, 7)})
plt.rcParams.update({'text.usetex': True})

# Set the path to save plots (ensure this directory exists)
plots_path = '/Users/mncavieres/Documents/2024-2/HVS/Plots/new_cmd'

class BayesianKDEClassifier:
    def __init__(self, simulated_data, gaia_data, alpha, threshold=0.5, bandwidth=1.0):
        """
        Initialize the classifier with simulated data and Gaia data separately.

        Parameters:
        - simulated_data (DataFrame): Simulated HVS data with required columns.
        - gaia_data (DataFrame): Gaia data with required columns.
        - alpha (float): IMF slope for computing weights.
        - threshold (float): Probability threshold for classification (default is 0.5).
        - bandwidth (float): Bandwidth for the KDE (default is 1.0).
        """
        self.threshold = threshold
        self.bandwidth = bandwidth
        self.alpha = alpha

        # Extract features from simulated data
        self.sim_X = simulated_data['bp_rp_corr'].values
        self.sim_Y = simulated_data['implied_M_g_corr'].values
        self.masses = simulated_data['mass'].values

        # Compute weights based on the IMF
        self.weights = self.compute_weights(self.masses, self.alpha)

        # Extract features from Gaia data
        self.gaia_X = gaia_data['bp_rp_corr'].values
        self.gaia_Y = gaia_data['implied_M_g_corr'].values

        # Prepare data
        self.X_class = np.column_stack((self.sim_X, self.sim_Y))
        self.X_not_class = np.column_stack((self.gaia_X, self.gaia_Y))

        # Calculate prior probabilities
        self.p_class = len(self.sim_X) / (len(self.sim_X) + len(self.gaia_X))
        self.p_not_class = 1 - self.p_class

        # Fit KDEs
        self.kde_class = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde_class.fit(self.X_class, sample_weight=self.weights)

        self.kde_not_class = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde_not_class.fit(self.X_not_class)

    def compute_weights(self, masses, alpha):
        """
        Compute weights based on the IMF slope alpha.

        Parameters:
        - masses (array-like): Stellar masses.
        - alpha (float): IMF slope.

        Returns:
        - weights (array-like): Normalized weights.
        """
        # Since the initial sampling is with alpha=0 (flat IMF),
        # the weight for each mass is proportional to M^(-alpha)
        weights = masses ** (-alpha)
        # Normalize weights
        weights /= np.sum(weights)
        return weights

    def plot_kde(self, x_range=(-1, 2.5), y_range=(15, -7), resolution=100, alpha_label=''):
        """
        Plot the 2D KDE distributions for both classes.

        Parameters:
        - x_range (tuple): Range of x values for the plot (default is (-1, 2.5)).
        - y_range (tuple): Range of y values for the plot (default is (15, -7)).
        - resolution (int): Number of points along each axis (default is 100).
        - alpha_label (str): Label to include in the plot title indicating the alpha value.
        """
        # Create a meshgrid
        x_i = np.linspace(x_range[0], x_range[1], resolution)
        y_i = np.linspace(y_range[0], y_range[1], resolution)
        X_mesh, Y_mesh = np.meshgrid(x_i, y_i)
        xy_sample = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T

        # Calculate KDE values for both classes
        log_prob_class = self.kde_class.score_samples(xy_sample)
        log_prob_not_class = self.kde_not_class.score_samples(xy_sample)

        # Reshape probabilities for plotting
        prob_class = np.exp(log_prob_class).reshape(X_mesh.shape)
        prob_not_class = np.exp(log_prob_not_class).reshape(X_mesh.shape)

        # Plot the KDEs
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        cbar_1 = ax[0].contourf(X_mesh, Y_mesh, prob_class, levels=20, cmap="Blues")
        ax[0].scatter(self.sim_X, self.sim_Y, color='yellow', alpha=0.1, s=0.1, label='Simulated HVS')
        ax[0].set_title(f"KDE for Simulated HVS (alpha={alpha_label})")
        ax[0].set_xlabel("$G_{bp} - G_{rp}$")
        ax[0].set_ylabel("$M_G$")
        ax[0].set_xlim(x_range)
        ax[0].set_ylim(y_range)
        fig.colorbar(cbar_1, ax=ax[0], label='Probability')

        cbar_2 = ax[1].contourf(X_mesh, Y_mesh, prob_not_class, levels=20, cmap="Reds")
        ax[1].scatter(self.gaia_X, self.gaia_Y, color='green', alpha=0.1, s=0.1, label='Gaia Data')
        ax[1].set_title(f"KDE for Gaia Data (alpha={alpha_label})")
        ax[1].set_xlabel("$G_{bp} - G_{rp}$")
        ax[1].set_xlim(x_range)
        ax[1].set_ylim(y_range)
        fig.colorbar(cbar_2, ax=ax[1], label='Probability')
        ax[1].legend()

        # Save the plot with alpha in the filename
        plt.savefig(os.path.join(plots_path, f'kde_classifier_fit_alpha_{alpha_label}.png'))
        plt.close()

if __name__ == '__main__':
    # Load your simulated HVS data and Gaia data
    # Ensure these DataFrames contain the required columns:
    # For simulated_data: 'mass', 'bp_rp_corr', 'implied_M_g_corr'
    # For gaia_data: 'bp_rp_corr', 'implied_M_g_corr'

    # Example paths (update these paths to your actual data files)
    simulated_data_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/simulated_data.csv'
    gaia_data_path = '/Users/mncavieres/Documents/2024-2/HVS/Data/gaia_data.csv'

    # Load the data
    simulated_data = pd.read_csv(simulated_data_path)
    gaia_data = pd.read_csv(gaia_data_path)

    # Define a range of alpha values
    alpha_list = [1.5, 2.0, 2.2, 2.5, 3.0]

    # Loop over alpha values
    for alpha in alpha_list:
        print(f"Processing alpha = {alpha}")

        # Initialize the classifier
        kde_classifier = BayesianKDEClassifier(
            simulated_data=simulated_data,
            gaia_data=gaia_data,
            alpha=alpha,
            threshold=0.9,  # Adjust the threshold as needed
            bandwidth=0.1   # Adjust the bandwidth based on your data
        )

        # Plot the KDEs
        kde_classifier.plot_kde(alpha_label=str(alpha))

    print("KDE plots for different alpha values have been saved.")
