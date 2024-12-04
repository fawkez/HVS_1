# Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

# Set up the plotting
# Set font size
plt.rcParams.update({'font.size': 18})
# Set the figure size
plt.rcParams.update({'figure.figsize': (10, 7)})
# Set the font to LaTeX
plt.rcParams.update({'text.usetex': True})

# Set the path to save plots
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

    def classify(self, x, y):
        """
        Classify points based on the KDE-based Bayesian model.

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

        # Calculate log-likelihoods
        log_p_data_given_class = self.kde_class.score_samples(xy)
        log_p_data_given_not_class = self.kde_not_class.score_samples(xy)

        # Convert log-likelihoods to probabilities
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
        ax[0].set_title("KDE for Simulated HVS")
        ax[0].set_xlabel("$G_{bp} - G_{rp}$")
        ax[0].set_ylabel("$G_{I}$")
        ax[0].set_xlim(x_range)
        ax[0].set_ylim(y_range)
        fig.colorbar(cbar_1, ax=ax[0], label='Probability')

        cbar_2 = ax[1].contourf(X_mesh, Y_mesh, prob_not_class, levels=20, cmap="Reds")
        ax[1].scatter(self.gaia_X, self.gaia_Y, color='green', alpha=0.1, s=0.1, label='Gaia Data')
        ax[1].set_title("KDE for Gaia Data")
        ax[1].set_xlabel("$G_{bp} - G_{rp}$")
        ax[1].set_xlim(x_range)
        ax[1].set_ylim(y_range)
        fig.colorbar(cbar_2, ax=ax[1], label='Probability')
        plt.legend()

        plt.savefig(os.path.join(plots_path, 'kde_classifier_fit.png'))

        plt.show()

if __name__ == '__main__':
    # Load your simulated HVS data and Gaia data
    # Make sure these DataFrames contain the required columns:
    # For simulated_data: 'mass', 'bp_rp_corr', 'implied_M_g_corr'
    # For gaia_data: 'bp_rp_corr', 'implied_M_g_corr'

    # Example paths (update these paths to your actual data files)
    simulated_data_path = '/path/to/your/simulated_data.csv'
    gaia_data_path = '/path/to/your/gaia_data.csv'

    # Load the data
    simulated_data = pd.read_csv(simulated_data_path)
    gaia_data = pd.read_csv(gaia_data_path)

    # Set the desired IMF slope alpha
    alpha = 2.2

    # Initialize the classifier
    kde_classifier = BayesianKDEClassifier(
        simulated_data=simulated_data,
        gaia_data=gaia_data,
        alpha=alpha,
        threshold=0.9,  # Adjust the threshold as needed
        bandwidth=0.1   # Adjust the bandwidth based on your data
    )

    # Optional: Plot the KDEs to visualize the fits
    kde_classifier.plot_kde()

    # Classify Gaia data
    x_test = gaia_data['bp_rp_corr'].values
    y_test = gaia_data['implied_M_g_corr'].values

    classification, p_class_given_data, p_not_class_given_data, p_data = kde_classifier.classify(x_test, y_test)

    # Add classification results to the Gaia data DataFrame
    gaia_data['classification'] = classification
    gaia_data['p_class_given_data'] = p_class_given_data

    # Save the classification results
    gaia_data.to_csv('/path/to/save/classified_gaia_data.csv', index=False)

    # Print some statistics
    num_candidates = np.sum(classification)
    print(f"Number of HVS candidates identified: {num_candidates}")
