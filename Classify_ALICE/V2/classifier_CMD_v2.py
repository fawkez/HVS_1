import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class HistogramClassifier2D:
    """
    A 2D-histogram-based classifier that estimates p(x,y|class) for two classes
    (HVS vs. background). 

    This version includes multiple classification approaches:
      1) classify(...): star-by-star 'local' convolution (the more correct approach).
      2) classify_brute_force(...): star-by-star summation over entire histogram.
      3) classify_gaussian_filter(...): naive approach that applies a global
         gaussian_filter for each star and picks the bin value. 
         (This does *not* properly shift the kernel to each star's (x,y).)
    """

    def __init__(self,
                 x_hvs, y_hvs,
                 x_bg,  y_bg,
                 bins_x=100, bins_y=100,
                 x_range=None, y_range=None):
        """
        Build 2D histograms (counts) for HVS and background; convert to density.
        """
        # Build 2D histograms (counts) for each class
        self.hvs_hist, self.xedges, self.yedges = np.histogram2d(
            x_hvs, y_hvs,
            bins=[bins_x, bins_y],
            range=[x_range, y_range]
        )
        self.bg_hist, _, _ = np.histogram2d(
            x_bg,  y_bg,
            bins=[self.xedges, self.yedges]  # ensure same bin edges
        )

        # Bin centers (1D arrays)
        self.xcenters = 0.5 * (self.xedges[:-1] + self.xedges[1:])
        self.ycenters = 0.5 * (self.yedges[:-1] + self.yedges[1:])

        # Compute bin "areas" (dx * dy per bin).
        dx = self.xedges[1:] - self.xedges[:-1]  # shape (bins_x,)
        dy = self.yedges[1:] - self.yedges[:-1]  # shape (bins_y,)
        bin_area = np.outer(dy, dx)  # shape (bins_y, bins_x)

        # Convert histogram counts to densities p_hist(x,y|class)
        total_hvs = np.sum(self.hvs_hist)
        total_bg  = np.sum(self.bg_hist)

        if total_hvs > 0:
            self.hvs_density = (self.hvs_hist / total_hvs) / bin_area
        else:
            self.hvs_density = np.zeros_like(self.hvs_hist)

        if total_bg > 0:
            self.bg_density  = (self.bg_hist / total_bg)  / bin_area
        else:
            self.bg_density  = np.zeros_like(self.bg_hist)

        # Transpose so densities are indexed as [iy, ix]
        self.hvs_density = self.hvs_density.T
        self.bg_density  = self.bg_density.T
        bin_area         = bin_area.T

        self.bin_area = bin_area

        # We'll store the bin-center mesh in 2D arrays for direct summation
        Xc2d, Yc2d = np.meshgrid(self.xcenters, self.ycenters)  # shape (nx, ny)
        self.Xc2d = Xc2d.T  # shape (ny, nx)
        self.Yc2d = Yc2d.T

    def classify(self, x, y, x_err, y_err,
                 prior_hvs=0.5, prior_bg=0.5,
                 nsigma=3.0):
        """
        Classify an array of points (x, y, x_err, y_err) by summing over
        local region +/- nsigma for each star. (More correct approach.)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_err = np.asarray(x_err)
        y_err = np.asarray(y_err)

        N = len(x)
        p_hvs_given_data = np.zeros(N, dtype=float)
        p_bg_given_data  = np.zeros(N, dtype=float)
        p_data_out       = np.zeros(N, dtype=float)

        # We'll reference the 2D arrays
        Xc2d = self.Xc2d
        Yc2d = self.Yc2d
        hvsD = self.hvs_density
        bgD  = self.bg_density
        area = self.bin_area

        nx = hvsD.shape[1]
        ny = hvsD.shape[0]

        for i in range(N):
            x_star  = x[i]
            y_star  = y[i]
            sx_star = x_err[i]
            sy_star = y_err[i]

            # local bounding box
            x_min = x_star - nsigma*sx_star
            x_max = x_star + nsigma*sx_star
            y_min = y_star - nsigma*sy_star
            y_max = y_star + nsigma*sy_star

            ix_min = max(0, np.searchsorted(self.xedges, x_min) - 1)
            ix_max = min(nx-1, np.searchsorted(self.xedges, x_max))
            iy_min = max(0, np.searchsorted(self.yedges, y_min) - 1)
            iy_max = min(ny-1, np.searchsorted(self.yedges, y_max))

            # Extract local subarrays
            local_hvs  = hvsD [iy_min:iy_max+1, ix_min:ix_max+1]
            local_bg   = bgD  [iy_min:iy_max+1, ix_min:ix_max+1]
            local_area = area [iy_min:iy_max+1, ix_min:ix_max+1]
            local_Xc   = Xc2d[iy_min:iy_max+1, ix_min:ix_max+1]
            local_Yc   = Yc2d[iy_min:iy_max+1, ix_min:ix_max+1]

            dx = (local_Xc - x_star)/sx_star
            dy = (local_Yc - y_star)/sy_star
            w_ij = (1.0/(2.0*np.pi*sx_star*sy_star)) * \
                   np.exp(-0.5*(dx*dx + dy*dy))

            p_data_given_hvs = np.sum(local_hvs * w_ij * local_area)
            p_data_given_bg  = np.sum(local_bg  * w_ij * local_area)

            p_data = p_data_given_hvs * prior_hvs + p_data_given_bg * prior_bg
            if p_data > 0.0:
                p_hvs = (p_data_given_hvs * prior_hvs) / p_data
            else:
                p_hvs = 0.0

            p_hvs_given_data[i] = p_hvs
            p_bg_given_data[i]  = 1.0 - p_hvs
            p_data_out[i]       = p_data

        return p_hvs_given_data, p_bg_given_data, p_data_out

    def classify_brute_force(self, x, y, x_err, y_err,
                             prior_hvs=0.5, prior_bg=0.5):
        """
        Classify an array of points by summing over the *entire* histogram
        (no nsigma bounding). Good for testing correctness vs. local method,
        but slower for large histograms.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_err = np.asarray(x_err)
        y_err = np.asarray(y_err)

        N = len(x)
        p_hvs_out = np.zeros(N, dtype=float)
        p_bg_out  = np.zeros(N, dtype=float)
        p_data_out= np.zeros(N, dtype=float)

        Xc2d = self.Xc2d
        Yc2d = self.Yc2d
        hvsD = self.hvs_density
        bgD  = self.bg_density
        area = self.bin_area

        ny, nx = hvsD.shape

        for i in range(N):
            x_star  = x[i]
            y_star  = y[i]
            sx_star = x_err[i]
            sy_star = y_err[i]

            dx = (Xc2d - x_star)/sx_star
            dy = (Yc2d - y_star)/sy_star
            w_ij = (1.0/(2.0*np.pi*sx_star*sy_star)) * \
                   np.exp(-0.5*(dx*dx + dy*dy))

            p_data_hvs = np.sum(hvsD * w_ij * area)
            p_data_bg  = np.sum(bgD  * w_ij * area)

            p_data = p_data_hvs*prior_hvs + p_data_bg*prior_bg
            if p_data > 0:
                p_hvs = (p_data_hvs*prior_hvs)/p_data
            else:
                p_hvs = 0.0

            p_hvs_out[i] = p_hvs
            p_bg_out[i]  = 1.0 - p_hvs
            p_data_out[i]= p_data

        return p_hvs_out, p_bg_out, p_data_out

    def classify_gaussian_filter(self, x, y, x_err, y_err,
                                 prior_hvs=0.5, prior_bg=0.5):
        """
        Naive approach that applies scipy.ndimage.gaussian_filter() on the entire
        HVS density & BG density for each star, using that star's sigma 
        translated into bin units. Then it looks up the bin nearest (x, y).

        WARNING:
          - This does NOT properly center the kernel on (x,y).
          - Each star's measurement error is globally applied to the entire histogram.
          - It's mainly for performance testing, not a correct star-by-star convolution.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_err = np.asarray(x_err)
        y_err = np.asarray(y_err)

        N = len(x)
        p_hvs_out = np.zeros(N, dtype=float)
        p_bg_out  = np.zeros(N, dtype=float)
        p_data_out= np.zeros(N, dtype=float)

        # For convenience, gather array shapes
        ny, nx = self.hvs_density.shape

        # For each star, do a global gaussian_filter
        for i in range(N):
            x_star  = x[i]
            y_star  = y[i]
            sx_star = x_err[i]
            sy_star = y_err[i]

            # Convert real errors -> approximate "bin-space" sigma
            # e.g. if x ranges from xedges[0] to xedges[-1], then
            # each bin ~ (range / nx). So star's x_err in physical units
            # => star_sigma_x_bins ~ ( x_err / bin_size_in_x ).
            # We'll do a rough approach here:
            bin_width_x = (self.xedges[-1] - self.xedges[0]) / nx
            bin_width_y = (self.yedges[-1] - self.yedges[0]) / ny
            sigma_x_bins = sx_star / bin_width_x
            sigma_y_bins = sy_star / bin_width_y

            # Now apply gaussian_filter to the entire histogram.
            # We must pass sigma in the order [sigma_y, sigma_x] 
            # if the array is shape (ny, nx).
            smoothed_hvs = gaussian_filter(self.hvs_density, 
                                           sigma=[sigma_y_bins, sigma_x_bins])
            smoothed_bg  = gaussian_filter(self.bg_density,    
                                           sigma=[sigma_y_bins, sigma_x_bins])

            # Next, locate the star's (x_star, y_star) in bin indices
            ix = np.searchsorted(self.xedges, x_star) - 1
            iy = np.searchsorted(self.yedges, y_star) - 1
            # clamp if out of range
            ix = max(0, min(ix, nx-1))
            iy = max(0, min(iy, ny-1))

            # The value in smoothed_hvs[iy, ix] is presumably the "blurred" density
            val_hvs = smoothed_hvs[iy, ix]
            val_bg  = smoothed_bg [iy, ix]

            # Combine with priors
            p_data = val_hvs*prior_hvs + val_bg*prior_bg
            if p_data > 0.0:
                p_hvs = (val_hvs*prior_hvs)/p_data
            else:
                p_hvs = 0.0

            p_hvs_out[i] = p_hvs
            p_bg_out[i]  = 1.0 - p_hvs
            p_data_out[i]= p_data

        return p_hvs_out, p_bg_out, p_data_out
    
    def plot_2d_histograms_and_posterior_no_errors(self, 
                                               prior_hvs=0.5, 
                                               prior_bg=0.5):
        """
        Plots:
        1. The HVS 2D density histogram (no errors).
        2. The BG (background) 2D density histogram (no errors).
        3. The posterior p(HVS | x, y) = [p(x,y|HVS)*prior_hvs] / 
            [p(x,y|HVS)*prior_hvs + p(x,y|BG)*prior_bg],
            *again without* convolving with measurement errors.

        Parameters
        ----------
        prior_hvs : float
            Prior probability of HVS.
        prior_bg : float
            Prior probability of background. Typically 1 - prior_hvs.
        """

        import matplotlib.pyplot as plt

        # 1) Prepare the figure and subplots
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("2D Histograms & Posterior (No Measurement Errors)")

        # 2) Plot HVS density
        # self.Xc2d, self.Yc2d hold the bin centers in shape (ny, nx)
        # self.hvs_density is also shape (ny, nx)
        c0 = ax[0].contourf(self.Xc2d, self.Yc2d, self.hvs_density,
                            levels=40, cmap='Blues')
        ax[0].set_title("HVS Density (no errors)")
        ax[0].set_xlabel("X (e.g. bp_rp)")
        ax[0].set_ylabel("Y (e.g. G)")

        # If you stored raw training data for HVS:
        if hasattr(self, 'x_hvs') and hasattr(self, 'y_hvs'):
            ax[0].scatter(self.x_hvs, self.y_hvs, s=1, c='k', alpha=0.2, label='HVS training')
            ax[0].legend(loc='upper right')

        fig.colorbar(c0, ax=ax[0], label='p(x,y|HVS)')

        # 3) Plot Background density
        c1 = ax[1].contourf(self.Xc2d, self.Yc2d, self.bg_density,
                            levels=40, cmap='Reds')
        ax[1].set_title("Background Density (no errors)")
        ax[1].set_xlabel("X (e.g. bp_rp)")
        ax[1].set_ylabel("Y (e.g. G)")

        # If you stored raw training data for BG:
        if hasattr(self, 'x_bg') and hasattr(self, 'y_bg'):
            ax[1].scatter(self.x_bg, self.y_bg, s=1, c='k', alpha=0.2, label='BG training')
            ax[1].legend(loc='upper right')

        fig.colorbar(c1, ax=ax[1], label='p(x,y|BG)')

        # 4) Compute and plot posterior p(HVS|x,y)
        # Posterior = [p(x,y|HVS)*prior_hvs] / [p(x,y|HVS)*prior_hvs + p(x,y|BG)*prior_bg]
        p_hvs_grid = self.hvs_density * prior_hvs
        p_bg_grid  = self.bg_density  * prior_bg
        p_sum_grid = p_hvs_grid + p_bg_grid

        posterior = np.zeros_like(p_sum_grid)
        mask = (p_sum_grid > 0)
        posterior[mask] = p_hvs_grid[mask] / p_sum_grid[mask]

        c2 = ax[2].contourf(self.Xc2d, self.Yc2d, posterior,
                            levels=40, cmap='coolwarm')
        ax[2].set_title("Posterior: p(HVS|x,y) (No errors)")
        ax[2].set_xlabel("X (e.g. bp_rp)")
        ax[2].set_ylabel("Y (e.g. G)")

        fig.colorbar(c2, ax=ax[2], label='p(HVS|x,y)')

        plt.tight_layout()
        plt.show()



# ----------------------------------------------------------------------
# Example usage (dummy data):
if __name__ == "__main__":

    import time

    np.random.seed(42)
    # Suppose x=bp_rp, y=G for training
    # HVS training set
    x_hvs = np.random.normal(loc=0.0, scale=0.3, size=300)
    y_hvs = np.random.normal(loc=2.0, scale=0.5, size=300)

    # BG training set
    x_bg  = np.random.normal(loc=1.0, scale=0.5, size=2000)
    y_bg  = np.random.normal(loc=5.0, scale=1.0, size=2000)

    # Build the classifier
    classifier = HistogramClassifier2D(
        x_hvs, y_hvs, x_bg, y_bg,
        bins_x=50, bins_y=50,
        x_range=(-2, 4),  # for example
        y_range=(-1, 10),
    )

    # Some random test points
    x_test = np.random.uniform(-2, 4, 10)
    y_test = np.random.uniform(-1, 10, 10)
    x_err  = 0.05 + 0.05*np.random.rand(10)  # random errors
    y_err  = 0.10 + 0.10*np.random.rand(10)

    # 1) "Local" classify
    start = time.time()
    p_hvs1, p_bg1, p_data1 = classifier.classify(
        x_test, y_test, x_err, y_err,
        prior_hvs=0.1,  # say 10% prior for HVS
        prior_bg=0.9,
        nsigma=3.0
    )
    end = time.time()
    print(f"\nLocal classify time: {end - start:.4f} s")

    # 2) Brute force classify
    start = time.time()
    p_hvs2, p_bg2, p_data2 = classifier.classify_brute_force(
        x_test, y_test, x_err, y_err,
        prior_hvs=0.1, prior_bg=0.9
    )
    end = time.time()
    print(f"Brute force time: {end - start:.4f} s")

    # 3) Gaussian filter classify (naive)
    start = time.time()
    p_hvs3, p_bg3, p_data3 = classifier.classify_gaussian_filter(
        x_test, y_test, x_err, y_err,
        prior_hvs=0.1, prior_bg=0.9
    )
    end = time.time()
    print(f"Gaussian filter time: {end - start:.4f} s")

    # Print out results for each method
    print("\nCompare for each star (x,y):\n")
    for i in range(len(x_test)):
        print(f"Star {i} at (x={x_test[i]:.2f}, y={y_test[i]:.2f}):")
        print(f"  Local         => pHVS={p_hvs1[i]:.4f}, pBG={p_bg1[i]:.4f}")
        print(f"  Brute force   => pHVS={p_hvs2[i]:.4f}, pBG={p_bg2[i]:.4f}")
        print(f"  gaussian_filt => pHVS={p_hvs3[i]:.4f}, pBG={p_bg3[i]:.4f}")
        print("  ---")

    # Plot the 2D histograms and posterior (no errors)
    classifier.plot_2d_histograms_and_posterior_no_errors(prior_hvs=0.1, prior_bg=0.9)