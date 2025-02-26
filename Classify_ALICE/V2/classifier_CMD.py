import numpy as np

class HistogramClassifier2D:
    """
    A 2D-histogram-based classifier that estimates p(x,y|class) for two classes
    (HVS vs. background), then for each input star convolving the histogram with
    a 2D Gaussian kernel derived from that star's measurement errors.

    Usage:
    1) Initialize with training data for HVS and background, plus bin definitions.
    2) Call 'classify' with arrays of (x, y, x_err, y_err) and an optional prior.
    """

    def __init__(self,
                 x_hvs, y_hvs,
                 x_bg,  y_bg,
                 bins_x=50, bins_y=50,
                 x_range=None, y_range=None):
        """
        Parameters
        ----------
        x_hvs, y_hvs : array-like
            Arrays of x,y training data points for HVS (e.g., bp_rp, G).
        x_bg, y_bg : array-like
            Arrays of x,y training data points for background stars.
        bins_x : int or sequence
            Number of bins (or explicit bin edges) along x dimension.
        bins_y : int or sequence
            Number of bins (or explicit bin edges) along y dimension.
        x_range : tuple or None
            (min_x, max_x) for the histogram if bins_x is an integer. If None,
            use the min/max of the data.
        y_range : tuple or None
            (min_y, max_y) for the histogram if bins_y is an integer. If None,
            use the min/max of the data.
        """

        # Build 2D histograms (counts) for each class
        self.hvs_hist, self.xedges, self.yedges = np.histogram2d(
            x_hvs, y_hvs,
            bins=[bins_x, bins_y],
            range=[x_range, y_range]
        )
        self.bg_hist,  _,            _         = np.histogram2d(
            x_bg,  y_bg,
            bins=[self.xedges, self.yedges]  # ensure same bin edges
        )

        # Store bin centers for quick access
        # xedges ~ length (bins_x+1), yedges ~ length (bins_y+1)
        self.xcenters = 0.5 * (self.xedges[:-1] + self.xedges[1:])
        self.ycenters = 0.5 * (self.yedges[:-1] + self.yedges[1:])

        # For each bin (i,j), the area = (dx_i)*(dy_j)
        # We'll store these for convolving. They are constant if uniform bins,
        # but we allow the possibility of non-uniform bin edges.
        dx = self.xedges[1:] - self.xedges[:-1]  # shape (bins_x,)
        dy = self.yedges[1:] - self.yedges[:-1]  # shape (bins_y,)

        # Create 2D array of bin areas by outer product
        # area[i,j] = dx[i] * dy[j]
        self.bin_area = np.outer(dx, dy).T  # shape (bins_y, bins_x)

        # Convert raw histogram counts to probability densities:
        # p_hist[i,j] = (counts[i,j] / total_counts) / bin_area[i,j]
        # So that sum p_hist[i,j] * bin_area[i,j] = 1
        total_hvs = np.sum(self.hvs_hist)
        total_bg  = np.sum(self.bg_hist)

        # Prevent zero-division if a class is empty
        if total_hvs > 0:
            self.hvs_density = (self.hvs_hist / total_hvs) / self.bin_area
        else:
            # degenerate case: no HVS in training set
            self.hvs_density = np.zeros_like(self.hvs_hist, dtype=float)

        if total_bg > 0:
            self.bg_density  = (self.bg_hist / total_bg)   / self.bin_area
        else:
            # degenerate case: no background in training set
            self.bg_density  = np.zeros_like(self.bg_hist, dtype=float)

        # shape of density arrays is (bins_x, bins_y) if we keep np.histogram2d default
        # but note that np.histogram2d returns [xbin, ybin], so we must track carefully
        # Actually, by default, histogram2d returns shape (bins_x, bins_y). We'll keep it:
        # self.hvs_density[i_x, i_y] ~ density in bin i_x, i_y
        # We'll keep that shape, but remember self.bin_area is (bins_y, bins_x). We'll need to transpose carefully or re-check indexing.
        # Let's reorder things so we keep the same orientation:
        # We'll do self.hvs_density = self.hvs_density.T so that it matches [i_y, i_x] -> We'll just handle carefully in the convolution.

        self.hvs_density = self.hvs_density.T  # shape (bins_y, bins_x)
        self.bg_density  = self.bg_density.T
        self.hvs_hist    = self.hvs_hist.T
        self.bg_hist     = self.bg_hist.T

        # Now self.hvs_density[i_y, i_x], self.bin_area[i_y, i_x], etc.
        # xcenters[i_x], ycenters[i_y]

    def classify(self, x, y, x_err, y_err, prior_hvs=0.5, prior_bg=0.5,
                 nsigma=3.0):
        """
        Classify an array of points (x, y, x_err, y_err) using the 2D histogram
        + measurement-error convolution method.

        Parameters
        ----------
        x, y : array-like of shape (N,)
            Positions (color, magnitude, etc.) for each star to classify.
        x_err, y_err : array-like of shape (N,)
            1-sigma uncertainties for each star.
        prior_hvs, prior_bg : float
            Prior probabilities p(HVS) and p(Background).
            Must sum to 1 if there are only two classes.
        nsigma : float
            Convolution cutoff in multiples of the measurement error. We only
            sum over the region +/- nsigma in x and y for efficiency.

        Returns
        -------
        p_hvs_given_data : np.ndarray of shape (N,)
            Posterior probability that each star is HVS.
        p_bg_given_data  : np.ndarray of shape (N,)
            Posterior probability that each star is background (=1 - p(HVS)).
        p_data : np.ndarray of shape (N,)
            The total probability of the data under both classes:
            p(data) = p(data|HVS)*prior_hvs + p(data|BG)*prior_bg.

        Notes
        -----
        - For each star, we do a local convolution by summing the histogram density
          in the relevant bins, weighted by the star's 2D Gaussian error kernel.
        """

        x = np.asarray(x)
        y = np.asarray(y)
        x_err = np.asarray(x_err)
        y_err = np.asarray(y_err)

        # Make output arrays
        N = len(x)
        p_hvs_given_data = np.zeros(N, dtype=float)
        p_bg_given_data  = np.zeros(N, dtype=float)
        p_data_out       = np.zeros(N, dtype=float)

        # For convenience, store bin centers in 2D arrays
        # shape (bins_y, bins_x)
        Xc2d, Yc2d = np.meshgrid(self.xcenters, self.ycenters)
        # Xc2d[j,i], Yc2d[j,i] correspond to bin (j, i) in self densities

        for idx in range(N):
            x_star  = x[idx]
            y_star  = y[idx]
            sx_star = x_err[idx]
            sy_star = y_err[idx]

            # 1) Identify local bins around (x_star, y_star) within +/- nsigma
            x_min = x_star - nsigma*sx_star
            x_max = x_star + nsigma*sx_star
            y_min = y_star - nsigma*sy_star
            y_max = y_star + nsigma*sy_star

            # Find bin indices that lie within these ranges
            # We'll use searchsorted on xedges, yedges
            ix_min = max(0, np.searchsorted(self.xedges, x_min) - 1)
            ix_max = min(len(self.xcenters)-1, np.searchsorted(self.xedges, x_max))
            iy_min = max(0, np.searchsorted(self.yedges, y_min) - 1)
            iy_max = min(len(self.ycenters)-1, np.searchsorted(self.yedges, y_max))

            # 2) Convolve locally
            # Extract the local region from the 2D arrays
            slice_x = slice(ix_min, ix_max+1)
            slice_y = slice(iy_min, iy_max+1)

            local_xc   = Xc2d[slice_y, slice_x]   # shape ~ (ny_local, nx_local)
            local_yc   = Yc2d[slice_y, slice_x]
            local_hvs  = self.hvs_density[slice_y, slice_x]
            local_bg   = self.bg_density[slice_y, slice_x]
            local_area = self.bin_area[slice_y, slice_x]

            # Compute 2D Gaussian weights
            dx = (local_xc - x_star)/sx_star
            dy = (local_yc - y_star)/sy_star
            # If we want the full normalized PDF:
            w_ij = (1.0/(2.0*np.pi*sx_star*sy_star)) * \
                    np.exp(-0.5*(dx*dx + dy*dy))

            # 3) Sum up p_data|class = sum( density_ij * w_ij * bin_area )
            p_data_given_hvs = np.sum(local_hvs * w_ij * local_area)
            p_data_given_bg  = np.sum(local_bg  * w_ij * local_area)

            # 4) Combine with priors
            p_data = p_data_given_hvs*prior_hvs + p_data_given_bg*prior_bg
            if p_data > 0.0:
                p_hvs = (p_data_given_hvs * prior_hvs) / p_data
            else:
                # If p_data = 0, numerical edge case: set p(HVS)=0 or handle gracefully
                p_hvs = 0.0

            p_hvs_given_data[idx] = p_hvs
            p_bg_given_data[idx]  = 1.0 - p_hvs
            p_data_out[idx]       = p_data

        return p_hvs_given_data, p_bg_given_data, p_data_out


# ----------------------------------------------------------------------
# Example usage (dummy data):
if __name__ == "__main__":

    # Suppose x=bp_rp, y=G for training
    # HVS training set
    np.random.seed(123)
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

    # Now classify some new stars
    # e.g., 10 random test points
    x_test = np.random.uniform(-2, 4, 10)
    y_test = np.random.uniform(-1, 10, 10)
    x_err  = 0.05 * np.ones(10)  # uniform error
    y_err  = 0.10 * np.ones(10)

    p_hvs, p_bg, p_data = classifier.classify(
        x_test, y_test, x_err, y_err,
        prior_hvs=0.1,  # say 10% prior for HVS
        prior_bg=0.9,
        nsigma=3.0
    )

    print("Test classification results:")
    for i in range(10):
        print(f"Star {i}: (x={x_test[i]:.2f}, y={y_test[i]:.2f}) => "
              f"p(HVS|data)={p_hvs[i]:.3f}, p(BG|data)={p_bg[i]:.3f}")
