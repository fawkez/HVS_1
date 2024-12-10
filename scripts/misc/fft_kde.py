import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

class WeightedFFTKDE:
    def __init__(self, kernel='gaussian', bw=0.2, grid_size=2560, margin=0.5):
        """
        Weighted FFT-based KDE for 2D data.

        Parameters
        ----------
        kernel : str
            Kernel type. Only 'gaussian' is implemented in this demo.
        bw : float
            Bandwidth of the kernel. For a Gaussian kernel, this is the standard deviation.
        grid_size : int
            Number of grid points in each dimension for the internal grid.
        margin : float
            Margin around the data range.
        """
        if kernel != 'gaussian':
            raise NotImplementedError("Only 'gaussian' kernel is implemented.")
        self.kernel = kernel
        self.bw = bw
        self.grid_size = grid_size
        self.margin = margin
        self.fitted = False

    def fit(self, X, weights=None):
        """
        Fit the weighted KDE on given data.

        Parameters
        ----------
        X : array_like, shape (n_samples, 2)
            2D input data.
        weights : array_like, shape (n_samples,)
            Weights for each data point. If None, all points have equal weight.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, 2).")

        n = X.shape[0]
        if weights is None:
            weights = np.ones(n)
        else:
            weights = np.asarray(weights)
            if weights.shape[0] != n:
                raise ValueError("Length of weights must match number of samples in X.")

        # Determine data range
        x_min, x_max = X[:,0].min(), X[:,0].max()
        y_min, y_max = X[:,1].min(), X[:,1].max()

        x_min -= self.margin
        x_max += self.margin
        y_min -= self.margin
        y_max += self.margin

        self.x_edges = np.linspace(x_min, x_max, self.grid_size)
        self.y_edges = np.linspace(y_min, y_max, self.grid_size)

        # Compute bin indices for each point
        # np.digitize returns indices for each dimension
        x_idx = np.searchsorted(self.x_edges, X[:,0]) - 1
        y_idx = np.searchsorted(self.y_edges, X[:,1]) - 1

        # Clip indices to be within grid
        x_idx = np.clip(x_idx, 0, self.grid_size-1)
        y_idx = np.clip(y_idx, 0, self.grid_size-1)

        # Create weighted histogram
        hist = np.zeros((self.grid_size, self.grid_size))
        np.add.at(hist, (y_idx, x_idx), weights)  # note y_idx first, as hist is indexed [row, col]

        # Construct the kernel grid (Gaussian)
        # We want a kernel that has the same shape as hist for convolution
        # but it's best to center it and let fftconvolve handle it
        half_size = self.grid_size // 2
        xv = np.linspace(-half_size, half_size, self.grid_size)
        # Scale xv to represent coordinates in original space
        # To do so, find the spacing of the grid
        dx = (x_max - x_min) / (self.grid_size - 1)
        dy = (y_max - y_min) / (self.grid_size - 1)

        # Meshgrid in units of data space
        XX, YY = np.meshgrid(xv*dx, xv*dy)
        # Gaussian kernel
        kernel = np.exp(-0.5 * ((XX**2 + YY**2) / (self.bw**2)))
        kernel /= (2.0 * np.pi * self.bw**2)

        # Convolve using FFT
        density = fftconvolve(hist, kernel, mode='same')

        # Store results
        self.density = density
        self.x_grid = (self.x_edges)
        self.y_grid = (self.y_edges)

        # Create interpolator for evaluate
        self._interpolator = RegularGridInterpolator((self.y_grid, self.x_grid), self.density, bounds_error=False, fill_value=0.0)
        
        self.fitted = True
        return self

    def evaluate(self, points):
        """
        Evaluate the KDE at given points.

        Parameters
        ----------
        points : array_like, shape (m, 2)
            Points at which to evaluate the KDE.

        Returns
        -------
        values : ndarray, shape (m,)
            KDE values at the given points.
        """
        if not self.fitted:
            raise RuntimeError("The KDE must be fitted before calling evaluate().")

        points = np.asarray(points)
        if points.ndim == 1 and points.shape[0] == 2:
            points = points[np.newaxis, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be shape (m, 2).")

        # Interpolator expects (y, x) order
        # Our points are (x, y), so we need to swap
        query_points = np.column_stack((points[:,1], points[:,0]))
        vals = self._interpolator(query_points)
        return vals

    def __call__(self, points):
        return self.evaluate(points)

# Example usage (assuming you have data and weights):
if __name__ == "__main__":
    # Generate synthetic data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(10000, 2))
    weights = rng.uniform(0.5, 1.5, size=10000)

    kde = WeightedFFTKDE(bw=0.3, grid_size=256, margin=1.0)
    kde.fit(X, weights=weights)

    # Evaluate on a grid for visualization
    xv = np.linspace(-3, 3, 100)
    yv = np.linspace(-3, 3, 100)
    XX, YY = np.meshgrid(xv, yv)
    Z = kde.evaluate(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

    import matplotlib.pyplot as plt

    plt.contourf(XX, YY, Z, levels=20)
    plt.scatter(X[:,0], X[:,1], s=1, color='red', alpha=0.1)

    plt.title("Weighted FFT-based KDE")
    plt.colorbar(label='Density')
    plt.show()
