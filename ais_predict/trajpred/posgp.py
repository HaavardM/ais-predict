import numpy as np
import gpflow as gpf
import geopandas as gpd
import tensorflow as tf

def samples_from_lag_n_df(df: gpd.GeoDataFrame, n: int):
    position = df.position.to_crs(epsg=3857)
    position_x = position.x.to_numpy()
    position_y = position.y.to_numpy()       
    cog = df.cog.to_numpy()
    sog = df.sog.to_numpy()

    X = np.empty((len(df.index), n, 5))
    Y = np.empty((len(df.index), n, 2))
    for i in range(n):
        
        postfix = f"_{i}"
        if i == 0:
            postfix = ""

        
        dt = (df["timestamp" + postfix] - df.timestamp).dt.seconds.to_numpy()
        next_position = df["position" + postfix].to_crs(epsg=3857)

        X[:, i, :] = np.vstack([position_x, position_y, cog, sog, dt]).T
        Y[:, i, :] = np.vstack([next_position.x.to_numpy(), next_position.y.to_numpy()]).T
    return X.reshape((-1, 5)), Y.reshape((-1, 2))


class PosGP():
    def __init__(self, X: np.ndarray, Y: np.ndarray, variance: float=50, center: np.ndarray = None, N: int = None):
        """ Constructor for PosGP

        Parameters:
        X (numpy.ndarray): Training inputs on the form (n, 5) where the innermost dimension corresponds to [p_x, p_y, cog, sog, t]
        Y (numpy.ndarray): Training outputs on the form (n, 2) where the innermost dimension is the true position at time t
        variance (float): Kernel amplitude/variance
        center (numpy.ndarray): Position used to calculate distance to each position in X. Used to sort the input data by distance.
        N (int): Number of samples to include in the model. If N is less than the number of samples in (X, Y), N samples are collected. If center != None, the N samples with shortest distance between center and X[:, :2] are used, otherwise random.
        """
        assert X.shape[0] == Y.shape[0], "Shapes not matching"
        N = N if N is not None else X.shape[0]
        if N < X.shape[0]:
            if center is not None:
                # Sort the indicies by distance and pick the first N
                dist = np.linalg.norm(X[:, :2]-center, axis=1)
                ix = np.argsort(dist)[:N]
            else:
                # Randomly select N samples
                ix = np.random.choice(N, size=N, replace=False)
            X = X[ix, :]
            Y = Y[ix, :]
        
        
        self.kernel = gpf.kernels.Constant(variance=variance) * (
            gpf.kernels.RBF(active_dims=[0, 1], lengthscales=200) * 
            gpf.kernels.RBF(active_dims=[2], lengthscales=2) *
            gpf.kernels.RBF(active_dims=[3], lengthscales=2) * (
                gpf.kernels.Matern32(active_dims=[4], lengthscales=500)
            ) 
        )

        self.gpx = gpf.models.GPR(
            data=(X, Y[:, 0].reshape((-1, 1))), 
            kernel=self.kernel, 
            mean_function=lambda x: tf.reshape(x[:, 0], (-1, 1)), 
            noise_variance=1
        )
        self.gpy = gpf.models.GPR(
            data=(X, Y[:, 1].reshape((-1, 1))), 
            kernel=self.kernel, 
            mean_function=lambda x: tf.reshape(x[:, 1], (-1, 1)), 
            noise_variance=1
        )

    def __call__(self, x: np.ndarray):
        pred_x, std_x = self.gpx.predict_f(x)
        pred_y, std_y = self.gpy.predict_f(x)
        return np.concatenate([pred_x.numpy(), pred_y.numpy()], axis=1), np.concatenate([std_x.numpy(), std_y.numpy()], axis=1)