import numpy as np
import gpflow as gpf
import geopandas as gdp


def samples_from_lag_1_df(df: gdp.GeoDataFrame) -> tuple:
    position = df.position.to_crs(epsg=3857)
    position_1 = df.position_1.to_crs(epsg=3857)
    dt = (df.timestamp_1 - df.timestamp).dt.seconds
    dx = (position_1.x - position.x) / dt
    dy = (position_1.y - position.y) / dt

    X = np.vstack([position.x.to_numpy(), 
                   position.y.to_numpy(),
                   df.cog.to_numpy(),
                   df.sog.to_numpy()
                   ]).T
    Y = np.vstack([dx.to_numpy(), dy.to_numpy()]).T
    return X, Y, dt.to_numpy()


class DynGP():
    def __init__(self, x: np.ndarray, dy: np.ndarray):
        """ Constructor for dynamical GP prediction

        This model learns the dynamical system x_dot = f(x), y = x + error from data, and simulate this system from any initial state\n

        Parameters: \n
        x: Current state of the vessel. ndarray with shape (n, 4) where the innermost dimension corresponds to [p_x, p_y, v_x, v_y]\n
        dy: Observed trajectory gradient corresponding to each sample in x. ndarray with shape (n, 2). 
         
        """

        assert x.shape[-1] == 4, f"Unexpected sample dimension, only expected 4 but got {x.shape[-1]}"
        assert dy.shape[-1] == 2, f"Unexpected sample dimention, only expected 2 but got {dy.shape[-1]}"
        self.x = x
        self.dy = dy
        kernel = gpf.kernels.Matern12(active_dims=[0, 1], lengthscales=10000) * gpf.kernels.RBF(active_dims=[2], lengthscales=2)*gpf.kernels.RBF(active_dims=[3], lengthscales=2)
        self.gpx = gpf.models.GPR((x, dy[:, 0].reshape((-1, 1))), kernel, noise_variance=1e-4)
        self.gpy = gpf.models.GPR((x, dy[:, 1].reshape((-1, 1))), kernel, noise_variance=1e-4)
        
    
    def __call__(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        pred_x, std_x = self.gpx.predict_f(x)
        pred_y, std_y = self.gpy.predict_f(x)
        return np.concatenate([pred_x.numpy(), pred_y.numpy()], axis=1), np.concatenate([std_x.numpy(), std_y.numpy()], axis=1)


def simulate(f: DynGP, x_0: np.ndarray, n, dt=0.1) -> np.ndarray:
    x = np.empty((n,) + x_0.shape)
    x[0] = x_0
    for i in range(1,n):
        x[i] = x[i-1] + f(x[i-1])*dt
    return x

