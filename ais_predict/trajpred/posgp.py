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

    def __init__(self, X: np.ndarray, Y: np.ndarray, variance=50):
        self.kernel = gpf.kernels.Constant(variance=variance) * (
            gpf.kernels.RBF(active_dims=[0, 1], lengthscales=200) * 
            gpf.kernels.RBF(active_dims=[2], lengthscales=2) *
            gpf.kernels.RBF(active_dims=[3], lengthscales=2) * (
                #gpf.kernels.Matern12(variance=1, active_dims=[4], lengthscales=60) + 
                gpf.kernels.RBF(active_dims=[4], lengthscales=500)
            ) 
            #+ (
            #    gpf.kernels.Linear(variance=1 / ((20*60)**2), active_dims=[4]) * 
            #    gpf.kernels.White(variance=10, active_dims=[4])
            #)
            #gpf.kernels.RBF(variance=5000, active_dims=[4], lengthscales=4000)
        )
        self.gpx = gpf.models.GPR((X, Y[:, 0].reshape((-1, 1))), self.kernel, mean_function=lambda x: tf.reshape(x[:, 0], (-1, 1)), noise_variance=1)
        self.gpy = gpf.models.GPR((X, Y[:, 1].reshape((-1, 1))), self.kernel, mean_function=lambda x: tf.reshape(x[:, 1], (-1, 1)), noise_variance=1)

        #optimizer = gpf.optimizers.Scipy()
        #optimizer.minimize(self.gpx.training_loss, self.gpx.trainable_variables)
        #optimizer.minimize(self.gpy.training_loss, self.gpy.trainable_variables)

    def __call__(self, x: np.ndarray):
        pred_x, std_x = self.gpx.predict_f(x)
        pred_y, std_y = self.gpy.predict_f(x)
        return np.concatenate([pred_x.numpy(), pred_y.numpy()], axis=1), np.concatenate([std_x.numpy(), std_y.numpy()], axis=1)