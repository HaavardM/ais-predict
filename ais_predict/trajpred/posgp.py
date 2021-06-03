from gpflow import kernels
from gpflow.kernels.statics import White
import numpy as np
import gpflow as gpf
from gpflow.utilities import print_summary, set_trainable
from gpflow.ci_utils import ci_niter
import geopandas as gpd
from numpy.core.fromnumeric import var
import tensorflow as tf
from typing import Tuple


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
        Y[:, i, :] = np.vstack(
            [next_position.x.to_numpy(), next_position.y.to_numpy()]).T
    return X.reshape((-1, 5)), Y.reshape((-1, 2))


class PosGP():
    def __init__(self, X: np.ndarray, Y: np.ndarray, variance: float = 50, center: np.ndarray = None, N: int = None, N_Z: int = 50, batch_size: int = 100, N_train: int = 10000, log_per: int = 1000):
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

        self.kernel = kernels.SharedIndependent(
            kernels.RBF(variance=1000.0, lengthscales=[2000.0, 2000.0, 50.0, 50.0, 60.0])
            + kernels.RationalQuadratic(lengthscales=[1.0, 1.0, 1.0, 1.0, 1.0])
            + kernels.White(1),
            output_dim=2
        )

        inducing = np.random.choice(X.shape[0], size=N_Z)
        inducing = X[inducing]
        inducing = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(inducing)
        )

        gp = gpf.models.SVGP(
            likelihood=gpf.likelihoods.Gaussian(),
            inducing_variable=inducing,
            kernel=self.kernel,
            mean_function=lambda x: Y.mean(axis=0),
            num_latent_gps=2
        )

        train_data = tf.data.Dataset.from_tensor_slices(
            (X, Y)).repeat().shuffle(X.shape[0])
        train_iter = iter(train_data.batch(batch_size))
        loss = gp.training_loss_closure(train_iter, compile=True)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        natgrad = gpf.optimizers.NaturalGradient(gamma=0.1)
        var_params = [(gp.q_mu, gp.q_sqrt)]

        #set_trainable(gp.q_mu, False)
        #set_trainable(gp.q_sqrt, False)

        @tf.function
        def step(model: gpf.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor]):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(gp.trainable_variables)
                loss = model.training_loss(batch)
            grads = tape.gradient(loss, gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, gp.trainable_variables))
            return loss

        for i in range(N_train):
            #natgrad.minimize(loss, var_list=var_params)
            step(gp, next(train_iter))
            if i % log_per == 0:
                elbo = -loss().numpy()
                print(i, elbo)
        self.gp = gp

    def __call__(self, x: np.ndarray):
        f, std = self.gp.predict_f(x)
        return f.numpy(), std.numpy()
