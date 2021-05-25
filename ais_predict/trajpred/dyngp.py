from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from scipy.linalg import solve_triangular
from dataclasses import dataclass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from typing import Optional, Tuple


def samples_from_lag_n_df(df: gpd.GeoDataFrame, n: int, flatten: bool = True):
    X = np.empty((len(df.index), n, 2))
    Y = np.empty((len(df.index), n, 2))
    dt = np.empty((len(df.index), n))
    T = np.empty((len(df.index), n))
    for i in range(n):

        l1 = f"_{i}"
        l2 = f"_{i+1}"
        if i == 0:
            l1 = ""
        p1 = df["position" + l1].to_crs(epsg=3857)
        p1_x, p1_y = p1.x.to_numpy(), p1.y.to_numpy()
        p2 = df["position" + l2].to_crs(epsg=3857)
        p2_x, p2_y = p2.x.to_numpy(), p2.y.to_numpy()
        dt[:, i] = (df["timestamp" + l2] -
                    df["timestamp" + l1]).dt.seconds.to_numpy()

        X[:, i, :] = np.vstack([p1_x, p1_y]).T
        T[:, i] = (df["timestamp" + l1] -
                   df["timestamp"]).dt.seconds.to_numpy()
        msk = dt[:, i] > 1
        Y[msk, i, :] = np.vstack(
            [p2_x - p1_x, p2_y - p1_y]).T[msk] / dt[msk, i].reshape((-1, 1))
    X = X[msk]
    Y = Y[msk]
    T = T[msk]
    if flatten:
        return X.reshape((-1, 2)), Y.reshape((-1, 2)), T.flatten()
    return X, Y, T


def kernel_rbf(X1: np.ndarray, X2: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    K = np.empty((X2.shape[0], X1.shape[0]))
    K = (X1.reshape((1, -1, 2)) - X2.reshape((-1, 1, 2))) / lengthscale
    K = np.sum(K * K, axis=-1)
    K = np.exp(-K)
    return (sigma) * K


def kernel(X1: np.ndarray, X2: np.ndarray, sigmas: np.ndarray, lengthscales: np.ndarray, weights: np.ndarray = None) -> np.ndarray:

    if weights is None:
        weights = np.ones_like(sigmas)

    assert sigmas.ndim == 1
    assert lengthscales.shape == sigmas.shape
    assert weights.shape == sigmas.shape

    K = np.zeros((X2.shape[0], X1.shape[0]))
    for i in range(len(sigmas)):
        K += kernel_rbf(X1, X2, sigmas[i], lengthscales[i]) * weights[i]
    return K


def dkernel(X1: np.ndarray, X2: np.ndarray, sigmas: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    assert X1.shape == (1, 2)
    assert X2.shape[-1] == 2
    return -(X1 - X2) * kernel(X1, X2, sigmas, lengthscales, weights=(1.0 / lengthscales**2))


def kf_log_likelihood(v: np.ndarray, S: np.ndarray) -> np.ndarray:
    v = v.T.reshape((-1, 2, 1))
    nis = v.transpose(0, 2, 1) @ np.linalg.solve(S, v)  # (N x 1 x 1)
    nis = nis.flatten()  # (N,)
    return - 0.5 * (nis + np.log(2.0 * np.pi * np.linalg.det(S)))


def _pdaf_update(x_hat: np.ndarray, P_hat: np.ndarray, z: np.ndarray, R: np.ndarray, clutter_rate: float, p_d: float, gate_size) -> Tuple[np.ndarray, np.ndarray, int]:

    # Prepare constants
    logClutter = np.log(clutter_rate)
    logPD = np.log(p_d)
    logPND = np.log(1 - p_d)

    # Reshape values to follow linalg conventions
    x_hat = x_hat.reshape((2, 1))  # Column vector
    z = z.T  # (2 x N)

    assert z.shape[0] == 2
    assert x_hat.shape[0] == 2

    vs = (z - x_hat)  # (2 x N) innovation vector, one row for each measurement z
    S = P_hat + R  # (2 x 2)

    # Gating
    gvs = vs.T[..., np.newaxis]  # (N x 2 x 1)
    g = gvs.transpose((0, 2, 1)) @ np.linalg.solve(S, gvs)  # (N x 1 x 1)
    g = g.squeeze()  # (N,)

    assert g.ndim == 1

    g_msk = g < gate_size**2
    vs = vs[:, g_msk]

    ll = np.empty(vs.shape[1]+1)
    ll[0] = logPND + logClutter
    ll[1:] = kf_log_likelihood(vs, S) + logPD
    beta = np.exp(ll)
    beta /= np.sum(beta)
    vk = np.sum(beta[1:].reshape((1, -1)) * vs, axis=1).reshape((2, 1))
    W = np.linalg.solve(S.T, P_hat.T).T

    # P with moment matching
    v = vs.T.reshape((-1, 2, 1))  # (N x 2 x 1)
    v = beta[1:].reshape((-1, 1, 1)) * \
        (v @ v.transpose(0, 2, 1))  # (N x 2 x 2)
    v = np.sum(v, axis=0) - vk @ vk.T  # (2 x 2)
    v = W @ v @ W.T
    P = P_hat - (1-beta[0]) * W @ S @ W + v
    # X with moment matching
    x = x_hat + W @ vk  # (2 x 1) updated state vector x
    assert x.shape == (2, 1)
    assert P.shape == (2, 2)
    return x.flatten(), P, g_msk.sum()


@dataclass
class Params:
    """ Parameters used by DynGP """

    # GP parameters
    lengthscales: np.ndarray
    sigmas: np.ndarray
    noise: float

    # PDAF parameters
    R: np.ndarray
    clutter_rate: float
    p_d: float
    gate_size: float


def get_default_mle_params(train_x: np.ndarray, train_y: np.ndarray, return_score: bool = False, test_x: np.ndarray = None, test_y: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """ Finds best MLE hyperparameters for GP and otherwise use defaults

    This will internally use scikit-learn's GaussianProcessRegressor implementation to perform MLE. 



    Params:
    ----------------
    train_x (np.ndarray): Training inputs with shape (N x 2).

    train_y (np.ndarray): Training outputs with shape (N x 2).

    return_score (bool): If true, the R^2 score of the training inputs is also returned. 

    test_x (np.ndarray): Test inputs for R^2 scoring, ignored if return_score is false.

    test_y (np.ndarray): Test outputs for R^2 scoring, ignored if return_score is false.

    **kwargs: Additional named parameters to pass to sklearn.GaussianProcessRegressor's constructor


    Returns:
    ------------------
    (params) if return_score is False.

    (params, train_r_squared) if return_score is True and either test_x or test_y is None.

    (params, train_r_squared, test_r_squared) if return_score is True and both test_x and test_y is not None.

    Additional notes: 
    - It might complain about too low noise parameters after optimizing, but this can safely be ignored as the noise component is not always neccessary.
    - It depends scikit-learn's implementation of the kernels are identical to the the implementation in DynGP. 


    """
    kernel = kernels.ConstantKernel() * kernels.RBF(length_scale=2000) \
        + kernels.ConstantKernel() * kernels.RBF() \
        + kernels.WhiteKernel()

    if "n_restarts_optimizer" not in kwargs:
        kwargs["n_restarts_optimizer"] = 10

    gpr = GaussianProcessRegressor(
        kernel, normalize_y=True, copy_X_train=False, **kwargs)
    gpr.fit(train_x, train_y)

    p = gpr.kernel_.get_params()

    params = Params(
        lengthscales=np.array([
            p["k1__k1__k2"].length_scale,
            p["k1__k2__k2"].length_scale
        ]),
        sigmas=np.array([
            p["k1__k1__k1"].constant_value,
            p["k1__k2__k1"].constant_value
        ]),
        noise=p["k2"].noise_level,
        clutter_rate=1e-4,
        R=1000**2 * np.eye(2),
        gate_size=2,
        p_d=0.80,
    )

    if return_score:
        train_r_squared = (gpr.score(train_x, train_y),)
        if test_x is not None and test_y is not None:
            test_r_squared = (gpr.score(test_x, test_y),)
            return params, train_r_squared, test_r_squared
        return params, train_r_squared
    return params


class DynGP():
    def __init__(self, train_x: np.ndarray, train_delta_y: np.ndarray, params: Params, normalize_y: bool = False):
        assert train_x.ndim == 2
        assert train_x.shape[-1] == 2
        assert train_delta_y.ndim == 2
        assert train_delta_y.shape[-1] == 2

        self.params = params

        # Set parameters
        self._train_x = train_x
        self._train_delta_y = train_delta_y

        # Precompute L and alpha for later use
        k = kernel(
            X1=train_x,
            X2=train_x,
            sigmas=self.params.sigmas,
            lengthscales=self.params.lengthscales,
        )

        self._y_mean: np.ndarray = train_delta_y.mean(
            axis=0) if normalize_y else np.zeros(2)
        self._y_var: np.ndarray = train_delta_y.var(
            axis=0) if normalize_y else np.ones(2)
        self._y_std: np.ndarray = np.sqrt(self._y_var)

        self.L = np.linalg.cholesky(
            k + (self.params.noise) * np.eye(k.shape[0]))
        self.alpha = solve_triangular(
            self.L, (train_delta_y - self._y_mean) / self._y_std, lower=True)
        self.alpha = solve_triangular(
            self.L, self.alpha, lower=True, trans="T", overwrite_b=True)

    def _predict(self, pred_x: np.ndarray, return_as_unit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Evaluates the function f, conditioned on training samples in ctx """
        pred_x = pred_x.reshape((-1, 2))
        k = kernel(
            X1=pred_x,
            X2=self._train_x,
            sigmas=self.params.sigmas,
            lengthscales=self.params.lengthscales
        )
        mean = (k.T @ self.alpha)
        v = solve_triangular(self.L, k, overwrite_b=True,
                             lower=True, check_finite=False)
        var = kernel(
            X1=pred_x,
            X2=pred_x,
            sigmas=self.params.sigmas,
            lengthscales=self.params.lengthscales
        ) - v.T @ v
        if return_as_unit:
            return mean, var
        return mean * self._y_std + self._y_mean, var * self._y_var.reshape((2, 1, 1))

    def _sample(self, pred_x: np.ndarray) -> np.ndarray:
        # mean = (N, 2), var = (N, N)
        mean, var = self._predict(pred_x, return_as_unit=True)
        A = np.linalg.cholesky(var + np.eye(var.shape[0]) * 1e-6)  # (N, N)
        X = mean + A @ np.random.normal(size=(pred_x.shape[0], 2))
        return X * self._y_std + self._y_mean

    def _predict_F(self, pred_x: np.ndarray) -> np.ndarray:
        pred_x = pred_x.reshape((1, 2))
        dk = dkernel(pred_x, self._train_x, self.params.sigmas,
                     self.params.lengthscales)
        return (dk.T @ self.alpha) * self._y_std

    def kalman(self, init_x: np.ndarray, end_time: float, dt: float = 1.0, pdaf_update: bool = True, return_gated: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        steps = int(np.ceil(end_time / dt))+1
        init_x = init_x.reshape((2,))
        x = np.empty((steps, init_x.shape[-1]))
        gated_measurements = np.empty((steps,))
        P = np.empty((steps, 2, 2))
        P[0] = np.eye(2)*100**2
        x[0] = init_x
        identity = np.eye(2)
        dt_squared_eye = identity * dt ** 2
        for i in range(1, steps):
            # Predict
            f, v = self._predict(
                pred_x=x[i-1],
            )
            x[i] = x[i-1] + f * dt  # g(x_{t-1})
            G = identity + self._predict_F(
                pred_x=x[i-1],
            ) * dt
            P[i] = (G.T @ P[i-1]
                    @ G) + v.squeeze() * dt_squared_eye
            # Update
            if pdaf_update:
                x[i], P[i], gated_measurements[i] = _pdaf_update(
                    x_hat=x[i],
                    P_hat=P[i],
                    z=self._train_x,
                    R=self.params.R,
                    p_d=self.params.p_d,
                    clutter_rate=self.params.clutter_rate,
                    gate_size=self.params.gate_size
                )
        ret = (x, P)
        if return_gated:
            ret += (gated_measurements,)
        return ret

    def particle(self, init_x: np.ndarray, end_time: float, dt: float = 1.0, N_particles: int = 100) -> np.ndarray:
        if init_x.ndim > 1:
            print("Flattening input")
            init_x = init_x.reshape((2,))
        assert init_x.shape == (2,), "Only support for one starting point atm"
        steps = int(np.ceil(end_time / dt))+1  # Adding one to include init_x

        X = np.empty((steps, N_particles, 2))
        X[0] = init_x.reshape((1, 2))

        for i in range(1, steps):
            X[i] = X[i-1] + self._sample(X[i-1]) * dt
        return X

    def plot_uncertianty_grid(self, xlim=(-1, 1), ylim=(-1, 1), size=(20, 20)):
        """ Plots the underlying uncertianty as a background color

        Darker areas corresponds to more uncertianty. 

        Parameters:
        train_x (np.ndarray): Training inputs
        train_delta_y (np.ndarray): Training vector samples
        xlim (tuple): range of x values
        ylim (tuple): range of y values
        size (tuple): number of x and y values between xlim and ylim
        noise (tuple): measurement noise in training samples 
        """
        x1, x2 = xlim
        y1, y2 = ylim
        sx, sy = size
        dX = x2 - x1
        dY = y2 - y1
        #x1 -= dX / (sx+1)
        x2 += dX / (sx+1)
        #y1 -= dY / (sy+1)
        y2 += dY / (sy+1)
        pX, pY = np.meshgrid(np.linspace(x1, x2, sx+1),
                             np.linspace(y1, y2, sy+1))
        pred_x = np.vstack([pX.ravel(), pY.ravel()]).T
        _, var = self._predict(pred_x)
        std = np.sqrt(var.diagonal(axis1=1, axis2=2)).reshape((2, sx+1, sy+1))
        std = np.linalg.norm(std, axis=0)
        plt.pcolormesh(pX, pY, std, cmap="Greys", shading="gouraud")
        #plt.quiver(*pred_x.T, *pred_f.T)
        plt.quiver(*self._train_x.T, *self._train_delta_y.T,
                   color="red", alpha=0.1)


if __name__ == "__main__":
    pass