from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import sklearn.gaussian_process.kernels as kernels
from scipy.linalg import solve_triangular
from numba import prange, njit
from dataclasses import dataclass


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


@njit(nogil=True, parallel=True)
def kernel_rbf(X1: np.ndarray, X2: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    K = np.empty((X2.shape[0], X1.shape[0]))
    for i in prange(X2.shape[0]):
        d = (X1 - X2[i]) / lengthscale
        norm_dist_squared = np.sum(d * d, axis=1)
        K[i] = np.exp(-norm_dist_squared)
    return (sigma) * K


def kernel(X1: np.ndarray, X2: np.ndarray, sigmas: np.ndarray, lengthscales: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """ Computes the kernel for inputs X1 and X2. 
    
    The kernel is the sum of n RBF kernels, given by the parameters in the ndarrays sigmas and lengthscales
    """
    assert sigmas.ndim == 1
    assert lengthscales.ndim == 1

    K = np.empty((sigmas.shape[0], X2.shape[0], X1.shape[0]))
    for i in range(K.shape[0]):
        K[i] = kernel_rbf(X1, X2, sigmas[i], lengthscales[i])
    if weights is not None:
        K = K * weights.reshape((-1, 1, 1))
    return np.sum(K, axis=0)

def dkernel(X1: np.ndarray, X2: np.ndarray, sigmas: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    assert X1.shape == (1, 2)
    assert X2.shape[-1] == 2
    return -(X1 - X2) * kernel(X1, X2, sigmas, lengthscales, weights=(1.0 / lengthscales**2))


def kf_log_likelihood(v: np.ndarray, S: np.ndarray) -> np.ndarray:
    v = v.T.reshape((-1, 2, 1))
    nis = v.transpose(0, 2, 1) @ np.linalg.solve(S, v)  # (N x 1 x 1)
    nis = nis.flatten()  # (N,)
    return - 0.5 * (nis + np.log(2.0 * np.pi * np.linalg.det(S)))


def _pdaf_update(x_hat: np.ndarray, P_hat: np.ndarray, z: np.ndarray, R: np.ndarray, clutter_rate: float, p_d: float, gate_size) -> tuple:

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
    W = P_hat @ np.linalg.inv(S)

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
        lengthscales: np.ndarray
        sigmas: np.ndarray
        noise: float
        R: np.ndarray
        clutter_rate: float
        p_d: float
        gate_size: float

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
        self._y_var: np.ndarray = train_delta_y.var(axis=0) if normalize_y else np.ones(2)
        self._y_std: np.ndarray = np.sqrt(self._y_var)

        self.L = np.linalg.cholesky(k + (self.params.noise) * np.eye(k.shape[0]))
        self.alpha = solve_triangular(
            self.L, (train_delta_y - self._y_mean) / self._y_std, lower=True)
        self.alpha = solve_triangular(
            self.L, self.alpha, lower=True, trans="T", overwrite_b=True)

    def _predict(self, pred_x: np.ndarray, return_as_unit: bool = False) -> tuple:
        """ Evaluates the function f, conditioned on training samples in ctx """
        pred_x = pred_x.reshape((-1, 2))
        k = kernel(
            X1=pred_x,
            X2=self._train_x,
            sigmas=self.params.sigmas,
            lengthscales=self.params.lengthscales
        )
        mean = (k.T @ self.alpha)
        v = solve_triangular(self.L, k, overwrite_b=True, lower=True)
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
        mean, var = self._predict(pred_x, return_as_unit=True) # mean = (N, 2), var = (N, N)
        A = np.linalg.cholesky(var) # (N, N)
        Z = np.random.normal(size=(pred_x.shape[0], 2)) #(N x 2) standard normal samples
        X = mean + A @ Z
        return X * self._y_std + self._y_mean


    def _predict_F(self, pred_x: np.ndarray) -> np.ndarray:
        pred_x = pred_x.reshape((1, 2))
        dk = dkernel(pred_x, self._train_x, self.params.sigmas, self.params.lengthscales)
        return (dk.T @ self.alpha) * self._y_std

    def kalman(self, init_x: np.ndarray, end_time: float, dt: float = 1.0, pdaf_update: bool = True, return_gated: bool = False) -> np.ndarray:
        assert init_x.shape[-1] == 2
        steps = int(np.ceil(end_time / dt))+1
        if init_x.ndim == 1:
            print("Reshaping input")
            init_x = init_x.reshape((1, 2))
        traj_shape = tuple(init_x.shape[:-1])

        x = np.empty(traj_shape + (steps, init_x.shape[-1]))
        gated_measurements = np.empty(traj_shape + (steps,))
        P = np.empty(traj_shape + (steps, 2, 2))
        P[:, 0] = np.eye(2)*100*100
        x[:, 0] = init_x
        identity = np.eye(2)
        dt_squared_eye = identity * dt ** 2
        for traj in np.ndindex(traj_shape):
            for i in range(1, steps):
                # Predict
                f, v = self._predict(
                    pred_x=x[traj + (i-1,)],
                )
                x[traj + (i,)] = x[traj + (i-1,)] + f * dt  # g(x_{t-1})
                G = identity + self._predict_F(
                    pred_x=x[traj + (i-1,)],
                ) * dt
                P[traj + (i,)] = (G.T @ P[traj + (i-1,)]
                                  @ G) + v.squeeze() * dt_squared_eye
                # Update
                if pdaf_update:
                    x[traj + (i,)], P[traj + (i,)], gated_measurements[traj + (i,)] = _pdaf_update(x[traj + (i,)], P[traj + (i,)], self._train_x,
                                                                                                   self.params.R, p_d=self.params.p_d, clutter_rate=self.params.clutter_rate, gate_size=self.params.gate_size)
        ret = (x, P)
        if return_gated:
            ret += (gated_measurements,)
        return ret
    
    def particle(self, init_x: np.ndarray, end_time: float, dt: float = 1.0, N_particles: int =100) -> np.ndarray:
        if init_x.ndim > 1:
            print("Flattening input")
            init_x = init_x.reshape((2,))
        assert init_x.shape == (2,), "Only support for one starting point atm"
        steps = int(np.ceil(end_time / dt))+1 # Adding one to include init_x

        X = np.empty((steps, N_particles, 2))
        X[0] = init_x.reshape((1, 2)) + np.random.normal(scale=10, size=(N_particles, 2))

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
    import timeit
    import sys

    theta, r = np.mgrid[-np.pi:np.pi:0.1, 0:1:0.1]
    X = np.cos(theta)*r
    Y = np.sin(theta)*r
    X = np.vstack([X.T.ravel(), Y.T.ravel()]).T
    msk = np.linalg.norm(X, axis=1) > 0.1
    X = X[msk]
    DY = X[1:] - X[:-1]
    DY /= np.linalg.norm(DY, axis=1).reshape((-1, 1))
    # print(DY)
    X = X[:-1]
    msk = np.random.choice(X.shape[0], 20)
    X = X[msk, :]
    DY = DY[msk, :]
