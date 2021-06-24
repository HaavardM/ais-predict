from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from scipy.special import logsumexp
from dataclasses import dataclass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels


from typing import Optional, Tuple


def samples_from_lag_n_df(df: gpd.GeoDataFrame, n: int, flatten: bool = True, use_cog: bool = False):
    X = np.empty((len(df.index), n, 3))
    Y = np.empty((len(df.index), n, 2))
    dt = np.empty((len(df.index), n))
    for i in range(n):

        l1 = f"_{i}"
        l2 = f"_{i+1}"
        if i == 0:
            l1 = ""
        p1 = df["position" + l1]
        p1_x, p1_y = p1.x.to_numpy(), p1.y.to_numpy()
        p2 = df["position" + l2]
        p2_x, p2_y = p2.x.to_numpy(), p2.y.to_numpy()
        dt[:, i] = (df["timestamp" + l2] -
                    df["timestamp" + l1]).dt.seconds.to_numpy()

        X[:, i, :2] = np.vstack([p1_x, p1_y]).T
        X[:, i, -1] = (df["timestamp" + l1] -
                       df["timestamp"]).dt.seconds.to_numpy()
        msk = dt[:, i] > 1
        if use_cog:
            rad = df["cog" + l1] * np.pi / 180  # + np.pi / 2.0
            sog = df["sog" + l1] * 0.514444444
            Y[msk, i, :] = np.vstack([
                np.sin(rad),
                np.cos(rad)
            ]).T[msk] * sog.to_numpy()[msk][..., np.newaxis]
        else:
            Y[msk, i, :] = np.vstack(
                [p2_x - p1_x, p2_y - p1_y]).T[msk] / dt[msk, i].reshape((-1, 1))
    msk = dt > 1 & ~np.any(
        np.isnan(X), axis=-1) & ~np.any(np.isnan(Y), axis=-1)
    X = X[msk]
    Y = Y[msk]
    if flatten:
        return X.reshape((-1, 3)), Y.reshape((-1, 2))
    return X, Y


def kernel_rbf(X1: np.ndarray, X2: np.ndarray, sigma: float, lengthscale: np.ndarray) -> np.ndarray:
    assert X1.shape[-1] == X2.shape[-1]
    assert lengthscale.ndim == 1
    assert lengthscale.shape[-1] == X1.shape[-1]

    D = X1.shape[-1]

    K = np.empty((X2.shape[0], X1.shape[0]))
    K = (X1.reshape((1, -1, D)) - X2.reshape((-1, 1, D))) / lengthscale
    K = np.sum(K * K, axis=-1) / 2.0
    K = np.exp(-K)
    return (sigma) * K


def kernel(X1: np.ndarray, X2: np.ndarray, sigmas: np.ndarray, lengthscales: np.ndarray, weights: np.ndarray = None) -> np.ndarray:

    if weights is None:
        weights = np.ones_like(sigmas)[..., np.newaxis]

    assert sigmas.ndim == 1
    assert lengthscales.ndim == 2
    assert lengthscales.shape[0] == sigmas.shape[0]
    #assert weights.shape == lengthscales.shape

    K = np.zeros((weights.shape[-1], X2.shape[0], X1.shape[0]))
    for i in range(len(sigmas)):
        K += kernel_rbf(X1, X2, sigmas[i], lengthscales[i]) * \
            weights[i][..., np.newaxis, np.newaxis]
    if K.shape[0] == 1:
        return K[0]
    else:
        return K


def dkernel(X1: np.ndarray, X2: np.ndarray, sigmas: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:

    assert X2.shape[-1] == X1.shape[-1] == 3
    k = kernel(X1, X2, sigmas, lengthscales, weights=(
        1.0 / lengthscales[:, :2]**2)).squeeze().T
    return -(X1 - X2)[:, :2] * k


def kf_log_likelihood(v: np.ndarray, S: np.ndarray) -> np.ndarray:
    v = v.T[..., np.newaxis]
    nis = v.transpose(0, 2, 1) @ np.linalg.solve(S, v)  # (N x 1 x 1)
    nis = nis.flatten()  # (N,)
    return - nis / 2.0 - np.log(2.0 * np.pi) - np.log(np.linalg.det(S)) / 2.0


def _pdaf_update(x_hat: np.ndarray, P_hat: np.ndarray, z: np.ndarray, R: np.ndarray, clutter_rate: float, p_d: float, gate_size, update_P: bool) -> Tuple[np.ndarray, np.ndarray, int]:

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
    beta = np.exp(ll - logsumexp(ll))
    assert abs(np.sum(beta) - 1.0) < 1e-6
    vk = np.sum(beta[1:][np.newaxis, ...] * vs, axis=1).reshape((2, 1))
    W = P_hat @ np.linalg.inv(S)

    x_upd = x_hat + W @ vs
    x_upd = x_upd.T[..., np.newaxis]

    P_upd = np.eye(2) - W

    P_upd = (1-beta[0]) * (P_upd @ P_hat @
                           P_upd.T + W @ R @ W) + beta[0] * P_hat

    # P with moment matching
    v = vs.T[..., np.newaxis]  # (N x 2 x 1)
    v = beta[1:][..., np.newaxis, np.newaxis] * \
        (v @ v.transpose(0, 2, 1))  # (N x 2 x 2)
    v = np.sum(v, axis=0) - vk @ vk.T  # (2 x 2)
    v = W @ v @ W.T
    P = P_hat - ((1-beta[0]) * W @ S @ W) + v
    P = P if update_P else P_hat
    # X with moment matching
    x = x_hat + W @ vk  # (2 x 1) updated state vector x

    assert x.shape == (2, 1)
    assert P.shape == (2, 2)
    return x.flatten(), P, g_msk.sum()


def _synthetic_update(x_hat: np.ndarray, P_hat: np.ndarray, z: np.ndarray, R: np.ndarray, search_radius: float, update_P: bool) -> Tuple[np.ndarray, np.ndarray, int]:

    dist = z - x_hat
    dist = np.linalg.norm(dist, axis=1)
    S = P_hat + R  # (2 x 2)
    W = np.linalg.solve(S.T, P_hat.T).T

    #print(0, x_hat)
    #print(1, np.sort(dist)[:5])
    ix = dist < search_radius
    z = z[ix]
    dist = dist[ix]

    ix = np.argsort(dist)

    # print(2,z.shape)
    if z.shape[0] == 0:
        return x_hat, P_hat
    median = z.mean(axis=0) # z[ix[len(ix) // 2]]

    #print(3,mean, x_hat)
    #print(4, W @ (mean - x_hat))
    x = x_hat + W @ (median - x_hat)
    #print(W @ (median - x_hat))
    P = (np.eye(2) - W) @ P_hat @ (np.eye(2) - W).T + W @ R @ W.T
    P = P if update_P else P_hat

    return x, P


@dataclass
class Params:
    """ Parameters used by DynGP """

    # GP parameters
    lengthscales: np.ndarray
    sigmas: np.ndarray
    noise: float

    # PDAF parameters
    pdaf_R: np.ndarray
    pdaf_clutter_rate: float
    pdaf_p_d: float
    pdaf_gate_size: float

    # Synthetic parameters
    synthetic_search_radius: float
    synthetic_R: float


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
    kernel = kernels.ConstantKernel(1.0) * kernels.RBF(length_scale=[4000.0, 4000.0, 1000.0], length_scale_bounds=(10, 1000)) \
        + kernels.WhiteKernel(noise_level_bounds=(0.0001, 10))

    if "n_restarts_optimizer" not in kwargs:
        kwargs["n_restarts_optimizer"] = 10

    gpr = GaussianProcessRegressor(
        kernel, normalize_y=True, copy_X_train=False, **kwargs)
    gpr.fit(train_x, train_y)

    p = gpr.kernel_.get_params()

    params = Params(
        lengthscales=np.array([
            p["k1__k2"].length_scale,
        ]),
        sigmas=np.array([
            p["k1__k1"].constant_value,
        ]),
        noise=p["k2"].noise_level,
        pdaf_R=np.eye(2) * 50**2,
        pdaf_p_d=0.8,
        pdaf_clutter_rate=2e-3,
        pdaf_gate_size=2.0,
        synthetic_search_radius= 500,
        synthetic_R=np.eye(2) * 2000**2
    )

    if return_score:
        train_r_squared = (gpr.score(train_x[:, :2], train_y),)
        if test_x is not None and test_y is not None:
            test_r_squared = (gpr.score(test_x[:, :2], test_y),)
            return params, train_r_squared, test_r_squared
        return params, train_r_squared
    return params


class DynGP():
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, params: Params, normalize_y: bool = True):
        assert train_x.ndim == 2
        assert train_x.shape[-1] == 3
        assert train_y.ndim == 2
        assert train_y.shape[-1] == 2

        self.params = params

        # Set parameters
        self._train_x = train_x
        self._train_y = train_y

        # Precompute L and alpha for later use
        k = kernel(
            X1=train_x,
            X2=train_x,
            sigmas=self.params.sigmas,
            lengthscales=self.params.lengthscales,
        )

        self._y_mean: np.ndarray = train_y.mean(
            axis=0) if normalize_y else np.zeros(2)
        self._y_var: np.ndarray = train_y.var(
            axis=0) if normalize_y else np.ones(2)
        self._y_std: np.ndarray = np.sqrt(self._y_var)

        self.L, _ = cf = cho_factor(
            k + (self.params.noise) * np.eye(k.shape[0]), lower=True)
        self.alpha = cho_solve(
            cf, (train_y - self._y_mean) / self._y_std)

    def f(self, pred_x: np.ndarray, return_as_unit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Evaluates the function f, conditioned on training samples in ctx """
        pred_x = pred_x.reshape((-1, 3))
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
        mean, var = self.f(pred_x, return_as_unit=True)
        v = np.diagonal(var)
        A = np.linalg.cholesky(var + np.eye(var.shape[0]) * 1e-6)  # (N, N)
        X = mean + A @ np.random.normal(size=(pred_x.shape[0], 2))
        return X * self._y_std + self._y_mean, v

    def _predict_F(self, pred_x: np.ndarray) -> np.ndarray:
        pred_x = pred_x.reshape((1, 3))
        dk = dkernel(pred_x, self._train_x, self.params.sigmas,
                     self.params.lengthscales)
        # print(abs(dk).max(axis=0))

        # print(self.alpha.min(axis=0))
        # print(dk.shape)
        # print(self.alpha.shape)
        return (dk.T @ self.alpha).T * self._y_std

    def kalman(self, init_x: np.ndarray, end_time: float, dt: float, pdaf_update: bool, synthetic_update: bool, P_0: np.ndarray, update_P: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        steps = int(np.ceil(end_time / dt))+1
        init_x = init_x.reshape((2,))
        x = np.zeros((steps, 3))
        gated_measurements = np.empty((steps,))
        P = np.empty((steps, 2, 2))
        P[0] = P_0
        x[0, :2] = init_x
        identity = np.eye(2)
        for i in range(1, steps):
            # Predict
            f, v = self.f(
                pred_x=x[i-1],
            )

            x[i, :2] = x[i-1, :2] + f * dt  # g(x_{t-1})
            x[i, -1] = i*dt
            G = identity + self._predict_F(
                pred_x=x[i-1],
            ) * dt


            #print(1, G, self.num_jac(x[i-1], epsilon=1e-6), sep="\n")

            P[i] = (G @ P[i-1]
                    @ G.T) + np.diag(v.squeeze() * dt**2)
            #print(f"Iteration {i}", P[i], 1, G, 2, G_num, 3, G @ P[i-1] @ G.T, 4, G_num @ P[i-1] @ G.T, sep="\n")
            #print()

            # Update
            if pdaf_update:
                x[i, :2], P[i], gated_measurements[i] = _pdaf_update(
                    x_hat=x[i, :2],
                    P_hat=P[i],
                    z=self._train_x[:, :2],
                    R=self.params.pdaf_R,
                    p_d=self.params.pdaf_p_d,
                    clutter_rate=self.params.pdaf_clutter_rate,
                    gate_size=self.params.pdaf_gate_size,
                    update_P=update_P
                )
            elif synthetic_update:
                x[i, :2], P[i] = _synthetic_update(
                    x_hat=x[i, :2],
                    P_hat=P[i],
                    z=self._train_x[:, :2],
                    R=self.params.synthetic_R,
                    update_P=update_P,
                    search_radius=self.params.synthetic_search_radius
                )
        ret = (x, P)
        return ret

    def particle(self, init_x: np.ndarray, end_time: float, dt: float = 1.0, N_particles: int = 100, cutoff: float = 100) -> np.ndarray:
        if init_x.ndim > 1:
            print("Flattening input")
            init_x = init_x.reshape((2,))
        assert init_x.shape == (2,), "Only support for one starting point atm"
        steps = int(np.ceil(end_time / dt))+1  # Adding one to include init_x

        X = np.zeros((steps, N_particles, 3))
        X[0, :, :2] = init_x.reshape(
            (1, 2)) + np.random.normal(loc=0, scale=100, size=(N_particles, 2))
        X[:, :, -1] = (np.arange(steps) * dt)[..., np.newaxis]
        w = np.zeros((steps, N_particles))

        for i in range(1, steps):
            p, v = self._sample(X[i-1])
            X[i, :, :2] = X[i-1, :, :2] + p * dt
            w[i-1] = v
        _, w[-1] = self._sample(X[-1])
        return X, w

    def num_jac(self, x: np.ndarray, epsilon: float = 0.001):
        x = x[np.newaxis, ...].repeat(3, axis=0)
        x[1][0] += epsilon
        x[2][1] += epsilon

        f, v = self.f(x)

        f0 = f[0]
        f_x1 = f[1]
        f_x2 = f[2]

        J = np.empty((2, 2))
        J[:, 0] = (f_x1 - f0) / epsilon
        J[:, 1] = (f_x2 - f0) / epsilon
        return J


if __name__ == "__main__":
    pass
