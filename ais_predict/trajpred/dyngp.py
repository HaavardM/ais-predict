import numpy as np
import geopandas as gpd
import sklearn.gaussian_process.kernels as kernels
from scipy.linalg import solve_triangular
from numba import prange, njit

def samples_from_lag_n_df(df: gpd.GeoDataFrame, n: int):
    cog = df.cog.to_numpy()
    sog = df.sog.to_numpy()

    X = np.empty((len(df.index), n, 2))
    Y = np.empty((len(df.index), n, 2))
    for i in range(n):
        
        p1 = f"_{i}"
        p2 = f"_{i+1}"
        if i == 0:
            p1 = ""
        p1 = df["position" + p1].to_crs(epsg=3857)
        p1_x, p1_y = p1.x.to_numpy(), p1.y.to_numpy()
        p2 = df["position" + p2].to_crs(epsg=3857)
        p2_x, p2_y = p2.x.to_numpy(), p2.y.to_numpy()


        X[:, i, :] = np.vstack([p1_x, p1_y]).T
        Y[:, i, :] = np.vstack([p2_x - p1_x, p2_y - p1_y]).T
    return X.reshape((-1, 2)), Y.reshape((-1, 2))

@njit(nogil=True, parallel=True)
def kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1 ) -> np.ndarray:
    K = np.empty((X2.shape[0], X1.shape[0]))
    for i in prange(X2.shape[0]):
        d = (X1 - X2[i]) / lengthscale
        norm_dist_squared = np.sum(d * d, axis=1)
        K[i] = np.exp(-norm_dist_squared)
    return K

@njit(nogil=True)
def solve_lower_triangular(L: np.ndarray, b: np.ndarray, lower: bool = True, replace_b: bool = False) -> np.ndarray:
    if not replace_b:
        b = b.copy()
    if lower:
        for i in range(b.shape[0]):
            b[i] = (b[i] - L[i, :i] @ b[:i]) / L[i, i]
    else:
        for i in range(b.shape[0]-1, -1, -1):
            b[i] = (b[i] - L[i, i+1:] @ b[i+1:]) / L[i, i]
    return b

@njit(nogil=True)
def init(train_x: np.ndarray, train_delta_y: np.ndarray, noise: float = 1.0, lengthscale: float = 1)->np.ndarray:
    k = kernel(train_x, train_x, lengthscale=lengthscale)
    L = np.linalg.cholesky(k + noise * np.eye(k.shape[0]))
    alpha = solve_lower_triangular(L, train_delta_y)
    alpha = solve_lower_triangular(L.T, alpha, lower=False)
    return L, alpha

@njit(nogil=True, parallel=True)
def dyngp_kalman(train_x: np.ndarray, train_delta_y: np.ndarray, init_x: np.ndarray, noise: float = 0.1, steps: int = 1000, dt: float = 1, lengthscale: float = 1) -> np.ndarray:
    N_trajectories = init_x.shape[0]
    
    L, alpha = init(train_x, train_delta_y, noise=noise, lengthscale=lengthscale)

    x = np.empty((N_trajectories, steps) + (init_x.shape[-1],))
    var = np.empty((N_trajectories, steps, 2))
    var[:, 0] = noise**2
    x[:, 0] = init_x 
    for traj in prange(N_trajectories):
        for i in range(1, steps):
            k = kernel(x[traj, i-1, :].reshape((1, -1)), train_x, lengthscale=lengthscale)
            x[traj][i] = x[traj, i-1, :] + (k.T @ alpha)*dt
            v = solve_lower_triangular(L, k, replace_b=True)
            var[traj][i] = kernel(x[traj, i-1, :].reshape((1, -1)), x[traj, i-1, :].reshape((1, 2)), lengthscale=lengthscale) - v.T @ v
    return x, var

import matplotlib.pyplot as plt
def plot_grid(train_x: np.ndarray, train_delta_y: np.ndarray, xlim=(-1, 1), ylim=(-1,1), size=(20, 20), noise: float = 0.1, lengthscale: float = 1):
    L, alpha = init(train_x, train_delta_y, noise=noise)
    
    x1, x2 = xlim
    y1, y2 = ylim
    sx, sy = size
    pX, pY = np.mgrid[
        x1:x2:(x2-x1)/sx,
        y1:y2:(y2-y1)/sy
    ]
    pred_x = np.vstack([pX.ravel(), pY.ravel()]).T
    k = kernel(pred_x, train_x, lengthscale=lengthscale)
    pred_f = k.T @ alpha
    v = solve_lower_triangular(L, k, replace_b=True)
    pred_var = kernel(pred_x, pred_x, lengthscale=lengthscale) - v.T @ v
    std = np.sqrt(pred_var.diagonal()).reshape(size)
    plt.pcolormesh(pX, pY, std, cmap="Greys", shading="gouraud")
    #plt.quiver(*pred_x.T, *pred_f.T)
    plt.quiver(*train_x.T, *train_delta_y.T, color="red", alpha=0.1)


if __name__ == "__main__":
    import timeit
    import sys


    theta, r = np.mgrid[-np.pi:np.pi:0.001, 0:1:0.001]
    X = np.cos(theta)*r
    Y = np.sin(theta)*r
    X = np.vstack([X.T.ravel(), Y.T.ravel()]).T
    msk = np.linalg.norm(X, axis=1) > 0.1
    X = X[msk]
    DY = X[1:] - X[:-1]
    DY /= np.linalg.norm(DY, axis=1).reshape((-1, 1))
    print(DY)
    X = X[:-1]
    msk = np.random.choice(X.shape[0], 20)
    X = X[msk, :]
    DY = DY[msk, :]
    print(DY.shape, X.shape)

    plot_grid(X, DY)
    plt.savefig("grid.png")

    sys.exit()
    f, std = dyngp_kalman(X, DY, np.array([[0.5, 0.0], [0.0, 0.7], [1.0, 1.0]]), dt=0.1, steps=10000)
    print(f.shape)
    import matplotlib.pyplot as plt
    #plt.plot(*Y.T, color="black")
    plt.quiver(*X.T, *DY.T, color="black")
    for traj in range(f.shape[0]):
        print("traj", traj)
        T = f[traj, :, :]
        print(np.min(T), np.max(T))
        plt.plot(T[:, 0], T[:, 1])
    plt.savefig("test2.png")