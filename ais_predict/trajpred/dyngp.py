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
        dt = (df["timestamp" + l2] - df["timestamp" + l1]).dt.seconds.to_numpy()



        X[:, i, :] = np.vstack([p1_x, p1_y]).T
        Y[:, i, :] = np.vstack([p2_x - p1_x, p2_y - p1_y]).T / dt.reshape((-1, 1))
        T[:, i] = (df["timestamp" + l1] - df["timestamp"]).dt.seconds.to_numpy()
    return X.reshape((-1, 2)), Y.reshape((-1, 2)), T.flatten()

@njit(nogil=True, parallel=True)
def kernel(X1: np.ndarray, X2: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    K = np.empty((X2.shape[0], X1.shape[0]))
    for i in prange(X2.shape[0]):
        d = (X1 - X2[i]) / lengthscale
        norm_dist_squared = np.sum(d * d, axis=1)
        K[i] = np.exp(-norm_dist_squared)
    return (sigma**2) * K

@njit(nogil=True)
def dkernel(X1: np.ndarray, X2:np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    K = np.empty((X1.shape[0], X2.shape[0], X1.shape[-1]))
    for d in range(K.shape[-1]):
        K[:, :, d] = (-(X1[:, d] - X2[:, d]) * kernel(X1, X2, sigma, lengthscale).T)
    return K / (lengthscale**2)

@njit(nogil=True)
def solve_lower_triangular(L: np.ndarray, b: np.ndarray, lower: bool = True, replace_b: bool = False) -> np.ndarray:
    """ Solve linear system Lx=b using forward or backward substitution, assuming L is either lower (lower==True) or upper triangular
    """
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
def init(train_x: np.ndarray, train_delta_y: np.ndarray,  sigma: float, lengthscale: float, noise: float)->tuple:
    k = kernel(
        X1=train_x, 
        X2=train_x,
        sigma=sigma, 
        lengthscale=lengthscale
    )
    L = np.linalg.cholesky(k + (noise**2) * np.eye(k.shape[0]))
    alpha = solve_lower_triangular(L, train_delta_y)
    alpha = solve_lower_triangular(L.T, alpha, lower=False)
    return L, alpha, train_x

@njit(nogil=True)
def predict(ctx: tuple, pred_x: np.ndarray, sigma: float, lengthscale: float) -> tuple:
    """ Evaluates the function f, conditioned on training samples in ctx """
    L, alpha, train_x = ctx
    k = kernel(
        X1=pred_x, 
        X2=train_x, 
        sigma=sigma,
        lengthscale=lengthscale
    )
    mean = (k.T @ alpha)
    v = solve_lower_triangular(L, k, replace_b=True)
    var = kernel(
        X1=pred_x,
        X2=pred_x,
        sigma=sigma,
        lengthscale=lengthscale
    ) - v.T @ v
    return mean, var

@njit(nogil=True)
def predict_F(ctx: tuple, pred_x: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    _, alpha, train_x = ctx
    dk = dkernel(pred_x, train_x, sigma, lengthscale)
    G = np.empty((pred_x.shape[0], pred_x.shape[-1], pred_x.shape[-1]))
    for t in prange(dk.shape[0]):
         G[t] = dk[t].T @ alpha
    return G

#@njit(nogil=True)
def sample_f(ctx: tuple, pred_x: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    f, std = predict(
        ctx=ctx, 
        pred_x=pred_x, 
        sigma=sigma, 
        lengthscale=lengthscale
    )
    return np.random.normal(loc=f, scale=std)

#@njit(nogil=True)
def dyngp_kalman(train_x: np.ndarray, train_delta_y: np.ndarray, init_x: np.ndarray, sigma: float, lengthscale: float, noise: float, steps: int = 1000, dt: float = 1) -> np.ndarray:
    N_trajectories = init_x.shape[0]
    
    ctx = init(
        train_x=train_x, 
        train_delta_y=train_delta_y, 
        sigma=sigma, 
        lengthscale=lengthscale,
        noise=noise
    )

    x = np.empty((N_trajectories, steps) + (init_x.shape[-1],))
    var = np.empty((N_trajectories, steps, 2, 2))
    var[:, 0] = np.eye(2)
    x[:, 0] = init_x 
    dt_squared_eye = np.eye(2) * dt ** 2
    for traj in prange(N_trajectories):
        for i in range(1, steps):
            m, v = predict(
                ctx=ctx, 
                pred_x=x[traj, i-1, :].reshape((1, -1)),
                sigma=sigma,
                lengthscale=lengthscale
            )
            x[traj, i, :] = x[traj, i-1, :] + m * dt
            F = predict_F(
                ctx=ctx, 
                pred_x=x[traj, i-1, :].reshape((1, -1)),
                sigma=sigma,
                lengthscale=lengthscale
            )[0] * dt
            var[traj, i] = F.T @ var[traj, i-1] @ F + v * dt_squared_eye
    return x, var

def dyngp_particle(train_x: np.ndarray, train_delta_y: np.ndarray, init_x: np.ndarray, sigma: float, lengthscale: float, noise: float, steps: int = 1000, dt: float = 1, particles: int = 1000) -> np.ndarray:
    ctx = init(
        train_x=train_x,
        train_delta_y=train_delta_y,
        sigma=sigma,
        lengthscale=lengthscale,
        noise=noise
    )
    x = np.empty((steps, particles) + (init_x.shape[-1],))
    x[0, :] = init_x 
    for step in range(1, steps):
        for particle in prange(particles):
            x[step, particle] = x[step-1, particle] + sample_f(ctx, x[step-1, particle].reshape((1, -1)), sigma=sigma, lengthscale=lengthscale)*dt
    return x


import matplotlib.pyplot as plt
def plot_uncertianty_grid(train_x: np.ndarray, train_delta_y: np.ndarray, sigma: float, lengthscale: float, noise: float, xlim=(-1, 1), ylim=(-1,1), size=(20, 20)):
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
    ctx = init(
        train_x=train_x, 
        train_delta_y=train_delta_y, 
        sigma=sigma,
        lengthscale=lengthscale,
        noise=noise
    )
    
    x1, x2 = xlim
    y1, y2 = ylim
    sx, sy = size
    dX = x2 - x1
    dY = y2 - y1
    #x1 -= dX / (sx+1)
    x2 += dX / (sx+1)
    #y1 -= dY / (sy+1)
    y2 += dY / (sy+1)
    pX, pY = np.meshgrid(np.linspace(x1, x2, sx+1), np.linspace(y1, y2, sy+1))
    pred_x = np.vstack([pX.ravel(), pY.ravel()]).T
    mean, var = predict(ctx, pred_x, sigma=sigma, lengthscale=lengthscale)
    std = np.sqrt(var.diagonal()).reshape((sx+1, sy+1))
    plt.pcolormesh(pX, pY, std, cmap="Greys", shading="gouraud")
    #plt.quiver(*pred_x.T, *pred_f.T)
    plt.quiver(*train_x.T, *train_delta_y.T, color="red", alpha=0.1)

def plot_kernel(x: np.ndarray, sigma: float, lengthscale: float, ax=None):
    k = kernel(x, x, lengthscale=lengthscale, sigma=sigma)
    plt.imshow(k, ax=ax)


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
    #print(DY)
    X = X[:-1]
    msk = np.random.choice(X.shape[0], 20)
    X = X[msk, :]
    DY = DY[msk, :]




