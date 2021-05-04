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
def _predict(ctx: tuple, pred_x: np.ndarray, sigma: float, lengthscale: float) -> tuple:
    """ Evaluates the function f, conditioned on training samples in ctx """
    L, alpha, train_x = ctx

    pred_x = pred_x.reshape((-1, 2))
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
def _predict_F(ctx: tuple, pred_x: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    _, alpha, train_x = ctx

    pred_x = pred_x.reshape((-1, 2))
    dk = dkernel(pred_x, train_x, sigma, lengthscale)
    G = np.empty((pred_x.shape[0], pred_x.shape[-1], pred_x.shape[-1]))
    for t in prange(dk.shape[0]):
         G[t] = dk[t].T @ alpha
    return G

#@njit(nogil=True)
def sample_f(ctx: tuple, pred_x: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    f, std = _predict(
        ctx=ctx, 
        pred_x=pred_x, 
        sigma=sigma, 
        lengthscale=lengthscale
    )
    return np.random.normal(loc=f, scale=std)

def kf_log_likelihood(v: np.ndarray, S: np.ndarray) -> np.ndarray:
    v = v.T.reshape((-1, 2, 1))
    nis = v.transpose(0, 2, 1) @ np.linalg.solve(S, v) # (N x 1 x 1)
    nis = nis.flatten() # (N,)
    return - 0.5 * (nis + np.log(2.0 * np.pi * np.linalg.det(S)))

def _pdaf_update(x_hat: np.ndarray, P_hat: np.ndarray, z: np.ndarray, R: np.ndarray, clutter_rate: float, p_d: float) -> tuple:

    # Prepare constants
    logClutter = np.log(clutter_rate)
    logPD = np.log(p_d)
    logPND = np.log(1 - p_d)

    # Reshape values to follow linalg conventions
    x_hat = x_hat.reshape((2, 1)) # Column vector
    z = z.T # (2 x N)

    assert z.shape[0] == 2
    assert x_hat.shape[0] == 2
    
    ll = np.empty(z.shape[1]+1)
    vs = (z - x_hat) # (2 x N) innovation vector, one row for each measurement z
    S = P_hat + R # (2 x 2)
    ll[0] = logPND + logClutter
    ll[1:] = kf_log_likelihood(vs, S) + logPD
    beta = np.exp(ll)
    beta /= np.sum(beta)
    vk = np.sum(beta[1:].reshape((1, -1)) * vs, axis=1).reshape((2, 1))
    W = P_hat @ np.linalg.inv(S)


    # P with moment matching
    v = vs.T.reshape((-1, 2, 1)) # (N x 2 x 1)
    v = beta[1:].reshape((-1, 1, 1)) * (v @ v.transpose(0, 2, 1)) # (N x 2 x 2)
    v = np.sum(v, axis=0) - vk @ vk.T # (2 x 2)
    v = W @ v @ W.T
    P = P_hat - (1-beta[0]) * W @ S @ W + v
    # X with moment matching
    x = x_hat+ W @ vk # (2 x 1) updated state vector x
    return x.flatten(), P




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

class DynGP():
    def __init__(self, train_x: np.ndarray, train_delta_y: np.ndarray):
        self._train_x = train_x
        self._train_delta_y = train_delta_y
        self.lengthscale = 500
        self.sigma = 1000
        self.noise = 100
        self.R = 1000 * np.eye(2)
        self.clutter_rate = 1e-3
        self.p_d = 0.9

    def kalman(self, init_x: np.ndarray, steps: int = 1000, dt: float = 1, pdaf_update: bool = True) -> np.ndarray:
        assert init_x.shape[-1] == 2

        traj_shape = tuple(init_x.shape[:-1])
        
        ctx = init(
            train_x=self._train_x, 
            train_delta_y=self._train_delta_y, 
            sigma=self.sigma, 
            lengthscale=self.lengthscale,
            noise=self.noise
        )

        x = np.empty(traj_shape + (steps, init_x.shape[-1]))
        var = np.empty(traj_shape + (steps, 2, 2))
        var[:, 0] = np.eye(2)*1000
        x[:, 0] = init_x 
        dt_squared_eye = np.eye(2) * dt ** 2
        for traj in np.ndindex(traj_shape):
            for i in range(1, steps):
                # Predict
                m, v = _predict(
                    ctx=ctx, 
                    pred_x=x[traj + (i-1,)],
                    sigma=self.sigma,
                    lengthscale=self.lengthscale
                )
                x[traj, i] = x[traj + (i-1,)] + m * dt
                F = _predict_F(
                    ctx=ctx, 
                    pred_x=x[traj + (i-1,)].reshape((1, -1)),
                    sigma=self.sigma,
                    lengthscale=self.lengthscale
                )[0] * dt
                var[traj + (i,)] = F.T @ var[traj + (i-1,)] @ F + v * dt_squared_eye
                # Update
                if pdaf_update:
                    x[traj + (i,)], var[traj + (i,)] = _pdaf_update(x[traj + (i,)], var[traj + (i,)], self._train_x, self.R, p_d=self.p_d, clutter_rate=self.clutter_rate)
        return x, var
    
    def plot_uncertianty_grid(self, **kwargs):
        plot_uncertianty_grid(self._train_x, self._train_delta_y, self.sigma, self.lengthscale, self.noise, **kwargs)



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
    mean, var = _predict(ctx, pred_x, sigma=sigma, lengthscale=lengthscale)
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




