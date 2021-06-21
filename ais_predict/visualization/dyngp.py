from ais_predict.trajpred.dyngp import DynGP
from .plotting import confidence_ellipse, limits
import numpy as np
import matplotlib.pyplot as plt

def plot_uncertianty_grid(dgp, xlim, ylim, tlim, size=(20, 20, 20)):
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
        t1, t2 = tlim
        sx, sy, st = size
        dX = x2 - x1
        dY = y2 - y1
        #x1 -= dX / (sx+1)
        x2 += dX / (sx+1)
        #y1 -= dY / (sy+1)
        y2 += dY / (sy+1)
        pX, pY = np.meshgrid(np.linspace(x1, x2, sx+1),
                             np.linspace(y1, y2, sy+1))
        pred_x = np.vstack([pX.ravel(), pY.ravel()]).T
        pred_x = np.concatenate([pred_x, np.zeros_like(pred_x[:, 0][...,np.newaxis])], axis=1)
        var = np.zeros((2, pred_x.shape[0]))
        for t in np.linspace(t1, t2, st):
            pred_x[:, -1] = t
            _, v = dgp.f(pred_x)
            var += v.diagonal(axis1=1, axis2=2)
        var = var.squeeze().reshape((2, sx+1, sy+1)) / st
        var = np.linalg.norm(var, axis=0)
        plt.pcolormesh(pX, pY, var, cmap="Greys", shading="gouraud", rasterized=True)
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(r'Mean $\mathbb{V}[\vec{f}]$ over $t$', rotation=270)
       #plt.quiver(*pred_x.T, *pred_f.T)
        plt.quiver(*dgp._train_x.T[:2], *dgp._train_y.T,
                   color="blue", alpha=0.2, label=r"$\vec{f}$", rasterized=True)

def kalman_figure(dgp: DynGP, res, true, label, plot_P: bool = False, add_legend: bool = False):
    fig = plt.figure()
    x, P = res
    max_t = max(np.concatenate([dgp._train_x[:, -1], true[:, -1]]))
    xlim, ylim = limits(x[:, :2], true[:, :2], dgp._train_x[:, :2])
    plot_uncertianty_grid(dgp, xlim=xlim, ylim=ylim, tlim=(0, max_t), size=(40, 40, 20))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.scatter(*true.T[:2], color="yellow", label="Ground Truth", zorder=3)
    ix = np.argmin(abs(x[:, -1][..., np.newaxis] - true[:, -1][np.newaxis, ...]), axis=0)
    plt.plot(*(x.T[:2]), alpha=0.9, linewidth=2, label=label, color="red", zorder=1)
    if plot_P: confidence_ellipse(*(x[ix].T[:2]), P[ix], plt.gca(), linestyle="--", linewidth=1, edgecolor="r", alpha=0.5, zorder=2)
    if add_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fancybox=True, bbox_to_anchor=(0.45, -0.2))
    plt.xlabel("Distance (East) [m]")
    plt.ylabel("Distance (North) [m]")
    return fig


def kalman_state_figure(dgp: DynGP, res, true, label, add_legend: bool = False):
    train_x = dgp._train_x
    x, P = res

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    s = np.sqrt(np.diagonal(P, axis1=-2, axis2=-1)).squeeze()
    ci_high = x[:, :2] + 2 * s
    ci_low = x[:, :2] - 2 * s
    t = x[:, -1] / 60
    
    for d in range(2):
        ax[d].scatter(train_x[:, -1] / 60, train_x[:, d], alpha=0.2, label="Available Training Data", rasterized=True)
        ax[d].plot(t, x[:, d],"--", color="red", label=label)
        ax[d].fill_between(t, ci_high[:, d], ci_low[:, d], alpha=0.2, color="red")
        ax[d].plot(t, ci_high[:, d], color="red", alpha=0.3)
        ax[d].plot(t, ci_low[:, d], color="red", alpha=0.3)
        ax[d].set_xlabel("Time [Minutes]")
        ax[d].set_ylabel("Distance [m]")
        ax[d].scatter(true[:, -1] / 60, true[:, d], label="Ground Truth", color="yellow")
    ax[0].set_title("East")
    ax[1].set_title("North")
    plt.tight_layout()
    if add_legend:
        handles, labels = ax[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.1))
    return fig


