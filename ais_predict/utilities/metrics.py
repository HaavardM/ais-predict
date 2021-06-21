import numpy as np

def compare_trajectories(pred_x, true_x):
    ix = np.argmin(abs(pred_x[:, -1][...,np.newaxis] - true_x[:, -1][np.newaxis, ...]), axis=0)
    traj_err = pred_x[ix] - true_x
    return pred_x[ix], traj_err

def path_error(pred_x, true_x):
    x = pred_x
    path_err = np.empty_like(true_x)
    for i in range(true_x.shape[0]):
        err = x - true_x[i]
        ix = np.argmin(np.linalg.norm(err[:, :2], axis=1))
        x = x[ix:]
        path_err[i] = err[ix]
    return path_err

def nees(pred_x, pred_P, true_x):
    ix = np.argmin(abs(pred_x[:, -1][...,np.newaxis] - true_x[:, -1][np.newaxis, ...]), axis=0)
    err = pred_x[ix] - true_x
    e = err[:, :2][...,np.newaxis]
    return (e.transpose((0, 2, 1)) @ np.linalg.solve(pred_P[ix], e)).squeeze()