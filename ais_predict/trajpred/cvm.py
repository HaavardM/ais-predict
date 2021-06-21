import numpy as np


def constant_velocity_model(init_pos: np.ndarray, t: np.ndarray, cog: float, sog: float):
    cog = cog / 180 * np.pi
    v = np.array([np.sin(cog), np.cos(cog)]) * sog * 0.514444444
    v = v.flatten()
    p = v * t[...,np.newaxis] + init_pos
    # Add t to vector to follow convention
    return np.concatenate([p, t[...,np.newaxis]], axis=1)