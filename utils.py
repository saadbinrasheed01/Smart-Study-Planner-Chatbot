import numpy as np


def normalize_weights(weights):
    w = np.array(weights, dtype=float)
    w = np.maximum(w, 1e-9)
    return w / w.sum()
