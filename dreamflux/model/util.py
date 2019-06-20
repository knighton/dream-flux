import numpy as np


def int_interpolate(begin, end, count, eps=1e-6):
    assert 2 <= count
    fracs = np.arange(count) / (count - 1)
    xx = []
    for frac in fracs:
        x = (1 - frac) * begin + frac * end
        x = int(x + eps)
        xx.append(x)
    return xx


def sharded_binary_cross_entropy(pred, true):
    return -true * pred.log() - (1 - true) * (1 - pred).log()
