# ECSE 316 - Assignment 2
# Dina Shoham and Roey Wine

import numpy as np

# naive 1D fourier transform


def naive_ft(x):
    X = []
    N = len(arr)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    return X
