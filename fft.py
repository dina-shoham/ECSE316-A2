# ECSE 316 - Assignment 2
# Dina Shoham and Roey Wine

import numpy as np
import matplotlib.pyplot as plt


# naive 1D discrete fourier transform
def naive_ft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)  # initialize result array as an array of 0s

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)

    return X


# naive 1D inverse discrete fourier transform
def naive_ift(X):
    X = np.asarray(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)  # initialize result array as an array of 0s

    for k in range(N):
        for n in range(N):
            x[n] = 1/N * X[k] * np.exp(1j * 2 * np.pi * k * n / N)

    return x


# 1D cooley-tukey fast fourier transform (divide and conquer algorithm)
# parameters are x (an array) and n_subproblems, which defines the base case for the algo (default value is 8)
def fft(x, n_subproblems=8):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    if N % 2 != 0:
        raise ValueError("array length must be a power of two")

    # base case
    elif N <= n_subproblems:
        return naive_ft(x)

    else:
        X_even = fft(x[::2])  # all elements at even indeces
        X_odd = fft(x[1::2])  # all elements at odd indeces
        coeff = np.exp(-j * 2 * np.pi * np.arange(N // 2) / N)

        X = np.concatenate(X_even + coeff * X_odd, X_even - coeff * X_odd)

        return X


# 1D cooley-tukey inverse FFT
def ifft(X, n_subproblems=8):
    X = np.asarray(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    if N % 2 != 0:
        raise ValueError("array length must be a power of two")

    # base case
    elif N <= n_subproblems:
        return naive_ift(x)

    else:
        x_even = ifft(X[::2])  # all elements at even indeces
        x_odd = ifft(X[1::2])  # all elements at odd indeces
        coeff = 1/N * np.exp(j * 2 * np.pi * np.arange(N // 2) / N)

        x = np.concatenate(x_even + coeff * x_odd, x_even - coeff * x_odd)

        return X


# 2D fft
def two_dim_fft(x):
    x = np.asarray(x, dtype=complex)
    N, M = len(x)
    X = np.zeros((N, M), dtype=complex)

    for m in range(M):
        X[:, m] = fft(x[:, m])  # fft of all elements in column m

    for n in range(N):
        X[n, :] = fft(X[n, :])  # fft of all elements in fft'ed row n

    return X


# 2D ifft
def two_dim_ifft(X):
    X = np.asarray(X, dtype=complex)
    N, M = len(x)
    x = np.zeros((N, M), dtype=complex)

    for m in range(M):
        x[:, m] = ifft(X[:, m])

    for n in range(N):
        x[n, :] = ifft(x[n, :])

    return x
