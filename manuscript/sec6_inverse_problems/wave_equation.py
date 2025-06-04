import numpy as np
import rJKOtt
import teneva

from scipy.stats import norm
import emcee

import matplotlib.pyplot as plt
import matplotlib as mpl

from geomloss import SamplesLoss
import torch

import os

from typing import List


def h(x: np.ndarray, theta: np.ndarray, k=1.0):
    if len(x.shape) < 4:
        # shape = (nsamples, t, x, theta)
        x = np.atleast_2d(x)
        theta = np.atleast_2d(theta)

        X = x[np.newaxis, :, :, np.newaxis]
        Theta = theta[:, np.newaxis, np.newaxis, :]
    else:
        X = x
        Theta = theta
    arg = np.abs(X - Theta) / k
    return np.sum(np.exp(-(arg**2)) / k, axis=-1)


def forward_wave(t: np.ndarray, x: np.ndarray, theta: np.ndarray, k=1.0):
    t = np.atleast_1d(t)
    x = np.atleast_1d(x)
    theta = np.atleast_2d(theta)

    T = t[np.newaxis, :, np.newaxis, np.newaxis]
    X = x[np.newaxis, np.newaxis, :, np.newaxis]
    Theta = theta[:, np.newaxis, np.newaxis, :]

    return (h(X + T, Theta, k=k) + h(X - T, Theta, k=k)) / 2.0

