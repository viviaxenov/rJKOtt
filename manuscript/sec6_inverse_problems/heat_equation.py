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


def forward_heat(t: np.ndarray, x: np.ndarray, theta: np.ndarray, k=1.0):
    t = np.atleast_1d(t)
    x = np.atleast_1d(x)
    theta = np.atleast_2d(theta)
    n = np.array(range(theta.shape[-1]))

    Theta = theta[:, np.newaxis, np.newaxis, :]
    T = t[np.newaxis, :, np.newaxis, np.newaxis]
    X = x[np.newaxis, np.newaxis, :, np.newaxis]
    N = n[np.newaxis, np.newaxis, np.newaxis, :]

    return (np.cos(np.pi*N*X)*Theta*np.exp(-(np.pi*N)**2*T)).sum(axis=-1)

def initial_heat(x: np.ndarray, theta: np.ndarray, k=1.0):
    return forward_heat(0., x, theta, k)[:, 0, :]



