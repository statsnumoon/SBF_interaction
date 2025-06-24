import os

__n_jobs = '1'

os.environ['MKL_NUM_THREADS'] = __n_jobs
os.environ['OPENBLAS_NUM_THREADS'] = __n_jobs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import warnings
import argparse
import pickle
import math

from datetime import timedelta
from tqdm.auto import tqdm
from scipy.stats import norm as normal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from itertools import product,repeat,combinations
from scipy.spatial import cKDTree
from scipy.integrate import trapz

"""
- pickle code
"""

def save_pickle(data,path):
    with open(path, 'wb') as file:
        pickle.dump(data,file)
        
def load_pickle(path):
    with open(path,'rb') as file:
        data = pickle.load(file)
    return data    

"""
- kernel function
"""

def K0(x):
    """
    - base kernel (epanechnikov)
    - K0 has the support [-1,1]
    """
    return 3/4*(1-x**2)*(abs(x)<1)


def intK0(x):
    """
    - exact interal of K0 from -infty to x
    """
    return 1/4*(3*x-x**3+2)*(abs(x)<1)+1.*(x>1)


def Kh(u,v,h):
    """
    - u: target points, (g,) or (q,g) (g: number of grids)
    - v: data points, (n,) or (n,q) (n: number of data, q: number of covariates)
    - h: bandwidth, scalar or (q,)
    - return elementwise result of K((u-v)/h), (g,n) or (q,g,n)
    """
    if len(u.shape)==2:
        n,q = v.shape
        k_all = []
        u,v = u[:,:,None],v[None,:,:]
        for j in range(q):
            k_all.append(K0((u[j]-v[:,:,j])/h[j])/h[j])
            
        return np.array(k_all)
    
    elif len(u.shape)==1:
        return K0((u[:,None]-v[None,:])/h)/h


def Kdenom(grids,k_all):
    """
    - grids: grid points, for numerical integral, (g,) or (q,g)
    - k_all: boundary corrected kernel Kh(xj,Xij), (g,n) or (q,g,n)
    - k_all[k,i] or k_all[j,k,i] : Kh(xj,Xij) where xj is the k-th grid
    - return numerical integral, int_0^1 Kh(u,v,h) du, (n,) or (q,n)
    """
        
    if len(v.shape)==2:
        n,q  = k_all.shape
        kdenom_all = []
        for j in range(q):
            kdenom_all.append(numerical_integral(grids[j],k_all[j]))
        
        return np.array(kdenom_all)
    
    elif len(v.shape)==1:
        return numerical_integral(grids,k_all)


def Kdenom_exact(grids,v,h):
    """
    - grids: dummy variable
    - v: data points, (n,) or (n,q)
    - h: bandwidth, scalar
    - return exact integral, int_0^1 Kh(u,v,h) du, (n,) or (q,n)
    - intK0 function is needed.
    """
    return (intK0((1-v)/h)-intK0((0-v)/h)).T

"""
- ordering code
"""

def get_order(array):
    """
    - array: 1D array-like of length l
    - return: array of integers of same length
        * order[i] = the rank (0-based) of array[i] in sorted(array)
    """
    array = np.asarray(array)
    return np.argsort(np.argsort(array))

def get_idx(array):
    """
    - array: 1D array-like
    - return:
        * res1: sorted array
        * res2: order index of each original element
    """
    array = np.asarray(array)
    res1 = np.sort(array)
    res2 = get_order(array)
    return res1, res2

"""
- rule of thumb bandwidth
"""

def h_RT(X, d, ngrid, h_scott_only=False):
    """
    - X: (n, d) array of data points, each row is an observation in [0,1]^d
    - d: number of variables (dimensions)
    - ngrid: number of grid points per dimension (for [0,1]^2 grids)
    
    - return: list of d bandwidths h = (h_1, ..., h_d)

    Steps:
    1. h1: standard univariate bandwidth estimate for each variable j based on Silverman's Rule of Thumb.
    2. h2: the maximal distance from each 2D grid point on [0,1]^2 to the nearest sample in (X_j, X_k) for any k ≠ j.
       This ensures that kernel density estimates do not vanish at any grid point.
    3. Return h_j = max(h1_j, h2_j) for each j.
    """

    h1 = np.zeros(d)
    for j in range(d):
        h1[j] = len(X[:, j])**(-1/6) * np.std(X[:, j])  
    
    if not h_scott_only:
        grid_2d = np.array(np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))).reshape(2, -1).T

        h2 = []
        for j in range(d):
            X1 = X[:, j]  
            dist_list = []  
            for k in range(d):
                if j != k:
                    X2 = X[:, k]
                    points_2d = np.vstack((X1, X2)).T
                    tree = cKDTree(points_2d)
                    nearest_dists = tree.query(grid_2d, k=1)[0] 
                    dist_list.extend(nearest_dists)
            h2.append(np.max(dist_list) + 0.01) 
    else:
        h2 = np.zeros(d)

    h = [max(h1[i], h2[i]) for i in range(d)]
    return h

"""
- interpolation code
"""

def linear_interpolation_1d(hatm_main, X):
    """
    - hatm_main: 1D array of grid values of shape (g,)
    - X: 1D array of normalized coordinates in [0,1], shape (n,)
    
    - return: 1D array of interpolated values at input locations X, shape (n,)
    """
    fine_grid_size = len(hatm_main)
    X_j = X * (fine_grid_size - 1)
    i = np.clip(X_j.astype(int), 0, fine_grid_size - 1)
    i_next = np.clip(i + 1, 0, fine_grid_size - 1)
    t = X_j - i
    f0 = hatm_main[i]
    f1 = hatm_main[i_next]
    results = (1 - t) * f0 + t * f1
    return results


def linear_interpolation_2d(hatm_int, X):
    """
    - hatm_int: 2D array of grid values over [0,1]^2, shape (g, g)
    - X: 2D array of normalized coordinates in [0,1]^2, shape (n, 2)
    
    - return: 1D array of bilinearly interpolated values at input points X, shape (n,)
    """
    fine_grid_size = hatm_int.shape[0]
    X_j = X[:, 0] * (fine_grid_size - 1)
    X_k = X[:, 1] * (fine_grid_size - 1)
    i = np.clip(X_j.astype(int), 0, fine_grid_size - 1)
    j = np.clip(X_k.astype(int), 0, fine_grid_size - 1)
    i_next = np.clip(i + 1, 0, fine_grid_size - 1)
    j_next = np.clip(j + 1, 0, fine_grid_size - 1)
    t_i = X_j - i
    t_j = X_k - j
    f00 = hatm_int[i, j]
    f01 = hatm_int[i, j_next]
    f10 = hatm_int[i_next, j]
    f11 = hatm_int[i_next, j_next]
    results = (
        (1 - t_i) * (1 - t_j) * f00 +
        (1 - t_i) * t_j * f01 +
        t_i * (1 - t_j) * f10 +
        t_i * t_j * f11
    )
    return results

"""
- bivariate function centering
"""

def convert_func_to_grid(func, ngrid):
    """
    - func: callable function f(x1, x2) supporting array broadcasting
    - ngrid: number of grid points per axis

    Returns:
    - func_grid: 2D array of shape (ngrid, ngrid), i.e., f(x1_i, x2_j)
    """
    x1 = np.linspace(0, 1, ngrid)
    x2 = np.linspace(0, 1, ngrid)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    func_grid = func(X1, X2)
    return func_grid

def inter_center(func, X, ngrid, max_iter=100):
    """
    Estimate the centered bivariate function f(x1, x2) - mean - g1(x1) - g2(x2)
    using iterative alternating projection and KDE smoothing.

    Parameters:
    - func: callable or (ngrid, ngrid) array, real-valued function of two arguments (x1, x2)
    - X: (n, 2) array of sample points
    - ngrid: number of grid
    - max_iter: maximum number of iterations for projection

    Returns:
    - result: array of centered values at Xpt_f
    - gjstar, gkstar: estimated additive components g1 and g2
    """
    n = X.shape[0]
    x = np.array([np.linspace(0,1,ngrid) for j in range(2)])
    
    # Bandwidth selection
    h = h_RT(X, 2, ngrid)

    # Evaluate function on grid
    if callable(func):
        funcval = convert_func_to_grid(func, ngrid)
    else:
        funcval = func

    # Kernel values and normalization
    kvalues = Kh(x, X, h)
    kdenom_all = Kdenom_exact(x, X, h)
    for j in range(2):
        kvalues[j] /= kdenom_all[j]

    # 1D KDEs
    kde_1d = np.zeros((2, ngrid, 1))
    for j in range(2):
        kde_j = kvalues[j] @ np.ones((n, 1)) / n
        kde_j[kde_j == 0] = 0
        integral = np.trapz(kde_j, x[j], axis=0)
        kde_1d[j] = kde_j / integral

    # 2D KDE
    kde_2d = np.zeros((2, 2, ngrid, ngrid))
    for j in range(2):
        for k in range(j + 1, 2):
            kde_jk = (kvalues[j] @ kvalues[k].T) / n
            kde_jk[kde_jk == 0] = 0
            kde_2d[j, k] = kde_jk
            kde_2d[k, j] = kde_jk.T

    hatpj = kde_1d[0].reshape(-1)
    hatpk = kde_1d[1].reshape(-1)
    hatpjk = kde_2d[0, 1]

    # Initialization
    gjstar = np.zeros((max_iter + 1, ngrid))
    gkstar = np.zeros((max_iter + 1, ngrid))

    joint_mean = np.trapz(np.trapz(funcval * hatpjk, x[0], axis=0), x[1], axis=0)
    gjstar[0] = np.trapz(funcval * hatpjk, x[1], axis=1) / (hatpj + 1e-10) - joint_mean
    gkstar[0] = np.trapz(funcval * hatpjk, x[0], axis=0) / (hatpk + 1e-10) - joint_mean

    for it in range(1, max_iter + 1):
        gjstar[it] = gjstar[0] - np.trapz(gkstar[it - 1][np.newaxis, :] * hatpjk, x[1], axis=1) / (hatpj + 1e-10)
        gkstar[it] = gkstar[0] - np.trapz(gjstar[it][:, np.newaxis] * hatpjk, x[0], axis=0) / (hatpk + 1e-10)

        gjstar[it] -= np.trapz(gjstar[it] * hatpj, x[0], axis=0)
        gkstar[it] -= np.trapz(gkstar[it] * hatpk, x[1], axis=0)

        H_norm = (
            np.trapz((gjstar[it] - gjstar[it - 1]) ** 2 * hatpj, x[0])
            + np.trapz((gkstar[it] - gkstar[it - 1]) ** 2 * hatpk, x[1])
        )

        if np.max(H_norm) < 1e-10:
            break

    # Final centering
    result = (
        funcval
        - joint_mean
        - gjstar[it][:, np.newaxis]
        - gkstar[it][np.newaxis, :]
    )


    return result, gjstar[it], gkstar[it]

