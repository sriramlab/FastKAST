import os.path
from joblib import dump, load
import glob
import time
import numpy as np
from scipy.stats import qmc
# from numba_stats import norm
from scipy.stats import norm
from numba import jit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import scale
from scipy.stats import chi2
from scipy.optimize import minimize
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import traceback
from scipy.optimize import minimize
import sys
import gc
from scipy.linalg import svd
import time
from numpy.linalg import inv
import scipy
from tqdm import tqdm
from scipy.linalg import pinvh
import fastlmmclib.quadform as qf
from chi2comb import chi2comb_cdf, ChiSquared
from sklearn.linear_model import LogisticRegression
import scipy
from numpy.core.umath_tests import inner1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from FastKAST.Test.stat_test import mix_chi_fit, mix_chi_quantile, fit_null



def standerr(U, y, Sii, UTy, g, e):
    L11 = np.sum(np.square(UTy.flatten()) * np.square(Sii) / (g * Sii + e)**3)
    L22 = np.sum(np.square(y - U @ UTy).flatten() / (e)**3) + np.sum(
        np.square(UTy).flatten() / ((e + Sii * g)**3))
    L12 = np.sum((np.square(UTy.flatten()) * Sii) / (g * Sii + e)**3)
    L = 0.5 * np.array([[L11, L12], [L12, L22]])
    cov = np.linalg.inv(L)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]


def standerr_dev(U, y, Sii, UTy, g, e):
    
    if cov:
        n = len(UTy)
        assert n>10
    else:
        n = len(y)
    print(g,e,n)
    # if isinstance(cov,int):
    #     n = len(y) - cov
    # else:
    #     n = len(y)
    nulity = max(0, n - len(Sii))
    L11 = -0.5*(np.sum(np.square(Sii) / (g * Sii + e)**2))+np.sum(np.square(UTy.flatten()) * np.square(Sii) / (g * Sii + e)**3) 
    L22 = -0.5*(np.sum(1. / (g * Sii + e)**2)+nulity*1./e**2) + np.sum(np.square(UTy.flatten()) / (g * Sii + e)**3) +  np.sum(np.square((y - U @ UTy).flatten()) / e**3)
    # print((np.square(UTy.flatten())).shape)
    # print(((g * Sii + e)**3).shape)
    # print(f'L22 is {L22}')
    L12 = -0.5*np.sum(Sii/np.square(g * Sii + e))+np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3)
    # print(np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3))
    # print(-0.5*np.sum(Sii/(g * Sii + e)**2))
    L = np.array([[L11, L12], [L12, L22]])
    # print(L)
    cov = np.linalg.inv(L)
    # print(cov)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]



def standerr_dev_cov(Sii, yt, LLadd1, n, g, e):
    '''
    This is the default standard deviation method to use
    '''
    nulity = max(0, n - len(Sii))
    # print(f'nulity is {nulity}')
    # print(f'Sii: {Sii}')
    # print(f'g: {g}')
    # print(f'e: {e}')
    L11 = -0.5*(np.sum(np.square(Sii) / np.square(g * Sii + e)))+np.sum(np.square(yt.flatten()) * np.square(Sii) / (g * Sii + e)**3) 
    L22 = -0.5*(np.sum(1. / np.square(g * Sii + e))+nulity*1./e**2) + np.sum(np.square(yt.flatten()) / (g * Sii + e)**3) - np.sum(np.square(yt.flatten())/e**3) +  LLadd1 / e**3
    L12 = -0.5*np.sum(Sii / np.square(g * Sii + e))+np.sum(np.square(yt.flatten()) * Sii / (g * Sii + e)**3)
    # print(np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3))
    # print(-0.5*np.sum(Sii/(g * Sii + e)**2))
    L = np.array([[L11, L12], [L12, L22]])
    # print(f'L is:')
    # print(L)
    cov = np.linalg.inv(L)
    # print(f'Cov is:')
    # print(cov)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]


